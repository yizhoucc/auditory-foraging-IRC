import os, warnings, json, time, random, pickle
import numpy as np
import torch
from typing import Optional, Type
from collections.abc import Iterable
from scipy.special import logsumexp
from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from jarvis import BaseJob
from jarvis.hashable import to_hashable
from jarvis.utils import flatten, nest, fill_defaults, time_str, progress_str, numpy_dict, tensor_dict

from .distributions import BaseDistribution
from .models import BeliefModel
from .utils import Array, GymEnv, SB3Policy, SB3Algo
from .defaults import EST_SPEC, ALGO_KWARGS, POLICY_KWARGS, LEARN_KWARGS, WARNING_OPTIMALITY, EVAL_KWARGS


class BeliefAgent:
    r"""An agent with an internal model and uses belief for decision."""

    def __init__(self,
        model: BeliefModel,
        algo: SB3Algo,
        gamma: float = 0.99,
    ):
        r"""
        Args
        ----
        model:
            The internal model of assumed environment. Belief about the states
            is updated by it.
        algo:
            The reinfocement learning algorithm for learning the policy based on
            belief.
        gamma:
            Reward discount factor. It does not have to be the same one in `algo`
            for calculating advantages.

        """
        self.model = model
        self.algo = algo
        self.gamma = gamma


    def state_dict(self):
        r"""Returns state dictionary."""
        return {
            'model_state': self.model.state_dict(),
            'policy_state': self.algo.policy.state_dict(),
        }

    def load_state_dict(self, state):
        r"""Loads state dictionary."""
        self.model.load_state_dict(state['model_state'])
        self.algo.policy.load_state_dict(state['policy_state'])

    def _return(self, rewards):
        r"""Returns cumulative discounted reward."""
        w = self.gamma**np.flip(np.arange(len(rewards)))
        g = (w*rewards).sum()
        return g

    def evaluate(self, num_episodes=10, num_steps=40):
        r"""Evaluates current policy with respect to internal model.

        Args
        ----
        num_episodes:
            Number of evaluation episodes.

        Returns
        -------
        eval_record: dict
            Evaluation record, used for keeping track of training progress.

        """
        returns, optimalities = [], []
        for _ in range(num_episodes):
            episode = self.run_one_episode(num_steps=num_steps)
            rewards = episode['rewards']
            returns.append(self._return(rewards))
            optimalities.append(np.nanmean(episode['optimalities']))
        eval_record = {
            'num_episodes': num_episodes,
            'num_steps': num_steps,
            'returns': returns,
            'optimalities': optimalities,
            }
        return eval_record

    def run_one_episode(self,
        env: Optional[GymEnv] = None,
        num_steps: int = 40,
    ):
        r"""Runs one episode.

        Args
        ----
        env:
            The environment to interact with.
        num_steps:
            Maximum number of time steps of each episode.

        Returns
        -------
        episode: dict
            Results of one episode.

        """
        _to_restore_train = self.algo.policy.training # policy will be set to evaluation mode temporarily
        self.algo.policy.set_training_mode(False)
        actions, rewards, states, obss, beliefs = [], [], [], [], []
        try:
            q_states = np.array(self.model.env.query_states())
            q_states = []
            q_probs = []
        except:
            q_probs = None
        optimalities, fvus = [], []

        belief, info = self.model.reset(return_info=True)
        states.append(info['state'])
        obss.append(info['obs'])
        beliefs.append(belief)
        if q_probs is not None:
            q_states.append(np.array(self.model.env.query_states()))
            q_probs.append(self.query_probs(belief, q_states[-1]))
        t = 0
        while True:
            action, _ = self.algo.predict(belief)
            actions.append(action)
            belief, reward, done, info = self.model.step(action, env)
            rewards.append(reward)
            states.append(info['state'])
            obss.append(info['obs'])
            beliefs.append(belief)
            if q_probs is not None:
                q_states.append(np.array(self.model.env.query_states()))
                q_probs.append(self.query_probs(belief, q_states[-1]))
            optimalities.append(self.model.p_s.est_stats['optimality'])
            fvus.append(self.model.p_s.est_stats['fvu'])
            t += 1
            if done or t==num_steps:
                break
        episode = {
            'num_steps': t,
            'actions': np.array(actions), # [0, t)
            'rewards': np.array(rewards), # [0, t)
            'states': np.array(states), # [0, t]
            'obss': np.array(obss), # [0, t]
            'beliefs': np.array(beliefs), # [0, t]
            'optimalities': np.array(optimalities),
            'fvus': np.array(fvus),
        }
        if q_probs is not None:
            episode['q_states'] = np.array(q_states)
            episode['q_probs'] = np.array(q_probs)
        self.algo.policy.set_training_mode(_to_restore_train)
        return episode

    def query_probs(self, belief: Array, states: Array):
        r"""Returns probabilities of queried states.

        Args
        ----
        belief: (num_vars_belief)
            Parameters of state distribution.
        states: (num_samples, num_vars_state)
            Queried states.

        Returns
        -------
        probs: (num_samples,) Array
            Probability mass/density values of queried states.

        """
        device = self.model.p_s.get_param_vec().device
        self.model.p_s.set_param_vec(torch.tensor(belief, device=device))
        with torch.no_grad():
            probs = np.exp(self.model.p_s.loglikelihood(states).cpu().numpy())
        return probs

    def episode_likelihood(self,
        actions: Array,
        obss: Array,
        num_repeats: int = 4,
    ):
        r"""Returns the likelihood of given episode.

        Args
        ----
        actions:
            Actions taken by the agent, in [0, t).
        obss:
            Observations to the agent, in [0, t]. The last one will not be used.
        num_repeats:
            The number of sampled belief trajectories.

        Returns
        -------
        logps: (num_repeats,) Array
            Log likelihood p(actions, obss|model) for each sampled belief
            trajectory.

        """
        num_steps = len(actions)
        logps = np.zeros((num_repeats, num_steps))
        device = self.model.p_s.get_param_vec().device
        self.algo.policy.eval().to(device)
        for i in range(num_repeats):
            for t in range(len(actions)):
                obs = obss[t]
                if t==0:
                    with torch.no_grad():
                        self.model.p_s.set_param_vec(
                            self.model.p_s_o.param_net(np.array(obs)[None])[0]
                        )
                else:
                    self.model.update_belief(actions[t-1], obs)
                belief = self.model.p_s.get_param_vec()
                pi = self.algo.policy.get_distribution(belief[None].to(device))
                logps[i, t] = pi.log_prob(torch.tensor(actions[t], dtype=torch.long, device=device)).item()
        return logps.sum(axis=1)


class BeliefAgentFamily(BaseJob):
    r"""A family of belief agents.

    The class uses a folder for saving training checkpoints. Batch processing
    over a set of assumed environment parameters is implemented.

    """

    def __init__(self,
        env_class: Type[GymEnv],
        store_dir: str = 'cache',
        *,
        env_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        state_dist_class: Optional[Type[BaseDistribution]] = None,
        state_dist_kwargs: Optional[dict] = None,
        obs_dist_class: Optional[Type[BaseDistribution]] = None,
        obs_dist_kwargs: Optional[dict] = None,
        est_spec: Optional[dict] = None,
        algo_class: Optional[Type[SB3Algo]] = None,
        algo_kwargs: Optional[dict] = None,
        policy_class: Optional[Type[SB3Policy]] = None,
        policy_kwargs: Optional[dict] = None,
        learn_kwargs: Optional[dict] = None,
        eval_kwargs: Optional[dict] = None,
        eval_interval: int = 5,
        save_interval: int = 10,
        device: str = 'cuda',
        save_kwargs: bool = False,
        **kwargs,
    ):
        super(BeliefAgentFamily, self).__init__(store_dir, **kwargs)
        json_path = f'{self.store_dir}/bafam_kwargs.json'
        bafam_kwargs = {'env_class': str(env_class)}
        if os.path.exists(json_path): # load default keyword arguments
            with open(json_path, 'r') as f:
                _bafam_kwargs = json.load(f)
            if _bafam_kwargs['env_class']==bafam_kwargs['env_class']:
                print(
                    f"Default keyword arguments detected in {json_path}, and will be used as if not "
                    "specified explicitly."
                )
                bafam_kwargs = _bafam_kwargs
            else:
                print(
                    f"Keyword arguments saved in {json_path} will not be used because the saved "
                    "environment class is different from the current one."
                )
        else:
            save_kwargs = True

        self.env_class = env_class
        self.env_kwargs = fill_defaults(env_kwargs or {}, bafam_kwargs.get('env_kwargs', {}))
        self.model_kwargs = model_kwargs or {}
        self.model_kwargs['state_dist_class'] = state_dist_class
        self.model_kwargs['state_dist_kwargs'] = fill_defaults(
            state_dist_kwargs or {}, bafam_kwargs.get('state_dist_kwargs', {}),
        )
        self.model_kwargs['obs_dist_class'] = obs_dist_class
        self.model_kwargs['obs_dist_kwargs'] = fill_defaults(
            obs_dist_kwargs or {}, bafam_kwargs.get('obs_dist_kwargs', {}),
        )
        self.model_kwargs['est_spec'] = fill_defaults(
            est_spec or {}, bafam_kwargs.get('est_spec', EST_SPEC),
        )
        self.algo_class = algo_class or PPO
        self.algo_kwargs = fill_defaults(
            algo_kwargs or {}, bafam_kwargs.get('algo_kwargs', ALGO_KWARGS),
        )
        self.policy_class = policy_class or ActorCriticPolicy
        self.policy_kwargs = fill_defaults(
            policy_kwargs or {}, bafam_kwargs.get('policy_kwargs', POLICY_KWARGS),
        )
        self.learn_kwargs = fill_defaults(
            learn_kwargs or {}, bafam_kwargs.get('learn_kwargs', LEARN_KWARGS),
        )
        self.eval_kwargs = fill_defaults(
            eval_kwargs or {}, bafam_kwargs.get('eval_kwargs', EVAL_KWARGS),
        )

        if save_kwargs:
            bafam_kwargs.update({
                'env_kwargs': self.env_kwargs,
                'state_dist_kwargs': self.model_kwargs['state_dist_kwargs'],
                'obs_dist_kwargs': self.model_kwargs['obs_dist_kwargs'],
                'est_spec': self.model_kwargs['est_spec'],
                'algo_kwargs': self.algo_kwargs,
                'policy_kwargs': self.policy_kwargs,
                'learn_kwargs': self.learn_kwargs,
                'eval_kwargs': self.eval_kwargs,
            })
            try:
                to_hashable(bafam_kwargs)
            except:
                raise RuntimeError("Keyword arguments for bafam is unhashable, and cannot be saved.")
            if os.path.exists(json_path):
                to_backup = True
                try:
                    if to_hashable(bafam_kwargs)==to_hashable(_bafam_kwargs):
                        to_backup = False
                except: # kwargs not hashable
                    pass
                if to_backup:
                    backup_path = json_path[:-4]+'bak'
                    print(f"Old keyword arguments are backed up to {backup_path}.")
                    with open(backup_path, 'w') as f:
                        json.dump(_bafam_kwargs, f)
            print(f"Current keyword arguments are saved in {json_path}.")
            with open(json_path, 'w') as f:
                json.dump(bafam_kwargs, f)

        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.catalog_path = f'{self.store_dir}/class_catalog.pickle'
        self._register()

        env = self.env_class(**self.env_kwargs)
        self.param_dim = len(env.get_env_param())
        print(f"Belief agent family initialized, with environment parameters of dimension {self.param_dim}.")

    def _register(self):
        r"""Registers base configuration and class objects."""
        self._config = flatten({
            'env_class': self.env_class,
            'env_kwargs': self.env_kwargs,
            'model_kwargs': self.model_kwargs,
            'algo_class': self.algo_class,
            'algo_kwargs': self.algo_kwargs,
            'learn_kwargs': self.learn_kwargs,
        })
        try:
            with open(self.catalog_path, 'rb') as f:
                self.catalog = pickle.load(f)
        except:
            self.catalog = {}
        updated = False
        for c_key, c_val in self._config.items():
            c_str = str(c_val)
            if c_str.startswith('<class '):
                if c_str not in self.catalog:
                    updated = True
                self.catalog[c_str] = c_val
                self._config[c_key] = c_str
        self._config = nest(self._config)
        if updated:
            with open(self.catalog_path, 'wb') as f:
                pickle.dump(self.catalog, f)

    def _raw_config(self, config):
        r"""Returns the raw configuration with class objects as values."""
        config = flatten(config)
        for key, val in config.items():
            if isinstance(val, str) and val in self.catalog:
                config[key] = self.catalog[val]
        return nest(config)

    def default_env_param(self):
        env_param = self.env_class(**self.env_kwargs).get_env_param()
        return env_param

    def create_agent(self, config):
        r"""Creates a new agent."""
        config = self._raw_config(config)
        env = config['env_class'](**config['env_kwargs'])
        try:
            env.seed(config['seed'])
        except:
            pass # seeding is not implemented
        env.set_env_param(config['env_param'])
        model = BeliefModel(
            env=env, device=self.device, rng=config['seed'],
            **config['model_kwargs'],
        )
        algo = config['algo_class'](
            env=model, policy=self.policy_class, policy_kwargs=self.policy_kwargs,
            device=self.device, seed=config['seed'],
            **config['algo_kwargs'],
        )
        agent = BeliefAgent(model, algo, gamma=config['algo_kwargs']['gamma'])
        return agent

    def main(self, config, num_epochs, verbose=1):
        if 'episode_path' in config:
            ckpt, preview = self._compute_logp(config, num_epochs, verbose)
        else:
            ckpt, preview = self._train_agent(config, num_epochs, verbose)
        return ckpt, preview

    def _train_agent(self, config, num_epochs=40, verbose=1):
        r"""Trains an agent."""
        agent = self.create_agent(config)
        if verbose>0:
            print("Belief agent (seed {}) initialized for environment parameter:".format(config['seed']))
            print("({})".format(', '.join(['{:g}'.format(p) for p in config['env_param']])))

        def _evaluate(epoch):
            tic = time.time()
            eval_record = agent.evaluate(**self.eval_kwargs)
            toc = time.time()
            if verbose>0:
                print("Epoch {}".format(progress_str(epoch, num_epochs)))
                print("Episode return {:.3f} ({:.2f}), evaluation time {}".format(
                    np.mean(eval_record['returns']),
                    np.std(eval_record['returns']),
                    time_str(toc-tic),
                ))
                print("Belief update optimality {:.1%}".format(
                    np.mean(eval_record['optimalities']),
                ))
            return eval_record
        def _get_preview(ckpt):
            preview = {
                'p_s_o_optimality': ckpt['p_s_o_est_stats']['optimality'],
                'p_o_s_optimality': ckpt['p_o_s_est_stats']['optimality'],
                'epochs': [], 'r_means': [], 'r_sems': [], 'belief_optimalities': [],
            }
            for epoch in sorted(ckpt['eval_records'].keys()):
                eval_record = ckpt['eval_records'][epoch]
                preview['epochs'].append(epoch)
                preview['r_means'].append(np.mean(eval_record['returns']))
                preview['r_sems'].append(np.std(eval_record['returns'])/len(eval_record['returns'])**0.5)
                preview['belief_optimalities'].append(np.mean(eval_record['optimalities']))
            for key in ['epochs', 'r_means', 'r_sems', 'belief_optimalities']:
                preview[key] = np.array(preview[key])
            return preview

        try:
            epoch, ckpt, preview = self.load_ckpt(config)
            agent.load_state_dict(tensor_dict(ckpt['agent_state']))
            if verbose>0:
                print(f"Checkpoint ({epoch}) loaded.")
        except:
            epoch = 0
            tic = time.time()
            _, info = agent.model.reset(return_info=True)
            toc = time.time()
            if verbose>0:
                print("Initial state distribution estimation optimality {:.1%}".format(
                    info['p_s_o_est_stats']['optimality'],
                ))
                if info['p_s_o_est_stats']['optimality']<WARNING_OPTIMALITY:
                    warnings.warn(
                        f"The estimation of initial state distribution p(s|o) is poor, please "
                        "consider increase 'num_samples' in est_spec['state_prior'], or 'num_epochs' "
                        "in est_spec['state_prior']['optim_kwargs']."
                    )
                print("Conditional observation distribution estimation optimality {:.1%}".format(
                    info['p_o_s_est_stats']['optimality'],
                ))
                if info['p_s_o_est_stats']['optimality']<WARNING_OPTIMALITY:
                    warnings.warn(
                        f"The estimation of conditional observation distribution p(o|s) is poor, "
                        "please consider increase 'num_samples' in est_spec['obs_conditional'], or "
                        "'num_epochs' in est_spec['obs_conditional']['optim_kwargs']."
                    )
                print("{} elapsed.".format(time_str(toc-tic)))
            ckpt = {
                'p_s_o_est_stats': info['p_s_o_est_stats'],
                'p_o_s_est_stats': info['p_o_s_est_stats'],
                'eval_records': {0: _evaluate(0)},
                'agent_state': numpy_dict(agent.state_dict()),
            }
            preview = _get_preview(ckpt)
            self.save_ckpt(config, epoch, ckpt, preview)
            if np.mean(ckpt['eval_records'][0]['optimalities'])<WARNING_OPTIMALITY:
                warnings.warn(
                    f"The estimation of new belief is poor, please consider increase 'num_samples' "
                    "in est_spec['belief'], or 'num_epochs' in est_spec['belief']['optim_kwargs']."
                )
            if verbose>0:
                print(f"Initial checkpoint saved.")
        t_train, count = 0., 0
        while epoch<num_epochs:
            tic = time.time()
            agent.algo.learn(**config['learn_kwargs'], log_interval=None, reset_num_timesteps=False)
            toc = time.time()
            t_train += toc-tic
            count += 1

            epoch += 1
            if epoch%self.eval_interval==0 or epoch==num_epochs:
                ckpt['eval_records'][epoch] = _evaluate(epoch)
                ckpt['agent_state'] = numpy_dict(agent.state_dict())
                preview = _get_preview(ckpt)
                if verbose>0 and count>0:
                    print('Average training time {}/epoch.'.format(time_str(t_train/count)))
                    t_train, count = 0., 0
            if epoch%self.save_interval==0:
                self.save_ckpt(config, epoch, ckpt, preview)
                if verbose>0:
                    print(f"Checkpoint {epoch} saved.")
        return ckpt, preview

    def _compute_logp(self, config, num_epochs, verbose):
        try:
            epoch, ckpt, preview = self.load_ckpt(config)
        except:
            epoch = 0
        if epoch<num_epochs:
            _config = dict( # create new config for agent training, `config` is not modified
                (key, val) for key, val in config.items()
                if key not in ['episode_path', 'num_repeats']
            )
            self._train_agent(_config, num_epochs, verbose)
            agent = self.create_agent(_config)
            _, ckpt, _ = self.load_ckpt(_config)
            agent.load_state_dict(tensor_dict(ckpt['agent_state']))
            if verbose>0:
                print(f"Agent trained for at least {num_epochs} epochs loaded.")

            episode_path = config['episode_path']
            num_repeats = config['num_repeats']
            with open(f'{self.store_dir}/{episode_path}', 'rb') as f:
                episode = pickle.load(f)['episode']
            actions = episode['actions']
            obss = episode['obss']
            if verbose>0:
                print("Data loaded.")
            logps = agent.episode_likelihood(actions, obss, num_repeats)
            if verbose>0:
                print("Log likelihoods {:.2f} of episode (length {}) is calculated for {} belief sequences.".format(
                    logsumexp(logps)-np.log(num_repeats), len(actions), num_repeats,
                ))

            ckpt, preview = {'logps': logps}, {}
            self.save_ckpt(config, num_epochs, ckpt, preview)
        return ckpt, preview

    def to_config(self, env_param, seed=0, **kwargs):
        r"""Converts environment parameter to configuration."""
        return to_hashable(dict(env_param=env_param, seed=seed, **self._config, **kwargs))

    def _random_configs(self, env_params, seeds, **kwargs):
        for env_param in env_params:
            for seed in random.sample(seeds, len(seeds)):
                yield self.to_config(env_param, seed, **kwargs)

    def _random_configs(self, env_params, seeds, **kwargs):
        for env_param in env_params:
            for seed in random.sample(seeds, len(seeds)):
                yield self.to_config(env_param, seed, **kwargs)

    def train_agents(self,
        env_params: Iterable[Array],
        seeds: Optional[Iterable[int]] = None,
        num_epochs: int = 40,
        verbose: int = 1,
        **kwargs,
    ):
        r"""Train agents for a list of environment parameters."""
        if seeds is None:
            seeds = [0]
            if verbose>0:
                print(f"Use default seeds {seeds} for each env_param.")
        self.batch(self._random_configs(env_params, seeds), num_epochs=num_epochs, verbose=verbose, **kwargs)

    def train_agents_on_param_grid(self,
        param_grid: list[list[float]],
        max_seed: int = 1,
        num_epochs: int = 40,
        verbose: int = 1,
        **kwargs,
    ):
        assert len(param_grid)==self.param_dim, (
            "The length of parameter grid should be environtment pameter dimension."
        )
        grid_dims = [len(v) for v in param_grid]
        num_params = np.prod(grid_dims)
        if verbose>0:
            print(f"{num_params} different environment parameters specified by the parameter grid.")

        def idx2param(idx):
            sub_idxs = np.unravel_index(idx, grid_dims)
            env_param = tuple(val_list[sub_idx] for sub_idx, val_list in zip(sub_idxs, param_grid))
            return env_param
        def env_param_generator():
            for idx in random.sample(range(num_params), num_params):
                env_param = idx2param(idx)
                yield env_param

        self.train_agents(
            env_param_generator(), range(max_seed),
            num_epochs=num_epochs, verbose=verbose, **kwargs,
        )

    def compute_logps(self,
        env_params: Iterable[Array],
        episode_path: str,
        num_repeats: int = 8,
        seeds: Optional[Iterable[int]] = None,
        **kwargs,
    ):
        if seeds is None:
            seeds = [0]
        self.batch(
            self._random_configs(
                env_params, seeds, episode_path=episode_path, num_repeats=num_repeats,
            ), **kwargs,
        )

    def optimal_agent(self,
        env_param: Array,
        seed: int = 0,
        num_epochs: int = 10,
        verbose: int = 1,
    ):
        r"""Returns the optimal agent of given environment parameter."""
        self.train_agents([env_param], [seed], num_epochs, patience=0, verbose=verbose)
        config = self.to_config(env_param, seed=seed)
        _, ckpt, preview = self.load_ckpt(config)
        agent = self.create_agent(config)
        agent.load_state_dict(tensor_dict(ckpt['agent_state']))
        return agent, preview

    def episode_likelihood(self,
        env_param: Array,
        episode_path: str,
        num_repeats: int = 8,
        seeds: Optional[Iterable[int]] = None,
        num_epochs: int = 40,
    ):
        if seeds is None:
            seeds = [0]
        logps = []
        for seed in seeds:
            config = self.to_config(
                env_param=env_param, seed=seed,
                episode_path=episode_path, num_repeats=num_repeats,
            )
            #Lokesh changed next 3 lines, so check
            logps_dict, _ = self._compute_logp(config, num_epochs=num_epochs, verbose=0)
            logps.append(logps_dict['logps'])
            # logps.append(self._compute_logp(config, num_epochs=num_epochs, verbose=0))
        logps = np.array(logps)
        return logps
