import warnings, time, yaml, pickle, tarfile, random
from pathlib import Path
import numpy as np
import torch
from collections.abc import Iterable
import itertools
from typing import Optional, Union

from jarvis import Config, Archive, Manager
from jarvis.config import _locate
from jarvis.utils import time_str

from .agent import BeliefAgent
from .utils import check_env, plot_agent_checkpoint, _exp_fit
from .alias import Array, Figure

defaults_dir = Path(__file__).parent/'defaults'
with open(defaults_dir/'manager.yaml') as f:
    MANAGER_CONFIG = Config(yaml.safe_load(f))


class AgentManager(Manager):
    r"""Manager for training rational agents."""

    def __init__(self,
        store_dir: str,
        defaults: Optional[dict] = None,
        **kwargs,
    ):
        runtime_config = Config(kwargs).fill(MANAGER_CONFIG.runtime.agent)
        device = runtime_config.pop('device')
        self.device = device if torch.cuda.is_available() else 'cpu'
        if self.device.startswith('cuda'):
            print(f"Using GPU device '{self.device}'.")
        self.eval_config = runtime_config.pop('eval_config')
        self.to_check_env = runtime_config.pop('to_check_env')
        super(AgentManager, self).__init__(store_dir, **runtime_config)

        self.defaults = defaults

    @staticmethod
    def _to_use_sample_model(config: Config):
        return config.model._target_=='irc.model.SamplingBeliefModel'

    def get_config(self, config=None):
        config = super(AgentManager, self).get_config(config)
        assert config.env._target_ is not None, "Environment class name needs to be specified."
        if self.to_check_env or config.env_param is None:
            _env = config.env.instantiate()
        # check environment is valid
        if self.to_check_env:
            check_env(_env)
        # assign default environment parameter
        if config.env_param is None:
            config.env_param = _env.get_param()
            if self.verbose>0:
                print(f"Using default environment parameter {config.env_param}.")
        # assign default agent seed
        if config.seed is None:
            config.seed = self.defaults.seed
        # assing default argument for sampling-based belief model
        if self._to_use_sample_model(config):
            config.model = config.model.fill(MANAGER_CONFIG.sampling_belief_model)
        return config

    def setup(self, config):
        super(AgentManager, self).setup(config)
        env = config.env.instantiate()
        env.set_param(config.env_param)
        model = config.model.instantiate(
            env, device=self.device, rng=config.seed,
        )
        policy = config.policy.pop('_target_')
        try:
            policy = _locate(policy)
        except:
            pass
        
        


        # Lokesh made this change for high entropy case - this is to add other hyperparameters to training.
        # algo = config.algo.instantiate(
        #     policy=policy, env=model, policy_kwargs=config.policy,
        #     device=self.device, seed=config.seed,
        # )
        algo = config.algo.instantiate(
            policy=policy, env=model, policy_kwargs=config.policy,
            device=self.device, seed=config.seed, gamma=config.algo.gamma, ent_coef=config.algo.ent_coef, learning_rate=config.algo.learning_rate
        )
        

        
        
        
        self.agent = BeliefAgent(model, algo)

    def init_ckpt(self):
        super(AgentManager, self).init_ckpt()
        if self.verbose>0:
            print(
                "Initializing an agent with internal environment parameter:"
                f"\n{tuple(self.agent.model.env.get_param())}"
            )
        tic = time.time()
        self.agent.model.reset()
        toc = time.time()
        if self._to_use_sample_model(self.config):
            self.ckpt['est_stats'] = {
                'p_s_o': self.agent.model.p_s_o.est_stats,
                'p_o_s': self.agent.model.p_o_s.est_stats,
            }
            if self.verbose>0:
                for tag in ['p_s_o', 'p_o_s']:
                    est_stats = self.ckpt['est_stats'][tag]
                    if tag=='p_s_o':
                        texts = [
                            "Initial state distribution",
                            "initial state distribution p(s|o)",
                            "model.estimate.p_s_o.num_samples",
                            "model.estimate.p_s_o.num_epochs",
                        ]
                    if tag=='p_o_s':
                        texts = [
                            "Conditional observation distribution",
                            "conditional observation distribution p(o|s)",
                            "model.estimate.p_o_s.num_samples",
                            "model.estimate.p_o_s.num_epochs",
                        ]
                    print("{} distribution estimation optimality {:.1%}".format(
                        texts[0], est_stats['optimality'],
                    ))
                    if est_stats['optimality']<MANAGER_CONFIG.warning.optimality:
                        num_samples = est_stats['num_samples']
                        num_epochs = est_stats['num_epochs']
                        warnings.warn(
                            f"The estimation of {texts[1]} is poor, please consider increasing "
                            f"'num_samples' (currently {num_samples}) in {texts[2]}, or 'num_epochs' "
                            f"(currently {num_epochs}) in {texts[3]}."
                        )
                print(f"Agent internal model is reset ({time_str(toc-tic)}).")

    def save_ckpt(self):
        self.ckpt['agent_state'] = self.agent.state_dict()
        if self._to_use_sample_model(self.config):
            self.ckpt['est_stats']['p_s'] = self.agent.model.running_stats

        eval_records = self.ckpt['eval_records']
        epochs = np.array(sorted(eval_records.keys()))
        if len(epochs)<4:
            optimality = fvu = np.nan
        else:
            r_means = np.array([np.mean(eval_records[e]['returns']) for e in epochs])
            _, optimality, fvu = _exp_fit(epochs, r_means)
        self.preview = {
            'optimality': optimality, 'fvu': fvu,
        }
        super(AgentManager, self).save_ckpt()

    def load_ckpt(self):
        super(AgentManager, self).load_ckpt()
        self.agent.load_state_dict(self.ckpt['agent_state'])

    def train(self):
        r"""Runs RL algorithm for one epoch.

        This method is only called in agent training mode. Each training epoch
        contains a fixed number of environment interactions specified by
        `config.learn.total_timesteps`.

        """
        tic = time.time()
        self.agent.algo.learn(
            **self.config.learn, log_interval=None, reset_num_timesteps=False,
        )
        toc = time.time()
        if self.verbose>0:
            print(
                f"Agent trained by {self.config.learn.total_timesteps} time steps "
                f"({time_str(toc-tic)})."
            )

    def eval(self):
        tic = time.time()
        eval_record = self.agent.evaluate(**self.eval_config)
        toc = time.time()
        if self.verbose>0:
            print(
                f"Agent evaluated for {self.eval_config.num_episodes} episodes of length "
                f"{self.eval_config.num_steps}. Average return "
                f"{'{:.2f}'.format(np.mean(eval_record['returns']))}. ({time_str(toc-tic)})"
            )
        self.ckpt['eval_records'][self.epoch] = eval_record

    def _to_choices(self,
        env_param_list: Optional[Iterable[list[float]]] = None,
        env_param_grid: Optional[list[Iterable[float]]] = None,
        seeds: Optional[Iterable[int]] = None,
    ) -> dict:
        r"""Returns choices option for sweep method.

        Args
        ----
        env_param_list:
            A list of environment parameters to sweep over. The order is random.
        env_param_grid:
            A list of value choices for each environment parameter dimension. A
            parameter list will be constructed by taking outer product of the
            choices. Only one of `env_param_list` and `env_param_grid` can be
            specified.
        seeds:
            A list of random seeds for different agent. For example, if the user
            needs 6 agents trained for each environment parameter, the method
            takes `seeds=range(6)` as input.

        Returns
        -------
        choices:
            A dictionary with `'env_param'` and `'seed'` keys, used for `sweep`
            method.

        """
        assert (env_param_list is None)!=(env_param_grid is None), (
            "One and only one of 'env_param_list' and 'env_param_grid' needs to be specified."
        )
        if env_param_list is None:
            env_param_list = itertools.product(*env_param_grid)
        if seeds is None:
            seeds = MANAGER_CONFIG.sweep.agent_seeds
            if self.verbose>0:
                print(f"Use default agent seeds {seeds} for each environment parameter.")
        choices = {
            'env_param': [tuple(env_param) for env_param in env_param_list],
            'seed': list(seeds),
        }
        return choices


class LikelihoodManager(Manager):
    r"""Manager for computing episode likelihoods.

    This is an evaluation-only work manager, therefore method `train` is not
    implemented. Both `process` or `sweep` should be called with
    `num_epochs = 0`.

    """

    def __init__(self,
        store_dir: str,
        defaults: Optional[dict] = None,
        **kwargs,
    ):
        runtime_config = Config(kwargs).fill(MANAGER_CONFIG.runtime.likelihood)
        super(LikelihoodManager, self).__init__(store_dir, **runtime_config)
        self.defaults = defaults

    def get_config(self, config=None):
        config = super(LikelihoodManager, self).get_config(config)
        assert config.agent_key in self.agent_manager.configs
        # fetch the latest trained_epochs
        config.trained_epochs = self.agent_manager.stats[config.agent_key]['epoch']
        assert config.episode_key in self.ep_traces
        return config

    def setup(self, config):
        super(LikelihoodManager, self).setup(config)
        self.agent_manager.setup(self.agent_manager.configs[config.agent_key])
        self.agent_manager.load_ckpt()
        self.agent: BeliefAgent = self.agent_manager.agent
        self.episode: dict = self.ep_traces[config.episode_key].native()

    def init_ckpt(self):
        super(LikelihoodManager, self).init_ckpt()
        self.ckpt = {}
        if self.verbose>0:
            env_param = self.agent.model.env.get_param()
            env_param = tuple(np.array(env_param).astype(float).flatten())
            print(f"Agent environment parameter:\n{tuple(self.agent.model.env.get_param())}")

    def eval(self):
        tic = time.time()
        logp = self.agent.episode_likelihood(
            self.episode['actions'], self.episode['observations'],
            seed=self.config.seed,
        )
        self.ckpt = self.preview = {'logp': logp}
        toc = time.time()
        if self.verbose>0:
            print(
                "Log likelihood of an agent (trained for {:2d} epochs): {:.3f} ({})".format(
                    self.config.trained_epochs, logp, time_str(toc-tic),
                )
            )

    def prune(self):
        # TODO remove likelihood computations of outdated agents
        super(LikelihoodManager, self).prune()


class IRCManager:
    r"""Main class of IRC manager.

    The manager contains an agent manager and a likelihood manager. Additional
    `Archive` objects are initialized to save episode traces and meta
    information.

    """

    def __init__(self,
        defaults: Union[dict, str, None] = None,
        **kwargs,
    ):
        r"""
        Args
        ----
        defaults:
            A default template of configuration. `defaults.agent.env._target_`
            must be specified in order to train agents or compute likelihoods.
            Keys not specified by the user will be filled with default values
            from the file 'defaults/manager.yaml'. If `defaults` is a string,
            it can refer to a yaml file with the same structure as `defaults`
            in 'defaults/manager.yaml'.
        kwargs:
            Additional runtime configuration for `agent_manager` and
            `likelihood_manager`. Default values are specified in
            'defaults/manager.yaml'. More details are described in `Manager`
            documentation.

        """
        if isinstance(defaults, str):
            with open(defaults) as f:
                defaults = yaml.safe_load(f)
        defaults = Config(defaults).fill(MANAGER_CONFIG.defaults)
        runtime_config = Config(kwargs).fill(MANAGER_CONFIG.runtime)
        self.store_dir = runtime_config.pop('store_dir')
        self.verbose = runtime_config.pop('verbose')

        # initialize agent manager
        self.agent_manager = AgentManager(
            f'{self.store_dir}/agents', defaults.agent,
            verbose=self.verbose, **runtime_config.agent,
        )
        # initialize likelihood manager
        self.likelihood_manager = LikelihoodManager(
            f'{self.store_dir}/likelihoods', defaults.likelihood,
            verbose=self.verbose, **runtime_config.likelihood,
        )
        # initialize episode data archives
        self.ep_traces = Archive(
            f'{self.store_dir}/episodes/traces', is_config=True,
            key_len=6, path_len=6,
        )
        self.ep_metas = Archive(
            f'{self.store_dir}/episodes/metas', key_len=self.ep_traces.key_len,
        )

        # link episode traces and agent manager to likelihood manager
        self.likelihood_manager.ep_traces = self.ep_traces
        self.likelihood_manager.agent_manager = self.agent_manager

    def export_tar(self, tar_path: str = 'store.tar.gz'):
        raise NotImplementedError

    def load_tar(self, tar_path: str):
        raise NotImplementedError

    def add_episode(self,
        actions: Array, observations: Array,
        meta_info: Optional[dict] = None,
    ):
        r"""Adds episode data to external storage.

        Episode traces are saved in `self.ep_traces` as hashable records, and
        the meta information is saved in `self.ep_metas` with the same key.

        Args
        ----
        actions: (num_steps,)
            Actions at time steps [1, t].
        observations: (num_steps+1, *)
            Observations at time steps [0, t].
        meta_info:
            Meta information for this episode, such as environment parameters of
            the external one (real) and the internal one (imaginary).

        """
        ep_trace = {'actions': actions, 'observations': observations}
        episode_key = self.ep_traces.get_key(ep_trace)
        if episode_key is None: # new traces
            episode_key = self.ep_traces._new_key()
            self.ep_traces[episode_key] = ep_trace
            self.ep_metas[episode_key] = meta_info
            if self.verbose>0:
                print(f"Episode data saved at key {episode_key}.")
        else: # existing traces
            if episode_key not in self.ep_metas or self.ep_metas[episode_key] is None:
                self.ep_metas[episode_key] = meta_info
            elif meta_info is not None and self.ep_metas[episode_key]!=meta_info:
                warnings.warn(
                    "Trace already exists, with conflicting meta information:\n"
                    f"Old: {self.ep_metas[episode_key]}\n"
                    f"New: {meta_info}"
                )
        return episode_key

    def train_agent(self,
        env_param: list[float] = None,
        seed: Optional[int] = None,
        num_epochs: int = 5,
        **kwargs,
    ) -> BeliefAgent:
        r"""Traines one agent.

        Args
        ----
        env_param:
            The environment parameter for agent internal model. Other
            specifications are based on `self.agent_manager.defaults`.
        seed:
            Random seed used for indexing different agent instances.
        num_epochs:
            Number of training epochs in reinforcement learning.
        kwargs:
            Addtional arguments for `process` method, see `Manager` class for
            more details.

        """
        
        config = self.agent_manager.get_config({'env_param': env_param, 'seed': seed})
        self.agent_manager.process(config, num_epochs=num_epochs, **kwargs)
        return self.agent_manager.agent

    def inspect_agent(self,
        env_param: list[float] = None,
        seed: Optional[int] = None,
        figsize: tuple[float, float] = (5, 4),
    ) -> tuple[BeliefAgent, Figure]:
        r"""Inspects one agent.

        This method generates a figure composed of the training progress of the
        agent, optimization progress of distributions estimation. It can be ran
        while the agent is being trained.

        Args
        ----
        env_param, seed:
            Environment parameters and seed of an agent, see `train_agent` for
            more details.
        figsize:
            Size of the summary figure.

        """
        if seed is None:
            # if agent seed is not specified, find the most trained one

            
            
            # Lokesh added for scheduling - cond is kind of like what is being read from saved file, which has info only about env_param and seed
            # cond = self.agent_manager.get_config({'env_param': env_param})
            cond = {}
            cond['env_param'] = self.agent_manager.get_config({'env_param': env_param})['env_param']
            cond['seed'] = self.agent_manager.get_config({'env_param': env_param})['seed']


            
            cond.pop('seed')
            keys = list(self.agent_manager.completed(cond=cond))
            random.shuffle(keys)
            max_epoch, best_key = 0, None
            for key in keys:
                if self.agent_manager.stats[key]['epoch']>max_epoch:
                    max_epoch = self.agent_manager.stats[key]['epoch']
                    best_key = key
            if best_key is None:
                raise RuntimeError(
                    f"No checkpoint found for environment parameter {cond.env_param}."
                )
            else:
                print(f"Fetching the most trained agent so far (epoch {max_epoch}).")
                config = self.agent_manager.configs[best_key]




                # Lokesh added for scheduling - since cond had info only about env_param and seed, after finding best key we need to add other things to config file from defaults yaml.
                config['env'] = self.agent_manager.get_config({'env_param': env_param})['env']
                config['model'] = self.agent_manager.get_config({'env_param': env_param})['model']
                config['policy'] = self.agent_manager.get_config({'env_param': env_param})['policy']
                config['algo'] = self.agent_manager.get_config({'env_param': env_param})['algo']




        else:
            config = self.agent_manager.get_config({'env_param': env_param, 'seed': seed})
            key = self.agent_manager.configs.get_key(config)
            if key is None:
                raise RuntimeError(
                    f"No checkpoint found for environment parameter {config.env_param} and seed "
                    f"{config.seed}."
                )
        self.agent_manager.setup(config)
        self.agent_manager.load_ckpt()
        optimality = self.agent_manager.preview['optimality']
        print("Agent (seed {}) was trained for {} epochs{}.".format(
            config.seed, self.agent_manager.epoch,
            "" if np.isnan(optimality) else ", RL optimality {:.2%}".format(optimality),
        ))
        fig = plot_agent_checkpoint(self.agent_manager.ckpt, figsize)
        return self.agent_manager.agent, fig

    def train_agents(self,
        *,
        env_param_list: Optional[Iterable[list[float]]] = None,
        env_param_grid: Optional[list[Iterable[float]]] = None,
        seeds: Optional[Iterable[int]] = None,
        **kwargs,
    ):
        r"""Train agents for a list/grid of environment parameters.

        This method can be run in parallel on multiple machines, training
        different agents simultaneously.

        Args
        ----
        env_param_list, env_param_grid, seeds:
            Specifications of agents to be trained, see
            `AgentManager._to_choices` for more details.
        kwargs:
            Additional arguments for `sweep` method, see `Manager` class for
            more details.

        Examples
        --------
        >>> manager.train_agents(
            env_param_grid=env_param_grid, seeds=seeds,
            num_epochs=num_epochs, count=count,
        )

        """
        choices = self.agent_manager._to_choices(env_param_list, env_param_grid, seeds)
        self.agent_manager.sweep(choices, **kwargs)

    def overview_agents(self,
        *,
        env_param_list: Optional[Iterable[list[float]]] = None,
        env_param_grid: Optional[list[Iterable[float]]] = None,
        seeds: Optional[Iterable[int]] = None,
        **kwargs,
    ) -> dict:
        r"""Returns an overview of the agents training status.

        Args
        ----
        env_param_list, env_param_grid, seeds:
            Specifications of agents to be trained, see
            `AgentManager._to_choices` for more details.
        kwargs:
            Additional arguments for `overview` method, see `Manager` class for
            more details.

        """
        choices = self.agent_manager._to_choices(env_param_list, env_param_grid, seeds)
        report = self.agent_manager.overview(choices, p_keys=['optimality'], **kwargs)
        if self.verbose>0:
            optimality = np.nanmean(report['optimality'])
            if not np.isnan(optimality):
                print("Average training optimality {:.1%}".format(optimality))
        return report

    def compute_logps(self,
        episode_key: Optional[str] = None,
        episode_path: Optional[str] = None,
        *,
        env_param_list: Optional[Iterable[list[float]]] = None,
        env_param_grid: Optional[list[Iterable[float]]] = None,
        agent_seeds: Optional[Iterable[int]] = None,
        belief_seeds: Optional[Iterable[int]] = None,
        min_epoch: int = 10, min_optimality: float = 0.95,
        disp_interval: int = 10,
        **kwargs,
    ) -> Array:
        r"""Compute log likelihoods of multiple agents.

        Over the given agents specifications, only the ones trained extensively
        are selected and used to compute likelihood p(episode|agent). Since the
        belief updates within an agent is stochastic, multiple random seeds need
        to be used when computing the likelihood.

        episode_key:
            The key of a saved episode in `self.ep_traces`.
            `self.ep_traces[episode_key]` is a dictionary with 'actions' and
            'observations' as keys.
        episode_path:
            The path to an external file, that can be opened by `pickle` and
            contains 'actions' and 'observations' as keys. External data will be
            saved in `self.ep_traces` at first encounter.
            Only one of `episode_path` and `episode_key` needs to be specified.
        env_param_list, env_param_grid, agent_seeds:
            Specifications of agents to be trained, see
            `AgentManager._to_choices` for more details.
        belief_seeds:
            The seeds used in stochastic belief updates given episode traces and
            an agent.
        min_epoch:
            Minimum number of trained epochs for an agent.
        min_optimality:
            Minimum reinforcement learning optimality of an agent.
        kwargs:
            Additional arguments for `sweep` method, see `Manager` class for
            more details. This is for likelihood computation instead of agent
            training.

        Returns
        -------
        logps:
            Log likelihoods of the episode data in episode_key/episode_path
            conditioned on all agents. If agents are specified by
            `env_param_list`, `logps`'s shape is (len(env_param_list),
            num_agent_seeds, num_belief_seeds). If agents are specified by
            `env_param_grid`, `logps`'s shape is (*env_param_grid_shape,
            num_agent_seeds, num_belief_seeds).
            Agents that are not qualified to compute will leave `np.nan` in
            `logps`.

        """
        # set up 'episode_key'
        assert (episode_key is None)!=(episode_path is None), (
            "One and only one of 'episode_key' and 'episode_path' needs to be specified."
        )
        if episode_key is None:
            try:
                with open(f'{self.store_dir}/{episode_path}', 'rb') as f:
                    saved = pickle.load(f)
                assert 'actions' in saved and 'observations' in saved
            except:
                raise RuntimeError(
                    f"'{episode_path}' is not path to a valid data file,  it should be a pickleable"
                    " file with 'actions' and 'observations' as keys."
                )
            episode_key = self.add_episode(
                saved['actions'], saved['observations'],
                meta_info={'file_path': episode_path},
            )
        # set up 'agent_key'
        a_choices = self.agent_manager._to_choices(
            env_param_list, env_param_grid, agent_seeds,
        )
        agent_keys = np.full(
            (len(a_choices['env_param']), len(a_choices['seed'])),
            dtype=object, fill_value=None,
        )
        _keys = dict((v, k) for k, v in self.agent_manager.configs.items())
        _stats = dict((k, v) for k, v in self.agent_manager.stats.items())
        _previews = dict((k, v) for k, v in self.agent_manager.previews.items())
        for a_config in self.agent_manager._config_gen(a_choices):
            try:
                agent_key = _keys[a_config]
                assert _stats[agent_key]['epoch']>=min_epoch
                assert _previews[agent_key]['optimality']>=min_optimality
            except:
                continue
            i = a_choices['env_param'].index(tuple(a_config.env_param))
            j = a_choices['seed'].index(a_config.seed)
            agent_keys[i, j] = agent_key
        # set up 'seed'
        if belief_seeds is None:
            belief_seeds = MANAGER_CONFIG.sweep.belief_seeds
            if self.verbose>0:
                print(f"Use default belief seeds {belief_seeds} for each run.")

        # sweep to compute likelihoods
        l_choices = {
            'episode_key': [episode_key],
            'agent_key': [v for v in agent_keys.flatten() if v is not None],
            'seed': belief_seeds,
        }
        if self.verbose>0:
            print(f"{len(l_choices['agent_key'])} valid agents found.")
            ep_len = len(self.ep_traces[episode_key].native()['actions'])
            print(f"Computing likelihood of episode {episode_key} ({ep_len} time steps)...")
        self.likelihood_manager.sweep(
            l_choices, num_epochs=0, disp_interval=disp_interval, **kwargs,
        )

        # fetch existing results
        shape = (
            len(a_choices['env_param']), len(a_choices['seed']),
            len(l_choices['seed']),
        )
        logps = np.full(shape=shape, fill_value=np.nan)
        _keys = dict((v, k) for k, v in self.likelihood_manager.configs.items())
        _previews = dict((k, v) for k, v in self.likelihood_manager.previews.items())
        for i in range(shape[0]):
            for j in range(shape[1]):
                agent_key = agent_keys[i, j]
                if agent_key is None:
                    continue
                for k in range(shape[2]):
                    l_config = self.likelihood_manager.get_config({
                        'agent_key': agent_key,
                        'episode_key': episode_key,
                        'seed': l_choices['seed'][k],
                    })
                    try:
                        key = _keys[l_config]
                        preview = _previews[key]
                        logps[i, j, k] = preview['logp']
                    except:
                        continue
        if env_param_list is None:
            shape = (
                *[len(x) for x in env_param_grid],
                len(a_choices['seed']), len(l_choices['seed']),
            )
            logps = logps.reshape(shape)
        print("{:.2%} of the likelihood entries have been computed.".format(1-np.isnan(logps).mean()))
        return logps
