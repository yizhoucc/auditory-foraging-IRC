import os, time, random, yaml, warnings
from pathlib import Path
import numpy as np
import torch
import itertools
from tqdm import tqdm

from typing import Optional, Union
from collections.abc import Iterable, Collection, Container

from jarvis.config import Config, _load_dict, _locate
from jarvis.manager import Manager
from jarvis.utils import progress_str, time_str

from . import rcParams
from .agent import BaseBeliefAgent, DistilledBeliefAgent
from .net import BaseDistributionNet
from .model import SamplingBeliefModel, DistilledBeliefModel
from .utils import check_env, plot_checkpoint, logmeanexp
from .alias import Array, Figure, EnvParam


class AgentManager(Manager):
    r"""Manager for training rational agents."""

    def __init__(self,
        store_dir: str = 'irc_store',
        defaults: Union[dict, Path, str, None] = None,
        *,
        eval_config: Optional[dict] = None,
        device: Optional[str] = None,
        to_check_env: Optional[bool] = None,
        eval_interval: Optional[int] = None,
        save_interval: Optional[int] = None,
        disp_interval: Optional[int] = None,
        **kwargs,
    ):
        _rcParams = Config(rcParams.get('manager.AgentManager._init_'))
        super().__init__(
            store_dir, defaults,
            eval_interval=eval_interval or _rcParams.eval_interval,
            save_interval=save_interval or _rcParams.save_interval,
            disp_interval=disp_interval or _rcParams.disp_interval,
            **kwargs,
        )
        self.defaults.fill(_rcParams.defaults)
        self.eval_config = Config(eval_config)
        self.eval_config.fill(_rcParams.eval_config)
        self.device = (device or _rcParams.device) if torch.cuda.is_available() else 'cpu'
        if self.verbose and self.device.startswith('cuda'):
            print(f"Using GPU device '{self.device}'.")
        self.to_check_env = to_check_env or _rcParams.to_check_env

    def get_config(self, config=None):
        _rcParams = Config(rcParams.get('manager.AgentManager.get_config'))
        config = super().get_config(config)
        if config.model._target_=='irc.model.SamplingBeliefModel':
            config.fill(_rcParams._sampling)
        assert config.env._target_ is not None, "Environment class name needs to be specified."
        if self.to_check_env or config.env_param is None:
            _env = config.env.instantiate()
        # check environment is valid
        if self.to_check_env:
            check_env(_env)
        # assign default environment parameter
        if config.env_param is None:
            config.env_param = _env.get_param()
        config.env_param = [*config.env_param]
        return config

    def setup(self, config):
        super().setup(config)
        config = config.clone()
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
        self.algo = config.algo.instantiate(
            policy=policy, env=model, policy_kwargs=config.policy,
            gamma=config.task.gamma, ent_coef=config.task.ent_coef,
            device=self.device, seed=config.seed,
        )
        self.agent = BaseBeliefAgent(model, self.algo.policy)

    def init_ckpt(self):
        super().init_ckpt()
        if self.verbose:
            print(
                "Training agent for environment parameter:"
                f"\n{tuple(self.config.env_param)}"
                f"\nReward discount rate {self.config.task.gamma}, "
                f"entropy loss coefficient {self.config.task.ent_coef}."
            )
        if isinstance(self.agent.model, SamplingBeliefModel):
            if self.verbose:
                print("Initializing SamplingBeliefModel...")
            tic = time.time()
            self.ckpt['train_stats'] = {}
            self.ckpt['train_stats']['init_net'] = \
                self.agent.model.train_init_net(**self.config.train.init_net)
            self.ckpt['train_stats']['observe_net'] = {
                self.epoch: self.agent.model.train_observe_net(**self.config.train.observe_net.init)
            }
            self.ckpt['collect_stats'] = {
                self.epoch: self.agent.model.collect_rollouts(
                    policy=self.agent._get_distribution, **self.config.collect.init,
                )
            }
            self.ckpt['train_stats']['update_net'] = {
                self.epoch: self.agent.model.train_update_net(**self.config.train.update_net.init)
            }
            toc = time.time()
            if self.verbose:
                print(
                    "Conditional distributions and belief update network initialized "
                    f"({time_str(toc-tic)})."
                )

    def save_ckpt(self):
        self.ckpt['state_dict'] = self.agent.state_dict()
        super().save_ckpt()

    def load_ckpt(self):
        super().load_ckpt()
        self.agent.load_state_dict(self.ckpt['state_dict'])

    def train(self):
        r"""Runs RL algorithm for one epoch.

        This method is only called in agent training mode. Each training epoch
        contains a fixed number of environment interactions specified by
        `config.learn.total_timesteps`.

        """
        tic = time.time()
        if isinstance(self.agent.model, SamplingBeliefModel):
            self.ckpt['train_stats']['observe_net'][self.epoch] = \
                self.agent.model.train_observe_net(**self.config.train.observe_net.tune)
            self.ckpt['collect_stats'][self.epoch] = self.agent.model.collect_rollouts(
                policy=self.agent._get_distribution, **self.config.collect.tune,
            )
            self.ckpt['train_stats']['update_net'][self.epoch] = \
                self.agent.model.train_update_net(**self.config.train.update_net.tune)
        self.agent.policy.set_training_mode(True)
        self.algo.learn(
            total_timesteps=self.config.learn.rl_steps, log_interval=None,
            reset_num_timesteps=False,
        )
        self.agent.policy.set_training_mode(False)
        toc = time.time()
        if self.verbose:
            print(
                f"Agent trained for {self.config.learn.rl_steps} time steps "
                f"({time_str(toc-tic)} per epoch)."
            )

    def eval(self):
        r"""Evaluates the current agent.

        Multiple episodes are run and discounted cumulative rewards are recorded
        for the agent at latest epoch.

        """
        num_episodes = self.eval_config.get('num_episodes')
        total_steps = self.eval_config.get('total_steps')
        max_steps = self.eval_config.get('max_steps')
        with torch.no_grad():
            episodes = self.agent.run_episodes(
                num_episodes, total_steps, max_steps, q_states=[],
            )
        rewards = [] # rewards
        rates = [] # reward rates
        returns = [] # cumulative discounted rewards
        ents = [] # entropies
        for episode in episodes:
            rewards.append(episode['rewards'])
            rates.append(episode['rewards'].mean())
            returns.append(self.agent._return(self.config.task.gamma, episode['rewards']))
            ents.append(episode['ents'].numpy())
        rewards = np.concatenate(rewards)
        rates = np.array(rates)
        returns = np.array(returns)
        ents = np.concatenate(ents)
        if self.verbose:
            ent_str = "mean policy entropy {:.2f}".format(np.mean(ents))
            if num_episodes is not None:
                if self.epoch==0:
                    print(f"Agent evaluated for {num_episodes} episodes of maximum {max_steps} steps.")
                print(f"Mean return {'{:.2f}'.format(returns.mean())}, {ent_str}.")
            if total_steps is not None:
                if self.epoch==0:
                    print(f"Agent evaluated for {total_steps} steps.")
                print(f"Reward rate {'{:.2f}'.format(rewards.mean())}, {ent_str}.")
        eval_record = {
            'num_episodes': num_episodes,
            'total_steps': total_steps,
            'max_steps': max_steps,
            'rates': rates,
            'returns': returns,
            'mean_ent': ents.mean(),
        }
        self.ckpt['eval_records'][self.epoch] = eval_record
        self.preview = {
            'mean_rate': rewards.mean(),
            'mean_return': returns.mean(),
            'mean_ent': ents.mean(),
        }

    def train_agent(self,
        env_param: EnvParam,
        seed: Optional[int] = None,
        num_epochs: Optional[int] = None,
        **kwargs,
    ) -> BaseBeliefAgent:
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
            Addtional arguments for `process` method, see `Manager` for more
            details.

        Returns
        -------
        agent:
            The agent trained after at least `num_epochs` epochs. If the saved
            checkpoint already exceeds `num_epochs`, the latest version will be
            returned.
        key:
            A unique string to identify the trained agent.

        """
        _rcParams = Config(rcParams.get('manager.AgentManager.train_agent'))
        num_epochs = num_epochs or _rcParams.num_epochs
        config = {'env_param': env_param}
        if seed is not None:
            config['seed'] = seed
        config = self.get_config(config)
        self.process(config, num_epochs=num_epochs, **kwargs)
        key = self.configs.get_key(config)
        return self.agent, key

    def _fetch_agent_by_config(self, config: Config) -> tuple[BaseBeliefAgent, str]:
        key = self.configs.get_key(config)
        self.setup(config)
        self.load_ckpt()
        return self.agent, key

    def _fetch_agent_by_key(self, key: str) -> tuple[BaseBeliefAgent, Config]:
        config = self.configs[key]
        agent, _ = self._fetch_agent_by_config(config)
        return agent, config

    @staticmethod
    def _same_env_param(param_0: EnvParam, param_1: EnvParam) -> bool:
        return len(param_0)==len(param_1) and np.allclose(param_0, param_1)

    def find_agents(self,
        env_param: EnvParam,
        min_epoch: Optional[int] = None,
        cond: Optional[dict] = None,
        strict: bool = False,
    ) -> list[str]:
        _rcParams = Config(rcParams.get('manager.AgentManager.find_agents'))
        min_epoch = min_epoch or _rcParams.min_epoch

        _config = self.get_config()
        if cond is None:
            cond = _config
            cond.pop('seed')
        else:
            cond = Config(cond)
            assert 'env_param' not in cond, (
                "Environment parameter should be conditioned by `env_param` argument."
            )
            for key in ['env', 'task']:
                if key not in cond:
                    cond[key] = {}
                cond[key].fill(_config[key])
            strict = True
        cond.env_param = lambda x: self._same_env_param(x, env_param)
        if not strict:
            cond = Config({
                k: v for k, v in cond.items() if k in ['env', 'task', 'env_param']
            })
        keys = [key for key, _ in self.completed(min_epoch=min_epoch, cond=cond)]
        returns = [self.previews[key]['mean_return'] for key in keys]
        keys, returns = zip(*sorted(
            [(k, r) for k, r in zip(keys, returns)],
            reverse=True, key=lambda x: x[1],
        ))
        return keys

    def inspect_agent(self,
        key: str,
        **kwargs,
    ) -> tuple[BaseBeliefAgent, Figure]:
        r"""Inspects one agent.

        This method generates a figure composed of the training progress of the
        agent, optimization progress of distributions estimation. It can be ran
        while the agent is being trained.

        Args
        ----
        key:
            Agent key.
        kwargs:
            Addtional arguments for `plot_checkpoint`.

        """
        agent, _ = self._fetch_agent_by_key(key)
        fig = plot_checkpoint(self.ckpt, **kwargs)
        return agent, fig

    def _grid2list(self,
        param_grid: Union[Collection[Iterable[float]], Path, str],
    ) -> Iterable[EnvParam]:
        r"""Converts environment parameter grid to list.

        Args
        ----
        param_grid:
            A list of value choices for each environment parameter dimension, or
            a yaml file path that contains one. An empty list means to use the
            default value provided by environment.

        Returns
        -------
        param_list:
            A list of environment parameters constructed by taking outer product
            of the choices in `param_grid`.

        """
        if isinstance(param_grid, (Path, str)):
            with open(param_grid, 'r') as f:
                param_grid = yaml.safe_load(f)
        _env_param = self.get_config().env_param
        assert len(param_grid)==len(_env_param), (
            f"Expected {len(_env_param)} lists to specify a parameter grid for "
            "the environment."
        )
        param_list = itertools.product(*[
            param_grid[i] if param_grid[i] else [_env_param[i]]
            for i in range(len(_env_param))
        ])
        return param_list

    def _grid2configs(self,
        param_grid: Union[Collection[Iterable[float]], Path, str],
        seeds: Optional[Iterable[int]] = None,
        choices: Union[Path, str, dict, None] = None,
        min_epoch: Optional[int] = None,
    ) -> Iterable[Config]:
        r"""Prepares agent configs from a parameter grid.

        Args
        ----
        param_grid:
            Environment parameter grid, see `_grid2list` for more details.
        seeds:
            A list of random seeds for different agents. If the user needs six
            agents trained for each environment parameter, the method can take
            `seeds=range(6)` as input.
        choices:
            Training config choices in addition to environment parameters and
            random seeds, see `Manager._config_gen` for more details.

        Yields
        ------
        config:
            A config on the parameter grid, in random order.

        """
        _rcParams = Config(rcParams.get('manager.AgentManager._grid2configs'))
        choices = _load_dict(choices)
        choices.update({'env_param': self._grid2list(param_grid)})
        if min_epoch is None:
            choices['seed'] = seeds or _rcParams.seeds
            for config in self._config_gen(choices):
                yield config
        else:
            if seeds is not None:
                choices['seed'] = seeds
            cond = Config()
            def _in_vals(vals):
                _vals = [self.configs._to_hashable(Config({'k': val})) for val in vals]
                return lambda x: self.configs._to_hashable(Config({'k': x})) in _vals
            for key, vals in choices.items():
                cond[key] = _in_vals(vals)
            for _, config in self.completed(min_epoch, cond):
                yield config

    def train_agents(self,
        configs: Iterable[Config],
        num_epochs: Optional[int] = None,
        **kwargs,
    ) -> None:
        r"""Train multiple agents sequentially.

        Args
        ----
        configs:
            Configurations of agents to be trained, usually a generator.
        num_epochs:
            Number of epochs to train each agent.
        kwargs:
            Additional arguments for `batch` method, see `Manager.batch` for
            more details.

        """
        _rcParams = Config(rcParams.get('manager.AgentManager.train_agents'))
        num_epochs = num_epochs or _rcParams.num_epochs
        self.batch(configs, num_epochs=num_epochs, **kwargs)

    def monitor_agents(self,
        configs: Iterable[Config],
        **kwargs,
    ) -> dict:
        r"""Monitor training progress of agents.

        Args
        ----
        configs:
            See `train_agents` for more details.
        kwargs:
            Additional arguments for `monitor` method, see `Manager.monitor` for
            more details.

        """
        return self.monitor(
            configs, p_keys=['mean_rate', 'mean_return', 'mean_ent'], **kwargs,
        )

    def distill_agents(self,
        configs: Iterable[Config],
        max_agents: Optional[int] = None,
        total_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
        to_train: Optional[Container[str]] = None,
        train_config: Optional[dict] = None,
        init_net: Union[BaseDistributionNet, dict, None] = None,
        update_net: Union[BaseDistributionNet, dict, None] = None,
        policy_net: Union[BaseDistributionNet, dict, None] = None,
        save_path: Optional[str] = None,
    ) -> tuple[DistilledBeliefModel, dict]:
        r"""Distills sampling-based agents.

        Each agent runs in its assumed environment for some time, and the
        transitions are collected to train three networks that take environment
        parameter as inputs: belief initialization network, belief update
        network and policy network.

        Args
        ----
        configs:
            Configurations of agents to be distilled.
        total_steps, max_steps:
            Arguments for self play during transition collection, see
            `BaseBeliefAgent.run_episodes` for more details.
        to_train:
            A tag to indicate which components to train. Default is 'IUP', with
            three letters corresponding to `init_net`, `update_net` and
            `policy_net` respectively.
        train_config:
            Configurations of network training, can contain 'init_net',
            'update_net' and 'policy_net' as keys.
        init_net, update_net, policy_net:
            Belief initialization network, belief update network and policy
            network or configurations of each.
        save_path:
            A file to load existing agent or save trained one. `torch.load` and
            `torch.save` are used for reading and writing.

        Returns
        -------
        agent:
            Distilled agent from specified configurations.
        train_stats:
            Statistics of all training procedures.

        """
        _rcParams = Config(rcParams.get('manager.AgentManager.distill_agents'))
        total_steps = total_steps or _rcParams.total_steps
        max_steps = max_steps or _rcParams.max_steps
        if to_train is None: # empty string '' is a valid input
            to_train = _rcParams.to_train
        train_config = Config(train_config)
        train_config.fill(_rcParams.get('train_config'))

        def _create_agent(
            model_config: dict, env_config: dict,
            init_net: dict, update_net: dict, policy_net: dict,
        ):
            agent = DistilledBeliefAgent(
                Config(model_config).instantiate(
                    env=Config(env_config).instantiate(),
                    init_net=init_net, update_net=update_net,
                    device=self.device,
                ), policy_net,
            )
            return agent

        if not('I' in to_train or 'U' in to_train or 'P' in to_train):
            print(
                r"None of {'I', 'U', 'P'} is contained in `to_train`, skipping distillation."
            )
            if save_path is not None and os.path.exists(save_path):
                saved = torch.load(save_path)
                agent = _create_agent(**{
                    k: v for k, v in saved.items()
                    if k in ['model_config', 'env_config', 'init_net', 'update_net', 'policy_net']
                })
                agent.load_state_dict(saved['state_dict'])
            else:
                agent = None
            return agent, {}

        # collect self play data
        train_stats = {'keys': []}
        o_0, b_0, p_0 = [], [], []
        b_t, l_t, a_t, o_tp1, b_tp1, p_t = [], [], [], [], [], []
        env_config, model_config = None, None
        configs = list(configs)
        if max_agents is not None:
            configs = random.sample(configs, min(len(configs), max_agents))
        for config in tqdm(configs, desc='Collect Rollouts', unit='agent', **rcParams.tqdm_kw):
            try:
                agent, key = self._fetch_agent_by_config(config)
                if env_config is None:
                    env_config: Config = config.env.clone()
                else:
                    assert self.configs.equals(env_config, config.env), (
                        "Inconsistent environment detected."
                    )
                if model_config is None:
                    model_config: Config = config.model.clone()
                else:
                    assert self.configs.equals(model_config, config.model), (
                        "Inconsistent belief model detected."
                    )
                train_stats['keys'].append(key)
            except KeyboardInterrupt as e:
                raise e
            except:
                continue
            param = np.array([*config.env_param, config.task.gamma, config.task.ent_coef])
            with torch.no_grad():
                episodes = agent.run_episodes(None, total_steps, max_steps, q_states=[])
            for episode in episodes:
                episode['beliefs'] = episode['beliefs'].cpu().numpy()
                episode['logits'] = episode['logits'].cpu().numpy()
                o_0.append(episode['observations'][0])
                b_0.append(episode['beliefs'][0])
                p_0.append(param)
                b_t.append(episode['beliefs'][:-1])
                l_t.append(episode['logits'])
                a_t.append(episode['actions'][:, None])
                o_tp1.append(episode['observations'][1:])
                b_tp1.append(episode['beliefs'][1:])
                p_t.append(np.tile(param, (episode['num_steps'], 1)))
        o_0 = np.stack(o_0)
        b_0 = np.stack(b_0)
        p_0 = np.stack(p_0)
        b_t = np.concatenate(b_t)
        l_t = np.concatenate(l_t)
        a_t = np.concatenate(a_t)
        o_tp1 = np.concatenate(o_tp1)
        b_tp1 = np.concatenate(b_tp1)
        p_t = np.concatenate(p_t)

        model_config._target_ = 'irc.model.DistilledBeliefModel'
        agent = _create_agent(
            model_config, env_config, init_net, update_net, policy_net,
        )
        if save_path is not None and os.path.exists(save_path):
            saved = torch.load(save_path)
            try:
                assert self.configs.equals(model_config, saved['model_config'])
                assert self.configs.equals(env_config, saved['env_config'])
                if init_net is None or isinstance(init_net, dict):
                    assert self.configs.equals(init_net, saved['init_net'])
                if update_net is None or isinstance(update_net, dict):
                    assert self.configs.equals(update_net, saved['update_net'])
                if policy_net is None or isinstance(policy_net, dict):
                    assert self.configs.equals(policy_net, saved['policy_net'])
                agent.load_state_dict(saved['state_dict'])
            except:
                warnings.warn(
                    f"Loading from {save_path} unsuccessful, will train from scratch."
                )

        try:
            if 'I' in to_train:
                train_stats['init_net'] = agent.model.train_init_net(
                    o_0, p_0, b_0, **train_config.get('init_net', {}),
                    tqdm_kw=dict(desc='Train Init Net', unit='epoch', **rcParams.tqdm_kw),
                )
            if 'U' in to_train:
                train_stats['update_net'] = agent.model.train_update_net(
                    b_t, a_t, o_tp1, p_t, b_tp1, **train_config.get('update_net', {}),
                    tqdm_kw=dict(desc='Train Update Net', unit='epoch', **rcParams.tqdm_kw),
                )
            if 'P' in to_train:
                train_stats['policy_net'] = agent.train_policy_net(
                    b_t, p_t, l_t, **train_config.get('policy_net', {}),
                    tqdm_kw=dict(desc='Train Policy Net', unit='epoch', **rcParams.tqdm_kw),
                )
        except KeyboardInterrupt:
            print("Training interrupted"+(
                "" if save_path is None else f", progress will be saved at {save_path}"
            )+".")
        except:
            raise
        if save_path is not None:
            torch.save({
                'env_config': env_config, 'model_config': model_config,
                'init_net': init_net, 'update_net': update_net, 'policy_net': policy_net,
                'state_dict': agent.state_dict(),
            }, save_path)
        return agent, train_stats

    def train_agents_on_grid(self,
        param_grid: Union[Collection[Iterable[float]], Path, str],
        seeds: Optional[Iterable[int]] = None,
        choices: Union[Path, str, dict, None] = None,
        **kwargs,
    ) -> None:
        r"""Train agents on a parameter grid.

        Args
        ----
        param_grid, seeds, choices:
            Arguments to specify agents, see `_grid2configs` for more details.
        kwargs:
            Additional arguments for `train_agents`.

        """
        self.train_agents(
            self._grid2configs(param_grid, seeds, choices), **kwargs,
        )

    def monitor_agents_on_grid(self,
        param_grid: Union[Collection[Iterable[float]], Path, str],
        seeds: Optional[Iterable[int]] = None,
        choices: Union[Path, str, dict, None] = None,
        **kwargs,
    ) -> dict:
        r"""Monitor agents on a parameter grid.

        Args
        ----
        param_grid, seeds, choices:
            Arguments to specify agents, see `_grid2configs` for more details.
        kwargs:
            Additional arguments for `monitor_agents`.

        """
        return self.monitor_agents(
            self._grid2configs(param_grid, seeds, choices), **kwargs,
        )

    def distill_agents_on_grid(self,
        param_grid: Union[Collection[Iterable[float]], Path, str],
        seeds: Optional[Iterable[int]] = None,
        choices: Union[Path, str, dict, None] = None,
        min_epoch: int = 0,
        **kwargs,
    ) -> tuple[DistilledBeliefModel, dict]:
        r"""Distills agents on a parameter grid.

        Args
        ----
        param_grid, seeds, choices:
            Arguments to specify agents, see `_grid2configs` for more details.
        kwargs:
            Additional arguments for `distill_agents`.

        """
        return self.distill_agents(
            self._grid2configs(param_grid, seeds, choices, min_epoch), **kwargs,
        )

    def similar_agent(self, config: Config, min_epoch: int = 0):
        cond = Config({
            'env': config.env,
            'task': config.task,
            'env_param': lambda x: self._same_env_param(x, config.env_param),
            'model': config.model,
            'policy': config.policy,
        })
        _key = self.configs.get_key(config)
        configs = []
        for key, config in self.completed(min_epoch, cond):
            if key!=_key:
                configs.append(config)
        return configs

    def compare_agents(self,
        states: Array,
        t_agent: BaseBeliefAgent,
        c_agents: list[BaseBeliefAgent],
        total_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        _rcParams = Config(rcParams.get('manager.AgentManager.compare_agents'))
        total_steps = total_steps or _rcParams.total_steps
        max_steps = max_steps or _rcParams.max_steps
        env_param = t_agent.model.get_param()
        for c_agent in c_agents:
            if not self._same_env_param(c_agent.model.get_param(), env_param):
                warnings.warn(
                    "Candidate agent should have the same environment parameter as target."
                )
        with torch.no_grad():
            episodes = t_agent.run_episodes(None, total_steps, max_steps, q_states=[])
        state_probs = [None]*(len(c_agents)+1)
        action_probs = [None]*(len(c_agents)+1)
        for i, agent in enumerate([t_agent]+c_agents):
            state_probs[i] = []
            action_probs[i] = []
            for episode in episodes:
                if i==0:
                    beliefs = episode['beliefs'][:-1]
                    logits = episode['logits']
                else:
                    with torch.no_grad():
                        beliefs = agent.model.compute_beliefs(
                            episode['observations'], episode['actions'],
                        )[:-1]
                        logits = agent._get_distribution(beliefs).logits
                with torch.no_grad():
                    for belief in beliefs:
                        state_probs[i].append(agent.model.query_probs(states, belief))
                action_probs[i].append(torch.softmax(logits, dim=1).data.cpu().numpy())
            state_probs[i] = np.stack(state_probs[i])
            action_probs[i] = np.concatenate(action_probs[i])
        return episodes, state_probs, action_probs
