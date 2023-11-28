import time, yaml
from pathlib import Path
import numpy as np
import torch
from collections.abc import Iterable, Sequence
import itertools
from typing import Optional, Union

from jarvis.config import Config, _locate
from jarvis.manager import Manager
from jarvis.utils import progress_str, time_str

from . import rcParams
from .agent import BeliefAgent
from .model import SamplingBeliefModel
from .utils import check_env, plot_agent_checkpoint, logmeanexp
from .alias import Array, Figure


class AgentManager(Manager):
    r"""Manager for training rational agents."""

    def __init__(self,
        store_dir: str = 'irc_store',
        defaults: Union[dict, Path, str, None] = None,
        eval_config: Optional[dict] = None,
        device: Optional[str] = None,
        to_check_env: Optional[bool] = None,
        eval_interval: Optional[int] = None,
        save_interval: Optional[int] = None,
        disp_interval: Optional[int] = None,
        **kwargs,
    ):
        _rcParams = rcParams['irc.manager.AgentManager']
        super().__init__(
            store_dir, defaults,
            eval_interval=eval_interval or _rcParams['eval_interval'],
            save_interval=save_interval or _rcParams['save_interval'],
            disp_interval=disp_interval or _rcParams['disp_interval'],
            **kwargs,
        )
        self.defaults.fill(_rcParams['defaults'])
        self.eval_config = Config(eval_config)
        self.eval_config.fill(_rcParams['eval_config'])
        self.device = (device or _rcParams['device']) if torch.cuda.is_available() else 'cpu'
        if self.device.startswith('cuda'):
            print(f"Using GPU device '{self.device}'.")
        self.to_check_env = to_check_env or _rcParams['to_check_env']

    def get_config(self, config=None):
        config = super().get_config(config)
        if config.model._target_=='irc.model.SamplingBeliefModel':
            config.fill(rcParams['defaults.SamplingBeliefModel'])
        assert config.env._target_ is not None, "Environment class name needs to be specified."
        if self.to_check_env or config.env_param is None:
            _env = config.env.instantiate()
        # check environment is valid
        if self.to_check_env:
            check_env(_env)
        # assign default environment parameter
        if config.env_param is None:
            config.env_param = _env.get_param()
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
        algo = config.algo.instantiate(
            policy=policy, env=model, policy_kwargs=config.policy,
            device=self.device, seed=config.seed,
        )
        self.agent = BeliefAgent(model, algo)

    def init_ckpt(self):
        super().init_ckpt()
        if isinstance(self.agent.model, SamplingBeliefModel):
            if self.verbose:
                print(
                    "Initializing a SamplingBeliefModel with internal environment parameter:"
                    f"\n{tuple(self.agent.model.env.get_param())}"
                )
            tic = time.time()
            self.ckpt['estimate_stats'] = {}
            self.ckpt['estimate_stats']['p_s_o'] = self.agent.model.estimate_p_s_o(**self.config.estimate.p_s_o)
            self.ckpt['estimate_stats']['p_o_s'] = {
                self.epoch: self.agent.model.estimate_p_o_s(**self.config.estimate.p_o_s.init)
            }
            self.ckpt['collect_stats'] = {
                self.epoch: self.agent.model.collect_rollouts(
                    policy=self.agent.algo.policy, **self.config.collect.init,
                )
            }
            self.ckpt['train_stats'] = {
                self.epoch: self.agent.model.train_belief_net(**self.config.train.init)
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
            self.ckpt['estimate_stats']['p_o_s'][self.epoch] = self.agent.model.estimate_p_o_s(
                **self.config.estimate.p_o_s.tune,
            )
            self.ckpt['collect_stats'][self.epoch] = self.agent.model.collect_rollouts(
                policy=self.agent.algo.policy, **self.config.collect.tune,
            )
            self.ckpt['train_stats'][self.epoch] = self.agent.model.train_belief_net(
                **self.config.train.tune,
            )
        self.agent.algo.learn(
            total_timesteps=self.config.learn.rl_steps, log_interval=None,
            reset_num_timesteps=False,
        )
        toc = time.time()
        if self.verbose:
            print(
                f"Agent trained for {self.config.learn.rl_steps} time steps "
                f"({time_str(toc-tic)})."
            )

    def eval(self):
        r"""Evaluates the current agent.

        Multiple episodes are run and discounted cumulative rewards are recorded
        for the agent at latest epoch.

        """
        eval_record = self.agent.evaluate(**self.eval_config)
        if self.verbose:
            print(
                f"Agent evaluated for {self.eval_config.num_episodes} episodes of maximum length "
                f"{self.eval_config.max_steps}, average return "
                f"{'{:.2f}'.format(np.mean(eval_record['returns']))}."
            )
        self.ckpt['eval_records'][self.epoch] = eval_record

    def train_agent(self,
        env_param: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
        num_epochs: Optional[int] = None,
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
            Addtional arguments for `process` method, see `Manager` for more
            details.

        Returns
        -------
        agent:
            The agent trained after at least `num_epochs` epochs. If the saved
            checkpoint already exceeds `num_epochs`, the latest version will be
            returned.

        """
        config = {}
        if env_param is not None:
            config['env_param'] = env_param
        if seed is not None:
            config['seed'] = seed
        if num_epochs is None:
            num_epochs = rcParams['irc.manager.AgentManager.train_agent']['num_epochs']
        config = self.get_config(config)
        self.process(config, num_epochs=num_epochs, **kwargs)
        return self.agent

    def _env_param_keys(self,
        env_param: Sequence[float],
        **kwargs,
    ):
        cond = self.get_config({'env_param': env_param})
        cond.pop('seed')
        for key in self.completed(cond=cond, **kwargs):
            yield key

    def fetch_agent(self,
        env_param: Sequence[float],
        seed: Optional[int] = None,
    ) -> tuple[BeliefAgent, str]:
        r"""Returns a trained agent.

        Manager attributes will be modified while loading the trained agent.

        Args
        ----
        env_param:
            The environment parameter for agent internal model, see
            `train_agent` for more details.
        seed:
            Random seed used for indexing agents. If `None`, the most trained
            one will be returned.

        Returns
        -------
        agent:
            The trained agent of interest.
        key:
            The key of agent, used to load data such as config, e.g.
            `self.configs[key]`.

        """
        assert env_param is not None, "Environment parameter must be specified."
        if seed is None:
            # if agent seed is not specified, find the most trained one
            max_epoch, best_key = 0, None
            for key in self._env_param_keys(env_param):
                epoch = self.stats[key]['epoch']
                if epoch>max_epoch:
                    max_epoch = epoch
                    best_key = key
            if best_key is None:
                raise RuntimeError(
                    f"No checkpoint found for environment parameter {env_param}."
                )
            else:
                key = best_key
                config = self.configs[key]
                print(f"Fetching one most trained agent (seed {config.seed}) so far (epoch {max_epoch}).")
        else:
            config = self.get_config({'env_param': env_param, 'seed': seed})
            key = self.configs.get_key(config)
            if key is None:
                raise RuntimeError(
                    f"No checkpoint found for environment parameter {config.env_param} and seed "
                    f"{config.seed}."
                )
        self.setup(config)
        self.load_ckpt()
        return self.agent, key

    def inspect_agent(self,
        env_param: Sequence[float],
        seed: Optional[int] = None,
        figsize: Optional[tuple[float, float]] = None,
    ) -> tuple[BeliefAgent, Figure]:
        r"""Inspects one agent.

        This method generates a figure composed of the training progress of the
        agent, optimization progress of distributions estimation. It can be ran
        while the agent is being trained.

        Args
        ----
        env_param, seed:
            Environment parameters and seed of an agent, see `get_agent` for
            more details.
        figsize:
            Size of the summary figure.

        """
        agent, _ = self.fetch_agent(env_param, seed)
        fig = plot_agent_checkpoint(self.ckpt, figsize)
        return agent, fig

    def _grid_to_list(self,
        env_param_grid: Union[Sequence[Iterable[float]], Path, str, None] = None,
    ):
        if isinstance(env_param_grid, (Path, str)):
            with open(env_param_grid, 'r') as f:
                env_param_grid = yaml.safe_load(f)
        default_env_param = self.get_config().env_param
        assert len(env_param_grid)==len(default_env_param), (
            f"Expected {len(default_env_param)} lists as parameter grid for the "
            "environment."
        )
        env_param_list = itertools.product(*[
            env_param_grid[i] if env_param_grid[i] else [default_env_param[i]]
            for i in range(len(default_env_param))
        ])
        return env_param_list

    def _get_env_param_list(self,
        env_param_list: Union[Iterable[Sequence[float]], Path, str, None] = None,
        env_param_grid: Union[Sequence[Iterable[float]], Path, str, None] = None,
    ) -> list[tuple[float]]:
        r"""Returns formated environment paramter list.

        If `env_param_grid` is provided instead of `env_param_list`, a parameter
        list is created by outer product.

        Args
        ----
        env_param_list:
            A list of environment parameters to sweep over.
        env_param_grid:
            A list of value choices for each environment parameter dimension. A
            parameter list will be constructed by taking outer product of the
            choices. Only one of `env_param_list` and `env_param_grid` can be
            specified.

        Returns
        -------
        env_param_list:
            The formated environment parameter list.

        """
        assert (env_param_list is None)!=(env_param_grid is None), (
            "One and only one of 'env_param_list' and 'env_param_grid' needs to be specified."
        )
        if env_param_grid is None and isinstance(env_param_list, (Path, str)):
            with open(env_param_list, 'r') as f:
                env_param_list = yaml.safe_load(f)
        if env_param_list is None:
            env_param_list = self._grid_to_list(env_param_grid)
        env_param_list = [tuple(env_param) for env_param in env_param_list]
        default_env_param = self.get_config().env_param
        for env_param in env_param_list:
            assert len(env_param)==len(default_env_param), (
                "Inconsistent environment parameter vector dimension detected."
            )
        return env_param_list

    def train_agents(self,
        *,
        env_param_list=None, env_param_grid=None,
        seeds: Optional[Iterable[int]] = None,
        num_epochs: Optional[int] = None,
        **kwargs,
    ):
        r"""Train agents for a list/grid of environment parameters.

        This method can be run in parallel on multiple machines, training
        different agents simultaneously.

        Args
        ----
        env_param_list, env_param_grid:
            Specifications of agents to be trained, see `_to_choices` for more
            details.
        seeds:
            A list of random seeds for different agents. E.g., if the user needs
            6 agents trained for each environment parameter, the method takes
            `seeds=range(6)` as input.
        kwargs:
            Additional arguments for `sweep` method, see `Manager` for  more
            details.

        Examples
        --------
        >>> manager.train_agents(
            env_param_grid=env_param_grid, seeds=seeds,
            num_epochs=num_epochs, count=count,
        )

        """
        if num_epochs is None:
            num_epochs = rcParams['irc.manager.AgentManager.train_agent']['num_epochs']
        if seeds is None:
            seeds = rcParams['irc.manager.IRCManager.sweep']['agent_seeds']
            if self.verbose:
                print(f"Use default agent seeds {seeds} for each environment parameter.")
        choices = {
            'env_param': self._get_env_param_list(env_param_list, env_param_grid),
            'seed': list(seeds),
        }
        self.sweep(choices, num_epochs=num_epochs, **kwargs)

    def overview_agents(self,
        *,
        env_param_list=None, env_param_grid=None,
        seeds: Optional[Iterable[int]] = None,
        **kwargs,
    ) -> dict:
        r"""Returns an overview of the agents training status.

        Args
        ----
        env_param_list, env_param_grid:
            Specifications of agents to be overview, see `_to_choices` for more
            details. 
        seeds:
            A list of random seeds for different agents. If not specified, all
            agents regardless of the seed will be gathered.
        kwargs:
            Additional arguments for `overview` method, see `Manager` class for
            more details.

        """
        env_param_list = self._get_env_param_list(env_param_list, env_param_grid)
        cond = self.get_config()
        cond['env_param'] = lambda env_param: tuple(env_param) in env_param_list
        if seeds is None:
            cond.pop('seed')
        else:
            cond['seed'] = lambda seed: seed in list(seeds)
        configs = (
            self.configs[key] for key in self.completed(cond=cond)
        )
        report = self.overview(configs, **kwargs)
        return report

    def compute_logps(self,
        observations: Array, actions: Array,
        *,
        env_param_list=None, env_param_grid=None,
        **kwargs,
    ) -> tuple[Array, Array]:
        r"""Computes log likelihood of episode data on environment parameters.

        Args
        ----
        observations, actions:
            Episode data, see `Agent.episode_logps` for more details.
        env_param_list, env_param_grid:
            Specifications of agents, see `overview_agents` for more details.

        Returns
        -------
        counts, logps:
            Number of trained agents and the mean log likelihood for each
            environment parameter. If `env_param_list` is provided, `counts` and
            `logps` are both 1-D arrays. If `env_param_grid` is provided,
            `counts` and `logps` are N-D arrays, of which N is the number of
            parameters whose possible values are more than 1.

        """
        env_param_list = self._get_env_param_list(env_param_list, env_param_grid)
        counts = [0]*len(env_param_list)
        logps = [[] for _ in range(len(env_param_list))]
        configs = {k: v for k, v in self.configs.items()}
        cond = self.get_config()
        cond['env_param'] = lambda env_param: tuple(env_param) in env_param_list
        cond.pop('seed')
        keys = list(self.completed(cond=cond, **kwargs))
        if self.verbose:
            print(f"{len(keys)} trained agents found, computing episode likelihood...")
        tic = time.time()
        num_prints = None if len(keys)>=60 else 1
        for k, key in enumerate(keys, 1):
            try:
                self.setup(configs[key])
                self.load_ckpt()
            except:
                continue
            _, action_logps = self.agent.episode_logps(observations, actions)
            i = env_param_list.index(tuple(self.config.env_param))
            counts[i] += 1
            logps[i].append(action_logps)
            if num_prints is None and k>=10:
                toc = time.time()
                num_prints = max(int((toc-tic)/k*len(keys)/300), 6)
            if (
                self.verbose and num_prints is not None and
                (k%(-(-len(keys)//num_prints))==0 or k==len(keys))
            ):
                toc = time.time()
                print('{} ({})'.format(
                    progress_str(k, len(keys)), time_str(toc-tic, k/len(keys)),
                ))
        counts = np.array(counts)
        logps = np.stack([
            logmeanexp(np.array(logps[i]), axis=0) if counts[i]>0 else np.nan
            for i in range(len(counts))
        ])
        if env_param_grid is not None:
            if isinstance(env_param_grid, (Path, str)):
                with open(env_param_grid, 'r') as f:
                    env_param_grid = yaml.safe_load(f)
            shape = [len(x) for x in env_param_grid if len(x)>1]
            counts = counts.reshape(shape)
            logps = logps.reshape(*shape, -1)
        return counts, logps
