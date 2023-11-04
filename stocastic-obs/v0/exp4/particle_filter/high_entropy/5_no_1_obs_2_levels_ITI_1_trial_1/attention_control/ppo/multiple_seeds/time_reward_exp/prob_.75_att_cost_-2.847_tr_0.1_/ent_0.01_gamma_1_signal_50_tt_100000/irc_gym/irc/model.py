from pathlib import Path
import yaml
import numpy as np
import torch

from typing import Optional, Union
from gym.spaces import MultiDiscrete, Box
from jarvis.config import Config, _locate
from jarvis.utils import numpy_dict, tensor_dict

from .distribution import BaseDistribution
from .alias import Array, RandGen, VarSpace, GymEnv

defaults_dir = Path(__file__).parent/'defaults'
with open(defaults_dir/'model.yaml') as f:
    MODEL_CONFIG = Config(yaml.safe_load(f))

class BaseBeliefModel(GymEnv):
    r"""Base class of a belief MDP model.

    The belief MDP is based on a POMDP `env` assumed by an agent. A 'belief'
    vector is updated by the latest action and observation, and serves as the
    input to a policy network.

    API compatible with gym==0.21.0, regardless of `env` API version.

    """

    def __init__(self,
        env: GymEnv,
        device: str = 'cpu',
        rng: Union[RandGen, int, None] = None,
    ):
        r"""
        Args
        ----
        env:
            The assumed environment, with a few utility methods implemented,
            such as `get_state`, `set_state` etc. Detailed requirements are
            specified in 'README.md'.
        device:
            Tensor device.
        rng:
            Random number generator or seed.

        """
        # check gym api of the environment
        try:
            observation, info = env.reset()
            observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
            self.api = 'v26'
        except:
            observation = env.reset()
            observation, reward, done, info = env.step(env.action_space.sample())
            self.api = 'v21'
        assert env.state_space is not None
        assert callable(env.get_state) and callable(env.set_state)
        self.env = env
        self.action_space = env.action_space
        self.observation_space = None # belief space to be specified

        self.device = device if torch.cuda.is_available() else 'cpu'
        self.rng = rng if isinstance(rng, RandGen) else np.random.default_rng(rng)

    def seed(self, seed: int) -> list[int]:
        r"""Sets random seed.

        Args
        ----
        seed:
            Random seed for both the belief MDP and the original POMDP.

        Returns
        -------
        A list of seeds, potentially used by vectorized environments.

        """
        self.rng = np.random.default_rng(seed)
        if self.api=='v26':
            self.env.reset(seed=seed)
            return [seed]
        else:
            return self.env.seed(seed)

    def state_dict(self) -> dict:
        r"""Returns state dictionary."""
        raise NotImplementedError

    def load_state_dict(self, state_dict) -> None:
        r"""Loads state dictionary."""
        raise NotImplementedError

    def reset(self,
        env: Optional[GymEnv] = None,
        return_info: bool = True,
    ) -> Union[Array, tuple[Array, dict]]:
        r"""Resets the environment.

        Args
        ----
        env:
            The actual environment to interact with. If not provided, the
            internal model will be used. If provided, an additional sampling
            step will be taken to update the internal model state, because
            sometimes `self.env` is equipped with a `query_states` method that
            returns state-dependent queries.
        return_info:
            Whether to return additional information about the environment.

        Returns
        -------
        belief:
            A vector describing agent's belief about the environment state.
        info:
            A dictionary containing additional information, such as observation
            and real environment state. It is only returned when `return_info`
            is ``True``.

        """
        if env is None:
            env = self.env
            _use_internal = True
        else:
            _use_internal = False
        # get initial observation
        if self.api=='v26':
            observation, info = env.reset()
        else:
            try:
                assert return_info
                observation, info = env.reset(return_info=True)
            except:
                observation = env.reset()
                info = {}
        info['observation'] = observation
        if return_info and 'state' not in info:
            try:
                info['state'] = env.get_state()
            except:
                pass
        # get initial belief
        self.init_belief(observation)
        belief = self.get_belief()
        # update internal environment to ensure valid query states
        if not _use_internal:
            try:
                self.env.set_state(self.sample_state())
            except:
                pass
        if return_info:
            return belief, info
        else:
            return belief

    def step(self,
        action: int,
        env: Optional[GymEnv] = None,
    ) -> tuple[Array, float, bool, dict]:
        r"""Runs one step.

        Args
        ----
        action:
            The discrete action taken by the agent.
        env:
            The actual environment to interact with. See `reset` for more
            details.

        Returns
        -------
        belief:
            A vector describing agent's belief about the environment state.
        reward:
            Immediate reward.
        done:
            Whether the episode is terminated.
        info:
            A dictionary containing additional information. See `reset` for more
            details.

        """
        if env is None:
            env = self.env
            _use_internal = True
        else:
            _use_internal = False
        # get observation
        if self.api=='v26':
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        else:
            observation, reward, done, info = env.step(action)
        info['observation'] = observation
        if 'state' not in info:
            try:
                info['state'] = env.get_state()
            except:
                pass
        # update belief
        self.update_belief(action, observation)
        belief = self.get_belief()
        # update internal environment to ensure valid query states
        if not _use_internal:
            try:
                self.env.set_state(self.sample_state())
            except:
                pass
        return belief, reward, done, info

    def get_belief(self) -> Array:
        r"""Returns belief vector."""
        raise NotImplementedError

    def init_belief(self, observation) -> None:
        r"""Initializes the belief.

        A belief about states is initialized based on the first observation.

        Args
        ----
        observation:
            Initial observation.

        """
        raise NotImplementedError

    def update_belief(self, action, observation) -> None:
        r"""Updates the belief.

        The belief is updated by the most recent action and the following
        observation.

        Args
        ----
        action:
            The discrete action taken by the agent.
        observation:
            The new observation returned by the environment.

        """
        raise NotImplementedError

    def sample_state(self) -> Array:
        r"""Returns a state sampled from current belief.

        Potentially used to reset internal model state while using an external
        environment for interaction.

        """
        raise NotImplementedError

    def query_probs(self, states: Array) -> Array:
        r"""Returns probablities of queried states defined by current belief.

        Args
        ----
        states: (num_queries, *state_dim)
            Queried states of interest, can be different at every time step.

        Returns
        -------
        probs: (num_queries,)
            Probabilities of each state translated from current belief.

        """
        raise NotImplementedError


class SamplingBeliefModel(BaseBeliefModel):
    r"""Sampling-based belief model.

    The agent samples states from the current belief `p_s`, uses the internal
    environment to get new states with the action, and estimates the new belief
    after weighting state samples with the new observation.

    The agent learns two conditional distribution `p_s_o` and `p_o_s` for one
    time. `p_s_o` is used to initialize the belief vector, and `p_o_s` is used
    to weight sampled states.

    """

    def __init__(self,
        env: GymEnv,
        action_depend: bool = False,
        dist_config: Optional[dict] = None,
        est_config: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        Args
        ----
        env:
            See `BaseBeliefModel` for more information.
        action_dependent:
            Whether the observation is dependent on the action, i.e. whether to
            model p(o|s, a) or p(o|s).
        dist_config:
            Configuration for distributions used in the model, including p(s|o),
            p(o|s) and p(s). Typically used to specify belief structure.
        est_config:
            Configuration for estimating distributions, including specifications
            such as number of samples, number of optimization epochs, etc.

        """
        super(SamplingBeliefModel, self).__init__(env, **kwargs)
        self.action_depend = action_depend
        self.est_config = Config(est_config).clone().fill(MODEL_CONFIG.est_config)
        self._to_est_p_s_o = self._to_est_p_o_s = True

        # set up distribution configuration
        dist_config = Config(dist_config).clone().fill(MODEL_CONFIG.dist_config)
        if '_target_' not in dist_config.state:
            dist_config.state._target_ = self._default_dist_class(self.env.state_space)
        if '_target_' not in dist_config.observation:
            dist_config.observation._target_ = self._default_dist_class(self.env.observation_space)
        # belief about current state
        self.p_s: BaseDistribution = dist_config.state.instantiate(
            x_space=self.env.state_space, rng=self.rng,
        )
        self.p_s.to(self.device)
        self.observation_space = Box(-np.inf, np.inf, shape=self.p_s.get_param_vec().shape)
        # conditional probability p(s|o)
        self.p_s_o: BaseDistribution = dist_config.state.instantiate(
            x_space=self.env.state_space, y_space=self.env.observation_space, rng=self.rng,
        )
        self.p_s_o.to(self.device)
        # conditional probability p(o|s)
        if self.action_depend:
            _y_space = MultiDiscrete(list(self.env.state_space.nvec)+[self.env.action_space.n])
        else:
            _y_space = self.env.state_space
        self.p_o_s: BaseDistribution = dist_config.observation.instantiate(
            x_space=self.env.observation_space, y_space=_y_space, rng=self.rng,
        )
        self.p_o_s.to(self.device)
        # recent belief estimation statistics
        self.running_stats = []

    @staticmethod
    def _default_dist_class(space: VarSpace) -> str:
        r"""Returns default distribution class."""
        if isinstance(space, MultiDiscrete):
            return 'irc.distribution.DiscreteDistribution'
        if isinstance(space, Box):
            raise NotImplementedError
        raise RuntimeError(f"Default distribution configuration for {space} is not defined.")

    def estimate_p_s_o(self, verbose=0):
        r"""Estimates p(s|o)."""
        config = self.est_config.p_s_o.clone()
        # use env to collect initial states
        _state_to_restore = self.env.get_state()
        states, observations = [], []
        for _ in range(config.pop('num_samples')):
            if self.api=='v26':
                observation, _ = self.env.reset()
            else:
                observation = self.env.reset()
            states.append(self.env.get_state())
            observations.append(observation)
        self.env.set_state(_state_to_restore)
        if verbose>0:
            print(f"{len(states)} states collected.")
        # estimate the initial belief
        self.p_s_o.estimate(
            np.array(states), np.array(observations), verbose=verbose, **config,
        )
        self._to_est_p_s_o = False

    def estimate_p_o_s(self, verbose=0):
        r"""Estimates p(o|s)."""
        config = self.est_config.p_o_s.clone()
        # use env to collect state/observation pairs
        _state_to_restore = self.env.get_state()
        observations, states, actions = [], [], []
        for _ in range(config.pop('num_samples')):
            # TODO specify state and action distributions
            state = self.env.state_space.sample() # TODO verify it is not a terminal state
            self.env.set_state(state)
            action = self.env.action_space.sample()
            observation, *_ = self.env.step(action)
            observations.append(observation)
            states.append(self.env.get_state())
            actions.append(action)
        self.env.set_state(_state_to_restore)
        if verbose>0:
            print(f"{len(observations)} observations collected.")
        # estimate the conditional distribution
        if self.action_depend:
            ys = np.concatenate([np.array(states), np.array(actions)[:, None]], axis=1)
        else:
            ys = np.array(states)
        self.p_o_s.estimate(
            np.array(observations), ys, verbose=verbose, **config,
        )
        self._to_est_p_o_s = False

    def state_dict(self):
        state_dict = {
            'p_s_o_state': numpy_dict(self.p_s_o.state_dict()),
            'p_o_s_state': numpy_dict(self.p_o_s.state_dict()),
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.p_s_o.load_state_dict(tensor_dict(state_dict['p_s_o_state'], self.device))
        self._to_est_p_s_o = False
        self.p_o_s.load_state_dict(tensor_dict(state_dict['p_o_s_state'], self.device))
        self._to_est_p_o_s = False

    def reset(self, env=None, return_info=False):
        if self._to_est_p_s_o:
            self.estimate_p_s_o()
        if self._to_est_p_o_s:
            self.estimate_p_o_s()
        return super(SamplingBeliefModel, self).reset(env, return_info)

    def step(self, action, env=None):
        belief, reward, done, info = super(SamplingBeliefModel, self).step(action, env)
        self.running_stats.append(self.p_s.est_stats)
        if len(self.running_stats)>40:
            self.running_stats = self.running_stats[-40:]
        return belief, reward, done, info

    def get_belief(self):
        return self.p_s.get_param_vec().cpu().numpy()

    def init_belief(self, observation):
        with torch.no_grad():
            self.p_s.set_param_vec(self.p_s_o.param_net(np.array(observation)[None])[0])

    def update_belief(self, action, observation):
        config = self.est_config.p_s.clone()
        _state_to_restore = self.env.get_state()
        states, weights = [], []
        for _ in range(config.pop('num_samples')):
            state = self.p_s.sample()
            self.env.set_state(state)
            self.env.step(action)
            next_state = self.env.get_state()
            states.append(next_state)
            logp = self.p_o_s.loglikelihood(
                np.array(observation)[None], np.concatenate([
                    np.array(next_state)[None], np.array([action])[None],
                ], axis=1) if self.action_depend else np.array(next_state)[None],
            )
            weights.append(np.exp(logp.item()))
        self.env.set_state(_state_to_restore)
        self.p_s.estimate(
            np.array(states), ws=np.array(weights), verbose=0, **config,
        )

    def sample_state(self):
        return self.p_s.sample()

    def query_probs(self, states):
        with torch.no_grad():
            probs = np.exp(self.model.p_s.loglikelihood(states).cpu().numpy())
        return probs


class FuncBeliefModel(BaseBeliefModel):

    def __init__(self,
        env: GymEnv,
        **kwargs,
    ):
        super(FuncBeliefModel, self).__init__(env, **kwargs)
        belief, _ = super(FuncBeliefModel, self).reset()
        assert len(belief.shape)==1
        self.belief_dim = len(belief)
        self.observation_space = Box(-np.inf, np.inf, shape=(self.belief_dim,))
        self.belief = np.full(shape=self.belief_dim, fill_value=np.nan)

    def get_belief(self) -> Array:
        return self.belief

    def init_belief(self, observation) -> None:
        self.belief = self.env.init_belief(observation)

    def update_belief(self, action, observation) -> None:
        self.belief = self.env.update_belief(self.belief, action, observation)

    def sample_state(self) -> Array:
        return self.env.sample_state(self.belief)

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def reset(self, env=None, return_info=False):
        return super(FuncBeliefModel, self).reset(env, return_info)

    def query_probs(self, states):
        return self.env.query_probs(self.belief, states)
