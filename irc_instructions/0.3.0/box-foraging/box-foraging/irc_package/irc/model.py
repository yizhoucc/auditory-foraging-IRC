import time
import numpy as np
from scipy.special import softmax
import torch
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset

from collections.abc import Callable, Iterable
from typing import Optional, Union
from gym.spaces import MultiDiscrete, Discrete, Box

from jarvis.config import Config
from jarvis.utils import sgd_optimizer, create_mlp_layers

from . import rcParams
from .distribution import (
    BaseDistribution, BaseParamNet, ConditionalDistribution, _get_dtype,
)
from .alias import Array, RandGen, VarSpace, GymEnv, SB3Policy, Module
from .utils import train_and_summarize


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
            such as `get_state`, `set_state` etc. See `utils.check_env` for more
            details.
        device:
            Device for torch tensors.
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

    def reset(self,
        env: Optional[GymEnv] = None,
        seed: Optional[int] = None,
        return_info: bool = False,
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
            observation, info = env.reset(seed=seed)
        else:
            env.seed(seed)
            if return_info:
                observation, info = env.reset(return_info=True)
            else:
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
            self.env.set_state(self.sample_state())
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
            self.env.set_state(self.sample_state())
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

    def state_dict(self) -> dict:
        r"""Returns state dictionary of model."""
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict):
        r"""Loads state dictionary."""
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

    def get_beliefs(self,
        observations: Array, actions: Array,
    ) -> Array:
        r"""Returns time series of beliefs given observations and actions.

        Args
        ----
        observations: (num_steps+1, observation_dim)
            Observations to the agent, in [0, t].
        actions: (num_steps,)
            Actions taken by the agent, in [0, t).

        Returns
        -------
        beliefs: (num_steps, belief_dim)
            Belief vectors, in [0, t).

        """
        beliefs = []
        for t in range(len(actions)):
            observation = observations[t] # last observation will not be used
            if t==0:
                self.init_belief(observation)
            else:
                self.update_belief(actions[t-1], observation)
            beliefs.append(self.get_belief())
        beliefs = np.stack(beliefs)
        return beliefs

    def run_one_episode(self,
        policy: Optional[SB3Policy] = None,
        max_steps: Optional[int] = None,
        env: Optional[GymEnv] = None,
        seed: Optional[int] = None,
        q_states: Union[Iterable[Array], Callable[[GymEnv], Iterable[Array]], None] = None,
    ) -> dict:
        r"""Runs one episode.

        Args
        ----
        policy:
            Policy used to draw actions, use uniform policy if not specified.
        max_steps:
            Maximum number of time steps of each episode.
        env:
            The actual environment to interact with, if not provided, the agent
            interacts with the internal model.
        seed:
            Seed for policy and environment.
        q_states: (num_queries, *state_dim)
            Query states for presenting the belief. Probabilities of each
            queried state represented by the belief vector will be returned. It
            can also be a callable function that takes the current environment
            as input, because sometimes the states of interest is dynamically
            decided based on current observation.

        Returns
        -------
        episode:
            Results of one episode.

        """
        rng = np.random.default_rng(seed=seed)
        if policy is not None:
            _to_restore_train = policy.training # policy will be set to evaluation mode temporarily
            policy.set_training_mode(False)
        if max_steps is None:
            max_steps = rcParams['irc.model.BaseBeliefModel.run_one_episode']['max_steps']

        states, observations, beliefs, actions, logits, rewards = [], [], [], [], [], []
        if q_states is None:
            get_queries = lambda env: env.query_states()
        elif isinstance(q_states, Iterable):
            get_queries = lambda env: q_states
        else:
            assert callable(q_states)
            get_queries = q_states
        try:
            assert len(get_queries(self.env))>0
        except:
            get_queries = None
        else:
            _q_states, _q_probs = [], []

        def append_data(with_action=True):
            states.append(info['state'])
            observations.append(info['observation'])
            beliefs.append(belief)
            if get_queries is not None:
                _q_states.append(np.array(get_queries(self.env)))
                _q_probs.append(self.query_probs(_q_states[-1]))
            if with_action:
                actions.append(action)
                logits.append(logit)
                rewards.append(reward)

        belief, info = self.reset(env, seed=seed, return_info=True)
        append_data(with_action=False)
        t = 0
        while True:
            if policy is None:
                logit = np.zeros(self.action_space.n)
            else:
                with torch.no_grad():
                    dist = policy.get_distribution(
                        torch.tensor(belief, dtype=torch.float, device=policy.device)[None],
                    )
                logit = dist.distribution.logits[0].cpu().numpy()
            action = rng.choice(self.action_space.n, p=softmax(logit))
            belief, reward, done, info = self.step(action, env)
            append_data()
            t += 1
            if done or t==max_steps:
                break
        episode = {
            'num_steps': t,
            'states': np.array(states), # [0, t]
            'observations': np.array(observations), # [0, t]
            'beliefs': np.array(beliefs), # [0, t]
            'actions': np.array(actions), # [0, t)
            'logits': np.array(logits), # [0, t)
            'rewards': np.array(rewards), # [0, t)
        }
        if get_queries is not None:
            diffs = ((_q_states-_q_states[0])**2).reshape(len(_q_states), -1).sum(axis=1)
            if np.all(diffs<1e-8): # merge fixed query set
                _q_states = _q_states[0]
            episode['q_states'] = np.array(_q_states) # (num_queries, state_dim, t+1) or (num_queries, state_dim)
            episode['q_probs'] = np.array(_q_probs) # (num_queries, t+1)
        if policy is not None:
            policy.set_training_mode(_to_restore_train)
        return episode


class ReplayBuffer:
    r"""Replay buffer for an agent."""
    KEYS = [
        'states', 'observations', 'beliefs', 'actions', 'logits', 'rewards',
    ]

    def __init__(self,
        capacity: Optional[int] = None,
        rng: Union[RandGen, int, None] = None,
    ):
        r"""
        Args
        ----
        capacity:
            Maximum number of time steps saved in the buffer. When adding new
            episode exceeds the capacity, the oldest ones will be discarded.
        rng:
            Random number generator or seed.

        """
        if capacity is None:
            capacity = rcParams['irc.model.ReplayBuffer']['capacity']
        self.capacity = capacity

        self._usage = 0
        self.episodes = []
        self._ts = []
        self.rng = rng if isinstance(rng, RandGen) else np.random.default_rng(rng)

    def __repr__(self):
        r = self._usage/self.capacity
        return "Replay buffer containing {} episodes{}.".format(
            len(self.episodes), ' ({:.1%} full)'.format(r) if 0<r<1 else '',
        )

    def __len__(self):
        return self._usage

    def add_episode(self, episode: dict):
        r"""Adds an episode.

        Args
        ----
        episode:
            A dictionary containing 'actions' and other optional keys defined
            in `KEYS`. 'actions', 'logits' and 'rewards' are expected for time
            interval [0, t). 'states', 'observations' and 'beliefs' are expected
            for time interval [0, t].

        """
        num_steps = len(episode['actions'])
        assert num_steps<=self.capacity, (
            f"Episode length {num_steps} exceeds buffer capacity {self.capacity}."
        )
        self.episodes.append(dict(
            (key, episode[key]) for key in self.KEYS if key in episode
        ))
        self._ts.append(num_steps)
        self._usage += num_steps
        # remove least recent episodes
        head = 0
        while self._usage>self.capacity:
            self._usage -= len(self.episodes[head]['actions'])
            head += 1
        if head>0:
            self.episodes = self.episodes[head:]
            self._ts = self._ts[head:]

    def state_dict(self):
        state_dict = {
            'episodes': self.episodes,
            'sizes': self._ts,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        assert sum(state_dict['sizes'])<=self.capacity, "Data to load exceeds buffer capacity."
        self.episodes = state_dict['episodes']
        self._ts = state_dict['sizes']
        self._usage = sum(self._ts)

    def sample(self):
        r"""Samples a transition from buffer.

        Keys that do not exist will return `None` correspondingly.

        Returns
        -------
        s_t, s_tp1:
            State at time t and t+1.
        o_t, o_tp1:
            Observation at time t and t+1.
        b_t, b_tp1:
            Belief at time t and t+1.
        l_t:
            Logit at time t.
        a_t:
            Action at time t.
        r_t:
            Reward at time t.

        """
        p = np.array(self._ts)/np.sum(self._ts)
        e_idx = self.rng.choice(len(self.episodes), p=p)
        episode = self.episodes[e_idx]
        t = self.rng.choice(self._ts[e_idx])
        if 'states' in episode:
            s_t = episode['states'][t]
            s_tp1 = episode['states'][t+1]
        else:
            s_t = s_tp1 = None
        if 'observations' in episode:
            o_t = episode['observations'][t]
            o_tp1 = episode['observations'][t+1]
        else:
            o_t = o_tp1 = None
        if 'beliefs' in episode:
            b_t = episode['beliefs'][t]
            b_tp1 = episode['beliefs'][t+1]
        else:
            b_t = b_tp1 = None
        l_t = episode['logits'][t] if 'logits' in episode else None
        a_t = episode['actions'][t]
        r_t = episode['rewards'][t] if 'rewards' in episode else None
        return s_t, o_t, b_t, l_t, a_t, r_t, s_tp1, o_tp1, b_tp1


class BeliefNet(Module):

    def __init__(self,
        belief_space: Box,
        action_space: Discrete,
        observation_space: VarSpace,
        mlp_features: Optional[list[int]] = None,
        **kwargs,
    ):
        super().__init__()
        self.belief_space = belief_space
        self.action_space = action_space
        self.observation_space = observation_space
        belief_dim, = self.belief_space.shape
        action_dim = self.action_space.n
        if isinstance(self.observation_space, MultiDiscrete):
            observation_dim = sum(self.observation_space.nvec)
        if isinstance(self.observation_space, Box):
            observation_dim, = self.observation_space.shape
        if mlp_features is None:
            mlp_features = rcParams['irc.model.BeliefNet']['mlp_features']
        self.layers = create_mlp_layers(
            in_features=belief_dim+action_dim+observation_dim, out_features=belief_dim,
            num_features=mlp_features, **kwargs,
        )

    def forward(self, b_t, a_t, o_tp1):
        a_t = one_hot(a_t, self.action_space.n)
        if isinstance(self.observation_space, MultiDiscrete):
            o_tp1 = torch.cat([
                one_hot(o_tp1[:, i], self.observation_space.nvec[i])
                for i in range(len(self.observation_space.nvec))
            ], dim=1)
        out = torch.cat([b_t, a_t, o_tp1], dim=1)
        for layer in self.layers:
            out = layer(out)
        return out


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
        p_s: Union[BaseDistribution, dict, None] = None,
        p_o: Union[BaseDistribution, dict, None] = None,
        p_s_o_net: Union[BaseParamNet, dict, None] = None,
        p_o_s_net: Union[BaseParamNet, dict, None] = None,
        belief_net: Union[BeliefNet, dict, None] = None,
        update_config: Optional[dict] = None,
        buffer: Union[ReplayBuffer, dict, None] = None,
        **kwargs,
    ):
        r"""
        Args
        ----
        env:
            See `BaseBeliefModel` for more information.
        action_depend:
            Whether the observation is dependent on the action, i.e. whether to
            model p(o|s) or p(o|s, a).
        p_s:
            Distribution or the configuration about state. `p_s` is used either
            for maintaining the belief, or providing the distribution structure
            for conditional distribution p(s|o) when initializing the belief.
        p_o:
            Distribution or the configuration about observation. `p_o` is used
            for the conditional distribution p(o|s) or p(o|s, a).
        p_s_o_net:
            Parameter network or the configuration for p(s|o).
        p_o_s_net:
            Parameter network or the configuration for p(o|s) or p(o|s, a).
        buffer:
            Replay buffer to keep track recent experience with internal model.

        """
        super().__init__(env, **kwargs)
        self.action_depend = action_depend

        self.p_s_o = ConditionalDistribution(
            self.env.state_space, self.env.observation_space, p_s, p_s_o_net,
        )
        self.p_s_o.param_net.to(self.device)
        self.p_s = self.p_s_o.p_x
        self.p_s.to(self.device)
        self.observation_space = Box(-np.inf, np.inf, shape=self.p_s.get_param_vec().shape)

        if self.action_depend:
            assert isinstance(self.env.state_space, MultiDiscrete)
            _y_space = MultiDiscrete(list(self.env.state_space.nvec)+[self.env.action_space.n])
        else:
            _y_space = self.env.state_space
        self.p_o_s = ConditionalDistribution(
            self.env.observation_space, _y_space, p_o, p_o_s_net,
        )
        self.p_o_s.param_net.to(self.device)

        if belief_net is None or isinstance(belief_net, dict):
            belief_net = Config(belief_net)
            if '_target_' not in belief_net:
                belief_net._target_ = 'irc.model.BeliefNet'
            belief_net = belief_net.instantiate(
                belief_space=self.observation_space,
                action_space=self.action_space,
                observation_space=self.env.observation_space,
            )
        else:
            assert belief_net.belief_space==self.observation_space
            assert belief_net.action_space==self.action_space
            assert belief_net.observation_space==self.env.observation_space
        self.belief_net: BeliefNet = belief_net

        self.update_config = Config(update_config)
        self.update_config.fill(rcParams['irc.model.SamplingBeliefModel']['update_config'])

        if buffer is None or isinstance(buffer, dict):
            buffer = Config(buffer)
            if '_target_' not in buffer:
                buffer._target_ = 'irc.model.ReplayBuffer'
            buffer = buffer.instantiate(rng=self.rng)
        else:
            assert isinstance(buffer, ReplayBuffer)
        self.buffer: ReplayBuffer = buffer

        self.update_mode = 'N'

    def estimate_p_s_o(self, num_samples: Optional[int] = None, **kwargs):
        r"""Estimates p(s|o)."""
        # use env to collect initial states
        if num_samples is None:
            num_samples = rcParams['irc.model.SamplingBeliefModel.estimate_p_s_o']['num_samples']
        _state_to_restore = self.env.get_state()
        states, observations = [], []
        for _ in range(num_samples):
            if self.api=='v26':
                observation, _ = self.env.reset()
            else:
                observation = self.env.reset()
            states.append(self.env.get_state())
            observations.append(observation)
        self.env.set_state(_state_to_restore)
        # estimate the initial belief
        return self.p_s_o.estimate(np.array(states), np.array(observations), **kwargs)

    def estimate_p_o_s(self,
        num_samples: Optional[int] = None,
        use_replay: Optional[bool] = None,
        **kwargs,
    ):
        r"""Estimates p(o|s)."""
        if num_samples is None:
            num_samples = rcParams['irc.model.SamplingBeliefModel.estimate_p_o_s']['num_samples']
        if use_replay is None:
            use_replay = rcParams['irc.model.SamplingBeliefModel.estimate_p_o_s']['use_replay']
        # use env to collect state/observation pairs
        _state_to_restore = self.env.get_state()
        observations, states, actions = [], [], []
        for _ in range(num_samples):
            if len(self.buffer)>0 and use_replay:
                _, _, _, _, a_t, _, s_tp1, o_tp1, _ = self.buffer.sample()
            else:
                while True:
                    try:
                        s_t = self.env.state_space.sample()
                        a_t = self.env.action_space.sample()
                        self.env.set_state(s_t)
                        o_tp1, *_ = self.env.step(a_t)
                        break
                    except:
                        continue # loop if starting from a terminal state
                s_tp1 = self.env.get_state()
            observations.append(o_tp1)
            states.append(s_tp1)
            actions.append(a_t)
        self.env.set_state(_state_to_restore)
        # estimate the conditional distribution
        if self.action_depend:
            ys = np.concatenate([np.array(states), np.array(actions)[:, None]], axis=1)
        else:
            ys = np.array(states)
        return self.p_o_s.estimate(np.array(observations), ys, **kwargs)

    def state_dict(self):
        state_dict = {
            'p_s': self.p_s.state_dict(),
            'p_s_o': self.p_s_o.param_net.state_dict(),
            'p_o_s': self.p_o_s.param_net.state_dict(),
            'belief_net': self.belief_net.state_dict(),
            'buffer': self.buffer.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.p_s.load_state_dict(state_dict['p_s'])
        self.p_s_o.param_net.load_state_dict(state_dict['p_s_o'])
        self.p_o_s.param_net.load_state_dict(state_dict['p_o_s'])
        self.belief_net.load_state_dict(state_dict['belief_net'])
        self.buffer.load_state_dict(state_dict['buffer'])

    def get_belief(self):
        return self.p_s.get_param_vec().cpu().numpy()

    def init_belief(self, observation):
        with torch.no_grad():
            observation = torch.tensor(
                np.array(observation), dtype=_get_dtype(self.env.observation_space),
            )
            param_vec = self.p_s_o.param_net(observation[None])[0]
        self.p_s.set_param_vec(param_vec)
        if self.update_mode=='S':
            self._update_stats = []

    def update_belief(self, action, observation):
        r"""

        The belief is updated by gathering compatible next states and estimate
        a new distribution from them. At each iteration, we first sample a
        state from the current belief, and then use the internal environment
        simulator to generate a new state given the actual action. This new
        state is collected, also weighted by the actual observation through the
        pre-learned distribution p(s|o). Finally we learn a new belief using
        maximum likelihood method with the weighted state samples.

        """
        assert self.update_mode in ['S', 'N']
        if self.update_mode=='S': # sampling-based update
            config = self.update_config.clone()
            _state_to_restore = self.env.get_state()
            xs = torch.tensor(
                np.array(observation), dtype=_get_dtype(self.env.observation_space),
            )[None]
            states, weights = [], []
            for _ in range(config.pop('num_samples')):
                state = self.p_s.sample()
                self.env.set_state(state)
                self.env.step(action)
                next_state = self.env.get_state()
                states.append(next_state)
                if self.action_depend:
                    ys = torch.tensor(np.concatenate([
                        np.array(next_state), np.array([action]),
                    ], axis=0), dtype=torch.long)[None]
                else:
                    ys = torch.tensor(
                        np.array(next_state), dtype=_get_dtype(self.env.state_space),
                    )[None]
                with torch.no_grad():
                    logp = self.p_o_s.loglikelihoods(xs, ys)[0]
                weights.append(logp.exp().item())
            self.env.set_state(_state_to_restore)
            self._update_stats.append(self.p_s.estimate(
                np.array(states), np.array(weights), **config,
            ))
        if self.update_mode=='N': # network-based update
            with torch.no_grad():
                b_t = self.p_s.get_param_vec()
                a_t = torch.tensor(action, dtype=torch.long, device=self.device)
                o_tp1 = torch.tensor(observation, dtype=_get_dtype(self.env.observation_space), device=self.device)
                b_tp1 = self.belief_net(
                    b_t[None], a_t[None], o_tp1[None]
                )[0]
            self.p_s.set_param_vec(b_tp1)
            # TODO keep track of actions and observations in a separate buffer
            # TODO deal with out-of-distribution b_tp1

    def collect_rollouts(self,
        policy: Optional[SB3Policy] = None,
        num_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        if num_steps is None:
            num_steps = rcParams['irc.model.SamplingBeliefModel.collect_rollouts']['num_steps']
        if max_steps is None:
            max_steps = rcParams['irc.model.SamplingBeliefModel.collect_rollouts']['max_steps']
        self.update_mode = 'S'
        count, num_episodes = 0, 0
        collect_stats = {
            'num_steps': num_steps, 'max_steps': max_steps,
            'steps': [], 'losses': [],
        }
        tic = time.time()
        while count<num_steps:
            episode = self.run_one_episode(policy=policy, max_steps=max_steps, q_states=[])
            self.buffer.add_episode({
                k: v for k, v in episode.items()
                if k in ['actions', 'states', 'observations', 'beliefs']
            })
            collect_stats['steps'].append(episode['update_stats']['steps']+count)
            collect_stats['losses'].append(episode['update_stats']['losses'])
            count += episode['num_steps']
            num_episodes += 1
        collect_stats['num_episodes'] = num_episodes
        for key in ['steps', 'losses']:
            collect_stats[key] = np.concatenate(collect_stats[key])
        toc = time.time()
        collect_stats['t_elapse'] = toc-tic
        self.update_mode = 'N'
        return collect_stats

    def prepare_dataset(self):
        b_t, a_t, o_tp1, b_tp1 = [], [], [], []
        for episode in self.buffer.episodes:
            b_t.append(episode['beliefs'][:-1])
            a_t.append(episode['actions'])
            o_tp1.append(episode['observations'][1:])
            b_tp1.append(episode['beliefs'][1:])
        dset = TensorDataset(
            torch.tensor(np.concatenate(b_t), dtype=torch.float),
            torch.tensor(np.concatenate(a_t), dtype=torch.long),
            torch.tensor(np.concatenate(o_tp1), dtype=_get_dtype(self.env.observation_space)),
            torch.tensor(np.concatenate(b_tp1), dtype=torch.float),
        )
        return dset

    def train_belief_net(self,
        *,
        mse_reg: Optional[float] = None,
        lr: Optional[float] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        r"""Trains belief network using replay buffer.

        Args
        ----
        lr, num_epochs, batch_size:
            See `utils.train_and_summarize` for more details.

        """
        if mse_reg is None:
            mse_reg = rcParams['irc.model.SamplingBeliefModel.train_belief_net']['mse_reg']
        if lr is None:
            lr = rcParams['irc.model.SamplingBeliefModel.train_belief_net']['lr']
        if num_epochs is None:
            num_epochs = rcParams['irc.model.SamplingBeliefModel.train_belief_net']['num_epochs']
        if batch_size is None:
            batch_size = rcParams['irc.model.SamplingBeliefModel.train_belief_net']['batch_size']
        dset = self.prepare_dataset()
        def belief_loss(b_t, a_t, o_tp1, b_tp1):
            b_est = self.belief_net(b_t, a_t, o_tp1)
            kl_loss = 0
            for i in range(len(b_tp1)):
                kl_loss += self.p_s.kl_loss(b_tp1[i], b_est[i])
            kl_loss /= len(b_tp1)
            mse_loss = ((b_est-b_tp1)**2).mean()
            loss = (1-mse_reg)*kl_loss+mse_reg*mse_loss
            return loss
        optimizer = sgd_optimizer(self.belief_net, lr=lr)

        train_stats = {'lr': lr}
        train_stats.update(train_and_summarize(
            dset, belief_loss, optimizer, num_epochs, batch_size, device=self.device,
        ))
        return train_stats

    def sample_state(self):
        return self.p_s.sample()

    def query_probs(self, states):
        states = torch.tensor(states, dtype=_get_dtype(self.env.state_space))
        with torch.no_grad():
            probs = np.exp(self.p_s.loglikelihoods(states).cpu().numpy())
        return probs

    def run_one_episode(self, **kwargs) -> dict:
        # TODO setup seed for sampling mode
        episode = super().run_one_episode(**kwargs)
        if self.update_mode=='S':
            episode['update_stats'] = {
                k: self._update_stats[0][k]
                for k in ['num_samples', 'num_epochs', 'lr', 'batch_size']
            }
            episode['update_stats']['steps'] = []
            episode['update_stats']['losses'] = []
            for e in range(episode['num_steps']):
                steps = self._update_stats[e]['steps']
                episode['update_stats']['steps'].append(steps/len(steps)+e)
                episode['update_stats']['losses'].append(self._update_stats[e]['losses'])
            for key in ['steps', 'losses']:
                episode['update_stats'][key] = np.concatenate(episode['update_stats'][key])
        return episode


class FuncBeliefModel(BaseBeliefModel):
    r"""Belief model with custom belief functions.

    The belief model calls relevant methods from the environment object, which
    typically provides faster computation of belief updates. It requires the
    environment is equipped with the following methods:
    - init_belief(observation) -> belief
    - update_belief(belief, action, observation) -> belief
    - sample_state(belief) -> state
    - query_probs(belief, states) -> probs
    More details can be found in the documents.

    """

    def __init__(self,
        env: GymEnv,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        # identify the belief dimension
        belief = super().reset()
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

    def query_probs(self, states):
        return self.env.query_probs(self.belief, states)
