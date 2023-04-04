import time
import numpy as np
import torch
from torch.distributions.categorical import Categorical

from collections.abc import Callable, Iterable
from typing import Optional, Union
from gym.spaces import MultiDiscrete, Box

from jarvis.config import Config

from . import rcParams
from .distribution import BaseDistribution, _create_distribution
from .net import BaseDistributionNet, _create_net
from .alias import (
    Array, Tensor, RandGen, GymEnv, V21GymEnv, EnvParam,
    Action, State, Observation, Belief,
)
from .utils import get_dtype


#TODO add Episode class


class ReplayBuffer:
    r"""Replay buffer for an agent."""
    KEYS = [
        'states', 'observations', 'beliefs', 'actions', 'logits', 'rewards',
    ]

    def __init__(self,
        capacity: int,
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

    def state_dict(self) -> dict:
        r"""Returns state dictionary."""
        state_dict = {
            'episodes': self.episodes,
            'sizes': self._ts,
        }
        return state_dict

    def load_state_dict(self, state_dict: dict):
        r"""Loads state dictionary."""
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


class BaseBeliefModel(V21GymEnv):
    r"""Base class of a belief MDP model.

    The belief MDP is based on a POMDP `env` assumed by an agent. A 'belief'
    vector is updated by the latest action and observation, and serves as the
    input to a policy network.

    API compatible with gym==0.21.0, regardless of `env` API version.

    """

    belief: Tensor # (param_dim,)
    belief_space: Box
    observation_space: Box

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
        *,
        env: Optional[GymEnv] = None,
        return_tensor: bool = False,
        return_info: bool = False,
    ) -> Union[Belief, tuple[Belief, dict]]:
        r"""Resets the environment.

        Args
        ----
        env:
            The actual environment to interact with. If not provided, the
            internal model will be used. If provided, an additional sampling
            step will be taken to update the internal model state, because
            sometimes `self.env` is equipped with a `query_states` method that
            returns state-dependent queries.
        return_tensor:
            Whether to return belief as a tensor. If ``False``, return numpy
            array instead.
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
        if return_tensor:
            belief = self.belief
        else:
            belief = self.belief.data.cpu().numpy()
        # update internal environment to obatin probable query states
        if not _use_internal:
            self.env.set_state(self.sample_state())
        if return_info:
            return belief, info
        else:
            return belief

    def step(self,
        action: Action,
        *,
        env: Optional[GymEnv] = None,
        return_tensor: bool = False,
    ) -> tuple[Belief, float, bool, dict]:
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
        if return_tensor:
            belief = self.belief
        else:
            belief = self.belief.data.cpu().numpy()
        # update internal environment to obatin probable query states
        if not _use_internal:
            self.env.set_state(self.sample_state())
        return belief, reward, done, info

    def get_param(self) -> Array:
        r"""Returns environment parameter."""
        return np.array([*self.env.get_param()])

    def init_belief(self, observation: Observation) -> None:
        r"""Initializes the belief.

        The belief `self.belief` about environment state is initialized based on
        the first observation.

        Args
        ----
        observation:
            Initial observation.

        """
        raise NotImplementedError

    def update_belief(self, action: Action, next_observation: Observation) -> None:
        r"""Updates the belief.

        The belief `self.belief` is updated by the most recent action and the
        following observation.

        Args
        ----
        action:
            The action taken by the agent.
        next_observation:
            The new observation returned by the environment.

        """
        raise NotImplementedError

    def sample_state(self) -> State:
        r"""Returns a state sampled from current belief.

        Can be used to reset internal model state while using an external
        environment for interaction.

        """
        raise NotImplementedError

    def state_dict(self) -> dict:
        r"""Returns state dictionary of model."""
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict):
        r"""Loads state dictionary."""
        raise NotImplementedError

    def query_probs(self, states: Iterable[State], belief: Optional[Tensor] = None) -> Array:
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

    def compute_beliefs(self,
        observations: Iterable[Observation],
        actions: Iterable[Action],
    ) -> Tensor:
        r"""Returns time series of beliefs given observations and actions.

        Args
        ----
        observations: (num_steps+1, observation_dim)
            Observations to the agent, in [0, t].
        actions: (num_steps,)
            Actions taken by the agent, in [0, t).

        Returns
        -------
        beliefs: (num_steps+1, belief_dim)
            Belief vectors, in [0, t].

        """
        self.init_belief(observations[0])
        beliefs = [self.belief]
        for t in range(len(actions)):
            self.update_belief(actions[t], observations[t+1])
            beliefs.append(self.belief)
        beliefs = torch.stack(beliefs)
        return beliefs

    def run_one_episode(self,
        policy: Optional[Callable[[Tensor], Categorical]] = None,
        env: Optional[GymEnv] = None,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
        q_states: Union[Iterable[State], Callable[[GymEnv], Iterable[State]], None] = None,
    ) -> dict:
        r"""Runs one episode.

        Args
        ----
        policy:
            Policy used to draw actions, use uniform policy if not specified.
        env:
            The actual environment to interact with, if not provided, the agent
            interacts with the internal model.
        max_steps:
            Maximum number of time steps of each episode.
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
        _rcParams = Config(rcParams.get('model.BaseBeliefModel.run_one_episode'))
        max_steps = max_steps or _rcParams.max_steps

        rng = np.random.default_rng(seed=seed)
        states, observations, actions, rewards = [], [], [], []
        beliefs, logits, ents = [], [], []
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
                logits.append(dist.logits)
                ents.append(dist.entropy())
                actions.append(action)
                rewards.append(reward)

        self.seed(seed)
        belief, info = self.reset(env=env, return_tensor=True, return_info=True)
        append_data(with_action=False)
        num_steps = 0
        while True:
            if policy is None:
                dist = Categorical(logits=torch.zeros(self.action_space.n))
            else:
                dist = policy(belief[None])
            action = rng.choice(self.action_space.n, p=dist.probs[0].data.cpu().numpy())
            belief, reward, done, info = self.step(action, env=env, return_tensor=True)
            append_data()
            num_steps += 1
            if done or num_steps==max_steps:
                break

        episode = {
            'num_steps': num_steps,
            'states': np.array(states), # [0, t]
            'observations': np.array(observations), # [0, t]
            'actions': np.array(actions), # [0, t)
            'rewards': np.array(rewards), # [0, t)
            'beliefs': torch.stack(beliefs), # [0, t]
            'logits': torch.cat(logits), # [0, t)
            'ents': torch.cat(ents), #[0, t)
        }
        if get_queries is not None:
            _q_states = np.stack(_q_states)
            _q_probs = np.stack(_q_probs)
            if np.allclose(_q_states, np.stack([_q_states[0]]*len(_q_states))): # merge fixed query set
                _q_states = _q_states[0]
            episode['q_states'] = _q_states # (t+1, num_queries, state_dim) or (num_queries, state_dim)
            episode['q_probs'] = _q_probs # (t+1, num_queries)
        return episode


class NetworkBeliefModel(BaseBeliefModel):
    r"""Belief model that uses network to initialize and update belief."""

    def __init__(self,
        env: GymEnv,
        action_depend: Optional[bool] = None,
        p_s: Union[BaseDistribution, dict, None] = None,
        p_o: Union[BaseDistribution, dict, None] = None,
        init_net: Union[BaseDistributionNet, dict, None] = None,
        update_net: Union[BaseDistributionNet, dict, None] = None,
        **kwargs,
    ):
        r"""
        Args
        ----
        env:
            See `BaseBeliefModel` for more details.
        action_depend:
            Whether the observation is dependent on the action.
        p_s:
            Distribution of state or a configuration of it.
        p_o:
            Distribution of observation or a configuration of it.
        init_net:
            Network for initializing belief or a configuration of it.
        update_net:
            Network for updating belief or a configuration of it.

        """
        _rcParams = Config(rcParams.get('model.NetworkBeliefModel._init_'))
        super().__init__(env, **kwargs)

        self.action_depend = action_depend or _rcParams.action_depend

        if p_s is None or isinstance(p_s, dict):
            p_s = Config(p_s)
            p_s.fill(_rcParams.get('p_s'))
        self.p_s = _create_distribution(p_s, self.env.state_space)
        self.p_s.rng = self.rng
        self.belief_space = Box(
            -np.inf, np.inf, shape=(self.p_s.param_dim,),
        )
        self.observation_space = self.belief_space

        if p_o is None or isinstance(p_o, dict):
            p_o = Config(p_o)
            p_o.fill(_rcParams.get('p_o'))
        self.p_o = _create_distribution(p_o, self.env.observation_space)
        self.p_o.rng = self.rng

        if init_net is None or isinstance(init_net, dict):
            init_net = Config(init_net)
            init_net.fill(_rcParams.get('init_net'))
        self.init_net = init_net # to instantiate in child class

        if update_net is None or isinstance(update_net, dict):
            update_net = Config(update_net)
            update_net.fill(_rcParams.get('update_net'))
        self.update_net = update_net # to instantiate in child class

    def _o_tensor(self, observation: Observation) -> Tensor:
        return torch.tensor(
            np.array(observation),
            dtype=get_dtype(self.env.observation_space), device=self.device,
        )

    def _s_tensor(self, state: State) -> Tensor:
        return torch.tensor(
            np.array(state),
            dtype=get_dtype(self.env.state_space), device=self.device,
        )

    def _a_tensor(self, action: Action) -> Tensor:
        return torch.tensor(
            np.array(action)[..., None], dtype=torch.long, device=self.device,
        )

    def init_belief(self, observation: Observation) -> None:
        observation = self._o_tensor(observation)
        self.belief = self.init_net(observation[None])[0]
        self.p_s.set_param_vec(self.belief)

    def update_belief(self, action: Action, next_observation: Observation) -> None:
        a_t = self._a_tensor(action)
        o_tp1 = self._o_tensor(next_observation)
        self.belief = self.update_net(
            self.belief[None], a_t[None], o_tp1[None],
        )[0]
        self.p_s.set_param_vec(self.belief)

    def state_dict(self) -> dict:
        state_dict = {
            'init_net': self.init_net.state_dict(),
            'update_net': self.update_net.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        self.init_net.load_state_dict(state_dict['init_net'])
        self.update_net.load_state_dict(state_dict['update_net'])

    def sample_state(self) -> State:
        return self.p_s.sample()

    def query_probs(self, states: Iterable[State], belief: Optional[Tensor] = None) -> Array:
        states = torch.tensor(states, dtype=get_dtype(self.env.state_space))
        with torch.no_grad():
            probs = self.p_s.loglikelihoods(states, belief).exp().cpu().numpy()
        return probs

    def train_init_net(self, **kwargs) -> dict:
        raise NotImplementedError

    def train_update_net(self, **kwargs) -> dict:
        raise NotImplementedError


class SamplingBeliefModel(NetworkBeliefModel):
    r"""Sampling-based belief model.

    The agent samples states from the current belief `p_s`, uses the internal
    environment to get new states with the action, and estimates the new belief
    after weighting state samples with the new observation.

    """

    def __init__(self,
        env: GymEnv,
        observe_net: Union[BaseDistributionNet, dict, None] = None,
        buffer_cap: Optional[int] = None,
        **kwargs,
    ):
        r"""
        Args
        ----
        env:
            See `BaseBeliefModel` for more information.
        observe_net:
            Network for describing conditional distribution of observation or a
            configuration of it.
        buffer_cap:
            Capacity of replay buffer.

        """
        _rcParams = Config(rcParams.get('model.SamplingBeliefModel._init_'))
        super().__init__(env, **kwargs)

        self.init_net = _create_net(
            self.init_net, self.env.state_space,
            [self.env.observation_space], self.p_s,
        )
        _action_space = MultiDiscrete([self.action_space.n])
        self.update_net = _create_net(
            self.update_net, self.env.state_space,
            [self.belief_space, _action_space, self.env.observation_space],
            self.p_s,
        )

        if observe_net is None or isinstance(observe_net, dict):
            observe_net = Config(observe_net)
            observe_net.fill(_rcParams.get('observe_net'))
        self.observe_net = _create_net(
            observe_net, self.env.observation_space,
            [self.env.state_space]+([_action_space] if self.action_depend else []),
            self.p_o,
        )
        self.buffer = ReplayBuffer(buffer_cap or _rcParams.buffer_cap)

        self.update_mode = 'N' # 'N' for network-based mode

    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        state_dict.update({
            'observe_net': self.observe_net.state_dict(),
            'buffer': self.buffer.state_dict(),
        })
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.observe_net.load_state_dict(state_dict['observe_net'])
        self.buffer.load_state_dict(state_dict['buffer'])

    def init_belief(self, observation: Observation) -> None:
        super().init_belief(observation)
        if self.update_mode=='S':
            self._update_stats = []

    def update_belief(self, action: Action, next_observation: Observation) -> None:
        r"""

        The belief is updated by gathering compatible next states and estimate
        a new distribution from them. At each iteration, we first sample a
        state from the current belief, and then use the internal environment
        simulator to generate a new state given the actual action. This new
        state is collected, also weighted by the actual observation through the
        pre-learned distribution p(s|o). Finally we learn a new belief using
        maximum likelihood method with the weighted state samples.

        """
        assert self.update_mode in ['N', 'S']
        if self.update_mode=='N': # network-based update
            super().update_belief(action, next_observation)
        if self.update_mode=='S': # sampling-based update
            _config = self._update_config.clone()
            _state_to_restore = self.env.get_state()
            xs = self._o_tensor(next_observation)[None]
            states, weights = [], []
            for _ in range(_config.pop('num_samples')):
                state = self.sample_state()
                self.env.set_state(state)
                self.env.step(action)
                next_state = self.env.get_state()
                states.append(next_state)
                ys = [self._s_tensor(next_state)[None]]
                if self.action_depend:
                    ys.append(self._a_tensor(action)[None])
                with torch.no_grad():
                    param_vec = self.observe_net(*ys)[0]
                    logp = self.p_o.loglikelihoods(xs, param_vec)[0]
                weights.append(logp.exp().item())
            self.env.set_state(_state_to_restore)
            self._update_stats.append(self.p_s.estimate(
                np.array(states), np.array(weights), **_config,
            ))
            self.belief = self.p_s.get_param_vec()

    def collect_rollouts(self,
        policy: Optional[Callable[[Tensor], Categorical]] = None,
        total_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
        update_config: Optional[dict] = None,
    ) -> dict:
        r"""Collects rollout using sampling-based belief updates.

        Args
        ----
        policy:
            Policy used to collect rollout.
        num_steps:
            Total number of time steps of rollouts.
        max_steps:
            Maximum number of time steps of each episode.
        update_config:
            Configuration for sampling-based belief updates.

        Returns
        -------
        collect_stats:
            Statistics of collection procedure, including estimation statistics
            of sampling-based update at all time steps.

        """
        _rcParams = Config(rcParams.get('model.SamplingBeliefModel.collect_rollouts'))
        total_steps = total_steps or _rcParams.total_steps
        max_steps = max_steps or _rcParams.max_steps
        update_config = Config(update_config)
        update_config.fill(_rcParams.update_config)

        self.update_mode = 'S'
        self._update_config = update_config
        count, num_episodes = 0, 0
        collect_stats = {
            'num_steps': total_steps, 'max_steps': max_steps,
            'steps': [], 'losses': [],
        }
        tic = time.time()
        while count<total_steps:
            episode = self.run_one_episode(
                policy=policy, max_steps=min(max_steps, total_steps-count), q_states=[],
            )
            episode['beliefs'] = episode['beliefs'].data.cpu().numpy()
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

    def train_init_net(self,
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> dict:
        r"""Trains initialization network.

        The method collects initial observations and states first, then estimate
        the conditional distribution p(s|o) from the data.

        Args
        ----
        num_samples:
            Number of initial observation-state pairs to collect.
        kwargs:
            Additional arguments for `BaseDistributionNet.estimate`.

        Returns
        -------
        train_stats:
            Statistics of training procedure, see `BaseDistributionNet.estimate`
            for more details.

        """
        _rcParams = Config(rcParams.get('model.SamplingBeliefModel.train_init_net'))
        num_samples = num_samples or _rcParams.num_samples
        # use env to collect initial states
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
        return self.init_net.estimate(
            xs=np.array(states), ys=[np.array(observations)], **kwargs,
        )

    def train_update_net(self, **kwargs) -> dict:
        r"""Trains update network.

        The method collects sampling-based belief updates in the replay buffer,
        and trains the update network based on the data.

        Args
        ----
        kwargs:
            Additional arguments for `BaseDistributionNet.regress`.

        Returns
        -------
        train_stats:
            Statistics of training procedure, see `BaseDistributionNet.regress`
            for more details.

        """
        kwargs = Config(kwargs)
        kwargs.fill(rcParams.model.SamplingBeliefModel.get('train_update_net'))

        b_t, a_t, o_tp1, b_tp1 = [], [], [], []
        for episode in self.buffer.episodes:
            b_t.append(episode['beliefs'][:-1])
            a_t.append(episode['actions'][:, None])
            o_tp1.append(episode['observations'][1:])
            b_tp1.append(episode['beliefs'][1:])
        targets = np.concatenate(b_tp1)
        ys = [
            np.concatenate(b_t), np.concatenate(a_t), np.concatenate(o_tp1),
        ]
        return self.update_net.regress(targets, ys, **kwargs)

    def train_observe_net(self,
        num_samples: Optional[int] = None,
        use_replay: Optional[bool] = None,
        **kwargs,
    ) -> dict:
        r"""Trains observation network.

        The method collects transitions and use them to estimate conditional
        distribution p(o|s) or p(o|s, a). Transitions can either be sampled from
        existing replay buffer, or created by interacting with the assumed
        environment.

        Args
        ----
        num_samples:
            Number of transitions to collect.
        use_replay:
            Whether to sample transitions from replay buffer.
        kwargs:
            Additional arguments for `BaseDistributionNet.estimate`.

        Returns
        -------
        train_stats:
            Statistics of training procedure, see `BaseDistributionNet.estimate`
            for more details.

        """
        _rcParams = Config(rcParams.get('model.SamplingBeliefModel.train_observe_net'))
        num_samples = num_samples or _rcParams.num_samples
        use_replay = use_replay or _rcParams.use_replay
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
                        continue # loop if started from a terminal state
                s_tp1 = self.env.get_state()
            observations.append(o_tp1)
            states.append(s_tp1)
            actions.append(a_t)
        self.env.set_state(_state_to_restore)
        # estimate the conditional distribution
        return self.observe_net.estimate(
            xs=np.array(observations),
            ys=[np.array(states)]+([np.array(actions)[:, None]] if self.action_depend else []),
            **kwargs,
        )

    def run_one_episode(self, **kwargs) -> dict:
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


class DistilledBeliefModel(NetworkBeliefModel):
    r"""Belief model distilled from multiple sampling-based models.

    Initialization network, update network and the policy network will be
    distilled from data gathered by multiple sampling-based models. All networks
    take environment parameters as inputs.

    """

    def __init__(self,
        env: GymEnv,
        **kwargs,
    ):
        r"""
        Args
        ----
        env:
            See `BaseBeliefModel` for more details.

        """
        super().__init__(env, **kwargs)
        env_param = self.env.get_param()
        try:
            low = np.array([*self.env.param_low])
        except:
            low = np.full((len(env_param),), fill_value=-np.inf)
        try:
            high = np.array([*self.env.param_high])
        except:
            high = np.full((len(env_param),), fill_value=np.inf)
        # additional two parameters gamma and ent_coef
        self.low = np.concatenate([low, np.array([0, 0])])
        self.high = np.concatenate([high, np.array([1, np.inf])])
        self.theta_space = Box(
            -np.inf, np.inf, shape=(len(env_param)+2,),
        )
        self.theta = torch.nn.Parameter(
            torch.randn((len(env_param)+2,), dtype=torch.float), # TODO setup default values
        )

        self.init_net = _create_net(
            self.init_net, self.env.state_space,
            [self.env.observation_space, self.theta_space], self.p_s,
        )
        _action_space = MultiDiscrete([self.action_space.n])
        self.update_net = _create_net(
            self.update_net, self.env.state_space,
            [self.belief_space, _action_space, self.env.observation_space, self.theta_space],
            self.p_s,
        )

    def _theta2param(self, thetas: Tensor) -> Tensor:
        params = []
        for i in range(thetas.shape[-1]):
            if self.low[i]>-np.inf and self.high[i]<np.inf:
                params.append(
                    (torch.tanh(thetas[..., i])+1)/2*(self.high[i]-self.low[i])+self.low[i]
                )
            elif self.low[i]>-np.inf:
                params.append(torch.exp(thetas[..., i])+self.low[i])
            elif self.high[i]<np.inf:
                params.append(self.high[i]-torch.exp(thetas[..., i]))
            else:
                params.append(thetas[..., i])
        params = torch.stack(params, dim=-1)
        return params

    def _param2theta(self, params: Tensor, eps=1e-8) -> Tensor:
        thetas = []
        for i in range(params.shape[-1]):
            if self.low[i]>-np.inf and self.high[i]<np.inf:
                thetas.append(
                    torch.atanh(((params[..., i]-self.low[i])/(self.high[i]-self.low[i])*2-1)*(1-eps))
                )
            elif self.low[i]>-np.inf:
                thetas.append(torch.log(params[..., i]-self.low[i]+eps))
            elif self.high[i]<np.inf:
                thetas.append(torch.log(self.high[i]-params[..., i]+eps))
            else:
                thetas.append(params[..., i])
        thetas = torch.stack(thetas, dim=-1)
        return thetas

    def get_param(self) -> Array:
        return self._theta2param(self.theta[None])[0, :-2].data.cpu().numpy()

    def init_belief(self, observation: Observation) -> None:
        observation = self._o_tensor(observation)
        self.belief = self.init_net(observation[None], self.theta[None])[0]
        self.p_s.set_param_vec(self.belief)

    def update_belief(self, action: Action, next_observation: Observation) -> None:
        a_t = self._a_tensor(action)
        o_tp1 = self._o_tensor(next_observation)
        self.belief = self.update_net(
            self.belief[None], a_t[None], o_tp1[None], self.theta[None],
        )[0]
        self.p_s.set_param_vec(self.belief)

    def train_init_net(self,
        observations: Array,
        params: Array,
        beliefs: Array,
        **kwargs,
    ):
        r"""Trains initialization network.

        Args
        ----
        observations: (*, observation_dim)
            Initial observations from different assumed environments.
        env_params: (*, param_dim)
            Paramters of different environments.
        beliefs: (*, belief_dim)
            Initial beliefs given by `init_net` of different agents.

        Returns
        -------
        train_stats:
            Statistics of training procedure.

        """
        _rcParams = Config(rcParams.get('model.DistilledBeliefModel.train_init_net'))
        kwargs = Config(kwargs)
        kwargs.fill(_rcParams)
        thetas = self._param2theta(torch.tensor(params)).numpy()
        return self.init_net.regress(beliefs, [observations, thetas], **kwargs)

    def train_update_net(self,
        beliefs: Array,
        actions: Array,
        next_observations: Array,
        params: Array,
        next_beliefs: Array,
        **kwargs,
    ):
        r"""Trains update network.

        Args
        ----
        beliefs: (*, belief_dim)
            Beliefs at time t of different agents.
        actions: (*,)
            Actions at time t.
        next_observations: (* belief_dim)
            New observations at time t+1.
        env_params: (*, param_dim)
            Paramters of different environments.
        next_beliefs: (*, belief_dim)
            New beliefs at time t+1 given by `update_net` of different agents.

        Returns
        -------
        train_stats:
            Statistics of training procedure.

        """
        _rcParams = Config(rcParams.get('model.DistilledBeliefModel.train_update_net'))
        kwargs = Config(kwargs)
        kwargs.fill(_rcParams)
        thetas = self._param2theta(torch.tensor(params)).numpy()
        return self.update_net.regress(
            next_beliefs, [beliefs, actions, next_observations, thetas], **kwargs,
        )


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
