from pathlib import Path
import yaml
import numpy as np
from gym import Env
from gym.spaces import Discrete, MultiDiscrete, Box, Tuple, Dict
from typing import Optional, Union
from jarvis.config import Config
from .alias import RandGen
from numpy import random
from scipy.stats import norm

with open(Path(__file__).parent/'defaults.yaml') as f:
    D_ENV_SPEC = Config(yaml.safe_load(f)).auditory_foraging_v1
"""
    POMDP model
    -----------
    State 0 (tone cloud without target) and states representing tone cloud with target are treated as partially observable states.
    Remaining states are considered fully observable. i.e. inter trial time interval states (pink noise duration), and penaltly states (penalty time duration).
    If the agent pays attention and the agent's next state is a partially observable state, the observation is 1/0 depending on whether food is present/absent.
    If the agent does not pay attention and the agent's next state is a partially observable state, the observation is 0.5.
    If the agent's next state is a fully observable state, the observation is the next state's node index in the graphical model.

    Definitions
    -----------
    prob_01: Probability of transitioning from state 0 (tone cloud without target) to state 1 (tone cloud with target)
    no_signal_nodes: Maximum duration of tone cloud with target
    no_penalty_nodes: Penalty duration, in the case the agent licks in state 0 (during tone cloud without target)
    no_ITI_nodes: Inter trial time interval when pink noise is played.
    lick_cost: Cost of performing a lick.
    food_reward: Reward of licking food.
    attention_cost: A numpy array of corresponding cost of attention, in the increasing order of attention.
    no_nodes: Total number of nodes in the graphical model.
    """


class AuditoryForaging(Env):

    def __init__(self,
        *,
        spec: Optional[dict] = None,
        rng: Union[RandGen, int, None] = None,
    ):
        r"""
        Args
        ----
        spec:
            Environment specification.
        rng:
            Random number generator or seed.

        """
        self.spec = Config(spec).fill(D_ENV_SPEC)
        self.rng = rng if isinstance(rng, RandGen) else random.default_rng(rng)

        # Experimental setup
        self.prob_01 = self.spec.experiment.prob_01
        self.no_signal_nodes = self.spec.experiment.no_signal_nodes
        self.no_penalty_nodes = self.spec.experiment.no_penalty_nodes
        self.no_ITI_nodes =  self.spec.experiment.no_ITI_nodes
        self.no_nodes = 1 + self.no_signal_nodes + self.no_penalty_nodes + self.no_ITI_nodes

        # Agent's RL model parameters
        self.lick_cost = self.spec.agent.lick_cost
        self.food_reward = self.spec.agent.food_reward
        self.attention_cost_coeff = self.spec.agent.attention_cost_coeff
        self.attention_cost_temp = self.spec.agent.attention_cost_temp
        
        # Lokesh removed fully observable case
        self.penalty_cost = self.spec.agent.penalty_cost
        self.iti_cost = self.spec.agent.iti_cost
        self.time_in_game_reward = self.spec.agent.time_in_game_reward

        # Agent's sensory model parameters
        self.noise_obs_mean = -10
        self.signal_obs_mean = 10
        self.penalty_obs = 80
        self.iti_obs = 100
        self.min_attention_std = 5

        # Agent's observation space
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=()) # maybe change min and max

        # Agent's action space
        # self.action_space = Box(low=np.array([-10, -10]), high=np.array([1, 10]))
        # self.action_space = Box(low=np.array([-np.inf, -np.inf]), high=np.array([1, 10]))
        # self.action_space = Dict({"lick": Discrete(2), "attention": Box(low=0, high=np.inf, shape=(), dtype=np.float32)})
        self.action_space = Box(low=np.array([0, -100]), high=np.array([1, 100])) # could do sigmoid version instead, also not sure if 0,1 gaussian is a good idea

        # State space
        self.state_space = MultiDiscrete([self.no_nodes])

        # Initial state is the beginning of pink noise
        self.state = 1 + self.no_signal_nodes + self.no_penalty_nodes
        self.time = 1

        # Initial rewards collected is 0
        self.collected_reward = 0

        self.food_reward_list = None  # need to be assigned from outside
   
    def get_param(self):
        """
        Returns environment parameters.
        """

        env_param = (
            self.lick_cost,
            self.food_reward,
            self.attention_cost_coeff,
            self.attention_cost_temp,
            self.penalty_cost,
            self.iti_cost,
            self.time_in_game_reward
        )
        return env_param

    def set_param(self, env_param):
        """
        Updates environment with parameters.
        """

        self.lick_cost = env_param[0]
        self.food_reward = env_param[1]
        self.attention_cost_coeff = env_param[2]
        self.attention_cost_temp = env_param[3]
        self.penalty_cost = env_param[4]
        self.iti_cost = env_param[5]
        self.time_in_game_reward = env_param[6]

    def get_state(self):
        """
        Returns environment state.
        """

        state_tuple = (self.state,)
        return state_tuple

    def set_state(self, state_tuple):
        """
        Sets environment state.
        """

        self.state, = state_tuple

    def find_reward(self, lick_choice, attention_choice):
        """
        Computes the reward, given the choice of licking and the amount of attention.
        """

        attention_cost_value = - self.attention_cost_coeff * abs(self.signal_obs_mean - self.noise_obs_mean)/attention_choice
        lick_cost_value = lick_choice * self.lick_cost
        if self.state>=1 and self.state<=self.no_signal_nodes and lick_choice == 1:
            food_reward_value = self.food_reward
        else:
            food_reward_value = 0

        if self.state == 0 and lick_choice == 1:
            penalty_cost_value = self.penalty_cost
        else:
            penalty_cost_value = 0

        if self.state == self.no_signal_nodes + 2:
            iti_cost_value = self.iti_cost
        else:
            iti_cost_value = 0

        # rw = food_reward_value/self.time + attention_cost_value + lick_cost_value + penalty_cost_value + iti_cost_value + self.time_in_game_reward
        rw = food_reward_value + attention_cost_value + lick_cost_value + penalty_cost_value + iti_cost_value + self.time_in_game_reward
        return rw

    def transition_step(self, lick_choice):
        """
        Based on the current state and the lick choice, the state value is updated
        from the current state value to the future state value.
        """

        # If current state is node 0 (tone cloud without target)
        if self.state == 0:
            if lick_choice == 1: #penalty
                next_state = 1 + self.no_signal_nodes
            else: #no penalty
                next_state = self.state + random.choice(2, p=[1-self.prob_01, self.prob_01])

        # If current state is in the beginning of tone cloud with target
        if self.state>=1 and self.state<self.no_signal_nodes:
            if lick_choice == 0: #time passes by
                next_state = self.state + 1
            else: #goes to ITI
                next_state = self.no_signal_nodes + self.no_penalty_nodes + 1

        # If current state is in the end of tone cloud without target
        if self.state == self.no_signal_nodes:
            next_state = self.no_signal_nodes + self.no_penalty_nodes + 1

        # If current state is anywhere in between beginning of penalty period or just before the end of ITI
        if self.state>=1 + self.no_signal_nodes and self.state<self.no_nodes-1:
            next_state = self.state + 1

        # If current state is in the end of ITI
        if self.state == self.no_nodes-1:
            next_state = 0

        self.state = next_state

    def observe_step(self, attention_choice):
        """
        Provides the observation given the choice of attention provided at the previous time step, and the current state.
        """

        if self.state == 0:
            obs = random.normal(loc=self.noise_obs_mean, scale=attention_choice)

        if self.state in range(1, self.no_signal_nodes + 1):
            obs = random.normal(loc=self.signal_obs_mean, scale=attention_choice)

        if self.state in range(self.no_signal_nodes + 1, self.no_signal_nodes + self.no_penalty_nodes + 1):
            obs  = self.penalty_obs
        
        if self.state in range(self.no_signal_nodes + self.no_penalty_nodes + 1, self.no_nodes):
            obs = self.iti_obs

        obs = (obs,)
        return obs

    def transform_action(self, action):
        lick_choice = int(action[0] > 0.5)
        attention_choice = abs(action[1])
        return lick_choice, attention_choice
    
    def step(self, action):
        """
        One time step in the POMDP.
        """

        done = False
        info = {}

        lick_choice, attention_choice = self.transform_action(action)
        # Reward
        rw = self.find_reward(lick_choice, attention_choice)
        self.collected_reward += rw
        
        # State transition
        self.transition_step(lick_choice)

        # Observation
        obs = self.observe_step(attention_choice)

        if self.state > self.no_signal_nodes:
            done = True

        self.time += 1

        return obs, rw, done, info

    def reset(self, food_reward_idx=None):
        """
        Resetting to beginning of ITI period.
        """

        self.state = self.no_signal_nodes + self.no_penalty_nodes + 1

        self.time = 1

        if food_reward_idx is not None:
            self.food_reward_idx = food_reward_idx
        else:
            self.food_reward_idx = random.choice(
                list(range(len(self.food_reward_list))))
        self.food_reward = self.food_reward_list[self.food_reward_idx]
        obs = self.observe_step(self.min_attention_std)
        return obs

    def render(self, current_state, lick_choice, attention_choice, rw, obs):
        """
        Printing trajectory
        """

        print(f"Current State : {current_state}\nLick Choice : {lick_choice}\nAttention Choice : {attention_choice}\nReward Received: {rw}\nNext State: {self.state}\nNext Observation: {obs}")
        print(f"Total Reward : {self.collected_reward}")
        print("=============================================================================")

    def _compute_obs_prob(self, state, obs, attention_choice):
        """Observation probability for a given state, observation value, and attention level."""
        special_obs = (self.iti_obs, self.penalty_obs)
        if obs not in special_obs:
            if state == 0:
                return norm.pdf(obs, loc=self.noise_obs_mean, scale=attention_choice)
            elif 1 <= state <= self.no_signal_nodes:
                return norm.pdf(obs, loc=self.signal_obs_mean, scale=attention_choice)
            return 0
        penalty_start = self.no_signal_nodes + 1
        iti_start = penalty_start + self.no_penalty_nodes
        if penalty_start <= state < iti_start and obs == self.penalty_obs:
            return 1
        if iti_start <= state < self.no_nodes and obs == self.iti_obs:
            return 1
        return 0

    def _init_core_belief(self, observation):
        """Core belief initialization without extra dimensions."""
        belief = np.zeros(shape=self.no_nodes)
        if observation[0] == self.penalty_obs:
            belief[self.no_signal_nodes+1:self.no_signal_nodes+1+self.no_penalty_nodes] = 1/self.no_penalty_nodes
        elif observation[0] == self.iti_obs:
            belief[self.no_signal_nodes+1+self.no_penalty_nodes:self.no_nodes] = 1/self.no_ITI_nodes
        else:
            belief[0] = norm.pdf(observation, loc=self.noise_obs_mean, scale=self.min_attention_std)
            belief[1:self.no_signal_nodes+1] = norm.pdf(observation, loc=self.signal_obs_mean, scale=self.min_attention_std)
            belief = belief/np.sum(belief)
        return belief

    def _update_core_belief(self, core_belief, action, observation):
        """Core belief update without extra dimensions. Takes belief with extras already stripped."""
        lick_choice, attention_choice = self.transform_action(action)
        transition_matrix = self.find_transition_matrix()
        new_belief = np.zeros(self.no_nodes)
        obs = observation[0]
        lick_idx = int(lick_choice)

        for state in range(self.no_nodes):
            obs_prob = self._compute_obs_prob(state, obs, attention_choice)
            new_belief[state] = obs_prob * (transition_matrix[:, state, lick_idx] @ core_belief)

        if np.sum(new_belief) == 0:
            raise ValueError("Mistake in belief update as all probabilities are coming out to be 0 somehow")
        return new_belief / np.sum(new_belief)

    def _normalize_idx(self, idx, list_len):
        """Normalize an index to [0, 1] range, safe for single-element lists."""
        return idx / max(list_len - 1, 1)

    def update_belief(self, previous_belief, action, observation):
        """Updating belief, given previous belief, new observation, and past action."""
        new_belief = self._update_core_belief(previous_belief[:-1], action, observation)
        return np.concatenate([new_belief, [self._normalize_idx(self.food_reward_idx, len(self.food_reward_list))]])

    def find_transition_matrix(self):
        """
        Function returns the transition matrix of the form transition_matrix(current_state,future_state,current_lick_choice).
        Note that although the usual convention is transition_matrix(next state, current state, action),
        we set it up as transition_matrix(current_state,future_state,current_lick_choice).
        Because of the above choice, some places we use np.transpose() while using this matrix.
        """

        transition_matrix = np.zeros((self.no_nodes,self.no_nodes,2))

        # no lick cases
        transition_matrix[(0,0,0)] = 1 - self.prob_01
        transition_matrix[(0,1,0)] = self.prob_01

        transition_matrix[(self.no_signal_nodes, self.no_signal_nodes + self.no_penalty_nodes + 1, 0)] = 1
        
        transition_matrix[(self.no_nodes-1,0,0)] = 1
        for i in range(1,self.no_nodes - 1):
            if i != self.no_signal_nodes:
                transition_matrix[(i,i+1,0)] = 1

        # lick cases
        transition_matrix[(0,self.no_signal_nodes+1,1)] = 1

        for i in range(1,self.no_signal_nodes+1):
            transition_matrix[(i, self.no_signal_nodes + self.no_penalty_nodes + 1, 1)] = 1
        
        for i in range(self.no_signal_nodes+1,self.no_nodes-1):
            transition_matrix[(i,i+1,1)] = 1
        transition_matrix[(self.no_nodes-1,0,1)] = 1

        return transition_matrix

    def init_belief(self, observation):
        r"""Initializes belief with observation."""
        belief = self._init_core_belief(observation)
        return np.concatenate([belief, [self._normalize_idx(self.food_reward_idx, len(self.food_reward_list))]])
    
    def sample_state(self, belief):
        r"""Samples a state from the distribution described by the belief vector.

        Args
        ----
        belief:
            A belief vector that describe a distribution over states. Currently
            `belief` is an array of state probabilities, summed up to 1.

        Returns
        -------
        state:
            The tuple of environment state.

        """
        state = (self.rng.choice(self.no_nodes, p=belief),)
        return state

    def query_probs(self, belief, states):
        r"""Returns probabilities of queried states given belief vector.

        Args
        ----
        belief: (no_nodes,)
            Probabilities of all states.
        states: (num_queries, 1)
            States of interest.

        Returns
        -------
        probs: (num_queries,)
            Probabilities of each queried state.

        """
        return belief[states[:, 0].astype(int)]


class AuditoryForagingReward2(AuditoryForaging):
    '''Reward index included in belief state.
    In yc2, this is already the default behavior of the base class.'''
    pass


class AuditoryForagingReward(AuditoryForaging):
    '''Appends raw food_reward value (not normalized index) to belief.'''

    def init_belief(self, observation):
        belief = self._init_core_belief(observation)
        return np.concatenate([belief, [self.food_reward]])

    def update_belief(self, previous_belief, action, observation):
        new_belief = self._update_core_belief(previous_belief[:-1], action, observation)
        return np.concatenate([new_belief, [self.food_reward]])


class AFnorate(AuditoryForagingReward2):
    '''Reward not normalized by time. In yc2, this is already the default.'''
    pass


class AF2p(AuditoryForaging):
    '''Varying observation quality (p1) across episodes.
    Belief includes food_reward_idx and p1id as extra dimensions.
    Adapted for continuous model: p1id controls min_attention_std.'''

    def __init__(self, *, spec=None, rng=None):
        super().__init__(spec=spec, rng=rng)
        self.p1_std_options = np.array([8.0, 6.0, 4.0, 2.0])

    def reset(self, food_reward_idx=None):
        self.state = self.no_signal_nodes + self.no_penalty_nodes + 1
        self.time = 1
        if food_reward_idx is not None:
            self.food_reward_idx = food_reward_idx
        else:
            self.food_reward_idx = random.choice(list(range(len(self.food_reward_list))))
        self.p1id = np.random.choice(len(self.p1_std_options))
        self.min_attention_std = self.p1_std_options[self.p1id]
        self.food_reward = self.food_reward_list[self.food_reward_idx]
        obs = self.observe_step(self.min_attention_std)
        return obs

    def _belief_extras(self):
        return [self._normalize_idx(self.food_reward_idx, len(self.food_reward_list)), self.p1id]

    def init_belief(self, observation):
        belief = self._init_core_belief(observation)
        return np.concatenate([belief, self._belief_extras()])

    def update_belief(self, previous_belief, action, observation):
        new_belief = self._update_core_belief(previous_belief[:-2], action, observation)
        return np.concatenate([new_belief, self._belief_extras()])


class AF2pp(AuditoryForaging):
    '''Varying p1 and p2 without varying reward.
    Belief includes p1 and p2 values as extra dimensions.
    Adapted for continuous model: p1/p2 represent attention std levels.'''

    def __init__(self, *, spec=None, rng=None):
        super().__init__(spec=spec, rng=rng)
        self.food_reward_idx = 0

    def reset(self):
        self.state = self.no_signal_nodes + self.no_penalty_nodes + 1
        self.time = 1
        self.p1 = np.random.choice(np.arange(3.0, 9.0, 1.0))
        p2 = self.p1
        while p2 >= self.p1:
            p2 = np.random.choice(np.arange(2.0, 8.0, 1.0))
        self.p2 = p2
        self.min_attention_std = self.p1
        obs = self.observe_step(self.min_attention_std)
        return obs

    def init_belief(self, observation):
        belief = self._init_core_belief(observation)
        return np.concatenate([belief, [self.p1, self.p2]])

    def update_belief(self, previous_belief, action, observation):
        new_belief = self._update_core_belief(previous_belief[:-2], action, observation)
        return np.concatenate([new_belief, [self.p1, self.p2]])


class AuditoryForagingEnergy(AuditoryForaging):
    '''Tracks cumulative attention energy usage.
    Belief includes food_reward_idx and energy level as extra dimensions.'''

    def __init__(self, *, spec=None, rng=None):
        super().__init__(spec=spec, rng=rng)
        self.energycap = 15

    def reset(self, food_reward_idx=None):
        self.state = self.no_signal_nodes + self.no_penalty_nodes + 1
        self.time = 1
        if food_reward_idx is not None:
            self.food_reward_idx = food_reward_idx
        else:
            self.food_reward_idx = random.choice(list(range(len(self.food_reward_list))))
        self.food_reward = self.food_reward_list[self.food_reward_idx]
        self.previous_high_attention_count = 0
        obs = self.observe_step(self.min_attention_std)
        return obs

    def _belief_extras(self):
        return [self._normalize_idx(self.food_reward_idx, len(self.food_reward_list)), self.previous_high_attention_count/self.energycap]

    def init_belief(self, observation):
        belief = self._init_core_belief(observation)
        return np.concatenate([belief, self._belief_extras()])

    def update_belief(self, previous_belief, action, observation):
        _, attention_choice = self.transform_action(action)
        if attention_choice < self.min_attention_std:
            self.previous_high_attention_count += 1
        new_belief = self._update_core_belief(previous_belief[:-2], action, observation)
        return np.concatenate([new_belief, self._belief_extras()])

    def find_reward(self, lick_choice, attention_choice):
        lick_cost_value = lick_choice * self.lick_cost
        attention_cost_value = -self.attention_cost_coeff * abs(self.signal_obs_mean - self.noise_obs_mean)/attention_choice * 1.1**(self.previous_high_attention_count)
        if self.state >= 1 and self.state <= self.no_signal_nodes and lick_choice == 1:
            food_reward_value = self.food_reward
        else:
            food_reward_value = 0
        if self.state == 0 and lick_choice == 1:
            penalty_cost_value = self.penalty_cost
        else:
            penalty_cost_value = 0
        if self.state == self.no_signal_nodes + 2:
            iti_cost_value = self.iti_cost
        else:
            iti_cost_value = 0
        rw = food_reward_value + attention_cost_value + lick_cost_value + penalty_cost_value + iti_cost_value + self.time_in_game_reward
        return rw


class AFVaryAttention(AuditoryForaging):
    '''Varying attention cost across episodes.
    Belief includes att_idx as extra dimension.'''

    def __init__(self, *, spec=None, rng=None):
        super().__init__(spec=spec, rng=rng)
        self.att_cost_list = None

    def reset(self, att_idx=None):
        self.state = self.no_signal_nodes + self.no_penalty_nodes + 1
        self.time = 1
        if att_idx is not None:
            self.att_idx = att_idx
        else:
            self.att_idx = random.choice(list(range(len(self.att_cost_list))))
        self.att_cost = self.att_cost_list[self.att_idx]
        obs = self.observe_step(self.min_attention_std)
        return obs

    def init_belief(self, observation):
        belief = self._init_core_belief(observation)
        return np.concatenate([belief, [self._normalize_idx(self.att_idx, len(self.att_cost_list))]])

    def update_belief(self, previous_belief, action, observation):
        new_belief = self._update_core_belief(previous_belief[:-1], action, observation)
        return np.concatenate([new_belief, [self._normalize_idx(self.att_idx, len(self.att_cost_list))]])

    def find_reward(self, lick_choice, attention_choice):
        lick_cost_value = lick_choice * self.lick_cost
        attention_cost_value = -self.att_cost * abs(self.signal_obs_mean - self.noise_obs_mean)/attention_choice
        if self.state >= 1 and self.state <= self.no_signal_nodes and lick_choice == 1:
            food_reward_value = self.food_reward
        else:
            food_reward_value = 0
        if self.state == 0 and lick_choice == 1:
            penalty_cost_value = self.penalty_cost
        else:
            penalty_cost_value = 0
        if self.state == self.no_signal_nodes + 2:
            iti_cost_value = self.iti_cost
        else:
            iti_cost_value = 0
        rw = food_reward_value + attention_cost_value + lick_cost_value + penalty_cost_value + iti_cost_value + self.time_in_game_reward
        return rw
