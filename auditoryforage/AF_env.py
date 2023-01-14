from pathlib import Path
import yaml
import numpy as np
from gym import Env
from gym.spaces import Discrete, MultiDiscrete
from typing import Optional, Union
from jarvis.config import Config
from .alias import RandGen

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
    attention_possible: A numpy array of all possible attention levels. in the increasing order of attention.
    attention_cost: A numpy array of corresponding cost of attention, in the increasing order of attention.
    no_nodes: Total number of nodes in the graphical model.
    observation_possible: A numpy array of possible values of observation.
    dict_observation_possible: Dictionary version of 'observation_possible'. (key, value) pair in the dicitonary is the (index,observation) pair in the array.
    dict_action_possible: Dictionary with values as action tuples, each tuple representing (lick_choice, attention_choice).
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
        self.rng = rng if isinstance(rng, RandGen) else np.random.default_rng(rng)

        # Experimental setup
        self.prob_01 = self.spec.experiment.prob_01
        self.no_signal_nodes = self.spec.experiment.no_signal_nodes
        self.no_penalty_nodes = self.spec.experiment.no_penalty_nodes
        self.no_ITI_nodes =  self.spec.experiment.no_ITI_nodes
        self.no_nodes = 1 + self.no_signal_nodes + self.no_penalty_nodes + self.no_ITI_nodes

        # Agent's RL model parameters
        self.lick_cost = self.spec.agent.lick_cost
        self.food_reward = self.spec.agent.food_reward
        self.high_attention_cost = self.spec.agent.high_attention_cost
        self.attention_cost = np.array([0,self.high_attention_cost])
        self.penalty_cost = self.spec.agent.penalty_cost
        self.iti_cost = self.spec.agent.iti_cost

        # Agent's sensory model parameters (may or may not be known)
        self.attention_possible = np.array(self.spec.agent.attention_possible)
        self.observation_possible = np.concatenate((np.array(self.spec.agent.attention_based_obs), np.arange(1 + self.no_signal_nodes, self.no_nodes)))

        # Look-up dictionaries to map numbers used in OpenAI version (dict keys) to physical quantities in the foraging task (dict values).
        self.dict_observation_possible = dict(enumerate(self.observation_possible))
        self.dict_action_possible = dict(enumerate([(lick_choice,attention_choice) for lick_choice in range(2) for attention_choice in self.attention_possible]))

        # Agent's observation space, which may be different from experimentalist's observation space!
        self.observation_space = MultiDiscrete([len(self.dict_observation_possible)])

        # Agent's action space, note that the experimentalist might not have direct access to attention choice!
        self.action_space = Discrete(len(self.dict_action_possible))

        # State space
        self.state_space = MultiDiscrete([self.no_nodes])

        # Initial state is the beginning of pink noise
        self.state = 1 + self.no_signal_nodes + self.no_penalty_nodes

        # Initial rewards collected is 0
        self.collected_reward = 0

    def get_param(self):
        """
        Returns environment parameters.
        """

        env_param = (
            self.lick_cost,
            self.food_reward,
            self.high_attention_cost,
        )
        return env_param

    def set_param(self, env_param):
        """
        Updates environment with parameters.
        """

        self.lick_cost = env_param[0]
        self.food_reward = env_param[1]
        self.high_attention_cost = env_param[2]
        self.attention_cost = np.array([0,self.high_attention_cost])

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

        attention_cost_value = self.attention_cost[list(self.attention_possible).index(attention_choice)]
        lick_cost_value = lick_choice * self.lick_cost

        if self.state>=1 and self.state<=self.no_signal_nodes and lick_choice == 1:
            food_reward_value = self.food_reward
        else:
            food_reward_value = 0

        if self.state == self.no_signal_nodes + 1:
            penalty_cost_value = self.penalty_cost
        else:
            penalty_cost_value = 0

        if self.state == self.no_signal_nodes + 2:
            iti_cost_value = self.iti_cost
        else:
            iti_cost_value = 0

        rw = food_reward_value + attention_cost_value + lick_cost_value + penalty_cost_value + iti_cost_value

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
                next_state = self.state + np.random.choice(2, p=[1-self.prob_01, self.prob_01])

        # If current state is in the beginning of tone cloud with target
        if self.state>=1 and self.state<self.no_signal_nodes:
            if lick_choice == 0: #time passes by
                next_state = self.state + 1
            else: #goes to ITI
                next_state = 1 + self.no_signal_nodes + self.no_penalty_nodes

        # If current state is in the end of tone cloud without target
        if self.state == self.no_signal_nodes:
            next_state = 1 + self.no_signal_nodes + self.no_penalty_nodes

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
            if attention_choice == 0:
                obs = list(self.observation_possible).index(0.5)
            else:
                obs = list(self.observation_possible).index(0)

        if self.state>=1 and self.state<=self.no_signal_nodes:
            if attention_choice == 0:
                obs = list(self.observation_possible).index(0.5)
            else:
                obs = list(self.observation_possible).index(1)

        if self.state>=1 + self.no_signal_nodes and self.state<self.no_nodes:
            obs = list(self.observation_possible).index(self.state)

        obs = (obs,)
        return obs

    def step(self, action):
        """
        One time step in the POMDP.
        """

        done = False
        info = {}

        lick_choice, attention_choice = self.dict_action_possible[action]
        current_state = self.state

        # Reward
        rw = self.find_reward(lick_choice, attention_choice)
        self.collected_reward += rw

        # State transition
        self.transition_step(lick_choice)

        # Observation
        obs = self.observe_step(attention_choice)

        # self.render(current_state, lick_choice, attention_choice, rw, obs)

        return obs, rw, done, info

    def reset(self):
        """
        Resetting to beginning of ITI period.
        """

        self.state = 1 + self.no_signal_nodes + self.no_penalty_nodes

        obs = self.observe_step(0)
        # return self.state
        return obs

    def render(self, current_state, lick_choice, attention_choice, rw, obs):
        """
        Printing trajectory
        """

        print(f"Current State : {current_state}\nLick Choice : {lick_choice}\nAttention Choice : {attention_choice}\nReward Received: {rw}\nNext State: {self.state}\nNext Observation: {self.observation_possible[obs]}")
        print(f"Total Reward : {self.collected_reward}")
        print("=============================================================================")


    #Had to add this for printing in ipynb. Not sure if it's needed.
    def query_states(self):
        r"""Query states for belief visualization."""
        return [(1,)]


    def update_belief(self, previous_belief, observation, action):
        """
        Updating belief, given previous belief, new observation, and past action.
        """

        lick_choice, attention_choice = self.dict_action_possible[action]

        transition_matrix = self.find_transition_matrix()
        observation_matrix = self.find_observation_matrix()
        new_belief = np.zeros(self.no_nodes)

        for state in range(self.no_nodes):
            # note the transpose below, because of the way we made transition_matrix: (current state, next state, action)
            new_belief[state] = observation_matrix[observation,state,int(attention_choice)] * np.reshape(np.transpose(transition_matrix[:,state,int(lick_choice)]),(1,self.no_nodes)) @ previous_belief
        new_belief = new_belief/np.sum(new_belief) #Normalization
        return new_belief


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
        transition_matrix[(self.no_signal_nodes,1 + self.no_signal_nodes + self.no_penalty_nodes,0)] = 1
        transition_matrix[(self.no_nodes-1,0,0)] = 1
        for i in range(1,self.no_nodes - 1):
            if i != self.no_signal_nodes:
                transition_matrix[(i,i+1,0)] = 1

        # lick cases
        transition_matrix[(0,self.no_signal_nodes+1,1)] = 1
        for i in range(1,self.no_signal_nodes+1):
            transition_matrix[(i,1 + self.no_signal_nodes + self.no_penalty_nodes,1)] = 1
        for i in range(self.no_signal_nodes+1,self.no_nodes-1):
            transition_matrix[(i,i+1,1)] = 1
        transition_matrix[(self.no_nodes-1,0,1)] = 1

        return transition_matrix


    def find_observation_matrix(self):
        """
        Function returns the observation matrix of the form observation_matrix(obs at (t+1), state at (t+1), action at (t)).
        Representing the probability O(obs at (t+1)|state at (t+1),action at (t))
        """

        stimulus = np.zeros(self.no_nodes)
        stimulus[1:self.no_signal_nodes+1] = 1
        observation_matrix = np.zeros((len(self.observation_possible),self.no_nodes,len(self.attention_possible)))
        # Considering 'trial' (partially observable) nodes
        for i in range(self.no_signal_nodes+1):
            observation_matrix[np.where(self.observation_possible == 0.5)[0][0],i,0] = 1
            observation_matrix[np.where(self.observation_possible == stimulus[i])[0][0],i,1] = 1
        # Considering 'non-trial' (fully observable) nodes
        for i in range(self.no_signal_nodes+1,self.no_nodes):
            observation_matrix[np.where(self.observation_possible == i)[0][0],i,0] = 1
            observation_matrix[np.where(self.observation_possible == i)[0][0],i,1] = 1
        return observation_matrix

    def init_belief(self, observation):
        r"""Initializes belief with observation.

        Args
        ----
        observation:
            Initial observation at the start of an episode, may not be provided
            by the current environment.

        Returns
        -------
        belief:
            A belief vector compatible with the given observation.

        """

        if not(self.no_nodes==7 and len(self.observation_possible)==5):
            raise NotImplementedError("Only the example environment is implemented.")
        belief = np.zeros(shape=7)
        if observation==0:
            belief[0] = 1
        elif observation==1:
            belief[:5] = 0.2
        elif observation==2:
            belief[1:5] = 0.25
        elif observation==3:
            belief[6] = 1
        elif observation==4:
            belief[7] = 1
        return belief

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
