(-1., 3., -0.1)

import numpy as np
from gym import Env
from gym.spaces import Discrete, MultiDiscrete

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

# Problem-setup parameters
# Note: Possible attention values and their costs are entered in the increasing order
# env_spec = {
#     'exp_setup': {
#         'prob_01': .7, 'no_signal_nodes': 4., 'no_penalty_nodes': 7., 'no_ITI_nodes': 4.,
#     },
#     'agent_reward': {
#         'lick_cost': -1., 'food_reward': 10., 'attention_cost': np.array([0, -2.]),
#     },
#     'agent_sensory': {
#         'attention_possible': np.array([0, 1]),
#     },
# }
# env_spec['exp_setup']['no_nodes'] = 1 + env_spec['exp_setup']['no_signal_nodes'] + env_spec['exp_setup']['no_penalty_nodes'] + env_spec['exp_setup']['no_ITI_nodes']
# env_spec['agent_sensory']['observation_possible'] = np.concatenate((np.array([0,0.5,1]), np.arange(1 + env_spec['exp_setup']['no_signal_nodes'], env_spec['exp_setup']['no_nodes'])))

class AuditoryForaging(Env):

    def __init__(self, prob_01 = 1, no_signal_nodes = 4, no_penalty_nodes = 7, no_ITI_nodes = 4, lick_cost = -1., food_reward = 3, high_attention_cost = -0.5, attention_possible = np.array([0, 1]), attention_based_obs=np.array([0,0.5,1])):
        """
        Args
        ----
        env_spec:
            Environment specification.
        """


        # Experimental setup
        self.prob_01 = prob_01
        self.no_signal_nodes = no_signal_nodes
        self.no_penalty_nodes = no_penalty_nodes
        self.no_ITI_nodes =  no_ITI_nodes
        self.no_nodes = 1 + self.no_signal_nodes + self.no_penalty_nodes + self.no_ITI_nodes

        # Agent's RL model parameters
        self.lick_cost = lick_cost
        self.food_reward = food_reward
        self.high_attention_cost = high_attention_cost
        self.attention_cost = np.array([0,self.high_attention_cost])

        # Agent's sensory model parameters (may or may not be known)
        self.attention_possible = attention_possible
        self.observation_possible = np.concatenate((attention_based_obs, np.arange(1 + self.no_signal_nodes, self.no_nodes)))

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

    def get_env_param(self):
        """
        Returns environment parameters.
        """

        env_param = (
            self.lick_cost,
            self.food_reward,
            self.high_attention_cost,
        )
        return env_param

    def set_env_param(self, env_param):
        """
        Updates environment with parameters.
        """

        self.lick_cost = env_param[0]
        self.food_reward = env_param[1]
        self.high_attention_cost = env_param[2]

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

        rw = food_reward_value + attention_cost_value + lick_cost_value
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


#
# env = AuditoryForaging()
# done = False
# state = env.reset()
# while not done:
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
