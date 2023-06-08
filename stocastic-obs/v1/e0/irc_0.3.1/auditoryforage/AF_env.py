from pathlib import Path
import yaml
import numpy as np
from gym import Env
from gym.spaces import Discrete, MultiDiscrete
from typing import Optional, Union
from jarvis.config import Config
from .alias import RandGen
import torch

"""
    POMDP model
    -----------
    State 0 (tone cloud without target) and states representing tone cloud with target are treated as partially observable states.
    Remaining states are considered fully observable. i.e. inter trial time interval states (pink noise duration), and penaltly states (penalty time duration).
    If the agent pays attention and the agent's next state is in tone cloud duration, the observation at next time step is 0 (1) with certainity (1-certainity) given by the level of attention.
    If the agent pays attention and the agent's next state is in signal duration, the observation at next time step is 1 (0) with certainity (1-certainity) given by the level of attention.
    If the agent's next state is in penalty duration, the observation at next time step is 2.
    If the agent's next state is in ITI duration, the observation at next time step is 3.

    Misc. definitions
    -----------
    attention_possible: A numpy array of all possible attention levels. in the increasing order of attention.
    attention_cost: A numpy array of corresponding cost of attention, in the increasing order of attention.
    no_nodes: Total number of nodes in the graphical model.
    observation_possible: A numpy array of possible values of observation.
    dict_observation_possible: Dictionary version of 'observation_possible'. (key, value) pair in the dicitonary is the (index,observation) pair in the array.
    dict_action_possible: Dictionary with values as action tuples, each tuple representing (lick_choice, attention_choice).
"""

class AuditoryForaging(Env):
    def __init__(self,
        prob_01: float = 0.004,

        # no_signal_nodes: int = 150,
        no_signal_nodes: int = 5,
        
        no_penalty_nodes: int = 1,
        
        # no_ITI_nodes: int = 150,
        no_ITI_nodes: int = 5,

        no_attention_modes: int = 6,
        lick_cost: float = -1.0,
        food_reward: float = 3.0,
        attention_cost_coeff: float = 1.0, 
        attention_cost_temp: float = 1.0,
        penalty_cost: float = -50.0,
        iti_cost: float = 0.0,
        seed: Optional[int] = None,
    ):
        r"""
        Args
        ----
        prob_01:
            Probability of transitioning from state 0 (tone cloud) to state 1 (signal start).
        no_signal_nodes:
            Total signal duration.
        no_penalty_nodes:
            Penalty duration, in the case the agent licks in state 0 (during tone cloud).
        no_ITI_nodes:
            Inter trial time interval.
        no_attention_modes:
            No. of attention modes, with lowest having certainity 0.5 and highest having certainity 1.0.
        lick_cost:
            Cost of performing a lick.
        food_reward:
            Reward of licking food.
        attention_cost_coeff:
            Attention cost coefficient, a parameter that describes the attention cost function.            
        attention_cost_temp:
            Attention cost temperature, a parameter that describes the attention cost function.
        penalty_cost:
            The cost of going into penalty state.
        iti_cost:
            The cost of going into iti state.
        seed:
            Random seed.

        """
        self.prob_01 = prob_01
        self.no_signal_nodes = no_signal_nodes
        self.no_penalty_nodes = no_penalty_nodes
        self.no_ITI_nodes =  no_ITI_nodes
        self.lick_cost = lick_cost
        self.food_reward = food_reward
        self.attention_cost_coeff = attention_cost_coeff
        self.attention_cost_temp = attention_cost_temp
        self.no_attention_modes = no_attention_modes
        self.penalty_cost = penalty_cost
        self.iti_cost = iti_cost

        self.no_nodes = 1 + self.no_signal_nodes + self.no_penalty_nodes + self.no_ITI_nodes
        self.obs_certainity_possible = 1/(2*(self.no_attention_modes-1)) * np.arange(self.no_attention_modes) + 0.5 
        self.attention_possible = np.arange(self.no_attention_modes)
        self.observation_possible = np.arange(4)
        self.dict_observation_possible = dict(enumerate(self.observation_possible))
        self.dict_action_possible = dict(enumerate([(lick_choice,attention_choice) for lick_choice in range(2) for attention_choice in self.attention_possible]))
        
        self.observation_space = MultiDiscrete([len(self.dict_observation_possible)])
        self.action_space = Discrete(len(self.dict_action_possible))
        self.state_space = MultiDiscrete([self.no_nodes])
        self.rng = np.random.default_rng(seed)

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
            self.no_attention_modes
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
        self.no_attention_modes = env_param[6]

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

        self.attention_cost = np.array([-self.attention_cost_coeff * np.exp(certainity/self.attention_cost_temp) for certainity in self.obs_certainity_possible])
        attention_cost_value = self.attention_cost[list(self.attention_possible).index(attention_choice)] - self.attention_cost[0]
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
        rw = food_reward_value + attention_cost_value + lick_cost_value + penalty_cost_value + iti_cost_value
        return rw

    def transition_step(self, lick_choice):
        """
        Based on the current state and the lick choice, the state value is updated
        from the current state to the future state.
        """
        
        if self.state == 0:
            if lick_choice == 1: 
                next_state = 1 + self.no_signal_nodes
            else: 
                next_state = self.state + np.random.choice(2, p=[1-self.prob_01, self.prob_01])
        elif self.state == self.no_signal_nodes:
            next_state = self.no_signal_nodes + self.no_penalty_nodes + np.random.randint(1, int(self.no_ITI_nodes/3)+1)
        elif self.state == self.no_nodes-1:
            next_state = 0
        else:
            next_state = self.state + 1
        self.state = next_state

    def observe_step(self, attention_choice):
        """
        Provides the observation given the choice of attention provided at the previous time step, and the current state.
        """

        if self.state == 0:
            obs = list(self.observation_possible).index(1 - np.random.binomial(size=1, n=1, p= self.obs_certainity_possible[attention_choice])[0])
        elif self.state in range(1, self.no_signal_nodes + 1):
            obs = list(self.observation_possible).index(np.random.binomial(size=1, n=1, p= self.obs_certainity_possible[attention_choice])[0])
        elif self.state in range(self.no_signal_nodes + 1, self.no_signal_nodes + self.no_penalty_nodes + 1):
            obs  = self.observation_possible[-2]
        elif self.state in range(self.no_signal_nodes + self.no_penalty_nodes + 1, self.no_nodes):
            obs = self.observation_possible[-1]
        else:
            raise Exception("Some mistake in observe_step method, missing some case.")
        obs = (obs,)
        return obs
    
    def step(self, action):
        """
        One time step in the POMDP.
        """

        done = False
        lick_choice, attention_choice = self.dict_action_possible[action]
        rw = self.find_reward(lick_choice, attention_choice)
        self.transition_step(lick_choice)
        obs = self.observe_step(attention_choice)
        if self.state == 1 + self.no_signal_nodes:
            done = True
        truncated, info = False, {}
        return obs, rw, done, truncated, info
    
    def reset(self, seed = None):
        """
        Resetting to beginning of ITI period.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.no_signal_nodes + self.no_penalty_nodes + np.random.randint(1, int(self.no_ITI_nodes/3)+1)
        obs = self.observe_step(0)
        info = {}
        return obs, info
        
    def belief_to_normalized_tensor(self, belief):
        ############################################################
        #LOOK INTO NORMALIZING TENSOR TO DESIRABEL FORM!
        ############################################################

        # return torch.from_numpy(belief)
        return belief
    
    def update_belief(self, previous_belief, action, observation):
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
        new_belief = self.belief_to_normalized_tensor(new_belief)
        return new_belief

    def find_transition_matrix(self):
        """
        Function returns the transition matrix of the form transition_matrix(current_state,future_state,current_lick_choice).
        Note that although the usual convention is transition_matrix(next state, current state, action),
        we set it up as transition_matrix(current_state,future_state,current_lick_choice).
        Because of the above choice, some places we use np.transpose() while using this matrix.
        """

        transition_matrix = np.zeros((self.no_nodes,self.no_nodes,2))

        transition_matrix[0,0,0] = 1 - self.prob_01
        transition_matrix[0,1,0] = self.prob_01
        transition_matrix[0, self.no_signal_nodes + 1, 1] = 1
        
        temp_list = [i for i in range(1, self.no_signal_nodes)] 
        temp_list += [self.no_signal_nodes + i for i in range(1, self.no_penalty_nodes)]
        temp_list += [self.no_signal_nodes + self.no_penalty_nodes + i for i in range(1, self.no_ITI_nodes)]
        for node in temp_list:
            transition_matrix[node,node+1,:] = 1

        for from_node in [self.no_signal_nodes, self.no_signal_nodes + self.no_penalty_nodes]:
            for to_node in range(self.no_signal_nodes + self.no_penalty_nodes + 1, self.no_signal_nodes + self.no_penalty_nodes + 1 + int(self.no_ITI_nodes/3)):
                transition_matrix[from_node, to_node, :] = 1/int(self.no_ITI_nodes/3)
        
        transition_matrix[self.no_nodes - 1, 0, :] = 1
        
        return transition_matrix
    
    def find_observation_matrix(self):
        """
        Function returns the observation matrix of the form observation_matrix(obs at (t+1), state at (t+1), action at (t)).
        Representing the probability O(obs at (t+1)|state at (t+1),action at (t))
        """

        observation_matrix = np.zeros((len(self.observation_possible),self.no_nodes,len(self.attention_possible)))
        
        for attention in range(len(self.attention_possible)):
            observation_matrix[0,0,attention] = self.obs_certainity_possible[attention]
            observation_matrix[1,0,attention] = 1 - self.obs_certainity_possible[attention]
        
        for i in range(1,self.no_signal_nodes+1):
            for attention in range(len(self.attention_possible)):
                observation_matrix[0,i,attention] = 1 - self.obs_certainity_possible[attention]
                observation_matrix[1,i,attention] = self.obs_certainity_possible[attention]
        
        for i in range(self.no_signal_nodes + 1, self.no_signal_nodes + self.no_penalty_nodes + 1):
            observation_matrix[2, i, :] = 1
        for i in range(self.no_signal_nodes + self.no_penalty_nodes + 1, self.no_nodes):
            observation_matrix[3, i, :] = 1
        
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
        if observation[0] not in range(len(self.observation_possible)):
            raise Exception("Only the example environment is implemented.")
        
        belief = np.zeros(shape = self.no_nodes)
        
        if observation[0] == self.observation_possible[-2]:
            belief[self.no_signal_nodes + 1: self.no_signal_nodes + self.no_penalty_nodes + 1] = 1/self.no_penalty_nodes
        elif observation[0] == self.observation_possible[-1]:
            belief[self.no_signal_nodes + self.no_penalty_nodes + 1: self.no_nodes] = 1/self.no_ITI_nodes
        else:
            certainity_sum = np.sum(self.obs_certainity_possible)
            if observation[0] == 0:
                normalization = (1 - self.no_signal_nodes) * certainity_sum + self.no_signal_nodes * self.no_attention_modes
                belief[0] = certainity_sum/normalization
                belief[1:self.no_signal_nodes+1] = (self.no_attention_modes - certainity_sum)/normalization
            elif observation[0] == 1:
                normalization = (self.no_signal_nodes - 1) * certainity_sum + self.no_attention_modes
                belief[0] = (self.no_attention_modes - certainity_sum)/normalization
                belief[1:self.no_signal_nodes+1] = certainity_sum/normalization
        
        belief = self.belief_to_normalized_tensor(belief)

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