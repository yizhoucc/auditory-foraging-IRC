from pathlib import Path
import yaml
import numpy as np
from gym import Env
from gym.spaces import Discrete, MultiDiscrete
from typing import Optional, Union
from jarvis.config import Config
from .alias import RandGen

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
        no_signal_nodes: int = 150,
        no_penalty_nodes: int = 1,
        no_ITI_nodes: int = 150,
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
        # Experimental setup
        self.prob_01 = prob_01
        self.no_signal_nodes = no_signal_nodes
        self.no_penalty_nodes = no_penalty_nodes
        self.no_ITI_nodes =  no_ITI_nodes
        self.no_nodes = 1 + self.no_signal_nodes + self.no_penalty_nodes + self.no_ITI_nodes

        # Agent's parameters
        self.lick_cost = lick_cost
        self.food_reward = food_reward
        self.attention_cost_coeff = attention_cost_coeff
        self.attention_cost_temp = attention_cost_temp
        self.no_attention_modes = no_attention_modes
        self.obs_certainity_possible = 1/(2*(self.no_attention_modes-1)) * np.arange(self.no_attention_modes) + 0.5 
        self.penalty_cost = penalty_cost
        self.iti_cost = iti_cost
        self.attention_possible = np.arange(self.no_attention_modes)
        self.observation_possible = np.arange(4)
        self.dict_observation_possible = dict(enumerate(self.observation_possible))
        self.dict_action_possible = dict(enumerate([(lick_choice,attention_choice) for lick_choice in range(2) for attention_choice in self.attention_possible]))
        
        self.observation_space = MultiDiscrete([len(self.dict_observation_possible)])
        self.action_space = Discrete(len(self.dict_action_possible))
        self.state_space = MultiDiscrete([self.no_nodes])
        
        self.rng = np.random.default_rng(seed)

        
