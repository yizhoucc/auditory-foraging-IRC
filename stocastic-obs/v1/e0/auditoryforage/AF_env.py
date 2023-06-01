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

    Definitions
    -----------
    prob_01: Probability of transitioning from state 0 (tone cloud) to state 1 (signal start).
    no_signal_nodes: Total signal duration.
    no_penalty_nodes: Penalty duration, in the case the agent licks in state 0 (during tone cloud)
    no_ITI_nodes: Inter trial time interval.
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
        seed: Optional[int] = None,
        rng: Union[RandGen, int, None] = None,
    ):
        self.rng = np.random.default_rng(seed)
