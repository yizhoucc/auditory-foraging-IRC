import numpy as np
from typing import Union

from gym.spaces import MultiDiscrete, Discrete
from .alias import GymEnv, RandGen


class FoodBoxEnv(GymEnv):
    r"""Single box foraging environment.

    A minimal example of foraging experiment. Food will exist in a single box
    and the agent needs to decide whether to open the box based on the
    monochromatic cue outside the box. Food appears with a fixed probability if
    it does not already exist and the agent does not open the box at this time
    step. Color cue is drawn from a binomial distribution with probability p if
    food exists, and 1-p otherwise.

    """

    def __init__(self,
        *,
        num_shades: int = 5,
        p_appear: float = 0.2,
        p_cue: float = 0.8,
        r_food: float = 10.,
        rng: Union[RandGen, int, None] = None,
    ):
        r"""
        Args
        ----
        num_shades:
            Number of shades for the color cue.
        p_appear:
            Probability of food appears at each time step.
        p_cue:
            Parameter of binomial distribution of color cue.
        r_food:
            Reward of the food, relative to the fetch cost.
        rng:
            Random number generator or seed.

        """
        self.num_shades = num_shades
        self.p_appear = p_appear
        self.p_cue = p_cue
        self.r_food = r_food

        self.state_space = MultiDiscrete([2]) # box state
        self.action_space = Discrete(2) # wait and fetch
        self.observation_space = MultiDiscrete([self.num_shades+1]) # color cue

        self.rng = rng if isinstance(rng, RandGen) else np.random.default_rng(rng)

    def reset(self, seed=None):
        self.has_food = 0
        self.rng = np.random.default_rng(seed)
        observation = self.observe_step()
        info = {}
        return observation, info

    def step(self, action):
        reward, terminated = self.transition_step(action)
        observation = self.observe_step()
        truncated, info = False, {}
        return observation, reward, terminated, truncated, info

    def transition_step(self, action):
        r"""Runs one transition step."""
        reward, terminated = 0., False
        if action==0: # wait
            if self.has_food==0 and self.rng.random()<self.p_appear:
                self.has_food = 1
        else: # fetch
            reward -= 1. # action cost
            if self.has_food==1:
                reward += self.r_food
                self.has_food = 0
        return reward, terminated

    def observe_step(self):
        r"""Runs one observation step."""
        color = self.rng.binomial(
            self.num_shades,
            self.p_cue if self.has_food==1 else 1-self.p_cue,
        )
        observation = (color,)
        return observation

    def get_param(self):
        r"""Returns environment parameters."""
        env_param = (self.p_appear, self.p_cue, self.r_food)
        return env_param

    def set_param(self, env_param):
        r"""Updates environment with parameters."""
        assert len(env_param)==3
        self.p_appear = env_param[0]
        self.p_cue = env_param[1]
        self.r_food = env_param[2]

    def get_state(self):
        r"""Returns environment state."""
        state = (self.has_food,)
        return state

    def set_state(self, state):
        r"""Sets environment state."""
        self.has_food, = state

    def query_states(self):
        r"""Query states for belief visualization."""
        return [(1,)]
