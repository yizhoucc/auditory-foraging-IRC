import gym
from gym.spaces import MultiDiscrete, Discrete
import numpy as np
from typing import Optional


class FoodBoxEnv(gym.Env):

    def __init__(self,
        num_cue_levels: int = 3,
        p_appear: float = 0.1,
        p_cue: float = 0.8,
        r_food: float = 10.,
        seed: Optional[int] = None,
    ):
        self.num_cue_levels = num_cue_levels
        self.p_appear, self.p_cue = p_appear, p_cue
        self.r_food, self.r_open = r_food, -1.

        self.state_space = MultiDiscrete([2]) # (has_food,)
        self.action_space = Discrete(2) # 'OPEN' or 'WAIT'
        self.observation_space = MultiDiscrete([self.num_cue_levels]) # (color,)

        self.reset(seed=seed)

    def reset(self, seed=None, return_info=False):
        self.rng = np.random.default_rng(seed)
        self.has_food = 0
        observation = self.observe_step()
        if return_info:
            return observation, {'state': self.get_state()}
        else:
            return observation

    def step(self, action):
        reward, done = self.transition_step(action)
        observation = self.observe_step()
        info = {'state': self.get_state()}
        return observation, reward, done, info

    def transition_step(self, action):
        r"""Runs one transition step."""
        reward, done = 0., False
        if action==0: # WAIT
            if self.has_food==0 and self.rng.random()<self.p_appear:
                self.has_food = 1
        else: # OPEN
            reward += self.r_open
            if self.has_food==1:
                reward += self.r_food
                self.has_food = 0
        return reward, done

    def observe_step(self):
        r"""Runs one observe step."""
        p = self.p_cue if self.has_food==1 else 1-self.p_cue
        color = self.rng.binomial(self.num_cue_levels-1, p)
        observation = (color,)
        return observation

    def get_env_param(self):
        r"""Returns environment parameters."""
        env_param = (self.p_appear, self.p_cue, self.r_food)
        return env_param

    def set_env_param(self, env_param):
        r"""Updates environment with parameters."""
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
