import numpy as np
from scipy.stats import binom

from typing import Optional, Union
from collections.abc import Iterable

from gym.spaces import MultiDiscrete, Discrete
from .alias import Array, GymEnv

class FoodBoxesEnv(GymEnv):
    r"""Foraging environment with multiple color-cued food boxes.

    In this environment, there are one or many food boxes uniformly located on a
    circle. An agent can travel to any box to open it, or to the center which is
    a vintage point. Each box is opaque, but with a 1-D color (red to blue) on
    the outside that correlates the food availability. Color cue is randomly
    drawn from a distribution depending whether there is food or not. In
    addition, when the agent is at the vintage point, color cues will become
    more reliable and return to normal when the agent leaves center.

    At each time step, the agent can choose to travel to one location, or open
    the box at its current location. If the agent is at one of the box and
    chooses to open it, he will take the food if it exists. All untouched boxes
    will randomly update the food availability (independently), i.e. food can
    appear or vanish at a fixed proability.

    """
    R_FETCH = -1.

    def __init__(self,
        num_boxes: int = 2,
        num_shades: int = 5,
        p_appear: Union[list[float], float] = 0.2,
        p_vanish: Union[list[float], float] = 0.05,
        p_cue: Union[list[float], float] = 0.6,
        lambda_center: float = 0.1,
        r_food: float = 10.,
        r_move: float = -1.,
        seed: Optional[int] = None,
    ):
        r"""
        Args
        ----
        num_boxes:
            Number of food boxes.
        num_shades:
            Number of possible colors. Color cues on each box is represented by
            an integer in [0, `num_shades`).
        p_appear:
            Food appear probability of all boxes.
        p_vanish:
            Food vanish probability of all boxes.
        p_cue:
            Color cue parameters of all boxes. When there is food, color cue
            is drawn from a binomial distribution with parameter `p_cue`.
            Otherwise, color cue is drawn with parameter `1-p_cue`.
        lambda_center:
            Advantage parameter for vintage point, in (0, 1). Discriminability
            of all color cues will be increased according to `lambda_center`.
            0 means no effect, and 1 means fully discriminable.
        r_food:
            Reward of food, relative to the fixed fetching cost.
        r_move:
            Reward rate of moving, needs to be multiplied by the distance moved.
        seed:
            Random seed.

        """
        self.num_boxes = num_boxes
        self.num_shades = num_shades
        self.p_appear = self._convert_or_check_array(p_appear, self.num_boxes)
        self.p_vanish = self._convert_or_check_array(p_vanish, self.num_boxes)
        self.p_cue = self._convert_or_check_array(p_cue, self.num_boxes)
        self.lambda_center = lambda_center
        self.r_food = r_food
        self.r_move = r_move

        # box state and position
        self.state_space = MultiDiscrete([2]*self.num_boxes+[self.num_boxes+1])
        # moving to boxes or center, fetch
        self.action_space = Discrete(self.num_boxes+2)
        # color cue and position
        self.observation_space = MultiDiscrete([self.num_shades]*self.num_boxes+[self.num_boxes+1])

        # box locations
        _thetas = np.arange(self.num_boxes)/self.num_boxes*2*np.pi
        self.locs = np.concatenate([
            np.stack([np.cos(_thetas), np.sin(_thetas)], axis=1),
            np.zeros((1, 2)),
        ])

        self.reset(seed=seed)

    @staticmethod
    def _convert_or_check_array(val: Union[list[float], float], num_boxes: int) -> Array:
        if isinstance(val, float):
            return np.ones(num_boxes)*val
        else:
            assert len(val)==num_boxes
            return np.array(val)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.has_food = np.zeros(self.num_boxes)
        self.pos = self.num_boxes
        observation = self.observe_step()
        info = {}
        return observation, info

    def step(self, action):
        reward, terminated = self.transition_step(action)
        observation = self.observe_step()
        truncated, info = False, {}
        return observation, reward, terminated, truncated, info

    def update_box(self, b_idx: int):
        r"""Updates food availability of one box."""
        if self.has_food[b_idx]==0:
            if self.rng.random()<self.p_appear[b_idx]:
                self.has_food[b_idx] = 1
        else:
            if self.rng.random()<self.p_vanish[b_idx]:
                self.has_food[b_idx] = 0

    def transition_step(self, action):
        r"""Runs one transition step."""
        reward, terminated = 0., False
        if action<=self.num_boxes: # move
            d = np.sum((self.locs[action]-self.locs[self.pos])**2)**0.5
            reward += self.r_move*d
            self.pos = action
        else: # fetch
            reward += self.R_FETCH # fixed action cost
            if self.pos<self.num_boxes and self.has_food[self.pos]==1:
                reward += self.r_food
                self.has_food[self.pos] = 0
        for b_idx in range(self.num_boxes):
            if action<=self.num_boxes or b_idx!=self.pos:
                self.update_box(b_idx)
        return reward, terminated

    def observe_step(self):
        r"""Runs one observation step."""
        observation = []
        for i in range(self.num_boxes):
            p_cue = self.p_cue[i]
            if self.pos==self.num_boxes: # better cue viewed from center
                if p_cue>0.5:
                    p_cue = self.lambda_center+(1-self.lambda_center)*p_cue
                else:
                    p_cue = (1-self.lambda_center)*p_cue
            color = self.rng.binomial(
                self.num_shades-1,
                p_cue if self.has_food[i]==1 else 1-p_cue,
            )
            observation.append(color)
        observation = tuple(observation+[self.pos])
        return observation

    def get_param(self):
        r"""Returns environment parameters."""
        env_param = np.array([
            *self.p_appear, *self.p_vanish, *self.p_cue,
            self.lambda_center, self.r_food, self.r_move,
        ])
        return env_param

    def set_param(self, env_param):
        r"""Updates environment with parameters."""
        assert len(env_param)==3*self.num_boxes+3
        self.p_appear = np.array(env_param[:self.num_boxes])
        self.p_vanish = np.array(env_param[self.num_boxes:2*self.num_boxes])
        self.p_cue = np.array(env_param[2*self.num_boxes:3*self.num_boxes])
        self.lambda_center = env_param[3*self.num_boxes]
        self.r_food = env_param[3*self.num_boxes+1]
        self.r_move = env_param[3*self.num_boxes+2]

    def get_state(self):
        r"""Returns environment state."""
        state = (*self.has_food.astype(int), self.pos)
        return state

    def set_state(self, state):
        r"""Sets environment state."""
        assert len(state)==self.num_boxes+1
        self.has_food = np.array(state[:self.num_boxes])
        self.pos = state[self.num_boxes]

    def query_states(self):
        r"""Returns query states.

        Since the agent location `self.pos` is fully observable, there are only
        2^num_boxes states of interest.

        """
        states = np.concatenate([
            np.stack(np.unravel_index(np.arange(2**self.num_boxes), [2]*self.num_boxes), axis=1),
            np.ones((2**self.num_boxes, 1))*self.pos,
        ], axis=1)
        return states
    
    def _interpret_belief(self, belief: Array) -> tuple[Array, int]:
        r"""Interprets a belief vector.
        
        Args
        ----
        belief: (2*num_boxes,)
            The belief vector following the custom format. The first half of
            `belief` is the probabilities of food exists in each box. The second
            half is the probability of agent being at each of the box, which can
            be sum up to less than 1 since the agent can also be at the center.

        Returns
        -------
        f_probs: (num_boxes,)
            Probabilities of food exists in each box.
        pos:
            Agent position based on the belief.
        
        """
        assert len(belief)==2*self.num_boxes
        f_probs = belief[:self.num_boxes] # probability of food in boxes
        p_probs = belief[self.num_boxes] # probability of agent position
        if np.all(p_probs==0):
            pos = self.num_boxes
        else:
            (pos,), = (p_probs==1).nonzero()
        return f_probs, pos
    
    def init_belief(self, observation: Array) -> Array:
        r"""Initializes the custom belief."""
        f_probs = np.ones(self.num_boxes) 
        p_probs = np.ones(self.num_boxes) 
        belief = np.concatenate([f_probs, p_probs])
        return belief
    
    def update_belief(self, belief: Array, action: int, observation: Array) -> Array:
        r"""Updates the custom belief with action and observation."""
        f_probs, pos = self._interpret_belief(belief)
        # update f_probs
        if action==self.num_boxes+1 and pos<self.num_boxes:
            f_probs[pos] = 0
        for b_idx in range(self.num_boxes):
            if action<=self.num_boxes or b_idx!=pos:
                f_probs[b_idx] = f_probs[b_idx]*(1-self.p_vanish[b_idx])\
                    +(1-f_probs[b_idx])*self.p_appear[b_idx]
        for b_idx in range(self.num_boxes):
            color = observation[b_idx]
            p_cue = self.p_cue[b_idx]
            if pos==self.num_boxes: # better cue viewed from center
                if p_cue>0.5:
                    p_cue = self.lambda_center+(1-self.lambda_center)*p_cue
                else:
                    p_cue = (1-self.lambda_center)*p_cue
            _p_true = f_probs[b_idx]*binom.pmf(color, self.num_shades-1, p_cue)
            _p_false = (1-f_probs[b_idx])*binom.pmf(color, self.num_shades-1, 1-p_cue)
            f_probs[b_idx] = _p_true/(_p_true+_p_false)
        # update p_probs
        p_probs = np.zeros(self.num_boxes)
        if action<self.num_boxes:
            p_probs[action] = 1.
        belief = np.concatenate([f_probs, p_probs])
        return belief
    
    def sample_state(self, belief: Array) -> Array:
        r"""Samples a state from a custom belief."""
        f_probs, pos = self._interpret_belief(belief)
        has_food = self.rng.uniform(size=self.num_boxes)<f_probs
        state = (*has_food.astype(int), pos)
        return state
    
    def query_probs(self, belief: Array, states: Iterable[Array]) -> Array:
        r"""Returns probabilities of queried states given a custom belief."""
        f_probs, pos = self._interpret_belief(belief)
        probs = np.ones(len(states))
        for i, state in enumerate(states):
            for b_idx in range(self.num_boxes):
                if state[b_idx]:
                    probs[i] *= f_probs[b_idx]
                else:
                    probs[i] *= 1-f_probs[b_idx]
            if states[self.num_boxes]!=pos:
                probs[i] = 0
        return probs


class IdenticalBoxesEnv(FoodBoxesEnv):
    r"""Foraging environment with identical boxes.

    Parameters of the boxes are the same, while food availability and color cues
    are still independent.

    """

    def __init__(self,
        p_appear: float = 0.2,
        p_vanish: float = 0.05,
        p_cue: float = 0.6,
        **kwargs,
    ):
        super().__init__(p_appear=p_appear, p_vanish=p_vanish, p_cue=p_cue, **kwargs)

    def get_param(self):
        env_param = np.array([
            self.p_appear[0], self.p_vanish[0], self.p_cue[0],
            self.lambda_center, self.r_food, self.r_move,
        ])
        return env_param

    def set_param(self, env_param):
        assert len(env_param)==6
        self.p_appear = np.full(self.num_boxes, fill_value=env_param[0])
        self.p_vanish = np.full(self.num_boxes, fill_value=env_param[1])
        self.p_cue = np.full(self.num_boxes, fill_value=env_param[2])
        self.lambda_center = env_param[3]
        self.r_food = env_param[4]
        self.r_move = env_param[5]
