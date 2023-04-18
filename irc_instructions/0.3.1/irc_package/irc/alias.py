from typing import Optional, Union, TypeVar
from collections.abc import Collection

from numpy import ndarray as Array
from numpy.random import Generator as RandGen

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from gym.spaces import MultiDiscrete, Discrete, Box
VarSpace = Union[MultiDiscrete, Box]
State = TypeVar('State', Collection[int], Collection[float])
Observation = TypeVar('Observation', Collection[int], Collection[float])
Action = TypeVar('Action', int, float) # TODO add support for float
EnvParam = Collection[float]
Belief = TypeVar('Belief', Tensor, Array)

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
SB3Algo = OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy as SB3Policy

from matplotlib.figure import Figure

from abc import abstractmethod
import gym

class BaseGymEnv(gym.Env):

    state_space: VarSpace
    action_space: Discrete
    observation_space: VarSpace

    @abstractmethod
    def get_param(self) -> EnvParam:
        ...
    @abstractmethod
    def set_param(self, env_param: EnvParam) -> None:
        ...
    @abstractmethod
    def get_state(self) -> State:
        ...
    @abstractmethod
    def set_state(self, state: State) -> None:
        ...

    # Optional
    param_low: EnvParam
    param_high: EnvParam

class V26GymEnv(BaseGymEnv):

    @abstractmethod
    def reset(self, seed: Optional[int]) -> tuple[Observation, dict]:
        # observation, info
        ...
    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        # observation, reward, terminated, truncated, info
        ...

class V21GymEnv(BaseGymEnv):

    @abstractmethod
    def seed(self, seed: Optional[int]) -> list[int]:
        # seeds
        ...
    @abstractmethod
    def reset(self) -> Observation:
        # observation
        ...
    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        # observation, reward, done, info
        ...

GymEnv = Union[V26GymEnv, V21GymEnv]
