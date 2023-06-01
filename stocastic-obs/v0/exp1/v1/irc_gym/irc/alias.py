from typing import Union

from numpy import ndarray as Array
from numpy.random import Generator as RandGen

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from gym import Env as GymEnv
from gym.spaces import MultiDiscrete, Discrete, Box
VarSpace = Union[MultiDiscrete, Box]

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
SB3Algo = Union[OnPolicyAlgorithm, OffPolicyAlgorithm]
from stable_baselines3.common.policies import BasePolicy as SB3Policy

from matplotlib.figure import Figure
