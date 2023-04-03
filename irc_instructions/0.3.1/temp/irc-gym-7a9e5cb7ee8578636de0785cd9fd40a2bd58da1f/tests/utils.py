
import numpy as np
import torch
from gym.spaces import MultiDiscrete, Box


def _create_tensor_samples(space, num_samples=1):
    xs = np.stack([space.sample() for _ in range(num_samples)])
    if isinstance(space, MultiDiscrete):
        xs = torch.tensor(xs, dtype=torch.long)
    if isinstance(space, Box):
        xs = torch.tensor(xs, dtype=torch.float)
    return xs
