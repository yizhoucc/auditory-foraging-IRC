import numpy as np
from scipy.optimize import curve_fit
import torch
from typing import Optional, Union
from gym import Env as GymEnv
from gym.spaces import MultiDiscrete, Discrete, Box
from gym.utils.env_checker import check_env as _check_env
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy as SB3Policy
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

Array = np.ndarray
Tensor = torch.Tensor
VarSpace = Union[MultiDiscrete, Box]
SB3Algo = Union[OnPolicyAlgorithm, OffPolicyAlgorithm]
RandGen = np.random.Generator


def _exp_fit(epochs: Array, vals: Array, queries: Optional[Array] = None):
    def exp_decay(x, a, b, k):
        y = a*np.exp(-k*x)+b
        return y
    e_shift = min(epochs)
    epochs = epochs-e_shift
    if vals[list(epochs).index(min(epochs))]<vals[list(epochs).index(max(epochs))]:
        vals = -vals
        to_reverse = True
    else:
        to_reverse = False
    a = np.max(vals)-np.min(vals)
    b = np.min(vals)
    k = 1/(min(epochs[vals<(a*np.exp(-1)+b)])+1)
    try:
        (a, b, k), _ = curve_fit(
            exp_decay, epochs, vals, p0=(a, b, k),
            bounds=((0, -np.inf, 0), (np.inf, np.inf, np.inf)),
        )
        optimality = 1-np.exp(-k*(max(epochs)+e_shift))
        fvu = ((vals-exp_decay(epochs, a, b, k))**2).mean()/vals.var()
    except:
        optimality, fvu = np.nan, np.nan
    if queries is None:
        return optimality, fvu
    else:
        preds = exp_decay(queries-e_shift, a, b, k)
        if to_reverse:
            preds = -preds
        return optimality, fvu, preds


def loss_summary(losses: Array, epochs: Optional[Array] = None):
    r"""Returns summary of training.

    An exponential decay function is fit to the sequence of losses. `optimality`
    is a number in [0, 1], it estimates how close is the final training towards
    equilibrium. Low `optimality` suggests more training epochs are needed.
    `fvu` is a number in [0, 1], it estimates how noisy the true losses are
    compared to the best exponential fit. High `fvu` suggests bigger batches or
    smaller learning rates are needed.

    """
    num_epochs = len(losses)
    if epochs is None:
        epochs = np.arange(1, num_epochs+1)
    optimality, fvu = _exp_fit(epochs, losses)
    return optimality, fvu


def plot_rl_progress(epochs, r_means, r_sems, ax=None, color=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    if color is None:
        kwargs = {}
    else:
        kwargs = {'color': color}
    line, = ax.plot(epochs, r_means, **kwargs)
    ax.fill_between(epochs, r_means-r_sems, r_means+r_sems, alpha=0.2, **kwargs)
    if len(epochs)>2:
        if 'color' in kwargs:
            kwargs['color'] = np.array(to_rgb(kwargs['color']))*0.6+0.4
        kwargs['linestyle'] = '--'
        queries = np.linspace(min(epochs), max(epochs), 20)
        *_, preds = _exp_fit(epochs, r_means, queries)
        ax.plot(queries, preds, **kwargs)
    ax.set_xlabel('Training epochs')
    ax.set_ylabel('Episode return')
    ax.set_title('RL with internal model')
    return ax, line

def check_env(env: GymEnv):
    _check_env(env)
    assert hasattr(env, 'state_space'), (
        "You must specify a state space."
    )
    assert isinstance(env.state_space, MultiDiscrete), (
        "The state space must be 'MultiDiscrete'."
    )
    assert isinstance(env.action_space, Discrete), (
        "The action space must be 'Discrete'."
    )
    assert isinstance(env.observation_space, MultiDiscrete), (
        "The observation space must be 'MultiDiscrete'."
    )

    try:
        env.reset()
        state = env.get_state()
        assert isinstance(state, tuple) and len(state)==len(env.state_space.nvec)
        for val in state:
            assert isinstance(val, int)
    except:
        raise AssertionError(
            "You must specify 'get_state()' that returns a tuple of ints."
        )
    try:
        env.set_state(state)
    except:
        raise AssertionError(
            "You must specify 'set_state(state)' that sets environment state."
        )

    try:
        env_param = env.get_env_param()
        assert isinstance(env_param, tuple)
        for val in env_param:
            assert isinstance(val, float)
    except:
        raise AssertionError(
            "You must specify 'get_env_param()' that returns a tuple of floats."
        )
    try:
        env.set_env_param(env_param)
    except:
        raise AssertionError(
            "You must specify 'set_env_param(env_param)' that sets environment parameters."
        )
