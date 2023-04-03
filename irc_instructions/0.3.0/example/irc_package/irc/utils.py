import time
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import logsumexp
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator
from torch.utils.data import (
    TensorDataset, DataLoader, WeightedRandomSampler,
)

from typing import Optional
from collections.abc import Callable, Iterable
from gym.spaces import MultiDiscrete, Discrete, Box

from .alias import Array, Tensor, Optimizer, GymEnv, VarSpace


def _exp_fit(
    steps: Array, vals: Array,
    *,
    ascending: Optional[bool] = None,
    queries: Optional[Array] = None,
) -> tuple[Array, float, float]:
    r"""Fits data with exponential decay function.

    Args
    ----
    steps:
        Argument 'x' in the function y=a*exp(-k*x)+b
    vals:
        Target 'y' in the function y=a*exp(-k*x)+b
    queries:
        Queries in the 'x' domain, whose fitting predictions will be returned.

    Returns
    -------
    fits:
        Value of the fitted function at `queries`, with the same shape.
    optimality:
        A float number in [0, 1), describing the progress of exponential decay.
    fvu:
        Fraction of variance unexplained, describing the fitting quality.

    """
    def exp_decay(x, a, b, k):
        y = a*np.exp(-k*x)+b
        return y
    if queries is None:
        queries = steps
    shift = min(steps)
    steps = steps-shift
    if ascending is None:
        ascending = vals[list(steps).index(min(steps))]<vals[list(steps).index(max(steps))]
    if ascending:
        vals = -vals
    a = np.max(vals)-np.min(vals)
    b = np.min(vals)
    k = 1/(min(steps[vals<=(a*np.exp(-1)+b)])+1)
    try:
        (a, b, k), _ = curve_fit(
            exp_decay, steps, vals, p0=(a, b, k),
            bounds=((0, -np.inf, 0), (np.inf, np.inf, np.inf)),
        )
        fits = exp_decay(queries-shift, a, b, k)
        if ascending:
            fits = -fits
        optimality = 1-np.exp(-k*max(steps))
        fvu = ((vals-exp_decay(steps, a, b, k))**2).mean()/vals.var()
    except:
        fits = np.full_like(queries, np.nan)
        optimality, fvu = np.nan, np.nan
    return fits, optimality, fvu


def summarize_progress(steps: Array, vals: Array, ascending: Optional[bool] = None):
    r"""Returns summary of an optimization process.

    An exponential decay function is fit to the sequence of losses. `optimality`
    is a number in [0, 1), it estimates how close is the final training towards
    equilibrium. Low `optimality` suggests more training epochs are needed.
    `fvu` is a number in [0, 1], it estimates how noisy the true losses are
    compared to the best exponential fit. High `fvu` suggests bigger batches or
    smaller learning rates are needed.

    """
    queries = np.linspace(steps.min(), steps.max(), 24)
    fits, optimality, fvu = _exp_fit(steps, vals, ascending=ascending, queries=queries)
    return queries, fits, optimality, fvu


def train_and_summarize(
    dset: TensorDataset, criterion: Callable[..., Tensor], optimizer: Optimizer,
    num_epochs: int, batch_size: int,
    *,
    weights: Optional[Iterable[float]] = None, device: str = 'cuda',
) -> dict:
    r"""Trains on a dataset and summarize the training losses.

    Args
    ----
    dset:
        Training dataset.
    criterion:
        A function that receives a sequence of tensors and returns the objective
        tensor to be minimized.
    optimizer:
        Optimizer for relevant parameters.
    num_epochs:
        Number of training epochs, each epoch contains a sweep over `dset`.
    batch_size:
        Batch size of the data loader.
    weights:
        Sampling weights for `dset`. `WeightedRandomSampler` with replacement
        will be used when specified.
    device:
        Device for torch tensors.

    Returns
    -------
    train_stats:
        Statistics about the training progress, including the training losses
        per epoch and an exponential fit to it. In addtion, `optimality` and
        `fvu` are computed based on the exponential fit, see
        `train_and_summarize` for more details.

    """
    if weights is None:
        loader = DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        sampler = WeightedRandomSampler(weights, len(dset))
        loader = DataLoader(dset, batch_size=batch_size, sampler=sampler, drop_last=True)

    gamma = 0.5**(10/len(loader)) # used for running average losses
    tic = time.time()
    losses = []
    for _ in range(num_epochs):
        loss_train = None
        for xs in loader:
            xs = (t.to(device) for t in xs)
            loss: Tensor = criterion(*xs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train = loss.item() if loss_train is None else gamma*loss_train+(1-gamma)*loss.item()
        losses.append(loss_train)
    toc = time.time()
    steps = np.arange(num_epochs)+1 # optimization steps
    losses = np.array(losses) # training losses at each step

    queries, fits, optimality, fvu = summarize_progress(steps, losses, ascending=False)
    train_stats = {
        'num_samples': len(dset),
        'num_epochs': num_epochs, 'batch_size': batch_size,
        'steps': steps, 'losses': losses,
        'queries': queries, 'fits': fits,
        'optimality': optimality, 'fvu': fvu,
        't_elapse': toc-tic,
    }
    return train_stats


def _is_in_space(val, space: VarSpace) -> bool:
    r"""Checks if a value is valid in a space."""
    if isinstance(space, MultiDiscrete):
        return len(val)==len(space.nvec)
    if isinstance(space, Box):
        return (len(val),)==space.shape
    raise TypeError(space)


def check_env(env: GymEnv):
    r"""Checks environment is valid for the package."""
    assert hasattr(env, 'state_space'), "You must specify a state space."
    assert isinstance(env.state_space, (MultiDiscrete, Box)), (
        "The state space must be 'MultiDiscrete' or 'Box'."
    )
    assert hasattr(env, 'observation_space'), "You must specify an observation space."
    assert isinstance(env.observation_space, (MultiDiscrete, Box)), (
        "The observation space must be 'MultiDiscrete' or 'Box'."
    )
    assert hasattr(env, 'action_space'), "You must specify an action space."
    assert isinstance(env.action_space, Discrete), (
        "The action space must be 'Discrete'."
    )

    m_names = ['get_state', 'set_state', 'get_param', 'set_param']
    for m_name in m_names:
        assert hasattr(env, m_name), f"You must specify the method '{m_name}'."

    env.reset()
    observation, *_ = env.step(env.action_space.sample())
    assert _is_in_space(observation, env.observation_space)

    state = env.get_state()
    assert _is_in_space(state, env.state_space)
    env.set_state(state)

    env_param = env.get_param()
    assert len(np.array(env_param).shape)==1, "Environment parameter must be a vector."
    env.set_param(env_param)


def logmeanexp(x: Array, axis=None, keepdims=False):
    n = (~np.isnan(x)).sum(axis=axis, keepdims=True)
    x = x.copy()
    x[np.isnan(x)] = -np.inf
    y = logsumexp(x, b=1/n, axis=axis, keepdims=keepdims)
    return y


def plot_progress(
    steps: Array, vals: Array,
    *,
    errs: Optional[Array] = None,
    queries: Optional[Array] = None,
    fits: Optional[Array] = None,
    ax=None, color=None,
    **kwargs,
):
    r"""Plots the progress of an optimization process.

    steps:
        Index of optimization steps, can be per epoch or per batch.
    vals:
        Values of interest for each step.
    errs:
        Errors of each `vals` value, will be used to plot a shaded error region.
    queries:
        If provided, an exponential fit will be plotted as well, using `queries`
        as horizontal coordinates.
    fits:
        The exponential fit values at `quereis`. If not provided, it will be
        computed from `steps` and `vals`.
    ax:
        Matplotlib axis.
    color:
        Matplotlib color, used for all elements in the function.
    kwargs:
        Addtional key-word arguments for plotting lines, such as 'linewidth'.

    """
    if ax is None:
        _, ax = plt.subplots()
    line, = ax.plot(steps, vals, color=color, **kwargs)
    color = np.array(to_rgb(line.get_color()))
    if errs is not None:
        ax.fill_between(
            steps, vals-errs, vals+errs, alpha=0.2, color=color, **kwargs,
        )
    if queries is not None:
        if fits is None:
            fits, *_ = _exp_fit(steps, vals, queries=queries)
        ax.plot(queries, fits, linestyle='--', color=color*0.6+0.4, **kwargs)


def plot_stats(stats, *, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    es, vals = [], []
    for e in sorted(stats.keys()):
        steps = stats[e]['steps']
        es.append(steps/np.max(steps)+e)
        vals.append(stats[e]['losses'])
    es = np.concatenate(es)
    vals = np.concatenate(vals)
    _es = np.linspace(es.min(), es.max(), 100)
    _vals = np.interp(_es, es, vals)
    f = signal.windows.hamming(11, sym=True)
    _vals = signal.filtfilt(f, f.sum(), _vals)
    ax.plot(_es, _vals)


def plot_agent_checkpoint(ckpt, figsize=None):
    to_plot_stats = set(['estimate_stats', 'collect_stats', 'train_stats']).issubset(ckpt.keys())
    if figsize is None:
        if to_plot_stats:
            figsize = (6, 2.5)
        else:
            figsize = (5, 2.5)
    fig = plt.figure(figsize=figsize)

    # reinforcement learning progress
    if to_plot_stats:
        ax = plt.axes([0.05, 0.05, 0.56, 0.9])
    else:
        ax = plt.axes([0.05, 0.05, 0.9, 0.9])
    eval_records = ckpt['eval_records']
    epochs = np.array(sorted(eval_records.keys()))
    use_rate = len(set(eval_records[e]['max_steps'] for e in epochs))>1
    if use_rate:
        print(
            "'max_steps' for evaluating agent is not consistent across all "
            "episodes, will plot reward rates instead of episode returns."
        )
    if use_rate:
        r_means = np.array([np.mean(eval_records[e]['rates']) for e in epochs])
        r_stds = np.array([np.std(eval_records[e]['rates']) for e in epochs])
    else:
        r_means = np.array([np.mean(eval_records[e]['returns']) for e in epochs])
        r_stds = np.array([np.std(eval_records[e]['returns']) for e in epochs])
    if len(epochs)>2:
        queries = np.linspace(min(epochs), max(epochs), 50)
    else:
        queries = None
    plot_progress(epochs, r_means, errs=r_stds, queries=queries, ax=ax)
    ax.set_xlim([0, None])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Training epochs', x=0.85)
    if use_rate:
        ax.set_ylabel('Reward rate')
    else:
        ax.set_ylabel('Episode return')
    ax.set_title('RL with internal model')

    # plot optimization stats
    if to_plot_stats:
        gap = 0.15
        height = (0.9-gap)/2
        for i in range(2):
            ax = plt.axes([0.67, 0.05+(gap+height)*i, 0.28, height])
            if i==0:
                stats = ckpt['train_stats']
            if i==1:
                stats = ckpt['estimate_stats']['p_o_s']
            plot_stats(stats, ax=ax)
            if i>0:
                ax.set_xticklabels([])
            else:
                # ax.set_xlabel('Epochs')
                ...
            ax.set_yticklabels([])
            if i==0:
                ylabel = r'$D(\hat{b}, b)$'
                title = 'Learning belief update net'
            if i==1:
                ylabel = r'$\ln p(o|s)$'
                title = 'Learning observation model'
            ax.set_ylabel(ylabel, fontsize='small')
            ax.set_title(title, fontsize='small')

    return fig
