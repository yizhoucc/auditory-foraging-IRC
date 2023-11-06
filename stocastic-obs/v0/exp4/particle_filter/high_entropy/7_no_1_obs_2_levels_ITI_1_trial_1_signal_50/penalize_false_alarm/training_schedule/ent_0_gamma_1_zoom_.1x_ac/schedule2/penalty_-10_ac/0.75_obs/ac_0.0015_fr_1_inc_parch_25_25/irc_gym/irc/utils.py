import numpy as np
from scipy.optimize import curve_fit
from scipy.special import logsumexp
from gym.utils.env_checker import check_env as _check_env
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.cm import get_cmap
from matplotlib.ticker import MaxNLocator

from typing import Optional
from .alias import Array, GymEnv, MultiDiscrete, Discrete


def _exp_fit(
    steps: Array, vals: Array,
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
    if vals[list(steps).index(min(steps))]<vals[list(steps).index(max(steps))]:
        vals = -vals # original values are gradually increasing
        to_reverse = True
    else:
        to_reverse = False
    a = np.max(vals)-np.min(vals)
    b = np.min(vals)
    k = 1/(min(steps[vals<=(a*np.exp(-1)+b)])+1)
    try:
        (a, b, k), _ = curve_fit(
            exp_decay, steps, vals, p0=(a, b, k),
            bounds=((0, -np.inf, 0), (np.inf, np.inf, np.inf)),
        )
        fits = exp_decay(queries-shift, a, b, k)
        if to_reverse:
            fits = -fits
        optimality = 1-np.exp(-k*max(steps))
        fvu = ((vals-exp_decay(steps, a, b, k))**2).mean()/vals.var()
    except:
        fits = np.full_like(queries, np.nan)
        optimality, fvu = np.nan, np.nan
    return fits, optimality, fvu


def progress_summary(steps: Array, vals: Array):
    r"""Returns summary of an optimization process.

    An exponential decay function is fit to the sequence of losses. `optimality`
    is a number in [0, 1), it estimates how close is the final training towards
    equilibrium. Low `optimality` suggests more training epochs are needed.
    `fvu` is a number in [0, 1], it estimates how noisy the true losses are
    compared to the best exponential fit. High `fvu` suggests bigger batches or
    smaller learning rates are needed.

    """
    queries = np.linspace(steps.min(), steps.max(), 24)
    fits, optimality, fvu = _exp_fit(steps, vals, queries)
    return queries, fits, optimality, fvu


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
        _, ax = plt.subplots(figsize=(3, 2))
    line, = ax.plot(steps, vals, color=color, **kwargs)
    color = np.array(to_rgb(line.get_color()))
    if errs is not None:
        ax.fill_between(
            steps, vals-errs, vals+errs, alpha=0.2, color=color, **kwargs,
        )
    if queries is not None:
        if fits is None:
            fits, *_ = _exp_fit(steps, vals, queries)
        ax.plot(queries, fits, linestyle='--', color=color*0.6+0.4, **kwargs)


def check_env(env: GymEnv):
    # _check_env(env)
    # TODO add basic check_env from gym, API conflicts at 'reset()'
    assert hasattr(env, 'state_space'), "You must specify a state space."
    assert isinstance(env.state_space, MultiDiscrete), "The state space must be 'MultiDiscrete'."
    assert isinstance(env.action_space, Discrete), "The action space must be 'Discrete'."
    assert isinstance(env.observation_space, MultiDiscrete), (
        "The observation space must be 'MultiDiscrete'."
    )

    try:
        env.reset()
        state = env.get_state()
        assert isinstance(state, tuple) and len(state)==len(env.state_space.nvec)
        for val in state:
            assert int(val)==val
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
        env_param = env.get_param()
        assert isinstance(env_param, tuple)
        for val in env_param:
            assert isinstance(val, float)
    except:
        raise AssertionError(
            "You must specify 'get_param()' that returns a tuple of floats."
        )
    try:
        env.set_param(env_param)
    except:
        raise AssertionError(
            "You must specify 'set_param(env_param)' that sets environment parameters."
        )


def logmeanexp(x: Array, axis=None, keepdims=False):
    n = (~np.isnan(x)).sum(axis=axis, keepdims=True)
    x = x.copy()
    x[np.isnan(x)] = -np.inf
    y = logsumexp(x, b=1/n, axis=axis, keepdims=keepdims)
    return y


def plot_agent_checkpoint(ckpt, figsize=(5, 4)):
    fig = plt.figure(figsize=figsize)

    # reinforcement learning progress
    if 'est_stats' in ckpt:
        ax = plt.axes([0.05, 0.5, 0.9, 0.5])
    else:
        ax = plt.axes([0.05, 0.05, 0.9, 0.9])
    eval_records = ckpt['eval_records']
    epochs = np.array(sorted(eval_records.keys()))
    r_means = np.array([np.mean(eval_records[e]['returns']) for e in epochs])
    r_stds = np.array([np.std(eval_records[e]['returns']) for e in epochs])
    if len(epochs)>2:
        queries = np.linspace(min(epochs), max(epochs), 50)
    else:
        queries = None
    plot_progress(epochs, r_means, errs=r_stds, queries=queries, ax=ax)
    ax.set_xlim([0, None])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Training epochs')
    ax.set_ylabel('Episode return')
    ax.set_title('RL with internal model')

    if 'est_stats' in ckpt:
        # optimzation progress for estimating p(s|o)
        margin, gap, width = 0.05, 0.03, 0.28
        ax = plt.axes([margin, 0.05, width, 0.2])
        est_stats = ckpt['est_stats']['p_s_o']
        plot_progress(
            est_stats['steps'], -est_stats['losses'],
            queries=est_stats['queries'], fits=-est_stats['fits'], ax=ax,
        )
        ax.set_xlim([0, None])
        ax.set_ylabel('log likelihood')
        ax.set_yticklabels([])
        ax.set_title(r'$p(s|o)$')

        # optimzation progress for estimating p(s|o)
        ax = plt.axes([margin+width+gap, 0.05, width, 0.2])
        est_stats = ckpt['est_stats']['p_o_s']
        plot_progress(
            est_stats['steps'], -est_stats['losses'],
            queries=est_stats['queries'], fits=-est_stats['fits'], ax=ax,
        )
        ax.set_xlim([0, None])
        ax.set_xlabel('Optimizing steps for distribution estimation')
        ax.set_yticklabels([])
        ax.set_title(r'$p(o|s)$')

        # optimzation progress for most recent updates of belief p(s)
        ax = plt.axes([margin+2*(width+gap), 0.05, width, 0.2])
        cmap = get_cmap('Blues')
        for i, est_stats in enumerate(ckpt['est_stats']['p_s']):
            color = cmap(i/len(ckpt['est_stats']['p_s']))
            plot_progress(
                est_stats['steps'], -est_stats['losses'],
                color=color, linewidth=1, ax=ax,
            )
        ax.set_xlim([0, None])
        x_offset = 0.7*ax.get_xlim()[0]+0.3*ax.get_xlim()[1]
        y_offset = 0.8*ax.get_ylim()[0]+0.2*ax.get_ylim()[1]
        n = len(ckpt['est_stats']['p_s'])
        ax.text(x_offset, y_offset, f'latest {n}\nupdates', fontsize=10)
        ax.set_yticklabels([])
        ax.set_title('$p(s)$')

    return fig
