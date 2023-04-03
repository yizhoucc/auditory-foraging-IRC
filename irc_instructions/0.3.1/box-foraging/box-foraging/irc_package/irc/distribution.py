import numpy as np
import torch
from torch.nn.functional import embedding
from torch.utils.data import TensorDataset
from gym.spaces import MultiDiscrete, Box
from typing import Optional, Union

from jarvis.config import Config
from jarvis.utils import sgd_optimizer, create_mlp_layers

from . import rcParams
from .alias import Array, Tensor, Module, VarSpace, RandGen
from .utils import train_and_summarize


class BasePotential(Module):
    r"""Base class for potential."""

    def __init__(self, space: VarSpace):
        super().__init__()
        self.space = space
        self.num_vars: int = None
        self.num_params: int = None

    def get_param_vec(self) -> Tensor:
        r"""Returns the parameter vector data.

        Returns
        -------
        param_vec: (num_params,)

        """
        raise NotImplementedError

    def set_param_vec(self, param_vec: Tensor):
        r"""Sets the parameter vector data.

        Args
        ----
        param_vec: (num_params,)
            Parameters vector.

        """
        raise NotImplementedError

    def forward(self,
        xs: Tensor,
        param_vec: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Returns energy values of data samples.

        Args
        ----
        xs: (num_samples, num_vars)
            Data samples.
        param_vec: (num_params,)
            Parameters of the potential. If `param_vec` is ``None``, use the
            internal parameters with gradient enabled.

        Returns
        -------
        e: (num_samples,)
            Energy values of data samples.

        """
        raise NotImplementedError


def _sub2ind(subs: Tensor, nvec: list[int]):
    r"""Converts subscripts to indices.

    Args
    ----
    subs: (num_samples, num_vars)
        Subscripts with each row corresponding to a dimension of MultiDiscrete
        variables.
    nvec:
        Number of values of each discrete variable.

    Returns
    -------
    inds: (num_samples,)
        Indices of all samples.

    """
    inds = subs[:, -1]
    for i in reversed(range(len(nvec)-1)):
        inds = inds+subs[:, i]*np.prod(nvec[i+1:])
    return inds.to(torch.long)


class BasicDiscretePotential(BasePotential):
    r"""Basic potential for categorical variables.

    MultiDiscrete samples are treated as one-hot vectors. Probabilities are thus
    encoded by an embedding layer with parameters as logits.

    """

    def __init__(self, space: MultiDiscrete):
        assert isinstance(space, MultiDiscrete)
        super().__init__(space)
        self.nvec = self.space.nvec
        self.num_vars = len(self.nvec)
        self.num_params = np.prod(self.nvec)
        self.param_vec = torch.nn.Parameter(torch.randn(self.num_params))

    def get_param_vec(self) -> Tensor:
        return self.param_vec.data

    def set_param_vec(self, param_vec: Tensor):
        self.param_vec.data = param_vec

    def forward(self, xs: Tensor, param_vec: Optional[Tensor] = None) -> Tensor:
        if param_vec is None:
            param_vec = self.param_vec
        xs = _sub2ind(xs, self.nvec)
        logits = embedding(xs, param_vec[:, None])[:, 0]
        return logits

    def set_from_prob(self,
        prob_dict: dict[tuple[int], float],
        eps: Optional[float] = None,
    ):
        r"""Sets parameters from a given probability mass function.

        Args
        ----
        prob_dict:
            Relative probabilities for different variable values. Values not
            included are assigned with a small probability.
        eps:
            The probability that all values not in `prob_dict` account for.

        """
        if eps is None:
            eps = rcParams['irc.distribution.BasicDiscretePotential.set_from_prob']['eps']
        param_vec = torch.ones(self.num_params, device=self.param_vec.device)
        n_small = self.num_params-len(prob_dict)
        if n_small>0:
            param_vec *= np.log(eps/n_small) # initiate probabilities with small values
        else:
            eps = 0 # all probabilities are specified
        z = sum(prob_dict.values())
        for x, p in prob_dict.items():
            idx = np.ravel_multi_index(x, self.nvec)
            param_vec[idx] = np.log(p/z*(1-eps))
        param_vec -= param_vec.mean()
        self.set_param_vec(param_vec)


class BasicContinuousPotential(BasePotential):
    r"""Basic potential for continuous variables."""

    def __init__(self, space: Box):
        assert isinstance(space, Box) and len(space.shape)==1
        super().__init__(space)
        self.num_vars, = self.space.shape


class BaseParamNet(Module):
    r"""Base class for parameter network."""

    def __init__(self,
        space: VarSpace,
        n_out: int,
    ):
        r"""
        Args
        ----
        space:
            Input variable space.
        n_out:
            Output dimension, i.e. number of distribution parameters.

        """
        super().__init__()
        self.space = space
        self.n_out = n_out

    def forward(self,
        xs: Tensor,
    ) -> Tensor:
        r"""
        Args
        ----
        xs: (num_samples, num_vars)
            Input variable samples.

        Returns
        -------
        param_vecs: (num_samples, n_out)
            Distribution parameters as outputs.

        """
        raise NotImplementedError


class MultiLayerPerceptronParamNet(BaseParamNet):
    r"""Parameter network using a multi-layer perceptron (MLP)."""

    def __init__(self,
        space: Box, n_out: int,
        mlp_sizes: Optional[list[int]] = None,
    ):
        r"""
        Args
        ----
        mlp_sizes:
            Hidden layer sizes of the MLP.

        """
        super().__init__(space, n_out)
        assert isinstance(space, Box) and len(space.shape)==1, "Only supports 1D box space."
        if mlp_sizes is None:
            mlp_sizes = rcParams['irc.distribution.MultiLayerPerceptronParamNet']['mlp_sizes']
        self.layers = create_mlp_layers(space.shape[0], self.n_out, mlp_sizes)

    def forward(self, xs: Tensor) -> Tensor:
        out = xs
        for layer in self.layers:
            out = layer(out)
        return out


class CompleteEmbedParamNet(BaseParamNet):
    r"""Parameter network using complete embedding followed by a multi-layer perceptron."""

    def __init__(self,
        space: MultiDiscrete, n_out: int,
        mlp_sizes: Optional[list[int]] = None,
    ):
        r"""
        Args
        ----
        mlp_sizes:
            Hidden layer sizes of the appended MLP.

        """
        super().__init__(space, n_out)
        if mlp_sizes is None:
            mlp_sizes = rcParams['irc.distribution.CompleteEmbedParamNet']['mlp_sizes']
        self.embed = torch.nn.Embedding(
            np.prod(self.space.nvec), mlp_sizes[0] if mlp_sizes else self.n_out,
        )
        if mlp_sizes:
            self.layers = create_mlp_layers(mlp_sizes[0], self.n_out, mlp_sizes[1:])
        else:
            self.layers = torch.nn.ModuleList()

    def forward(self, xs: Tensor) -> Tensor:
        xs = _sub2ind(xs, self.space.nvec)
        out = self.embed(xs)
        for layer in self.layers:
            out = layer(out)
        return out


def _get_dtype(space: VarSpace):
    r"""Returns torch dtype."""
    if isinstance(space, MultiDiscrete):
        return torch.long
    if isinstance(space, Box):
        return torch.float


class BaseDistribution(Module):
    r"""Base class for energy based distributions.

    The probability (density) for each sample is characterized by a scalar of
    energy, which is the summation of local potentials defined over a small
    subset of variables.

    """

    def __init__(self,
        space: VarSpace,
        *,
        idxs: Optional[list[list[int]]] = None,
        phis: Optional[list[Union[BasePotential, dict, None]]] = None,
        rng: Union[RandGen, int, None] = None,
    ):
        r"""
        Args
        ----
        space:
            Variable space for distribution p(x). Currently only supports
            `MultiDiscrete` or `Box` with dimension 1.
        idxs:
            Variable indices for each potential. When `idxs` is ``None``, it
            will be initialized as the full set.
        phis:
            Local potentials or the configurations to instantiate one. If not
            provided, default basic potentials will be instantiated.
        rng:
            Random number generator or seed, used only in sampling but not in
            parameter estimation.

        """
        super().__init__()

        if isinstance(space, MultiDiscrete):
            self.num_vars = len(space.nvec)
        if isinstance(space, Box):
            self.num_vars, = space.shape
        self.space = space

        if idxs is None:
            assert phis is None, "Variable indices of potentials are not specified."
            idxs = [list(range(self.num_vars))] # one global potential by default
        else:
            for idx in idxs:
                assert set(idx).issubset(range(self.num_vars)), f"Variable index {idx} is invalid."
            assert frozenset(i for idx in idxs for i in idx)==frozenset(range(self.num_vars)), (
                f"Potentials fail to cover all variables."
            )
        self.idxs: list[list[int]] = idxs

        if phis is None:
            phis = [None]*len(self.idxs)
        else:
            assert len(phis)==len(self.idxs), (
                "List of variable indices and potentials should be of the same length."
            )
        self.phis: list[BasePotential] = torch.nn.ModuleList()
        self.num_params: list[int] = []
        for idx, phi in zip(self.idxs, phis):
            if isinstance(self.space, MultiDiscrete):
                _space = MultiDiscrete(nvec=self.space.nvec[idx])
            if isinstance(self.space, Box):
                _space = Box(low=self.space.low[idx], high=self.space.high[idx], shape=(len(idx),))
            if phi is None or isinstance(phi, dict):
                phi = Config(phi)
                if '_target_' not in phi:
                    if isinstance(self.space, MultiDiscrete):
                        phi._target_ = 'irc.distribution.BasicDiscretePotential'
                    if isinstance(self.space, Box):
                        phi._target_ = 'irc.distribution.BasicContinuousPotential'
                phi = phi.instantiate(space=_space)
            else:
                assert isinstance(phi, BasePotential)
                assert phi.space==_space
            self.phis.append(phi)
            self.num_params.append(len(phi.get_param_vec()))

        self.rng = rng if isinstance(rng, RandGen) else np.random.default_rng(rng)

    def get_param_vec(self) -> Tensor:
        r"""Returns the concatenated parameter vector.

        Returns
        -------
        param_vec: (num_params,)

        """
        param_vec = torch.cat([phi.get_param_vec() for phi in self.phis])
        return param_vec

    def set_param_vec(self, param_vec: Tensor):
        r"""Sets the concatenated parameter vector.

        Args
        ----
        param_vec: (num_params,)
            Concatenated parameters for all potentials.

        """
        assert param_vec.shape==(sum(self.num_params),)
        c_p = 0
        for phi, n_p in zip(self.phis, self.num_params):
            phi.set_param_vec(param_vec[c_p:c_p+n_p])
            c_p += n_p

    def energy(self,
        xs: Tensor,
        param_vec: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Returns energy values of data samples.

        Energy values from each local poentials are summed up.

        Args
        ----
        xs, param_vec:
            See `loglikelihoods` for more details.

        Returns
        -------
        es: (num_samples,)
            Energy values of data samples.

        """
        es, c_p = 0, 0
        for idx, phi, n_p in zip(self.idxs, self.phis, self.num_params):
            es += phi(xs[:, idx], None if param_vec is None else param_vec[c_p:c_p+n_p])
            c_p += n_p
        return es

    def logpartition(self,
        param_vec: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Returns log partition function.

        Args
        ----
        param_vec:
            See `set_param_vec` for more details.

        Returns
        -------
        logz:
            A scalar whose gradient with respect to distribution parameters is
            enabled.

        """
        raise NotImplementedError

    def loglikelihoods(self,
        xs: Tensor,
        param_vec: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Returns log likelihood of data samples.

        Args
        ----
        xs: (num_samples, num_vars)
            Data samples.
        param_vec:
            See `set_param_vec` for more details.

        Returns
        -------
        logps: (num_samples,)
            Log likelihood of data samples.

        """
        logps = self.energy(xs, param_vec)-self.logpartition(param_vec)
        return logps

    def kl_loss(self, param_vec_p: Tensor, param_vec_q: Tensor) -> Tensor:
        r"""Returns KL-divergence D(P||Q)."""
        raise NotImplementedError

    def sample(self,
        num_samples: Optional[int] = None,
    ) -> Array:
        r"""Returns data samples of the distribution.

        Args
        ----
        num_samples:
            Number of samples returned. When `num_samples` is ``None``, only one
            sample is returned.

        Returns
        -------
        xs: (num_vars,) or (num_samples, num_vars)
            Samples drawn from the distribution.

        """
        # TODO Gibbs sampling
        raise NotImplementedError

    def estimate(self,
        xs: Array,
        ws: Optional[Array] = None,
        *,
        lr: Optional[float] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> dict:
        r"""Estimates p(x) by gradient ascent on log likelihood.

        Args
        ----
        xs: (num_samples, num_vars)
            Data samples.
        ws: (num_samples,)
            Weights of samples, needs to be positive.
        lr:
            Learning rate of SGD optimizer.
        num_epochs, batch_size:
            See `utils.train_and_summarize` for more details.

        Returns
        -------
        estimate_stats:
            See `utils.train_and_summarize` for more details.

        """
        if lr is None:
            lr = rcParams['irc.distribution.BaseDistribution.estimate']['lr']
        if num_epochs is None:
            num_epochs = rcParams['irc.distribution.BaseDistribution.estimate']['num_epochs']
        if batch_size is None:
            batch_size = rcParams['irc.distribution.BaseDistribution.estimate']['batch_size']

        dset = TensorDataset(torch.tensor(xs, dtype=_get_dtype(self.space)))
        def nll(xs: Tensor) -> Tensor: # negative log likelihood
            loss = -self.loglikelihoods(xs).mean()
            return loss
        optimizer = sgd_optimizer(self, lr=lr)
        device = next(iter(self.parameters())).device

        estimate_stats = {'lr': lr}
        estimate_stats.update(train_and_summarize(
            dset, nll, optimizer, num_epochs, batch_size,
            weights=ws, device=device,
        ))
        return estimate_stats


class DiscreteDistribution(BaseDistribution):
    r"""Distribution for discrete variables."""

    def __init__(self,
        space: MultiDiscrete,
        **kwargs,
    ):
        r"""
        Args
        ----
        space:
            A MultiDiscrete space.

        """
        super().__init__(space, **kwargs)
        self._all_xs = np.arange(np.prod(self.space.nvec))
        self._all_xs = torch.tensor(
            np.stack(np.unravel_index(self._all_xs, self.space.nvec)).T,
            dtype=torch.long, device=next(iter(self.parameters())).device,
        )

    def __repr__(self):
        return 'Discrete distribution over space {}'.format(self.space.nvec)

    def logpartition(self,
        param_vec: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Returns log partition function.

        Calculate `logz = log(sum(exp(energy(x))))` directly over all possible
        variable values.

        """
        # TODO more efficient implementation by using phis structure
        logz = torch.logsumexp(self.energy(self._all_xs, param_vec), dim=0)
        return logz

    def kl_loss(self, param_vec_p: Tensor, param_vec_q: Tensor) -> Tensor:
        logps_p = self.loglikelihoods(self._all_xs, param_vec_p)
        logps_q = self.loglikelihoods(self._all_xs, param_vec_q)
        loss = (logps_p.exp()*(logps_p-logps_q)).sum()
        return loss

    def sample(self,
        num_samples: Optional[int] = None
    ) -> Array:
        with torch.no_grad():
            p = torch.nn.functional.softmax(self.energy(self._all_xs), dim=0)
        xs = self.rng.choice(
            np.prod(self.space.nvec),
            size=None if num_samples is None else (num_samples,),
            p=p.cpu().numpy(),
        )
        xs = np.stack(np.unravel_index(xs, self.space.nvec)).T
        return xs


class ConditionalDistribution:
    r"""Conditional distribution.

    A conditional distribution p(x|y) is a composition of a parameter network on
    y space and a distribution on x space. The output of the parameter network
    will serve as the parameter vector for the distribution of x.

    """

    def __init__(self,
        x_space: VarSpace,
        y_space: VarSpace,
        p_x: Union[BaseDistribution, dict, None] = None,
        param_net: Union[BaseParamNet, dict, None] = None,
    ):
        r"""
        Args
        ----
        x_space, y_space:
            Variable spaces for p(x|y).
        p_x:
            A distribution on x space or a configuration to instantiate one. The
            parameters of `p_x` will not be used, only the potential class and
            structure will be used.
        param_net:
            A parameter network defined on y space or a configuration to
            instantiate one.

        """
        self.x_space = x_space
        self.y_space = y_space

        if p_x is None or isinstance(p_x, dict):
            p_x = Config(p_x)
            if '_target_' not in p_x:
                if isinstance(self.x_space, MultiDiscrete):
                    p_x._target_ = 'irc.distribution.DiscreteDistribution'
                if isinstance(self.x_space, Box):
                    raise NotImplementedError
            p_x = p_x.instantiate(space=self.x_space)
        else:
            assert p_x.space==self.x_space
        self.p_x: BaseDistribution = p_x

        if param_net is None or isinstance(param_net, dict):
            param_net = Config(param_net)
            if '_target_' not in param_net:
                if isinstance(self.y_space, MultiDiscrete):
                    param_net._target_ = 'irc.distribution.CompleteEmbedParamNet'
                if isinstance(self.y_space, Box):
                    param_net._target_ = 'irc.distribution.MultiLayerPerceptronParamNet'
            param_net = param_net.instantiate(space=self.y_space, n_out=sum(self.p_x.num_params))
        else:
            assert isinstance(param_net, BaseParamNet)
            assert param_net.space==self.y_space
            assert param_net.n_out==sum(self.num_params)
        self.param_net: BaseParamNet = param_net

    def loglikelihoods(self, xs: Tensor, ys: Tensor) -> Tensor:
        r"""Returns log likelihood of data pair samples.

        Same ys are merged first because they correspond to the same parameter
        vector for p(x).

        Args
        ----
        xs: (num_samples, num_vars_x)
            x data samples.
        ys: (num_samples, num_vars_y)
            y data samples.

        Returns
        -------
        logps: (num_samples,)
            Log likelihood of data pair samples.

        """
        groups = {}
        for idx, y in enumerate(ys):
            if y not in groups:
                groups[y] = [idx]
            else:
                groups[y].append(idx)
        logps = torch.zeros(len(xs), device=xs.device)
        for y, idxs in groups.items():
            param_vec = self.param_net(y[None])[0]
            logps[idxs] = self.p_x.loglikelihoods(xs[idxs], param_vec)
        return logps

    def estimate(self,
        xs: Array,
        ys: Array,
        ws: Optional[Array] = None,
        *,
        lr: Optional[float] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> dict:
        r"""Estimates p(x|y) from samples.

        Args
        ----
        xs: (num_samples, num_vars_x)
            x data samples.
        ys: (num_samples, num_vars_y)
            y data samples.
        ws, lr, num_epochs, batch_size:
            See `BaseDistribution.estimate` for more details.

        Returns
        -------
        estimate_stats:
            See `BaseDistribution.estimate` for more details.

        """
        if lr is None:
            lr = rcParams['irc.distribution.ConditionalDistribution.estimate']['lr']
        if num_epochs is None:
            num_epochs = rcParams['irc.distribution.ConditionalDistribution.estimate']['num_epochs']
        if batch_size is None:
            batch_size = rcParams['irc.distribution.ConditionalDistribution.estimate']['batch_size']

        dset = TensorDataset(
            torch.tensor(xs, dtype=_get_dtype(self.x_space)),
            torch.tensor(ys, dtype=_get_dtype(self.y_space)),
        )
        def nll(xs: Tensor, ys: Tensor) -> Tensor:
            loss = -self.loglikelihoods(xs, ys).mean()
            return loss
        optimizer = sgd_optimizer(self.param_net, lr=lr)
        device = next(iter(self.param_net.parameters())).device

        estimate_stats = {'lr': lr}
        estimate_stats.update(train_and_summarize(
            dset, nll, optimizer, num_epochs, batch_size,
            weights=ws, device=device,
        ))
        return estimate_stats
