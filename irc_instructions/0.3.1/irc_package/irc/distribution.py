import numpy as np
import torch
from torch.nn.functional import embedding
from torch.utils.data import TensorDataset
from gym.spaces import MultiDiscrete, Box
from typing import Optional, Union
from collections.abc import Collection

from jarvis.config import Config
from jarvis.utils import sgd_optimizer

from . import rcParams
from .alias import Array, Module, RandGen, Tensor, VarSpace
from .utils import sub2ind, get_dtype, train_and_summarize


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

    def set_param_vec(self, param_vec: Tensor) -> None:
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
        return self.param_vec

    def set_param_vec(self, param_vec: Tensor) -> None:
        self.param_vec.data = param_vec

    def forward(self, xs: Tensor, param_vec: Optional[Tensor] = None) -> Tensor:
        if param_vec is None:
            param_vec = self.param_vec
        xs = sub2ind(xs, self.nvec)
        logits = embedding(xs, param_vec[:, None])[:, 0]
        return logits

    def set_from_prob(self,
        prob_dict: dict[tuple[int], float],
        eps: Optional[float] = None,
    ) -> None:
        r"""Sets parameters from a given probability mass function.

        Args
        ----
        prob_dict:
            Relative probabilities for different variable values. Values not
            included are assigned with a small probability.
        eps:
            The probability that all values not in `prob_dict` account for.

        """
        _rcParams = Config(rcParams.get('distribution.BasicDiscretePotential.set_from_prob'))
        eps = eps or _rcParams.eps
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


class BaseDistribution(Module):
    r"""Base class for energy based distributions.

    The probability (density) for each sample is characterized by a scalar of
    energy, which is the summation of local potentials defined over a small
    subset of variables.

    """

    def __init__(self,
        space: VarSpace,
        *,
        idxs: Optional[Collection[list[int]]] = None,
        phis: Optional[Collection[Union[BasePotential, dict, None]]] = None,
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
                assert isinstance(idx, list), (
                    f"Expect variable indices as a list of integers, but received {idx}."
                )
                assert set(idx).issubset(range(self.num_vars)), (
                    f"Values of variable indices should be integers in [0, {self.num_vars}), "
                    f"current value {idx} is invalid."
                )
            assert frozenset(i for idx in idxs for i in idx)==frozenset(range(self.num_vars)), (
                f"Potentials fail to cover all variables."
            )
        self.idxs = idxs

        if phis is None:
            phis = [None]*len(self.idxs)
        else:
            assert len(phis)==len(self.idxs), (
                "List of variable indices and potentials should be of the same length."
            )
        self.phis: list[BasePotential] = torch.nn.ModuleList()
        self.num_params = []
        for idx, phi in zip(self.idxs, phis):
            if isinstance(self.space, MultiDiscrete):
                _space = MultiDiscrete(nvec=self.space.nvec[idx])
            if isinstance(self.space, Box):
                _space = Box(low=self.space.low[idx], high=self.space.high[idx], shape=(len(idx),))
            phi = _create_potential(phi, _space)
            self.phis.append(phi)
            self.num_params.append(len(phi.get_param_vec()))
        self.param_dim = sum(self.num_params)

        self.rng = rng if isinstance(rng, RandGen) else np.random.default_rng(rng)

    def get_param_vec(self) -> Tensor:
        r"""Returns the concatenated parameter vector.

        Returns
        -------
        param_vec: (num_params,)

        """
        param_vec = torch.cat([phi.get_param_vec() for phi in self.phis])
        return param_vec

    def set_param_vec(self, param_vec: Tensor) -> None:
        r"""Sets the concatenated parameter vector.

        Args
        ----
        param_vec: (num_params,)
            Concatenated parameters for all potentials.

        """
        assert param_vec.shape==(self.param_dim,)
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
        r"""Returns KL-divergence D(P||Q).

        Args
        ----
        param_vec_p, param_vec_q: (param_dim,)
            Parameter vectors for distribution P and Q respectively.

        Returns
        -------
        loss: scalar
            KL loss value.

        """
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
        l2_reg: Optional[float] = None,
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
        l2_reg:
            Ratio of L2 regularization on the distribution parameters.
        lr:
            Learning rate of SGD optimizer.
        num_epochs, batch_size:
            See `utils.train_and_summarize` for more details.

        Returns
        -------
        estimate_stats:
            See `utils.train_and_summarize` for more details.

        """
        _rcParams = Config(rcParams.get('distribution.BaseDistribution.estimate'))
        l2_reg = l2_reg or _rcParams.l2_reg
        lr = lr or _rcParams.lr
        num_epochs = num_epochs or _rcParams.num_epochs
        batch_size = batch_size or _rcParams.batch_size

        dset = TensorDataset(torch.tensor(xs, dtype=get_dtype(self.space)))
        def get_loss(xs: Tensor) -> Tensor: # negative log likelihood
            nll_loss = -self.loglikelihoods(xs).mean()
            l2_loss = self.get_param_vec().pow(2).mean()
            loss = (1-l2_reg)*nll_loss+l2_reg*l2_loss
            return loss
        optimizer = sgd_optimizer(self, lr=lr)
        device = next(iter(self.parameters())).device

        estimate_stats = {'l2_reg': l2_reg, 'lr': lr}
        estimate_stats.update(train_and_summarize(
            dset, get_loss, optimizer, num_epochs, batch_size,
            ws=ws, device=device,
        ))
        return estimate_stats


class BasicDiscreteDistribution(BaseDistribution):
    r"""Basic distribution for discrete variables."""

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

    def __repr__(self) -> str:
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

    def kl_loss(self,
        param_vec_p: Tensor, param_vec_q: Tensor,
    ) -> Tensor:
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


def _create_potential(
    phi: Union[BasePotential, dict, None],
    space: VarSpace,
) -> BasePotential:
    _rcParams = Config(rcParams.get('distribution._create_potential'))
    if phi is None or isinstance(phi, dict):
        phi = Config(phi)
        if isinstance(space, MultiDiscrete):
            phi.fill(_rcParams.discrete)
        if isinstance(space, Box):
            phi.fill(_rcParams.continuous)
        phi = phi.instantiate(space=space)
    else:
        assert isinstance(phi, BasePotential)
        assert phi.space==space
    return phi


def _create_distribution(
    dist: Union[BaseDistribution, dict, None],
    space: VarSpace,
) -> BaseDistribution:
    _rcParams = Config(rcParams.get('distribution._create_distribution'))
    if dist is None or isinstance(dist, dict):
        dist = Config(dist)
        if isinstance(space, MultiDiscrete):
            dist.fill(_rcParams.discrete)
        if isinstance(space, Box):
            dist.fill(_rcParams.continuous)
        dist = dist.instantiate(space=space)
    else:
        assert isinstance(dist, BaseDistribution)
        assert dist.space==space
    return dist
