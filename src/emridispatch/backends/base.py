"""Sampler-agnostic problem description + backend protocol.

Stores the RAW physical-space likelihood/prior; reparam wrapping (for
whitened-space MCMC backends) happens only via wrapped(), not here. Fixed
parameters (config prior.fixed) are likewise removed only in wrapped(): the
problem itself stays full-dimensional.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Optional, Protocol

import numpy as np

from emridispatch.priors import JointPrior
from emridispatch.reparam import Reparam, ReparamCallable

__all__ = ["SamplingProblem", "SamplerBackend"]


@dataclass
class SamplingProblem:
    ndim: int
    param_names: list
    lnlike: Callable            # RAW physical-space likelihood: f(x_vec) -> float
    prior: JointPrior           # physical-space structure (bounds/periodic/sample)
    reparam: Reparam            # identity when reparam_mode == "off"
    reparam_mode: str
    x0: np.ndarray              # physical-space start point
    proposal_cov: Optional[np.ndarray]   # physical-space proposal covariance
    periodic: dict              # index -> period, physical space
    truth: Optional[np.ndarray]          # physical-space truth (None: blind run)
    outdir: str
    seed: int
    meta: dict = field(default_factory=dict)
    # index -> pinned value. Fixed indices lie outside the reparam block, where
    # sampling and physical coords coincide, so one value serves both.
    fixed: dict = field(default_factory=dict)

    @property
    def whitened(self) -> bool:
        return self.reparam_mode in ("auto", "grid")

    @property
    def free_idx(self) -> np.ndarray:
        return np.array([i for i in range(self.ndim) if i not in self.fixed],
                        dtype=int)

    def wrapped(self):
        """Sampling-space view for MCMC backends: reparam-wrapped callables plus
        the transformed start / proposal covariance / truth.

        jacobian=True adds the grid-mode change-of-variables term (no-op
        for linear "auto"); periodic is untouched since it's disjoint
        from the reparam's intrinsic-parameter block.

        With fixed parameters everything is reduced to the free coordinates:
        ndim/free_idx give the reduced layout, periodic is re-indexed, and
        embed(v) rebuilds the full sampling-space vector.
        """
        if self.whitened:
            lnlike = ReparamCallable(self.lnlike, self.reparam)
            lnprior = ReparamCallable(self.prior, self.reparam, jacobian=True)
            x0 = self.reparam.to_u(self.x0)
            cov = (self.reparam.transform_cov(self.proposal_cov)
                   if self.proposal_cov is not None else None)
            truth = self.reparam.to_u(self.truth) if self.truth is not None else None
        else:
            lnlike, lnprior = self.lnlike, self.prior
            x0 = np.asarray(self.x0, dtype=float)
            cov, truth = self.proposal_cov, self.truth
        periodic = dict(self.periodic)
        free = self.free_idx
        if self.fixed:
            base = np.array(x0, dtype=float)
            for i, v in self.fixed.items():
                base[i] = v
            embed = FixedEmbedding(base, free)
            lnlike, lnprior = Reduced(lnlike, embed), Reduced(lnprior, embed)
            x0 = x0[free]
            cov = cov[np.ix_(free, free)] if cov is not None else None
            truth = truth[free] if truth is not None else None
            periodic = {int(np.searchsorted(free, i)): p
                        for i, p in periodic.items() if i not in self.fixed}
        else:
            embed = FixedEmbedding(None, free)
        return SimpleNamespace(lnlike=lnlike, lnprior=lnprior, x0=x0,
                               proposal_cov=cov, periodic=periodic,
                               truth=truth, ndim=len(free), free_idx=free,
                               embed=embed)


class FixedEmbedding:
    """Reduced (free) sampling vector -> full one, fixed rows from `base`.

    base None is the identity (no fixed parameters). Batch-safe on the last
    axis; a class rather than a closure so samplers can pickle it.
    """

    def __init__(self, base, free_idx):
        self.base = None if base is None else np.asarray(base, dtype=float)
        self.free_idx = np.asarray(free_idx, dtype=int)

    def __call__(self, v):
        v = np.asarray(v, dtype=float)
        if self.base is None:
            return v
        out = np.broadcast_to(self.base, v.shape[:-1] + self.base.shape).copy()
        out[..., self.free_idx] = v
        return out


class Reduced:
    """f(full vector) -> f(free vector) via a FixedEmbedding."""

    def __init__(self, inner, embed):
        self.inner = inner
        self.embed = embed

    def __call__(self, params):
        return self.inner(self.embed(params))

    def initial_sample(self):
        return np.asarray(self.inner.initial_sample())[self.embed.free_idx]


class SamplerBackend(Protocol):
    name: str

    def run(self, problem: SamplingProblem, cfg, resume: bool = True) -> dict:
        """Build the concrete sampler, sample, write run outputs under
        problem.outdir, and return a summary dict."""
