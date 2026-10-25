"""Fisher-matrix providers: how the prior box and proposal covariance get sized.

Optional infrastructure; providers are sef (StableEMRIFisher), manual
(prior.sigmas / prior.covariance_file from config), or heuristic (rough
fallback). Selected via prior.fisher: auto | sef | manual | none, where
auto resolves to manual if configured, else sef if importable, else
heuristic. Providers implement

    compute(injection_parameters, *, duration, delta_t,
            use_gpu=None) -> FisherResult

carrying diagonal 1-sigma errors and the 6x6 intrinsic covariance in
linear coords, ordered [mass_1, mass_2, a, p, e, luminosity_distance].
With prior.fisher_full (sef only) the Fisher matrix spans all 12 sampled
parameters; cov is then the intrinsic block of its inverse (marginal over
the angles and phases) and cov_full the whole inverse.
"""

from dataclasses import dataclass

import numpy as np

from emridispatch.noise import PSD_NOTCH, PSD_NOTCH_DEPTH
from emridispatch.parameters import INTRINSIC_ORDER, NDIM

__all__ = ["FisherResult", "check_snr", "get_fisher_provider"]


@dataclass
class FisherResult:
    """Fisher products in linear coordinates, ordered as INTRINSIC_ORDER."""

    sigmas: dict          # injection-parameter name -> 1-sigma error
    cov: np.ndarray       # (6, 6) covariance, linear coords
    order: list           # ["mass_1", "mass_2", "a", "p", "e", "luminosity_distance"]
    # (12, 12) covariance over the sampling-vector rows (masses linear), or None
    cov_full: np.ndarray = None
    snr: float = None     # optimal SNR under the provider's own inner product

    def __post_init__(self):
        self.cov = np.asarray(self.cov, dtype=float)
        if self.cov_full is not None:
            self.cov_full = np.asarray(self.cov_full, dtype=float)
            if self.cov_full.shape != (NDIM, NDIM):
                raise ValueError(f"FisherResult cov_full must be ({NDIM}, {NDIM}), "
                                 f"got {self.cov_full.shape}")
        if list(self.order) != INTRINSIC_ORDER:
            raise ValueError(
                f"FisherResult order must be {INTRINSIC_ORDER}, got {self.order}")
        missing = [p for p in INTRINSIC_ORDER if p not in self.sigmas]
        if missing:
            raise ValueError(f"FisherResult sigmas missing {missing}")


def _sef_importable():
    try:
        import stableemrifisher  # noqa: F401

        return True
    except ImportError:
        return False


def get_fisher_provider(cfg):
    """Resolve the provider from cfg.prior.fisher (see module docstring)."""
    from emridispatch.fisher import manual as _manual

    kind = str(getattr(cfg.prior, "fisher", "auto")).lower()
    full = bool(getattr(cfg.prior, "fisher_full", False))
    has_manual_cfg = (getattr(cfg.prior, "sigmas", None) is not None
                      or getattr(cfg.prior, "covariance_file", None) is not None)

    if kind == "auto":
        if has_manual_cfg:
            kind = "manual"
        elif _sef_importable():
            kind = "sef"
        else:
            kind = "none"

    if kind == "sef":
        from emridispatch.fisher.sef import SEFFisherProvider

        data = cfg.data
        _channels = getattr(data, "channels", None)
        return SEFFisherProvider(
            tdi=str(getattr(data, "tdi", "2nd generation")),
            foreground=bool(getattr(data, "foreground", True)),
            channels=None if _channels is None else list(_channels),
            full=full,
            notch=float(getattr(data, "psd_notch", PSD_NOTCH)),
            notch_depth=float(getattr(data, "psd_notch_depth", PSD_NOTCH_DEPTH)))
    if full:
        raise ValueError(f"prior.fisher_full needs the 'sef' provider, not {kind!r}")
    if kind == "manual":
        return _manual.ManualFisherProvider.from_config(cfg)
    if kind == "none":
        return _manual.HeuristicFisherProvider()
    raise ValueError(
        f"unknown prior.fisher setting {kind!r} "
        "(auto | sef | manual | none)")


def check_snr(fisher, snr, rtol=0.1):
    """Raise if the provider's SNR differs from the likelihood's by > rtol:
    its Fisher would then describe a different inner product (Gamma ~ SNR^2)."""
    if fisher.snr is None:
        return
    if not abs(fisher.snr / snr - 1.0) <= rtol:
        raise ValueError(
            f"Fisher provider SNR {fisher.snr:.6g} != likelihood SNR {snr:.6g}: "
            "the two inner products differ (PSD, notch, band or response), so "
            "the Fisher box does not describe this likelihood")
