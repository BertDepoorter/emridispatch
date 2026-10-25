"""StableEMRIFisher provider (optional: `pip install emridispatch[fisher]`).

Needs stableemrifisher, few and lisatools only at compute() time; the
module itself imports clean.
"""

import logging

import numpy as np

from emridispatch.fisher import FisherResult
from emridispatch.noise import (
    MIN_FREQ, PSD_NOTCH, PSD_NOTCH_DEPTH, channel_noise_psd,
    per_channel_noise_kwargs, sensitivity_spec)
from emridispatch.parameters import FEW_TO_INJECTION, INTRINSIC_ORDER, few_params

logger = logging.getLogger(__name__)


def _gpu_available():
    """True iff a CUDA device is actually usable (not just cupy importable)."""
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class _SharedWaveform:
    """Callable standing in for a waveform class, returning one shared instance.

    StableEMRIFisher instantiates the waveform class twice -- once for the
    generator, once for StableEMRIDerivative -- and each copy of the FEW
    amplitude tables costs ~5 GB of device memory.
    """

    def __init__(self, waveform_class):
        self._waveform_class = waveform_class
        self._instance = None

    def __call__(self, *args, **kwargs):
        if self._instance is None:
            self._instance = self._waveform_class(*args, **kwargs)
        return self._instance


class SEFFisherProvider:
    name = "sef"

    def __init__(self, tdi="2nd generation", foreground=True, channels=None,
                 full=False, notch=PSD_NOTCH, notch_depth=PSD_NOTCH_DEPTH):
        self.tdi = tdi
        self.foreground = foreground
        self.channels = channels
        self.full = full
        self.notch = notch                  # same PSD-null mask as the likelihood
        self.notch_depth = notch_depth

    def compute(self, injection_parameters, *, duration, delta_t, use_gpu=None):
        sigmas, cov, order, snr = get_parameter_precision(
            injection_parameters, duration=duration, delta_t=delta_t,
            use_gpu=use_gpu, tdi=self.tdi, foreground=self.foreground,
            channels=self.channels, full=self.full, notch=self.notch,
            notch_depth=self.notch_depth)
        if not self.full:
            return FisherResult(sigmas=sigmas, cov=cov, order=order, snr=snr)
        k = len(INTRINSIC_ORDER)
        return FisherResult(sigmas=sigmas, cov=cov[:k, :k], order=order[:k],
                            cov_full=cov, snr=snr)


def get_parameter_precision(input_parameters, duration, delta_t,
                            use_gpu=None, tdi="2nd generation",
                            foreground=True, channels=None, full=False,
                            notch=PSD_NOTCH, notch_depth=PSD_NOTCH_DEPTH):
    """(sigmas, cov, order, snr) from SEF. full=False differentiates the six
    intrinsic parameters only (angles and phases held at the injection);
    full=True all twelve sampled ones, so cov is marginal over the angles."""
    if use_gpu is None:
        use_gpu = _gpu_available()
    force_backend = None if use_gpu else "cpu"

    if use_gpu:
        try:
            import cupy

            cupy.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    try:
        from few.waveform import FastKerrEccentricEquatorialFlux
        from stableemrifisher.fisher import StableEMRIFisher
    except ImportError as err:
        raise ImportError(
            "StableEMRIFisher / FastEMRIWaveforms are required for the 'sef' "
            "Fisher provider. Install with `pip install emridispatch[fisher]` (plus "
            "the lisatools extra), or configure prior.fisher: manual with your "
            "own sigmas/covariance."
        ) from err

    dt = delta_t
    T = duration
    wave_params = few_params(input_parameters)

    waveform_class_kwargs = {
        "sum_kwargs": {"pad_output": True},
        "force_backend": force_backend,
    }
    waveform_generator_kwargs = {"frame": "detector"}

    INDEX_LAMBDA = 8
    INDEX_BETA = 7

    t0 = 30000.0

    sens_classes, channels = sensitivity_spec(tdi, channels)
    extra_sef_kwargs = dict(
        noise_model=channel_noise_psd,
        noise_kwargs=per_channel_noise_kwargs(T, foreground, sens_classes,
                                              notch, notch_depth),
        channels=channels,
    )

    logger.info("fisher: tdi %s (channels %s, foreground=%s, gpu=%s)",
                tdi, channels, foreground, use_gpu)

    if tdi == "off":
        ResponseWrapper = None
        ResponseWrapper_kwargs = None
    else:
        from lisatools.response import ResponseWrapper
        from lisatools.detector import EqualArmlengthOrbits

        tdi_kwargs = dict(
            orbits=EqualArmlengthOrbits(force_backend=force_backend),
            order=25,
            tdi=tdi,
            tdi_chan="".join(channels),
        )
        ResponseWrapper_kwargs = dict(
            Tobs=T,
            dt=dt,
            index_lambda=INDEX_LAMBDA,
            index_beta=INDEX_BETA,
            t0=t0,
            flip_hx=True,
            force_backend=force_backend,
            is_ecliptic_latitude=False,
            remove_garbage="zero",
            **tdi_kwargs,
        )

    Ndelta = 8
    sef = StableEMRIFisher(
        waveform_class=_SharedWaveform(FastKerrEccentricEquatorialFlux),
        waveform_class_kwargs=waveform_class_kwargs,
        waveform_generator_kwargs=waveform_generator_kwargs,
        ResponseWrapper=ResponseWrapper,
        ResponseWrapper_kwargs=ResponseWrapper_kwargs,
        stats_for_nerds=True,
        use_gpu=use_gpu,
        T=T,
        dt=dt,
        der_order=4,
        **extra_sef_kwargs,
    )

    param_names = ["m1", "m2", "a", "p0", "e0", "dist"]
    if full:
        # Same order as the sampling vector rows 6..11; SEF picks the angle and
        # phase steps itself (relative grid 1e-1..1e-6 of the value).
        param_names += ["qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_r0"]

    delta_range = dict(
        m1=np.geomspace(1e3, 1e-5, Ndelta),
        m2=np.geomspace(1e-2, 1e-8, Ndelta),
        a=np.geomspace(1e-5, 1e-9, Ndelta),
        p0=np.geomspace(1e-5, 1e-9, Ndelta),
        e0=np.geomspace(1e-5, 1e-9, Ndelta),
        dist=np.geomspace(1e-5, 1e-9, Ndelta),
    )

    fisher_matrix = sef(
        wave_params,
        param_names=param_names,
        delta_range=delta_range,
        fmin=MIN_FREQ,
    )

    snr2 = getattr(sef, "SNR2", None)
    snr = None if snr2 is None else float(np.sqrt(snr2))
    logger.info("fisher: %d params, SNR (SEF) = %s, cond(Gamma) = %.3g",
                len(param_names), snr, np.linalg.cond(fisher_matrix))
    check_positive_definite(fisher_matrix, param_names)

    param_cov = np.linalg.inv(fisher_matrix)
    param_cov = 0.5 * (param_cov + param_cov.T)

    order = [FEW_TO_INJECTION[k] for k in param_names]
    sigmas = {name: param_cov[i, i] ** (1 / 2) for i, name in enumerate(order)}

    return sigmas, param_cov, order, snr


def check_positive_definite(fisher_matrix, param_names):
    """Raise unless Gamma is positive definite, naming the parameters that
    make up each non-positive eigen-direction of the normalised matrix
    D^-1/2 Gamma D^-1/2, D = diag(Gamma) (so units drop out)."""
    F = 0.5 * (np.asarray(fisher_matrix) + np.asarray(fisher_matrix).T)
    d = np.diag(F)
    if np.any(~np.isfinite(F)) or np.any(d <= 0):
        bad = [n for n, v in zip(param_names, d) if not v > 0]
        raise ValueError(f"Fisher matrix is not finite or has Gamma_ii <= 0 "
                         f"for {bad}")
    s = 1.0 / np.sqrt(d)
    w, V = np.linalg.eigh(F * np.outer(s, s))
    if w[0] > 0:
        return
    lines = []
    for k in np.flatnonzero(w <= 0):
        top = np.argsort(-np.abs(V[:, k]))[:3]
        lines.append(f"  eigenvalue {w[k]:.3g}: "
                     + ", ".join(f"{param_names[i]} {V[i, k]:+.2f}" for i in top))
    raise ValueError(
        "Fisher matrix is not positive definite (normalised eigenvalues "
        f"{w[0]:.3g} .. {w[-1]:.3g}); degenerate directions:\n" + "\n".join(lines))
