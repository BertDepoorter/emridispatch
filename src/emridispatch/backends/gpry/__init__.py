"""GPry active-learning GP surrogate backend (optional: `pip install emridispatch[gpry]`).

Not an MCMC: GPry regresses the log-posterior with a Gaussian process, picks
batches of true evaluations by acquisition, and finally samples the surrogate
(nested sampling). The backend stores an equal-weight resample of that
surrogate sample, plus every true evaluation it made.

GPry's prior is uniform on an axis-aligned box, so the sampling space u of
problem.wrapped() is mapped affinely to a box coordinate z, u = u0 + L z
(sampler.gpry.domain):

  prior   L = 1, u0 = 0. Box = bounding box of the prior support in u, with the
          true lnprior folded into the likelihood (-inf outside the support,
          learned by GPry's infinities classifier). Same posterior as every
          other backend, but u is rotated, not rescaled: on a strongly
          correlated posterior (a real EMRI) it is a thin slab in the box and
          acquisition stalls. Only for near-uncorrelated problems.
  fisher  L L^T = proposal covariance in u (Fisher for the intrinsics, the
          angle_sigma heuristic for the angles), u0 = start point. Box =
          |z_i| <= fisher_k, clipped to the support's bounding box. This
          TRUNCATES the prior at fisher_k proposal sigmas: posterior mass
          outside is lost by construction.

Periodic parameters are plain intervals to GPry (no periodic boundary).

Config: sampler.gpry (domain, fisher_k, ref_scale, kernel, n_initial,
max_initial, max_total, n_points_per_acq, mc, nlive, prior_draws, plots,
verbose); sampler.nsamples is the number of posterior draws stored.
"""

import itertools
import json
import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)

SAMPLES_NAME = "gpry_samples.npz"
EVALS_NAME = "gpry_evals.npz"
PROGRESS_NAME = "gpry_progress.csv"
CHECKPOINT_DIR = "gpry_checkpoint"
DOMAINS = ("prior", "fisher")
# Above this many free dimensions the support box comes from draws only.
MAX_CORNER_DIM = 14


def _coerce(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return float(o)


class BoxMap:
    """Affine map between sampling coords u and GPry box coords z: u = u0 + L z."""

    def __init__(self, u0, L):
        self.u0 = np.asarray(u0, dtype=float)
        self.L = np.asarray(L, dtype=float)
        self._Linv = np.linalg.inv(self.L)

    @classmethod
    def identity(cls, ndim):
        return cls(np.zeros(ndim), np.eye(ndim))

    @classmethod
    def whitening(cls, u0, cov):
        """z = L^{-1}(u - u0), L the Cholesky factor of cov.

        Cholesky (not an eigenbasis) keeps a diagonal sub-block, like the
        angle rows, on its own axes.
        """
        cov = np.asarray(cov, dtype=float)
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            jitter = 1e-12 * np.max(np.diag(cov))
            logger.warning("gpry: proposal covariance not positive definite; "
                           "adding %.3g jitter to the diagonal", jitter)
            L = np.linalg.cholesky(cov + jitter * np.eye(len(cov)))
        return cls(u0, L)

    def to_u(self, z):
        return self.u0 + np.asarray(z, dtype=float) @ self.L.T

    def to_z(self, u):
        return (np.asarray(u, dtype=float) - self.u0) @ self._Linv.T

    def cov_to_z(self, cov):
        return self._Linv @ np.asarray(cov, dtype=float) @ self._Linv.T


def support_box(problem, w, boxmap, n_draws, rng):
    """(d, 2) bounding box in z of the prior support, or None if unbounded.

    Exact for a linear reparam (the support is a parallelotope whose extreme
    points are corners of the physical box); for "grid" it is estimated from
    corners plus prior draws and padded by 1% of the width.
    """
    free = w.free_idx
    mins, maxes = problem.prior.mins, problem.prior.maxes
    if not (np.all(np.isfinite(mins[free])) and np.all(np.isfinite(maxes[free]))):
        return None

    pts = []
    if len(free) <= MAX_CORNER_DIM:
        base = w.embed(boxmap.to_u(np.zeros(len(free))))   # fixed rows set
        for corner in itertools.product((0, 1), repeat=len(free)):
            x = problem.reparam.to_x(base)
            x[free] = np.where(np.array(corner), maxes[free], mins[free])
            pts.append(x)
    if n_draws > 0:
        draws = problem.prior.sample(rng, size=int(n_draws))
        for i, v in problem.fixed.items():
            draws[:, i] = v
        pts.extend(draws)

    z = []
    for x in pts:
        try:
            z.append(boxmap.to_z(problem.reparam.to_u(x)[free]))
        except ValueError:      # outside the grid reparam's domain
            continue
    z = np.array(z)
    lo, hi = z.min(axis=0), z.max(axis=0)
    if problem.reparam_mode == "grid":
        pad = 0.01 * (hi - lo)
        lo, hi = lo - pad, hi + pad
    return np.column_stack([lo, hi])


def _detached(u):
    raise RuntimeError("Target restored from a GPry checkpoint has no "
                       "likelihood; pass the live one to Runner on resume")


class Target:
    """GPry log-likelihood in box coords: true lnlike + lnprior at u(z).

    Points with a non-finite prior skip the (expensive) likelihood. Every call
    is recorded, with its wall time, for the evaluation-count bookkeeping.
    """

    def __init__(self, lnlike, lnprior, boxmap):
        self.lnlike = lnlike
        self.lnprior = lnprior
        self.boxmap = boxmap
        self.z, self.lnl, self.lnp, self.seconds = [], [], [], []

    def __call__(self, z):
        z = np.asarray(z, dtype=float)
        u = self.boxmap.to_u(z)
        lp = float(self.lnprior(u))
        t0 = time.perf_counter()
        if np.isfinite(lp):
            ll = float(self.lnlike(u))
            if np.isnan(ll):
                ll = -np.inf
        else:
            ll = -np.inf
        self.z.append(z)
        self.lnl.append(ll)
        self.lnp.append(lp)
        self.seconds.append(time.perf_counter() - t0 if np.isfinite(lp) else 0.0)
        return ll + lp if np.isfinite(lp) else -np.inf

    def __getstate__(self):
        """GPry pickles its likelihood into every checkpoint, but on resume it
        uses the Target passed to Runner (this backend always passes one). The
        likelihood can hold unpicklable GPU objects, so only the box map is
        stored; an unpickled Target refuses to evaluate."""
        return {"boxmap": self.boxmap}

    def __setstate__(self, state):
        self.__init__(_detached, _detached, state["boxmap"])

    def records(self):
        d = len(self.boxmap.u0)
        return dict(z=np.reshape(self.z, (-1, d)), lnlike=np.array(self.lnl),
                    lnprior=np.array(self.lnp), seconds=np.array(self.seconds))


def resample_equal(weights, n, rng):
    """Systematic resampling: n indices drawn with probability ~ weights."""
    w = np.asarray(weights, dtype=float)
    cdf = np.cumsum(w / w.sum())
    cdf[-1] = 1.0
    return np.searchsorted(cdf, (rng.random() + np.arange(n)) / n)


def wrap_periodic(u, periodic, lows):
    """Map periodic rows of u into [low, low + period)."""
    u = np.array(u, dtype=float)
    for i, period in periodic.items():
        u[..., i] = np.mod(u[..., i] - lows[i], period) + lows[i]
    return u


def _append_evals(path, rec):
    """Accumulate true-evaluation records across resumed invocations."""
    if os.path.exists(path):
        with np.load(path) as old:
            rec = {k: np.concatenate([old[k], rec[k]]) for k in rec}
    np.savez(path, **rec)
    return rec


class GPryBackend:
    name = "gpry"

    def run(self, problem, cfg, resume=True):
        """Run GPry, store the surrogate posterior, and write run_summary.json."""
        try:
            from gpry import mpi
            from gpry import __version__ as gpry_version
            from gpry.run import Runner
        except ImportError as err:
            raise ImportError(
                "GPry is required for the 'gpry' backend. "
                "Install with `pip install emridispatch[gpry]`."
            ) from err

        g = cfg.sampler.gpry
        domain = str(g.domain).lower()
        if domain not in DOMAINS:
            raise ValueError(
                f"sampler.gpry.domain must be one of {DOMAINS} (got {g.domain!r})")
        nsamples = int(cfg.sampler.nsamples)
        outdir = problem.outdir
        os.makedirs(outdir, exist_ok=True)
        rng = np.random.default_rng(problem.seed)

        w = problem.wrapped()
        if domain == "fisher":
            if w.proposal_cov is None:
                raise ValueError("sampler.gpry.domain 'fisher' needs a proposal "
                                 "covariance; none was built for this problem")
            boxmap = BoxMap.whitening(w.x0, w.proposal_cov)
        else:
            boxmap = BoxMap.identity(w.ndim)

        support = support_box(problem, w, boxmap, int(g.prior_draws), rng)
        if domain == "prior":
            if support is None:
                raise ValueError(
                    "sampler.gpry.domain 'prior' needs finite prior bounds on "
                    "every free parameter; use domain 'fisher' or bound the "
                    "override priors")
            bounds = support
        else:
            k = float(g.fisher_k)
            bounds = np.tile([-k, k], (w.ndim, 1))
            if support is not None:
                bounds[:, 0] = np.maximum(bounds[:, 0], support[:, 0])
                bounds[:, 1] = np.minimum(bounds[:, 1], support[:, 1])
            if np.any(bounds[:, 1] <= bounds[:, 0]):
                raise ValueError("gpry: fisher box and prior support do not "
                                 "overlap; check the start point")
        log_box_volume = float(np.sum(np.log(bounds[:, 1] - bounds[:, 0])))

        # Fisher-informed initial design: ref_scale proposal sigmas around the
        # start point, clipped to the box (null: draw from the whole box).
        ref_bounds = None
        if g.ref_scale is not None and w.proposal_cov is not None:
            zc = boxmap.to_z(w.x0)
            half = float(g.ref_scale) * np.sqrt(np.diag(boxmap.cov_to_z(w.proposal_cov)))
            ref = np.column_stack([np.maximum(zc - half, bounds[:, 0]),
                                   np.minimum(zc + half, bounds[:, 1])])
            ref_bounds = [list(r) for r in ref]

        target = Target(w.lnlike, w.lnprior, boxmap)
        names = [problem.param_names[i] for i in w.free_idx]
        options = {key: (int(getattr(g, key)) if getattr(g, key) is not None else None)
                   for key in ("n_initial", "max_initial", "max_total",
                               "n_points_per_acq")}
        options = {k: v for k, v in options.items() if v is not None}
        if "max_total" in options and "max_initial" not in options:
            # GPry rejects max_total < max_initial (default 30 d^1.5).
            options["max_initial"] = min(int(30 * w.ndim ** 1.5),
                                         options["max_total"])
        mc = str(g.mc)
        mc_spec = {mc: ({"nlive": int(g.nlive)} if g.nlive is not None else {})}

        ckpt = os.path.join(outdir, CHECKPOINT_DIR)
        logger.info("gpry: domain=%s ndim=%d ln(box volume)=%.3g ref_scale=%s "
                    "kernel=%s options=%s mpi_size=%d", domain, w.ndim,
                    log_box_volume, g.ref_scale, g.kernel, options, mpi.SIZE)

        runner = Runner(
            loglike=target, bounds=[list(b) for b in bounds],
            ref_bounds=ref_bounds, params=names, surrogate=g.kernel,
            initial_proposer="reference", mc=mc_spec, options=options or None,
            checkpoint=ckpt, load_checkpoint="resume" if resume else "overwrite",
            seed=problem.seed, plots=bool(g.plots), verbose=int(g.verbose))

        t0 = time.perf_counter()
        runner.run()
        if runner.last_mc_samples() is None:
            runner.generate_mc_sample(sampler=mc_spec)
        wall = time.perf_counter() - t0

        # Every rank evaluated part of each batch; collect on the main rank.
        all_recs = mpi.gather(target.records())
        if not mpi.is_main_process:
            return {"mpi_rank": mpi.RANK}   # not None: the CLI reads that as failure
        rec = {k: np.concatenate([r[k] for r in all_recs]) for k in all_recs[0]}
        u_eval = boxmap.to_u(rec["z"])
        evals = _append_evals(
            os.path.join(outdir, EVALS_NAME),
            dict(samples=u_eval, lnlike=rec["lnlike"], lnprior=rec["lnprior"],
                 seconds=rec["seconds"]))

        mcs = runner.last_mc_samples()
        weights = (np.ones(len(mcs["X"])) if mcs["w"] is None
                   else np.asarray(mcs["w"], dtype=float))
        ess = float(weights.sum() ** 2 / np.sum(weights ** 2))
        if nsamples > ess:
            logger.warning("gpry: storing %d draws from an MC sample of effective "
                           "size %.0f; draws repeat", nsamples, ess)
        pick = resample_equal(weights, nsamples, rng)
        lows = problem.prior.mins[w.free_idx]
        u = wrap_periodic(boxmap.to_u(mcs["X"][pick]), w.periodic, lows)
        lnprior = np.array([float(w.lnprior(row)) for row in u])
        # GPry's MC "loglike" is its regressed target, i.e. our lnlike + lnprior.
        lnlike = np.asarray(mcs["loglike"], dtype=float)[pick] - lnprior
        np.savez(os.path.join(outdir, SAMPLES_NAME), samples=u,
                 lnlike=lnlike, lnprior=lnprior)

        progress = runner.progress.data
        progress.to_csv(os.path.join(outdir, PROGRESS_NAME), index=False)
        finite = np.isfinite(evals["lnlike"])
        logz = runner.last_mc_logZ()
        summary = {
            "config": dict(problem.meta, backend=self.name, domain=domain,
                           fisher_k=float(g.fisher_k), ref_scale=g.ref_scale,
                           kernel=g.kernel, options=options, mc=mc_spec,
                           nsamples=nsamples, seed=problem.seed,
                           gpry_version=gpry_version, mpi_size=mpi.SIZE),
            "converged": bool(runner.has_converged),
            "n_truth_calls": int(len(evals["lnlike"])),
            "n_likelihood_evals": int(np.sum(np.isfinite(evals["lnprior"]))),
            "n_finite": int(finite.sum()),
            "lnlike_max": float(np.max(evals["lnlike"][finite])) if finite.any() else None,
            "seconds_likelihood": float(np.sum(evals["seconds"])),
            "seconds_wall": wall,
            "seconds_by_stage": {c: float(progress[c].astype(float).sum())
                                 for c in ("time_acquire", "time_truth",
                                           "time_fit", "time_mc",
                                           "time_convergence")},
            "ln_box_volume": log_box_volume,
            "mc_ess": ess,
            "logZ": None if logz is None else list(logz),
        }
        summary_path = os.path.join(outdir, "run_summary.json")
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2, default=_coerce)
        logger.info("gpry: %d likelihood evaluations, converged=%s, wall %.1f s "
                    "-> %s", summary["n_likelihood_evals"], summary["converged"],
                    wall, summary_path)
        return summary
