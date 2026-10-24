"""GPry backend: gpry-free adapter unit tests + a small toy end-to-end run.

The box map, support box, target and resampling helpers are pure numpy; the
end-to-end test is skipped when gpry is not installed.
"""

import json
import os

import numpy as np
import pytest

from emridispatch.backends.gpry import (
    SAMPLES_NAME, BoxMap, Target, resample_equal, support_box, wrap_periodic)
from emridispatch.parameters import NDIM, PARAM_NAMES
from emridispatch.pipeline import build_problem

ANGLES = ["q_s", "phi_s", "q_k", "phi_k", "phi_phi", "phi_r"]


def test_boxmap_whitening_round_trip_and_unit_cov():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4))
    cov = A @ A.T + 0.1 * np.eye(4)
    bm = BoxMap.whitening(np.arange(4.0), cov)
    u = rng.standard_normal((5, 4))
    assert np.allclose(bm.to_u(bm.to_z(u)), u)
    assert np.allclose(bm.cov_to_z(cov), np.eye(4))


def test_boxmap_whitening_keeps_diagonal_block_on_axes():
    cov = np.diag([1.0, 4.0, 0.25])
    cov[0, 1] = cov[1, 0] = 1.0
    bm = BoxMap.whitening(np.zeros(3), cov)
    # The uncorrelated last row maps to its own (rescaled) axis.
    assert np.allclose(bm.L[2], [0.0, 0.0, 0.5])
    assert np.allclose(bm.L[:, 2], [0.0, 0.0, 0.5])


def test_support_box_prior_domain_is_exact(toy_cfg):
    toy_cfg.prior.fixed = ANGLES
    problem = build_problem(toy_cfg)
    w = problem.wrapped()
    bm = BoxMap.identity(w.ndim)
    box = support_box(problem, w, bm, 2000, np.random.default_rng(0))
    assert box.shape == (6, 2)
    # Prior draws mapped to u lie inside the box ...
    draws = problem.prior.sample(np.random.default_rng(1), size=500)
    u = problem.reparam.to_u(draws)[:, :6]
    assert np.all(u >= box[:, 0] - 1e-12) and np.all(u <= box[:, 1] + 1e-12)
    # ... and the box is tight: each face is touched by a corner of the
    # physical box, i.e. sum_j |A_ij| * half-width_j for a linear reparam.
    A = problem.reparam._A
    half = 0.5 * (problem.prior.maxes - problem.prior.mins)[:6]
    centre = problem.reparam.to_u(0.5 * (problem.prior.mins + problem.prior.maxes))[:6]
    expect = np.abs(A) @ half
    assert np.allclose(box[:, 1] - centre, expect, rtol=1e-9)
    assert np.allclose(centre - box[:, 0], expect, rtol=1e-9)


def test_support_box_unbounded_prior_returns_none(toy_cfg):
    toy_cfg.prior.fixed = ANGLES
    toy_cfg.priors = {"dist": {"type": "gaussian", "mu": 1.0, "sigma": 0.1,
                               "min": -np.inf, "max": np.inf}}
    problem = build_problem(toy_cfg)
    w = problem.wrapped()
    assert support_box(problem, w, BoxMap.identity(w.ndim), 10,
                       np.random.default_rng(0)) is None


class _Counting:
    def __init__(self, value):
        self.value, self.calls = value, 0

    def __call__(self, u):
        self.calls += 1
        return self.value


def test_target_skips_likelihood_outside_prior():
    lnlike = _Counting(-3.0)
    inside = Target(lnlike, lambda u: -1.0, BoxMap.identity(2))
    assert inside(np.zeros(2)) == -4.0
    outside = Target(lnlike, lambda u: -np.inf, BoxMap.identity(2))
    assert outside(np.zeros(2)) == -np.inf
    assert lnlike.calls == 1
    rec = outside.records()
    assert rec["z"].shape == (1, 2) and rec["seconds"][0] == 0.0


def test_target_maps_nan_likelihood_to_minus_inf():
    t = Target(lambda u: np.nan, lambda u: 0.0, BoxMap.identity(1))
    assert t(np.zeros(1)) == -np.inf


def test_resample_equal_follows_weights():
    w = np.array([0.0, 1.0, 3.0])
    idx = resample_equal(w, 4000, np.random.default_rng(0))
    assert not np.any(idx == 0)
    assert np.isclose(np.mean(idx == 2), 0.75, atol=0.01)


def test_wrap_periodic():
    u = np.array([[7.0, -0.5]])
    out = wrap_periodic(u, {0: 2 * np.pi, 1: 2 * np.pi}, np.zeros(2))
    assert np.allclose(out, [[7.0 - 2 * np.pi, 2 * np.pi - 0.5]])


@pytest.mark.parametrize("domain", ["prior", "fisher"])
def test_gpry_backend_end_to_end(toy_cfg, domain):
    pytest.importorskip("gpry")
    pytest.importorskip("h5py")
    from emridispatch.backends import get_backend
    from emridispatch.results import convert

    # 2 free dimensions keep the run (and its nested-sampling MC) quick.
    toy_cfg.prior.fixed = [n for n in PARAM_NAMES if n not in ("a", "e")]
    toy_cfg.reparam.idx = [2, 4]
    toy_cfg.prior.fisher = "manual"
    toy_cfg.prior.sigmas = {"mass_1": 1.0e4, "mass_2": 1.0e-2, "a": 0.05,
                            "p": 0.1, "e": 0.05, "luminosity_distance": 0.05}
    toy_cfg.prior.box_scale = 5.0
    toy_cfg.sampler.backend = "gpry"
    toy_cfg.sampler.nsamples = 500
    toy_cfg.sampler.gpry.domain = domain
    toy_cfg.sampler.gpry.max_total = 60
    toy_cfg.sampler.gpry.nlive = 50
    toy_cfg.sampler.gpry.verbose = 1
    problem = build_problem(toy_cfg)
    summary = get_backend("gpry").run(problem, toy_cfg, resume=False)

    assert summary["config"]["backend"] == "gpry"
    assert summary["config"]["domain"] == domain
    assert 0 < summary["n_likelihood_evals"] <= summary["n_truth_calls"] <= 60
    assert os.path.exists(os.path.join(problem.outdir, SAMPLES_NAME))
    with open(os.path.join(problem.outdir, "run_summary.json")) as fh:
        assert json.load(fh)["config"]["backend"] == "gpry"

    results = convert(problem.outdir)
    assert results.backend == "gpry"
    assert results.samples.shape == (1, 500, NDIM)
    x = results.rung(0, physical=True)
    # Toy posterior: N(truth, sigma), sigma = 0.05 * max(|truth|, 1) for a, e.
    for i in (2, 4):
        assert abs(x[:, i].mean() - problem.truth[i]) < 0.02


class _OrbitsWrapGPU:
    """Mimics lisatools' C++ GPU orbit wrapper: neither pickle nor dill can
    serialise it."""

    def __reduce__(self):
        raise TypeError("cannot pickle 'OrbitsWrapGPU' object")


class _GPUBoundLikelihood:
    """Stands in for the lisatools model: evaluates fine, cannot be pickled."""

    def __init__(self, inner):
        self.inner = inner
        self.orbits = _OrbitsWrapGPU()

    def __call__(self, x):
        return self.inner(x)


def _minus_one(u):
    return -1.0


def _zero(u):
    return 0.0


def test_target_pickles_without_likelihood():
    import pickle

    t = Target(_GPUBoundLikelihood(_minus_one), _zero, BoxMap.identity(2))
    assert t(np.zeros(2)) == -1.0
    restored = pickle.loads(pickle.dumps(t))
    assert np.allclose(restored.boxmap.u0, t.boxmap.u0)
    with pytest.raises(RuntimeError, match="no likelihood"):
        restored(np.zeros(2))


def test_gpry_backend_checkpoint_and_resume_unpicklable_likelihood(toy_cfg):
    pytest.importorskip("gpry")
    from emridispatch.backends import get_backend

    toy_cfg.prior.fixed = [n for n in PARAM_NAMES if n not in ("a", "e")]
    toy_cfg.reparam.idx = [2, 4]
    toy_cfg.prior.fisher = "manual"
    toy_cfg.prior.sigmas = {"mass_1": 1.0e4, "mass_2": 1.0e-2, "a": 0.05,
                            "p": 0.1, "e": 0.05, "luminosity_distance": 0.05}
    toy_cfg.prior.box_scale = 5.0
    toy_cfg.sampler.backend = "gpry"
    toy_cfg.sampler.nsamples = 200
    toy_cfg.sampler.gpry.max_total = 40
    toy_cfg.sampler.gpry.nlive = 50
    toy_cfg.sampler.gpry.verbose = 1
    problem = build_problem(toy_cfg)
    problem.lnlike = _GPUBoundLikelihood(problem.lnlike)

    backend = get_backend("gpry")
    first = backend.run(problem, toy_cfg, resume=False)
    assert os.path.isdir(os.path.join(problem.outdir, "gpry_checkpoint"))
    # Resume from the checkpoint with the live likelihood, as a resubmitted job does.
    second = backend.run(problem, toy_cfg, resume=True)
    assert second["n_truth_calls"] >= first["n_truth_calls"]
