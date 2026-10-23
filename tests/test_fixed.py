"""prior.fixed: resolution, the reduced wrapped() view, and the round trip
through a backend and its converter back to the full 12-D results."""

import dataclasses
import json
import os

import numpy as np
import pytest

from emridispatch.parameters import NDIM, PARAM_NAMES
from emridispatch.pipeline import FIXED_NAME, build_problem, resolve_fixed

ANGLES = ["q_s", "phi_s", "q_k", "phi_k", "phi_phi", "phi_r"]
TRUTH = np.linspace(0.1, 1.2, NDIM)


def test_resolve_fixed_list_pins_truth():
    fixed = resolve_fixed(["q_s", "phi_r"], TRUTH, range(6), "auto")
    assert fixed == {6: TRUTH[6], 11: TRUTH[11]}


def test_resolve_fixed_mapping_explicit_values():
    fixed = resolve_fixed({"phi_s": 0.5}, TRUTH, range(6), "auto")
    assert fixed == {7: 0.5}


def test_resolve_fixed_empty():
    assert resolve_fixed(None, TRUTH, range(6), "auto") == {}
    assert resolve_fixed([], TRUTH, range(6), "auto") == {}


def test_resolve_fixed_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown parameter"):
        resolve_fixed(["q_sky"], TRUTH, range(6), "auto")


def test_resolve_fixed_rejects_reparam_block_when_whitened():
    with pytest.raises(ValueError, match="inside the reparam block"):
        resolve_fixed(["a"], TRUTH, range(6), "auto")
    # The identity reparam does not mix rows, so this is allowed.
    assert resolve_fixed(["a"], TRUTH, range(6), "off") == {2: TRUTH[2]}


def test_wrapped_reduced_view(toy_cfg):
    toy_cfg.prior.fixed = ANGLES
    problem = build_problem(toy_cfg)
    full = dataclasses.replace(problem, fixed={}).wrapped()
    w = problem.wrapped()

    assert w.ndim == 6
    assert list(w.free_idx) == list(range(6))
    assert w.x0.shape == (6,)
    assert w.proposal_cov.shape == (6, 6)
    assert w.truth.shape == (6,)
    assert w.periodic == {}     # all periodic rows were angles
    # embed reinstates the fixed rows; callables agree with the full view.
    assert np.allclose(w.embed(w.x0)[6:], problem.truth[6:])
    assert np.isclose(w.lnlike(w.x0), full.lnlike(w.embed(w.x0)))
    assert np.isclose(w.lnprior(w.x0), full.lnprior(w.embed(w.x0)))
    # Batch-safe embedding.
    assert w.embed(np.zeros((3, 6))).shape == (3, NDIM)

    with open(os.path.join(problem.outdir, FIXED_NAME)) as fh:
        spec = json.load(fh)
    assert spec["names"] == ANGLES
    assert problem.meta["fixed"] == ANGLES


def test_periodic_reindexed(toy_cfg):
    toy_cfg.prior.fixed = ["q_s"]   # index 6: shifts every later row down one
    w = build_problem(toy_cfg).wrapped()
    assert w.ndim == NDIM - 1
    assert w.periodic == {i - 1: 2 * np.pi for i in (7, 9, 10, 11)}


def test_unfixed_view_unchanged(toy_cfg):
    problem = build_problem(toy_cfg)
    w = problem.wrapped()
    assert w.ndim == NDIM
    assert not os.path.exists(os.path.join(problem.outdir, FIXED_NAME))
    assert np.allclose(w.embed(w.x0), w.x0)


def test_eryn_fixed_round_trip(toy_cfg, tmp_path):
    pytest.importorskip("eryn")
    pytest.importorskip("h5py")
    from emridispatch.backends import get_backend
    from emridispatch.results import convert
    from emridispatch.viz import make_plots

    toy_cfg.prior.fixed = ANGLES
    toy_cfg.sampler.backend = "eryn"
    toy_cfg.sampler.nsamples = 10
    toy_cfg.sampler.eryn.nwalkers = 12     # 2 * 6 free dims suffices
    problem = build_problem(toy_cfg)
    assert get_backend("eryn").run(problem, toy_cfg, resume=False) is not None

    results = convert(problem.outdir)
    assert results.ndim == NDIM
    assert results.param_names == PARAM_NAMES
    pinned = results.samples[0][:, 6:]
    assert np.allclose(pinned, problem.truth[6:])
    assert np.all(np.ptp(results.samples[0][:, :6], axis=0) > 0)

    pytest.importorskip("corner")
    written = make_plots(results, outdir=str(tmp_path / "plots"))
    assert all(os.path.exists(p) for p in written)
