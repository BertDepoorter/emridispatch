import logging
from types import SimpleNamespace

import numpy as np
import pytest

import emridispatch.fisher.sef as sef_mod
from emridispatch.fisher.sef import SEFFisherProvider

INJ = {
    "mass_1": 1e6, "mass_2": 10.0, "a": 0.9, "p": 12.0, "e": 0.2, "x": 1.0,
    "luminosity_distance": 1.0, "q_s": 1.0, "phi_s": 1.5, "q_k": 1.0,
    "phi_k": 1.5, "phi_phi": 1.5, "phi_theta": 1.5, "phi_r": 1.5,
}


def _fake_gpp_factory(seen):
    def fake_gpp(params, *, duration, delta_t, use_gpu=None,
                 tdi="2nd generation", foreground=True, channels=None,
                 full=False, notch=0.0, notch_depth=2.0):
        seen["tdi"] = tdi
        seen["foreground"] = foreground
        seen["channels"] = channels
        sig = dict.fromkeys(
            ["mass_1", "mass_2", "a", "p", "e", "luminosity_distance"], 1.0)
        return sig, np.eye(6), list(sig), 30.0

    return fake_gpp


def test_provider_stores_and_forwards_tdi(monkeypatch):
    seen = {}
    monkeypatch.setattr(sef_mod, "get_parameter_precision",
                        _fake_gpp_factory(seen))
    provider = SEFFisherProvider(tdi="off")
    provider.compute({}, duration=0.3, delta_t=10.0)
    assert seen["tdi"] == "off"


def test_provider_forwards_foreground(monkeypatch):
    seen = {}
    monkeypatch.setattr(sef_mod, "get_parameter_precision",
                        _fake_gpp_factory(seen))
    SEFFisherProvider(tdi="off", foreground=False).compute(
        {}, duration=0.3, delta_t=10.0)
    assert seen["foreground"] is False


def test_provider_forwards_channels(monkeypatch):
    seen = {}
    monkeypatch.setattr(sef_mod, "get_parameter_precision",
                        _fake_gpp_factory(seen))
    SEFFisherProvider(channels=["A", "E", "T"]).compute(
        {}, duration=0.3, delta_t=10.0)
    assert seen["channels"] == ["A", "E", "T"]


def test_provider_defaults():
    provider = SEFFisherProvider()
    assert provider.tdi == "2nd generation"
    assert provider.foreground is True
    assert provider.channels is None


def test_get_fisher_provider_passes_tdi_and_foreground(monkeypatch):
    import emridispatch.fisher as fisher_mod

    monkeypatch.setattr(fisher_mod, "_sef_importable", lambda: True)
    cfg = SimpleNamespace(
        prior=SimpleNamespace(fisher="sef", sigmas=None, covariance_file=None),
        data=SimpleNamespace(tdi="off", foreground=False))
    provider = fisher_mod.get_fisher_provider(cfg)
    assert provider.tdi == "off"
    assert provider.foreground is False


def test_get_fisher_provider_foreground_defaults_true(monkeypatch):
    import emridispatch.fisher as fisher_mod

    monkeypatch.setattr(fisher_mod, "_sef_importable", lambda: True)
    cfg = SimpleNamespace(
        prior=SimpleNamespace(fisher="sef", sigmas=None, covariance_file=None),
        data=SimpleNamespace(tdi="off"))
    provider = fisher_mod.get_fisher_provider(cfg)
    assert provider.foreground is True
    assert provider.channels is None


def test_get_fisher_provider_passes_channels(monkeypatch):
    import emridispatch.fisher as fisher_mod

    monkeypatch.setattr(fisher_mod, "_sef_importable", lambda: True)
    cfg = SimpleNamespace(
        prior=SimpleNamespace(fisher="sef", sigmas=None, covariance_file=None),
        data=SimpleNamespace(tdi="2nd generation", channels=["A", "E", "T"]))
    assert fisher_mod.get_fisher_provider(cfg).channels == ["A", "E", "T"]


def test_get_fisher_provider_channels_none_stays_none(monkeypatch):
    import emridispatch.fisher as fisher_mod

    monkeypatch.setattr(fisher_mod, "_sef_importable", lambda: True)
    cfg = SimpleNamespace(
        prior=SimpleNamespace(fisher="sef", sigmas=None, covariance_file=None),
        data=SimpleNamespace(tdi="2nd generation", channels=None))
    assert fisher_mod.get_fisher_provider(cfg).channels is None


class _Recorder:
    seen = {}
    called = {}

    def __init__(self, **kwargs):
        type(self).seen = kwargs

    def __call__(self, *args, **kwargs):
        type(self).called = kwargs
        return np.eye(len(kwargs["param_names"]))


@pytest.fixture
def recorded_sef(monkeypatch):
    pytest.importorskip("few")
    stableemrifisher = pytest.importorskip("stableemrifisher.fisher")
    monkeypatch.setattr(stableemrifisher, "StableEMRIFisher", _Recorder)

    def run(**kwargs):
        sef_mod.get_parameter_precision(
            INJ, duration=0.3, delta_t=10.0, use_gpu=False, **kwargs)
        return _Recorder.seen

    return run


def test_wiring_tdi_off(recorded_sef):
    from lisatools.sensitivity import LISASens

    seen = recorded_sef(tdi="off", foreground=False)
    assert seen["channels"] == ["I", "II"]
    assert seen["ResponseWrapper"] is None
    assert seen["ResponseWrapper_kwargs"] is None
    assert seen["noise_model"] is sef_mod.channel_noise_psd
    assert [kw["sens_cls"] for kw in seen["noise_kwargs"]] == [LISASens, LISASens]
    assert all(kw["model"] == "scirdv1" for kw in seen["noise_kwargs"])
    assert all("stochastic_params" not in kw for kw in seen["noise_kwargs"])


def test_wiring_tdi_2nd_generation_aet(recorded_sef):
    pytest.importorskip("lisatools.response")
    from lisatools.sensitivity import A2TDISens, E2TDISens, T2TDISens

    seen = recorded_sef(tdi="2nd generation", foreground=True,
                        channels=["A", "E", "T"])
    assert seen["channels"] == ["A", "E", "T"]
    assert seen["ResponseWrapper_kwargs"]["tdi_chan"] == "AET"
    assert seen["ResponseWrapper_kwargs"]["tdi"] == "2nd generation"
    assert seen["noise_model"] is sef_mod.channel_noise_psd
    assert ([kw["sens_cls"] for kw in seen["noise_kwargs"]]
            == [A2TDISens, E2TDISens, T2TDISens])
    assert all(kw["model"] == "scirdv1" for kw in seen["noise_kwargs"])
    assert all("stochastic_params" in kw for kw in seen["noise_kwargs"])
    assert seen["waveform_generator_kwargs"]["frame"] == "detector"


@pytest.mark.parametrize("tdi", ["off", "2nd generation"])
def test_the_fisher_log_records_the_device_it_resolved(recorded_sef, caplog, tdi):
    if tdi != "off":
        pytest.importorskip("lisatools.response")
    with caplog.at_level(logging.INFO, logger="emridispatch"):
        recorded_sef(tdi=tdi, foreground=False)
    fisher_lines = [r.getMessage() for r in caplog.records
                    if r.getMessage().startswith("fisher: tdi")]
    assert len(fisher_lines) == 1
    assert "gpu=False" in fisher_lines[0]


def test_fisher_rows_are_differentiated_in_intrinsic_order(recorded_sef):
    from emridispatch.parameters import FEW_TO_INJECTION, INTRINSIC_ORDER

    recorded_sef(tdi="off", foreground=False)
    # SEF fills row i from param_names[i], so this list IS the covariance order.
    assert ([FEW_TO_INJECTION[k] for k in _Recorder.called["param_names"]]
            == INTRINSIC_ORDER)


def _ill_conditioned_fisher():
    rot = np.linalg.qr(np.arange(1.0, 37.0).reshape(6, 6) + 7.0 * np.eye(6))[0]
    return rot @ np.diag(np.geomspace(1.0, 1e-12, 6)) @ rot.T


def test_covariance_is_exactly_symmetric(monkeypatch):
    pytest.importorskip("few")
    stableemrifisher = pytest.importorskip("stableemrifisher.fisher")

    gamma = _ill_conditioned_fisher()
    raw = np.linalg.inv(gamma)
    assert not np.array_equal(raw, raw.T)

    class _FixedFisher(_Recorder):
        def __call__(self, *args, **kwargs):
            return gamma

    monkeypatch.setattr(stableemrifisher, "StableEMRIFisher", _FixedFisher)
    sigmas, cov, order, _snr = sef_mod.get_parameter_precision(
        INJ, duration=0.3, delta_t=10.0, use_gpu=False, tdi="off")

    assert np.array_equal(cov, cov.T)
    np.testing.assert_allclose(cov, 0.5 * (raw + raw.T), rtol=0, atol=0)
    np.testing.assert_allclose(
        [sigmas[k] for k in order], np.sqrt(np.diag(cov)), rtol=0, atol=0)


def test_wiring_tdi_default_channels(recorded_sef):
    pytest.importorskip("lisatools.response")

    seen = recorded_sef(tdi="1st generation", foreground=False)
    assert seen["channels"] == ["A", "E"]
    assert seen["ResponseWrapper_kwargs"]["tdi_chan"] == "AE"
    assert len(seen["noise_kwargs"]) == 2


def test_full_fisher_differentiates_every_sampled_parameter(recorded_sef):
    from emridispatch.parameters import (
        FEW_TO_INJECTION, PARAM_NAMES, VECTOR_TO_PHYSICAL)

    sigmas, cov, order, _snr = sef_mod.get_parameter_precision(
        INJ, duration=0.3, delta_t=10.0, use_gpu=False, tdi="off", full=True)
    # Row i of the covariance is sampling-vector row i (masses linear).
    assert ([FEW_TO_INJECTION[k] for k in _Recorder.called["param_names"]]
            == [VECTOR_TO_PHYSICAL[n] for n in PARAM_NAMES])
    assert order == [VECTOR_TO_PHYSICAL[n] for n in PARAM_NAMES]
    assert cov.shape == (12, 12) and set(order) <= set(sigmas)


def test_full_provider_splits_intrinsic_block(monkeypatch):
    from emridispatch.parameters import INTRINSIC_ORDER, PARAM_NAMES, VECTOR_TO_PHYSICAL

    order = [VECTOR_TO_PHYSICAL[n] for n in PARAM_NAMES]
    cov = np.diag(np.arange(1.0, 13.0))
    seen = {}

    def fake_gpp(params, *, full=False, **kwargs):
        seen["full"] = full
        return dict(zip(order, np.sqrt(np.diag(cov)))), cov, order, 30.0

    monkeypatch.setattr(sef_mod, "get_parameter_precision", fake_gpp)
    res = SEFFisherProvider(tdi="off", full=True).compute({}, duration=0.3, delta_t=10.0)
    assert seen["full"] is True
    assert res.order == INTRINSIC_ORDER
    np.testing.assert_array_equal(res.cov, cov[:6, :6])
    np.testing.assert_array_equal(res.cov_full, cov)


def test_fisher_full_needs_sef(monkeypatch):
    import emridispatch.fisher as fisher_mod

    cfg = SimpleNamespace(
        prior=SimpleNamespace(fisher="manual", fisher_full=True,
                              sigmas={"mass_1": 1.0}, covariance_file=None),
        data=SimpleNamespace(tdi="off"))
    with pytest.raises(ValueError, match="fisher_full"):
        fisher_mod.get_fisher_provider(cfg)


def test_provider_forwards_the_likelihood_notch(monkeypatch):
    import emridispatch.fisher as fisher_mod

    seen = {}

    def fake_gpp(params, *, notch, notch_depth, **kwargs):
        seen.update(notch=notch, notch_depth=notch_depth)
        sig = dict.fromkeys(
            ["mass_1", "mass_2", "a", "p", "e", "luminosity_distance"], 1.0)
        return sig, np.eye(6), list(sig), 30.0

    monkeypatch.setattr(sef_mod, "get_parameter_precision", fake_gpp)
    cfg = SimpleNamespace(
        prior=SimpleNamespace(fisher="sef", sigmas=None, covariance_file=None),
        data=SimpleNamespace(tdi="off", psd_notch=3e-5, psd_notch_depth=1.5))
    res = fisher_mod.get_fisher_provider(cfg).compute({}, duration=0.3, delta_t=10.0)
    assert seen == {"notch": 3e-5, "notch_depth": 1.5}
    assert res.snr == 30.0


def test_wiring_passes_the_notch_to_the_sef_psd(recorded_sef):
    seen = recorded_sef(tdi="off", foreground=False)
    assert all(kw["notch"] == 1e-5 for kw in seen["noise_kwargs"])


def test_positive_definite_fisher_passes():
    sef_mod.check_positive_definite(_ill_conditioned_fisher(), list("abcdef"))


def test_singular_fisher_names_the_degenerate_parameters():
    # qK with (numerically) no curvature and phiK fully degenerate with Phi_phi0:
    # the face-on injection's Fisher.
    names = ["m1", "qK", "phiK", "Phi_phi0"]
    F = np.diag([1.0, 1e-15, 1.0, 1.0])
    F[2, 3] = F[3, 2] = 1.0 + 1e-9
    with pytest.raises(ValueError, match="not positive definite") as err:
        sef_mod.check_positive_definite(F, names)
    assert "phiK" in str(err.value) and "Phi_phi0" in str(err.value)


def test_fisher_with_zero_curvature_is_rejected():
    with pytest.raises(ValueError, match="Gamma_ii <= 0.*qK"):
        sef_mod.check_positive_definite(np.diag([1.0, 0.0]), ["m1", "qK"])


def test_snr_mismatch_is_rejected():
    from emridispatch.fisher import FisherResult, check_snr

    sig = dict.fromkeys(
        ["mass_1", "mass_2", "a", "p", "e", "luminosity_distance"], 1.0)
    check_snr(FisherResult(sig, np.eye(6), list(sig), snr=31.0), 30.0)
    check_snr(FisherResult(sig, np.eye(6), list(sig)), 30.0)       # unknown: pass
    with pytest.raises(ValueError, match="378782"):
        check_snr(FisherResult(sig, np.eye(6), list(sig), snr=378782.4), 30.0)
