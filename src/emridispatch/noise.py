import numpy as np

NOISE_MODEL = "scirdv1"

MIN_FREQ = 1.0e-5

DEFAULT_TDI_CHANNELS = ["A", "E"]

DIRECT_CHANNELS = ["I", "II"]

# TDI-2 PSD nulls (f = k c / 4L): bins within PSD_NOTCH Hz of a run lying
# PSD_NOTCH_DEPTH decades below the local median are dropped from inner
# products. The likelihood and the SEF Fisher share this definition; without
# it the null bins dominate <h|h> (SEF reported SNR 3.8e5 for an SNR-30 source).
PSD_NOTCH = 1.0e-5
PSD_NOTCH_DEPTH = 2.0


def load_sensitivity_table():
    from lisatools.sensitivity import (
        LISASens,
        A1TDISens, E1TDISens, T1TDISens,
        A2TDISens, E2TDISens, T2TDISens,
    )

    return {
        "off": {"I": LISASens, "II": LISASens},
        "1st generation": {"A": A1TDISens, "E": E1TDISens, "T": T1TDISens},
        "2nd generation": {"A": A2TDISens, "E": E2TDISens, "T": T2TDISens},
    }


def sensitivity_spec(tdi, channels=None, sens_table=None):
    if sens_table is None:
        sens_table = load_sensitivity_table()

    gen_sens = sens_table.get(tdi)
    if gen_sens is None:
        raise ValueError(
            f"unknown TDI generation {tdi!r}; choose from "
            f"{sorted(k for k in sens_table if k != 'off')}")

    if tdi == "off":
        names = list(DIRECT_CHANNELS)
    else:
        names = list(DEFAULT_TDI_CHANNELS if channels is None else channels)

    try:
        sens = [gen_sens[c] for c in names]
    except KeyError as err:
        raise ValueError(
            f"unknown TDI channel {err.args[0]!r}; choose from "
            f"{sorted(gen_sens)}") from err
    return sens, names


def noise_sens_kwargs(duration, foreground):
    kwargs = {"model": NOISE_MODEL}
    if foreground:
        from lisatools.utils.constants import YRSID_SI

        kwargs["stochastic_params"] = (duration * YRSID_SI,)
    return kwargs


def psd_null_mask(f, S, width, depth=PSD_NOTCH_DEPTH, block=4096):
    """(mask, centres) of the bins within width Hz of a PSD null, or (None, []).

    A null is a run of bins where some row of S (channels x freqs) lies more
    than depth decades below the median of its block of bins; a run starting
    at bin 0 is the band edge, not a null.
    """
    if width <= 0.0:
        return None, []
    f = np.asarray(f.get() if hasattr(f, "get") else f)
    S = np.asarray(S.get() if hasattr(S, "get") else S)
    S = S.reshape(-1, S.shape[-1])
    nf = S.shape[-1]
    nblk = int(np.ceil(nf / block))
    pad = nblk * block - nf
    bad = np.zeros(nf, dtype=bool)
    with np.errstate(all="ignore"):
        for row in S:
            v = np.log10(np.where(np.isfinite(row) & (row > 0), row, np.nan))
            vp = np.concatenate([v, np.full(pad, np.nan)]).reshape(nblk, block)
            med = np.nanmedian(vp, axis=1)[:, None]
            bad |= (vp < med - depth).reshape(-1)[:nf]
    if not bad.any():
        return None, []
    idx = np.flatnonzero(bad)
    groups = [g for g in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
              if g[0] != 0]
    if not groups:
        return None, []
    mask = np.zeros(nf, dtype=bool)
    for g in groups:
        a = np.searchsorted(f, f[g[0]] - width, side="left")
        b = np.searchsorted(f, f[g[-1]] + width, side="right")
        mask[a:b] = True
    return mask, [float(f[g].mean()) for g in groups]


def channel_noise_psd(f, sens_cls, notch=0.0, notch_depth=PSD_NOTCH_DEPTH, **kwargs):
    """One channel's PSD; notch > 0 sets the bins near its nulls to +inf."""
    S = sens_cls.get_Sn(f, **kwargs)
    if notch > 0.0:
        mask, _ = psd_null_mask(f, S, notch, notch_depth)
        if mask is not None:
            S = np.where(mask, np.inf, S)
    return S


def per_channel_noise_kwargs(duration, foreground, sens_classes, notch=0.0,
                             notch_depth=PSD_NOTCH_DEPTH):
    shared = noise_sens_kwargs(duration, foreground)
    if notch > 0.0:
        shared.update(notch=notch, notch_depth=notch_depth)
    return [dict(shared, sens_cls=cls) for cls in sens_classes]
