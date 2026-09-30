"""Tensor builders for learned baselines (history window + future covariates)."""
import numpy as np
from common import HORIZONS, EPS
import pathosim as P

WIN = 60
HMAX = 21


def daily_past(b, idx, feat="mae"):
    wx = (b.wx[idx] - b.wx_mu) / b.wx_sd
    miss = b.miss[idx][..., None]
    iot = np.where(miss > 0.5, 0.0, (b.iot[idx] - b.iot_mu) / b.iot_sd)
    sprk = b.static[idx, 4][:, None] * b.irr[idx]
    T = b.T
    days = b.img_days
    j = np.clip(np.searchsorted(days, np.arange(T), side="right") - 1, 0, None)
    mu = b.hmu[idx] if feat == "mae" else b.cmu[idx]
    pca = (b.emb[idx] - b.emb_mu) @ b.pca.T / 3.0 if feat == "mae" else b.colorz[idx]
    today = np.isin(np.arange(T), days).astype(np.float32)
    since = (np.arange(T) - days[j]) / 10.0
    st = np.broadcast_to(b.static[idx][:, None], (len(idx), T, 5))
    f = np.concatenate([wx, iot, miss, b.spray[idx][..., None], sprk[..., None],
                        np.broadcast_to(today[None, :, None], (len(idx), T, 1)),
                        (mu[:, j] / 3.0)[..., None], pca[:, j],
                        np.broadcast_to(since[None, :, None], (len(idx), T, 1)),
                        np.broadcast_to((np.arange(T) / 140.0)[None, :, None], (len(idx), T, 1)),
                        st], -1)
    return f.astype(np.float32)


def windows(b, idx, sdays, mode="forecast", action=None, feat="mae"):
    """Returns past (B*O, WIN, Fp), mask (B*O, WIN), fut (B*O, HMAX, Ff)."""
    D = daily_past(b, idx, feat)
    B, O = len(idx), len(sdays)
    Fp = D.shape[-1]
    past = np.zeros((B, O, WIN, Fp), np.float32); mask = np.zeros((B, O, WIN), np.float32)
    for o, s in enumerate(sdays):
        lo = max(0, s - WIN + 1)
        seg = D[:, lo:s + 1]
        past[:, o, WIN - seg.shape[1]:] = seg
        mask[:, o, WIN - seg.shape[1]:] = 1
    if mode == "forecast":
        oi = [list(b.origins).index(int(s)) for s in sdays]
        wf = b.wxf[idx][:, oi, :HMAX]
    else:
        wf = np.stack([b.wx[idx, s + 1:s + 1 + HMAX] for s in sdays], 1)
    wf = (wf - b.wx_mu) / b.wx_sd
    spray = np.stack([b.spray[idx, s + 1:s + 1 + HMAX] for s in sdays], 1)
    sprk = np.stack([b.static[idx, 4][:, None] * b.irr[idx, s + 1:s + 1 + HMAX] for s in sdays], 1)
    if action is not None:
        spray = np.zeros_like(spray)
        if action in ("spray_now", "spray_twice"):
            spray[..., 0] = 1
        if action == "spray_twice":
            spray[..., 10] = 1
        if action == "drip_only":
            sprk = np.zeros_like(sprk)
    lead = np.broadcast_to((np.arange(1, HMAX + 1) / HMAX)[None, None, :, None], (B, O, HMAX, 1))
    tt = np.stack([(s + np.arange(1, HMAX + 1)) / 140.0 for s in sdays], 0)
    tt = np.broadcast_to(tt[None, ..., None], (B, O, HMAX, 1))
    st = np.broadcast_to(b.static[idx][:, None, None], (B, O, HMAX, 5))
    fut = np.concatenate([wf, spray[..., None], sprk[..., None], lead, tt, st], -1).astype(np.float32)
    return (past.reshape(B * O, WIN, Fp), mask.reshape(B * O, WIN), fut.reshape(B * O, HMAX, -1),
            spray.reshape(B * O, HMAX))


def targets(b, idx, sdays, steps=HORIZONS):
    """log-severity targets at s+h."""
    return np.stack([np.stack([np.log(b.S[idx, s + h] + EPS) for h in steps], -1) for s in sdays], 1)


def tab_features(b, idx, sdays, mode="forecast", action=None, feat="mae"):
    """Engineered features for GBM / static MLP (B, O, F)."""
    mu = b.hmu[idx] if feat == "mae" else b.cmu[idx]
    days = b.img_days
    Tm = 0.5 * (b.wx[idx, :, 0] + b.wx[idx, :, 1])
    lwdw = b.wx[idx, :, 4]
    iotw = np.where(b.miss[idx] > 0.5, lwdw, b.iot[idx, :, 1])
    rows = []
    for o, s in enumerate(sdays):
        j = np.searchsorted(days, s, side="right") - 1
        cur = mu[:, j]; prev = mu[:, max(j - 1, 0)]; prev3 = mu[:, max(j - 3, 0)]
        lo = max(0, s - 13)
        wetwarm = ((iotw[:, lo:s + 1] > 8) & (Tm[:, lo:s + 1] > 20)).sum(1)
        if mode == "forecast":
            wf = b.wxf[idx][:, o, :HMAX]
        else:
            wf = b.wx[idx, s + 1:s + 1 + HMAX]
        fT = 0.5 * (wf[..., 0] + wf[..., 1])
        fww = ((wf[..., 4] > 8) & (fT > 20)).sum(1)
        spray = b.spray[idx, s + 1:s + 1 + HMAX].copy()
        sprk = b.static[idx, 4][:, None] * b.irr[idx, s + 1:s + 1 + HMAX]
        if action is not None:
            spray[:] = 0
            if action in ("spray_now", "spray_twice"):
                spray[:, 0] = 1
            if action == "spray_twice":
                spray[:, 10] = 1
            if action == "drip_only":
                sprk = np.zeros_like(sprk)
        past_spray = b.spray[idx, :s + 1]
        last = np.where(past_spray.any(1), s - (s - np.argmax(past_spray[:, ::-1], 1)), 99)
        dsl = np.where(past_spray.any(1), np.argmax(past_spray[:, ::-1], 1), 60)
        rows.append(np.stack([cur, cur - prev, cur - prev3, wetwarm, Tm[:, lo:s + 1].mean(1),
                              np.log1p(b.wx[idx, lo:s + 1, 2]).sum(1), fww, fT.mean(1),
                              np.log1p(wf[..., 2]).sum(1), spray[:, :7].sum(1), spray[:, 7:14].sum(1),
                              spray[:, 14:].sum(1), sprk.sum(1), dsl, past_spray.sum(1),
                              np.full(len(idx), s)] + [b.static[idx, k] for k in range(5)], -1))
    return np.stack(rows, 1).astype(np.float32)
