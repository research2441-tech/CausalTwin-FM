"""Conformal layers: split, exposure-localised weighted, and exposure-localised
with adaptive (ACI) level tracking per region-season stream, plus referral."""
import numpy as np
from pathosim import beta_T


def exposure(b, idx, sdays, H=21, mode="forecast"):
    """Model-agnostic climate-exposure index over the forecast window: mean
    generic infection efficiency (literature-default cardinal temperatures and
    wetness requirement) of the forecast weather, sprinkler wetness included."""
    out = np.zeros((len(idx), len(sdays)))
    for o, s in enumerate(sdays):
        if mode == "forecast":
            wf = b.wxf[idx][:, o, :H]
        else:
            wf = b.wx[idx, s + 1:s + 1 + H]
        T = 0.5 * (wf[..., 0] + wf[..., 1])
        sprk = b.static[idx, 4][:, None] * b.irr[idx, s + 1:s + 1 + H]
        W = wf[..., 4] + 4.0 * sprk
        fT = beta_T(T, 10.0, 25.0, 35.0)
        wreq = np.minimum(8.0 / np.maximum(fT, 1e-3), 48)
        out[:, o] = (fT / (1 + np.exp(-(W - wreq) / 2.0))).mean(1)
    return out


def wquantile(scores, w, level):
    o = np.argsort(scores); s = scores[o]; ww = w[o]
    c = np.cumsum(ww) / (ww.sum() + 1.0)   # +1: test-point mass at +inf
    k = np.searchsorted(c, level)
    return s[k] if k < len(s) else np.inf


def split_q(cal_scores, alpha):
    n = len(cal_scores)
    return np.quantile(cal_scores, min(1.0, np.ceil((1 - alpha) * (n + 1)) / n), method="higher")


def localized(cal_scores, cal_phi, test_phi, alpha, bw=0.5, ess_min=40):
    """Kernel-weighted conformal quantile per test point; returns q, ess."""
    q = np.empty(len(test_phi)); ess = np.empty(len(test_phi))
    for i, f in enumerate(test_phi):
        d2 = np.sum((cal_phi - f) ** 2, 1)
        w = np.exp(-0.5 * d2 / bw ** 2)
        ess[i] = w.sum() ** 2 / max((w ** 2).sum(), 1e-12)
        q[i] = wquantile(cal_scores, w, 1 - alpha)
    return q, ess


def aci_stream(q_fn, scores_by_time, alpha, gamma=0.02):
    """Generic ACI: q_fn(alpha_t, t) -> per-item quantiles for time t;
    scores_by_time: list of (t_mature_list) handled by caller. Kept for clarity."""
    raise NotImplementedError


def run_aci(order_keys, origins, horizons, mu, sd, ztrue, qbase_fn, alpha, gamma=0.02):
    """ACI per stream. order_keys: (N,) stream id per field; origins: origin days;
    qbase_fn(alpha_level, n_idx, o, h) -> quantile multipliers for those items.
    Returns covered (N,O,H) and q (N,O,H)."""
    N, O, Hn = mu.shape
    Q = np.zeros((N, O, Hn))
    for key in np.unique(order_keys):
        ids = np.where(order_keys == key)[0]
        a_t = np.full(Hn, alpha)
        for o, s in enumerate(origins):
            for h in range(Hn):
                Q[ids, o, h] = qbase_fn(np.clip(a_t[h], 1e-3, 0.5), ids, o, h)
            # feedback: forecasts that have matured by the next origin
            nxt = origins[o + 1] if o + 1 < O else 10 ** 6
            for h, hd in enumerate(horizons):
                mat = [oo for oo, ss in enumerate(origins) if s < ss + hd <= nxt]
                if mat:
                    err = np.mean([np.mean(np.abs(ztrue[ids, oo, h] - mu[ids, oo, h]) >
                                           Q[ids, oo, h] * sd[ids, oo, h]) for oo in mat])
                    a_t[h] = a_t[h] + gamma * (alpha - err)
    return Q
