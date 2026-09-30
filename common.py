"""Shared data access, forecast-origin construction and metrics."""
import numpy as np
from weather import leaf_wetness

HORIZONS = (7, 14, 21)
ORIGIN_MIN, ORIGIN_MAX = 21, 117
ONSET = 0.05
EPS = 0.005
COST = {"no_spray": 0.0, "spray_now": 0.01, "spray_twice": 0.02, "drip_only": 0.004}


def z(s):
    return np.log(np.asarray(s) + EPS)


def unz(zv):
    return np.exp(zv) - EPS


class Bench:
    def __init__(self, path="bench.npz", emb="emb.npz"):
        Z = np.load(path)
        self.__dict__.update({k: Z[k] for k in Z.files})
        E = np.load(emb)
        self.emb = E["emb"]; self.hmu = E["mae_mu"]; self.hls = E["mae_ls"]
        self.cmu = E["color_mu"]; self.cls_ = E["color_ls"]
        C = E["color"]; cm = C[self.split == "train"].reshape(-1, C.shape[-1])
        self.colorz = ((C - cm.mean(0)) / (cm.std(0) + 1e-6))[..., :8].astype(np.float32)
        self.N, self.T = self.S.shape
        self.origins = np.array([d for d in self.img_days if ORIGIN_MIN <= d <= ORIGIN_MAX])
        self.oidx = np.array([list(self.img_days).index(d) for d in self.origins])
        tr = self.split == "train"
        e = self.emb[tr].reshape(-1, self.emb.shape[-1])
        self.emb_mu = e.mean(0)
        C = np.cov((e - self.emb_mu).T) + 1e-4 * np.eye(e.shape[1])
        self.emb_prec = np.linalg.inv(C)
        U, s_, Vt = np.linalg.svd(e - self.emb_mu, full_matrices=False)
        self.pca = Vt[:8]
        self.wx_mu = self.wx[tr].reshape(-1, 5).mean(0); self.wx_sd = self.wx[tr].reshape(-1, 5).std(0)
        self.iot_mu = self.iot[tr].reshape(-1, 3).mean(0); self.iot_sd = self.iot[tr].reshape(-1, 3).std(0)
        self.wxf = self._forecasts()

    def idx(self, split):
        return np.where(self.split == split)[0]

    def _forecasts(self, seed=77):
        """Forecast weather issued at each origin for the next 21 days (lead 1..21).
        Error grows with lead; identical for every model."""
        rng = np.random.default_rng(seed)
        N, O, Hm = self.N, len(self.origins), max(HORIZONS)
        out = np.zeros((N, O, Hm, 5), np.float32)
        # one forecast per region-season (fields share station weather)
        key = {}
        for n in range(N):
            k = (self.region[n], int(self.year[n]), self.scen[n])
            if k not in key:
                F = np.zeros((O, Hm, 5), np.float32)
                for o, d in enumerate(self.origins):
                    w = self.wx[n, d + 1:d + 1 + Hm]
                    lead = np.arange(1, Hm + 1)
                    tmin = w[:, 0] + rng.normal(0, 0.6 + 0.08 * lead)
                    tmax = np.maximum(w[:, 1] + rng.normal(0, 0.6 + 0.08 * lead), tmin + 0.5)
                    p = w[:, 2] * np.exp(rng.normal(0, 0.5, Hm))
                    flip = rng.random(Hm) < 0.02 + 0.01 * lead
                    p = np.where(flip, np.where(p > 0.5, 0.0, rng.exponential(6, Hm)), p)
                    eto = w[:, 3] * np.exp(rng.normal(0, 0.1, Hm))
                    F[o] = np.stack([tmin, tmax, p, eto, leaf_wetness(tmin, tmax, p)], -1)
                key[k] = F
            out[n] = key[k]
        return out

    def ood_score(self, n, j):
        e = self.emb[n, j] - self.emb_mu
        return np.einsum("...i,ij,...j->...", e, self.emb_prec, e)


# ------------------------------------------------------------------ metrics
def mae_pp(pred, true):
    return float(100 * np.mean(np.abs(pred - true)))


def rmse_log(pz, tz):
    return float(np.sqrt(np.mean((pz - tz) ** 2)))


def auroc(score, y):
    y = np.asarray(y).astype(bool); score = np.asarray(score)
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # ties
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks); ranks = sums[inv] / cnt[inv]
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def auprc(score, y):
    y = np.asarray(y).astype(bool)
    o = np.argsort(-np.asarray(score)); yy = y[o]
    tp = np.cumsum(yy); prec = tp / np.arange(1, len(yy) + 1)
    return float(np.sum(prec[yy]) / max(yy.sum(), 1))


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(p, y, bins=10):
    p = np.asarray(p); y = np.asarray(y).astype(float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
