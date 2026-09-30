"""Train/evaluate CausalTwin-FM, its ablations and the process-model DA baseline.
usage: python exp_twin.py SEED [config ...]"""
import sys, time, pickle
import numpy as np
import jax
from common import Bench, HORIZONS, EPS
import twin as TW
import twin_amort as TA
import run_twin as RT

CONFIGS = {
    "twin": {},
    "abl_no_attn": dict(attn="none"),
    "abl_calendar": dict(attn="calendar"),
    "abl_no_ports": dict(ports=False),
    "abl_joint": dict(joint=True),
    "abl_pf": dict(abduction="pf"),
    "abl_pf_no_obsop": dict(abduction="pf", obsop=False),
    "abl_no_images": dict(images=False),
    "abl_color": dict(feat="color"),
    "abl_no_iot": dict(iot=False),
    "process_da": dict(attn="none", abduction="pf", untrained=True),
}
SPLITS = ["cal", "test", "warm", "geo"]


def ztrue(b, idx):
    return np.stack([np.stack([np.log(b.S[idx, s + h] + EPS) for h in HORIZONS], -1) for s in b.origins], 1)


def fit_sig(zt, mu, raw):
    grid = np.linspace(0.02, 1.5, 150); sig, nll = [], 0.0
    for h in range(3):
        e2 = (zt[..., h] - mu[..., h]) ** 2; r2 = raw[..., h] ** 2
        L = [np.mean(np.log(r2 + g * g) + e2 / (r2 + g * g)) for g in grid]
        k = int(np.argmin(L)); sig.append(float(grid[k])); nll += L[k]
    return sig, nll


def main(seed, names):
    b = Bench()
    tr = b.idx("train"); cal = b.idx("cal")
    for name in names:
        t0 = time.time()
        cfg = dict(TW.DEFAULT); extra = dict(CONFIGS[name])
        untrained = extra.pop("untrained", False); joint = extra.pop("joint", False)
        abduction = extra.pop("abduction", "amortized")
        cfg.update(extra)
        if untrained:
            p = TW.init_params(jax.random.PRNGKey(seed), cfg); hist = []
            lat = dict(theta=np.zeros(len(tr)), iota=np.zeros(len(tr)))
        else:
            p, hist, lat = TW.train(b, cfg, tr, steps=1500, seed=seed)
        p = jax.tree_util.tree_map(np.asarray, p)
        out = {}
        q = None
        if abduction == "amortized":
            if joint:
                q, h2, p = TA.train_encoder(b, cfg, p, tr, seed=seed, joint=True)
            else:
                q, h2 = TA.train_encoder(b, cfg, p, tr, seed=seed)
            q = jax.tree_util.tree_map(np.asarray, q)
            best = None
            for sxi in [0.1, 0.2, 0.35]:
                mu, sd, raw = TA.forecast(b, cfg, p, q, cal, (0, 0, 0), sxi=sxi, seed=seed)
                sig, nll = fit_sig(ztrue(b, cal), mu, raw)
                if best is None or nll < best[0]:
                    best = (nll, sxi, sig)
            _, sxi, sig = best
            fc = lambda idx: TA.forecast(b, cfg, p, q, idx, sig, sxi=sxi, seed=seed)
            cff = lambda idx: TA.counterfactual(b, cfg, p, q, idx, sxi=sxi, seed=seed)
        else:
            if untrained:
                pri = None
                prior_fn = lambda idx: None
            else:
                prior_fn = lambda idx: TW.region_prior(b, tr, lat, idx)
            best = None
            for sxi in [0.15, 0.25, 0.4]:
                mu, sd, raw = RT.forecast(b, cfg, p, cal, sxi=sxi, sig_h=(0, 0, 0), seed=seed, prior=prior_fn(cal))
                sig, nll = fit_sig(ztrue(b, cal), mu, raw)
                if best is None or nll < best[0]:
                    best = (nll, sxi, sig)
            _, sxi, sig = best
            fc = lambda idx: RT.forecast(b, cfg, p, idx, sxi=sxi, sig_h=sig, seed=seed, prior=prior_fn(idx))
            cff = lambda idx: RT.counterfactual(b, cfg, p, idx, sxi=sxi, seed=seed, prior=prior_fn(idx))[..., 2]
        out.update(sxi=sxi, sig=np.array(sig))
        for sp in SPLITS:
            idx = b.idx(sp)
            mu, sd, raw = fc(idx)
            out[f"{sp}_mu"] = mu; out[f"{sp}_sd"] = sd; out[f"{sp}_raw"] = raw
            if sp != "cal":
                out[f"{sp}_cf"] = cff(idx)
        np.savez_compressed(f"pred_{name}_s{seed}.npz", **out)
        pickle.dump(dict(params=p, q=q, hist=hist, cfg=cfg, lat=jax.tree_util.tree_map(np.asarray, lat)),
                    open(f"params_{name}_s{seed}.pkl", "wb"))
        print(name, seed, "done %.0fs" % (time.time() - t0), "sxi", sxi, "sig", np.round(sig, 3), flush=True)


if __name__ == "__main__":
    seed = int(sys.argv[1])
    names = sys.argv[2:] or list(CONFIGS)
    main(seed, names)
