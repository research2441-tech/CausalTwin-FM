"""Aggregate metrics for all models/seeds: forecasting, onset warning,
counterfactual effects, conformal calibration, statistics."""
import glob, json, os
import numpy as np
from scipy import stats
from common import (Bench, HORIZONS, EPS, ONSET, COST, z, unz, mae_pp, auroc, auprc, brier, ece, spearman)
import conformal as CP
import pathosim as P

EVAL = ["test", "warm", "geo"]
MAIN = ["twin", "abl_pf", "process_da", "persistence", "gbm", "static_mlp", "gru", "transformer", "crn"]
ALPHA = 0.1


def ztrue(b, idx):
    return np.stack([np.stack([np.log(b.S[idx, s + h] + EPS) for h in HORIZONS], -1) for s in b.origins], 1)


def load(name, seed):
    f = f"pred_{name}_s{seed}.npz"
    return dict(np.load(f)) if os.path.exists(f) else None


def onset_prob(mu, sd):
    return 1 - stats.norm.cdf((np.log(ONSET + EPS) - mu) / sd)


def forecast_metrics(b, sp, R):
    idx = b.idx(sp); zt = ztrue(b, idx); st = unz(zt)
    mu, sd = R[f"{sp}_mu"], R[f"{sp}_sd"]
    m = {}
    for h, H in enumerate(HORIZONS):
        m[f"mae_{H}"] = mae_pp(unz(mu[..., h]), st[..., h])
        m[f"rmsez_{H}"] = float(np.sqrt(np.mean((mu[..., h] - zt[..., h]) ** 2)))
        zz = (zt[..., h] - mu[..., h]) / sd[..., h]
        m[f"crps_{H}"] = float(np.mean(sd[..., h] * (zz * (2 * stats.norm.cdf(zz) - 1) + 2 * stats.norm.pdf(zz)
                                                       - 1 / np.sqrt(np.pi))))
        m[f"cov_raw_{H}"] = float(np.mean(np.abs(zz) < stats.norm.ppf(1 - ALPHA / 2)))
    # onset within 14 days among not-yet-onset
    cur = b.S[idx][:, b.origins]
    elig = cur < ONSET
    y = (st[..., 1] >= ONSET)[elig]
    p = onset_prob(mu[..., 1], sd[..., 1])[elig]
    m.update(onset_auroc=auroc(p, y), onset_auprc=auprc(p, y), onset_brier=brier(p, y), onset_ece=ece(p, y),
             onset_prev=float(y.mean()))
    m["per_field_mae"] = np.mean(np.abs(unz(mu) - st), (1, 2)) * 100
    return m


def lead_time(b, sp, R, Rcal):
    cal = b.idx("cal"); zc = ztrue(b, cal)
    pc = onset_prob(Rcal["cal_mu"][..., 1], Rcal["cal_sd"][..., 1])
    neg = (b.S[cal][:, b.origins] < ONSET) & (unz(zc[..., 1]) < ONSET)
    thr = np.quantile(pc[neg], 0.9)
    idx = b.idx(sp)
    p = onset_prob(R[f"{sp}_mu"][..., 1], R[f"{sp}_sd"][..., 1])
    leads = []
    for i, n in enumerate(idx):
        on = np.where(b.S[n] >= ONSET)[0]
        if len(on) == 0 or on[0] <= b.origins[0] + 7:
            continue
        D = on[0]
        al = [s for o, s in enumerate(b.origins) if s < D and p[i, o] >= thr]
        leads.append(D - al[0] if al else 0)
    leads = np.array(leads)
    return dict(lead_mean=float(leads.mean()), lead_median=float(np.median(leads)),
                det7=float(np.mean(leads >= 7)), n_onset=int(len(leads)))


def cf_metrics(b, sp, R):
    idx = b.idx(sp)
    tru = b.cf[idx][..., 2]             # (B,3,4) severity at d+20
    pr = R[f"{sp}_cf"]
    et = tru[..., 1:] - tru[..., :1]; ep = pr[..., 1:] - pr[..., :1]
    cost = np.array([COST[a] for a in P.ACTIONS])
    ch = np.argmin(pr + cost, -1)
    val = np.take_along_axis(tru + cost, ch[..., None], -1)[..., 0]
    regret = val - (tru + cost).min(-1)
    big = np.abs(et) > 0.002
    rho = np.nanmean([spearman(ep[:, d, 0], et[:, d, 0]) for d in range(et.shape[1])])
    return dict(eff_rmse=float(100 * np.sqrt(np.mean((et - ep) ** 2))),
                eff_rmse_by_action=(100 * np.sqrt(np.mean((et - ep) ** 2, (0, 1)))).tolist(),
                sign_acc=float(np.mean(np.sign(et[big]) == np.sign(ep[big]))),
                spearman=float(rho), regret=float(100 * regret.mean()),
                eff_bias=(100 * (ep - et).mean((0, 1))).tolist(),
                per_field_cf=100 * np.sqrt(np.mean((et - ep) ** 2, (1, 2))),
                per_field_regret=100 * regret.mean(1))


def _aci(b, idx, O, H, err, qfun, region_level, alpha=ALPHA, gamma=0.02):
    """Adaptive level tracking. Streams are region-seasons; with region_level the level is
    carried chronologically from one season to the next within a region."""
    keys = np.array([f"{b.region[n]}{int(b.year[n]):04d}" for n in idx])
    Q = np.zeros(err.shape)
    state = {}
    for k in sorted(np.unique(keys)):
        ids = np.where(keys == k)[0]
        rk = k[:3] if region_level else k
        a_t = state.get(rk, alpha)
        for o, s in enumerate(O):
            Q[ids, o] = qfun(np.clip(a_t, 0.005, 0.5), ids, o)
            nxt = O[o + 1] if o + 1 < len(O) else 10 ** 6
            mat = [oo for oo, ss in enumerate(O) if s < ss + H <= nxt]
            if mat:
                e = np.mean([np.mean(err[ids, oo] > Q[ids, oo]) for oo in mat])
                a_t = a_t + gamma * (alpha - e)
        state[rk] = a_t
    return Q


def conformal_metrics(b, R, bw=0.5):
    cal = b.idx("cal"); zc = ztrue(b, cal)
    sc = np.abs(zc - R["cal_mu"]) / R["cal_sd"]
    ec = CP.exposure(b, cal, b.origins)
    O = list(b.origins)
    out = {}
    for sp in EVAL:
        idx = b.idx(sp); zt = ztrue(b, idx)
        mu, sd = R[f"{sp}_mu"], R[f"{sp}_sd"]
        et = CP.exposure(b, idx, b.origins)
        ood = b.ood_score(idx[:, None], b.oidx[None, :])
        ood_c = b.ood_score(cal[:, None], b.oidx[None, :])
        res = {"ood_flag_rate": float(np.mean(ood > np.quantile(ood_c, 0.99)))}
        tb = np.digitize(et, np.quantile(ec, [1 / 3, 2 / 3]))
        for h in range(3):
            s_cal = sc[..., h].ravel()
            fc = np.stack([ec.ravel(), R["cal_mu"][..., h].ravel()], -1)
            m_, s_ = fc.mean(0), fc.std(0)
            fc = (fc - m_) / s_
            ft = ((np.stack([et, mu[..., h]], -1) - m_) / s_)
            q_loc, ess = CP.localized(s_cal, fc, ft.reshape(-1, 2), ALPHA, bw=bw)
            q_loc = q_loc.reshape(mu.shape[:2]); ess = ess.reshape(mu.shape[:2])
            err = np.abs(zt[..., h] - mu[..., h]) / sd[..., h]
            abst_ess = ess < 40
            qloc_fn = lambda a, ids, o: CP.localized(s_cal, fc, ft[ids, o], a, bw=bw)[0]
            variants = [("split", np.full(mu.shape[:2], CP.split_q(s_cal, ALPHA)), np.zeros(mu.shape[:2], bool)),
                        ("local", q_loc, abst_ess),
                        ("aci_season", _aci(b, idx, O, HORIZONS[h], err, qloc_fn, False), abst_ess),
                        ("aci_region", _aci(b, idx, O, HORIZONS[h], err, qloc_fn, True), abst_ess),
                        ("split_aci_region", _aci(b, idx, O, HORIZONS[h], err,
                                                  lambda a, ids, o: np.full(len(ids), CP.split_q(s_cal, a)), True),
                         np.zeros(mu.shape[:2], bool))]
            for nm, q, abst in variants:
                acc = ~abst & np.isfinite(q)
                cov = err[acc] <= q[acc]
                lo = unz(mu[..., h] - q * sd[..., h]); hi = unz(mu[..., h] + q * sd[..., h])
                wid = 100 * (hi - lo)
                terc = [float(np.mean((err <= q)[acc & (tb == k)])) if (acc & (tb == k)).any() else np.nan
                        for k in range(3)]
                yrs = b.year[idx]; regs = b.region[idx]
                first = np.array([yrs[i] == yrs[regs == regs[i]].min() for i in range(len(idx))])
                cov_first = float(np.mean((err <= q)[acc & first[:, None]]))
                res[f"{nm}_{HORIZONS[h]}"] = dict(cov=float(cov.mean()), abst=float(abst.mean()), cov_first=cov_first,
                                                  width_pp=float(np.median(wid[acc])), cov_terc=terc,
                                                  worst_terc=float(np.nanmin(terc)))
        out[sp] = res
    return out


def main(models, seeds=(0, 1, 2), conformal_models=None):
    b = Bench()
    summary = {}
    perfield = {}
    for mname in models:
        base, sds = ("twin", (0, 1)) if mname == "twin_s01" else (mname, seeds)
        runs = [(s, load(base, s)) for s in sds]
        runs = [(s, r) for s, r in runs if r is not None]
        if not runs:
            continue
        agg = {}
        for s, R in runs:
            for sp in EVAL:
                fm = forecast_metrics(b, sp, R)
                perfield.setdefault((mname, sp, "mae"), []).append(fm.pop("per_field_mae"))
                fm.update(lead_time(b, sp, R, R))
                if f"{sp}_cf" in R:
                    cm = cf_metrics(b, sp, R)
                    perfield.setdefault((mname, sp, "cf"), []).append(cm.pop("per_field_cf"))
                    perfield.setdefault((mname, sp, "regret"), []).append(cm.pop("per_field_regret"))
                    fm.update({"cf_" + k: v for k, v in cm.items()})
                for k, v in fm.items():
                    agg.setdefault(sp, {}).setdefault(k, []).append(v)
        summ = {}
        for sp, d in agg.items():
            summ[sp] = {k: dict(mean=np.nanmean(np.array(v, dtype=float), 0).tolist(),
                                sd=np.nanstd(np.array(v, dtype=float), 0).tolist(), n=len(v)) for k, v in d.items()}
        if conformal_models is None or mname in conformal_models:
            summ["conformal"] = conformal_metrics(b, runs[0][1])
        summary[mname] = summ
        print(mname, "done", flush=True)
    # paired statistics vs twin (per-field, averaged over seeds)
    st = {}
    for sp in EVAL:
        for kind in ["mae", "cf", "regret"]:
            if ("twin", sp, kind) not in perfield:
                continue
            ref = np.mean(perfield[("twin", sp, kind)], 0)
            rows = []
            for mname in models:
                if mname not in MAIN or mname == "twin" or (mname, sp, kind) not in perfield:
                    continue
                oth = np.mean(perfield[(mname, sp, kind)], 0)
                d = oth - ref
                w = stats.wilcoxon(oth, ref) if np.any(d != 0) else None
                rng = np.random.default_rng(0)
                bs = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]
                r_rb = float((np.sum(d > 0) - np.sum(d < 0)) / max(np.sum(d != 0), 1))
                rows.append(dict(model=mname, mean_diff=float(d.mean()), ci=[float(np.quantile(bs, .025)),
                                                                                float(np.quantile(bs, .975))],
                                 p=float(w.pvalue) if w else 1.0, rank_biserial=r_rb))
            ps = np.array([r["p"] for r in rows]); order = np.argsort(ps); m = len(ps)
            adj = np.zeros(m); run = 0
            for k, i in enumerate(order):
                run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
            for r, a in zip(rows, adj):
                r["p_holm"] = float(a)
            st[f"{sp}_{kind}"] = rows
    # Friedman on test MAE across models with all seeds
    ms = [m for m in models if m in MAIN and (m, "test", "mae") in perfield]
    mat = np.stack([np.mean(perfield[(m, "test", "mae")], 0) for m in ms], 1)
    fr = stats.friedmanchisquare(*mat.T)
    ranks = np.mean(np.argsort(np.argsort(mat, 1), 1) + 1, 0)
    st["friedman_test_mae"] = dict(stat=float(fr.statistic), p=float(fr.pvalue), models=ms, mean_ranks=ranks.tolist())
    summary["_stats"] = st
    json.dump(summary, open("summary.json", "w"), indent=1, default=float)
    np.save("perfield.npy", {str(k): v for k, v in perfield.items()}, allow_pickle=True)
    return summary


if __name__ == "__main__":
    import sys
    models = ["twin", "process_da", "persistence", "gbm", "static_mlp", "gru", "transformer", "crn",
              "abl_no_attn", "abl_calendar", "abl_no_ports", "abl_joint", "abl_pf", "abl_pf_no_obsop",
              "abl_no_images", "abl_color", "abl_no_iot"]
    main(models, conformal_models=["twin", "gru", "transformer", "gbm", "crn", "process_da"])
