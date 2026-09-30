"""Generate the trace-driven benchmark: factual seasons, CRN counterfactuals,
warming replays, rendered canopy images and IoT streams.  Output: data/bench.npz"""
import numpy as np
import weather as W
import pathosim as P
from render import render

IMG_EVERY = 3
FIELDS = {"HYD": 20, "COR": 16, "TUN": 12}


def split_of(region, year):
    if region == "TUN":
        return "geo"
    if region == "COR":
        return "train" if year <= 2009 else ("cal" if year <= 2013 else "test")
    return "train" if year <= 2005 else ("cal" if year <= 2006 else "test")


def iot_streams(rng, o, w, F):
    n, T = o["S"].shape
    Tm = 0.5 * (w.Tmin.values + w.Tmax.values)
    tcan = Tm[None] + 0.3 * F["micro"][:, None] - 0.4 * o["lai"] + rng.normal(0, 0.5, (n, T))
    drift = F["sens_drift"][:, None] * np.arange(T)[None] * 24 / T * 5
    lwd_s = np.clip(o["lwd"] + drift + rng.normal(0, 1.5, (n, T)), 0, 24)
    smo = np.clip(o["sw"] + rng.normal(0, 0.05, (n, T)), 0, 1)
    miss = rng.random((n, T)) < 0.08
    return np.stack([tcan, lwd_s, smo], -1), miss


def main(seed=2026):
    rng = np.random.default_rng(seed)
    lwd_stats = W.calibrate_lwd()
    _, delta = W.cmip6_deltas()
    rec = {k: [] for k in ["region", "year", "split", "scen"]}
    arr = {k: [] for k in ["S", "Lat", "I", "R", "lai", "spray", "irr", "wx", "iot", "miss",
                           "static", "hidden", "img", "S_expert", "cf"]}

    def add(region, year, split, scen, w, F, Nn):
        o = P.simulate(w, F, Nn, region)
        cf = P.counterfactuals(w, F, Nn, region)
        n = len(F["skill"])
        iot, miss = iot_streams(rng, o, w, F)
        lwd_w = W.leaf_wetness(w.Tmin.values, w.Tmax.values, w.P.values)
        wx = np.stack([w.Tmin.values, w.Tmax.values, w.P.values, w.ETo.values, lwd_w], -1)
        chl = np.clip(0.15 * (2 - F["nlev"])[:, None] + 0.5 * (1 - o["ws"]), 0, 0.8)
        days = np.arange(0, P.SEASON_DAYS, IMG_EVERY)
        holes = np.exp(rng.normal(0.8, 0.6, n))
        imgs = np.zeros((n, len(days), 48, 48, 3), np.uint8)
        for i in range(n):
            for j, d in enumerate(days):
                imgs[i, j] = (255 * render(rng, o["lai"][i, d], o["I"][i, d], o["R"][i, d],
                                           chl[i, d], holes[i], region)).astype(np.uint8)
        s_exp = np.clip(o["S"][:, days] * np.exp(rng.normal(0, 0.15, (n, len(days))))
                        + rng.normal(0, 0.003, (n, len(days))), 0, 1)
        static = np.stack([F["soil"], F["nlev"], F["resistant"], F["irrigated"], F["sprinkler"]], -1)
        hidden = np.stack([F["skill"], F["log_inoc"], F["micro"], F["sens_drift"]], -1)
        cfa = np.stack([np.stack([cf[(d, a)] for a in P.ACTIONS], 1) for d in P.CF_DAYS], 1)
        for k, v in [("S", o["S"]), ("Lat", o["Lat"]), ("I", o["I"]), ("R", o["R"]),
                     ("lai", o["lai"]), ("spray", o["spray"]), ("irr", o["irr"]),
                     ("wx", np.repeat(wx[None], n, 0)), ("iot", iot), ("miss", miss),
                     ("static", static), ("hidden", hidden), ("img", imgs),
                     ("S_expert", s_exp), ("cf", cfa)]:
            arr[k].append(v.astype(np.float32) if k != "img" else v)
        for _ in range(n):
            rec["region"].append(region); rec["year"].append(year)
            rec["split"].append(split); rec["scen"].append(scen)

    for region in W.REGIONS:
        for year in W.REGIONS[region]["years"]:
            w = W.season_weather(region, year)
            if w is None:
                continue
            n = FIELDS[region]
            F = P.sample_fields(rng, n, region)
            Nn = P.season_noise(rng, n)
            sp = split_of(region, year)
            add(region, year, sp, "hist", w, F, Nn)
            if sp == "test":
                add(region, year, "warm", "ssp585", W.apply_delta(w, delta), F, Nn)
            print(region, year, sp, flush=True)
    out = {k: np.concatenate(v, 0) for k, v in arr.items()}
    for k, v in rec.items():
        out[k] = np.array(v)
    out["img_days"] = np.arange(0, P.SEASON_DAYS, IMG_EVERY)
    out["cf_days"] = np.array(P.CF_DAYS)
    np.savez_compressed("bench.npz", **out)
    import json
    json.dump(dict(lwd_calibration=lwd_stats, cmip6_delta=delta,
                   counts={s: int(np.sum(out["split"] == s)) for s in np.unique(out["split"])}),
              open("bench_meta.json", "w"), indent=1)
    print({s: int(np.sum(out["split"] == s)) for s in np.unique(out["split"])})


if __name__ == "__main__":
    main()
