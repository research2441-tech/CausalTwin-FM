"""Trace-driven cotton foliar-disease testbed (target-spot-like pathosystem).

Ground-truth generator, NOT the proposed model.  It deliberately contains
mechanisms that the proposed twin does not encode (three-stage Erlang latency,
rain-splash amplification, temperature-dependent infectious period, curative
kill, rain wash-off of residue, canopy-closure wetness) so that the learned
twin is misspecified in a controlled way.

Everything is vectorised over the fields that share one region-season weather
trace.  Common random numbers make counterfactual replays exact.
"""
import numpy as np
from weather import leaf_wetness, SEASON_DAYS

T_BASE_COTTON = 15.6
REGION_PRIOR = {"HYD": dict(mu_inoc=-7.8, irrig_p=0.45, spray_bias=0.5),
                "COR": dict(mu_inoc=-5.0, irrig_p=0.70, spray_bias=0.0),
                "TUN": dict(mu_inoc=-6.0, irrig_p=1.00, spray_bias=-0.3)}
ACTIONS = ["no_spray", "spray_now", "spray_twice", "drip_only"]
CF_DAYS = (45, 65, 85)
CF_WIN = 21
ONSET = 0.05
BETA0 = 0.36


def beta_T(T, tmin=12.0, topt=27.0, tmax=35.0):
    T = np.clip(T, tmin + 1e-3, tmax - 1e-3)
    a = (topt - tmin) / (tmax - topt)
    return ((tmax - T) / (tmax - topt)) * ((T - tmin) / (topt - tmin)) ** a


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sample_fields(rng, n, region):
    pr = REGION_PRIOR[region]
    skill = rng.normal(0, 1, n)
    soil = rng.normal(0, 1, n)                        # observed soil index
    awc = 130 + 30 * soil                              # mm
    irrigated = rng.random(n) < pr["irrig_p"]
    sprinkler = irrigated & (rng.random(n) < sigmoid(0.6 - 1.0 * skill))
    nlev = np.digitize(0.3 - 0.8 * skill + 0.5 * soil + rng.normal(0, 0.8, n), [-0.6, 0.6])
    resistant = rng.random(n) < sigmoid(-0.3 + 0.7 * skill)
    log_inoc = pr["mu_inoc"] - 0.5 * skill + rng.normal(0, 0.6, n)
    micro = rng.normal(0, 1.5, n) - 0.4 * skill       # canopy humidity offset (h)
    sens_drift = (rng.random(n) < 0.2) * rng.normal(0, 0.04, n)
    return dict(skill=skill, soil=soil, awc=awc, irrigated=irrigated,
                sprinkler=sprinkler, nlev=nlev, resistant=resistant,
                log_inoc=log_inoc, micro=micro, sens_drift=sens_drift)


def season_noise(rng, n, T=SEASON_DAYS):
    return dict(xi=rng.normal(0, 0.25, (n, T)), scout=rng.normal(0, 1, (n, T)),
                u=rng.random((n, T)))


def simulate(w, F, N, region, override=None):
    """w: weather DataFrame (T rows). F: field dict. N: noise dict.
    override: None (behavioural policy) or dict(day=d, action=a).
    Returns dict of daily trajectories."""
    T = len(w)
    n = len(F["skill"])
    tmin, tmax, P, ETo = (w[c].values for c in ["Tmin", "Tmax", "P", "ETo"])
    Tm = 0.5 * (tmin + tmax)
    lwd_w = leaf_wetness(tmin, tmax, P)
    dd_crop = np.clip(Tm - T_BASE_COTTON, 0, 14.4)
    dd10 = np.clip(Tm - 10.0, 0, 25)
    pr = REGION_PRIOR[region]

    H = np.ones(n); L = np.zeros((n, 3)); I = np.zeros(n); R = np.zeros(n)
    sw = 0.8 * F["awc"]; gdd = 0.0; lai = np.full(n, 0.05); rho = np.zeros(n)
    last_spray = np.full(n, -99); nspray = np.zeros(n, int)
    lai_max = 3.0 + 0.7 * F["nlev"]
    res = np.where(F["resistant"], 0.55, 1.0)
    out = {k: np.zeros((n, T)) for k in
           ["S", "Lat", "I", "R", "lai", "sw", "lwd", "spray", "irr", "beta", "ws"]}
    ws_cum = np.ones(n)
    for t in range(T):
        # --------------------------------------------------- crop and water
        gdd += dd_crop[t]
        pot = sigmoid((gdd - 450) / 110) * (1 - 0.5 * sigmoid((gdd - 1500) / 120))
        kc = 0.3 + 0.85 * np.clip(lai / 3.0, 0, 1)
        sw = sw + P[t] - kc * ETo[t]
        irr = F["irrigated"] & (sw < 0.45 * F["awc"])
        sw = np.where(irr, sw + 35.0, sw)
        sw = np.clip(sw, 0, F["awc"])
        ws = np.clip(sw / (0.5 * F["awc"]), 0, 1)
        ws_cum = 0.97 * ws_cum + 0.03 * ws
        lai = lai_max * pot * (0.55 + 0.45 * ws_cum) * (1 - 0.7 * R)
        # --------------------------------------------------- management
        spray = np.zeros(n, bool)
        in_win = override is not None and override["day"] <= t < override["day"] + CF_WIN
        if in_win:
            d0 = override["day"]; a = override["action"]
            if a in ("spray_now", "spray_twice") and t == d0:
                spray[:] = True
            if a == "spray_twice" and t == d0 + 10:
                spray[:] = True
        elif t >= 25:
            wetwarm = np.zeros(n)
            lo = max(0, t - 5)
            wetwarm += np.sum((lwd_w[lo:t] > 8) & (Tm[lo:t] > 22))
            s_scout = (I + R) * np.exp(0.35 * N["scout"][:, t])
            logit = (-5.2 + 70 * s_scout + 0.30 * wetwarm + 0.5 * F["skill"]
                     + pr["spray_bias"])
            want = N["u"][:, t] < sigmoid(logit)
            spray = want & (t - last_spray >= 10) & (nspray < 5)
        sprinkler_today = irr & F["sprinkler"]
        if in_win and override["action"] == "drip_only":
            sprinkler_today = np.zeros(n, bool)
        last_spray = np.where(spray, t, last_spray)
        nspray = nspray + spray
        rho = np.where(spray, 1.0, rho * np.exp(-np.log(2) / 7.0))
        if P[t] > 15:
            rho *= 0.5
        prot = 0.9 * rho
        # --------------------------------------------------- microclimate
        lwd = np.clip(lwd_w[t] + F["micro"] + 1.2 * np.clip(lai - 2, 0, None)
                      + 5.0 * sprinkler_today, 0, 24)
        # --------------------------------------------------- epidemic
        fT = beta_T(Tm[t])
        wreq = np.minimum(6.0 / np.maximum(fT, 1e-3), 48)
        gW = sigmoid((lwd - wreq) / 1.5)
        splash = 1 + 0.8 * float(np.any(P[max(0, t - 3):t] > 5)) if t > 0 else 1.0
        beta = BETA0 * fT * gW * splash * res * (1 - prot) * np.exp(N["xi"][:, t])
        inoc = np.exp(F["log_inoc"]) * np.exp(-t / 40.0) + 5e-4
        new = np.minimum(beta * H * (I + 0.6 * inoc), H)
        rate = np.clip(3 * dd10[t] / 110.0, 0, 1)
        flow1 = L[:, 0] * rate; flow2 = L[:, 1] * rate; flow3 = L[:, 2] * rate
        kill = np.where(spray, 0.3, 0.0)
        gam = np.clip((1 / 8.0) * (1 + 0.04 * (Tm[t] - 25)), 0.03, 0.4)
        rem = gam * I
        H = H - new
        L = np.stack([L[:, 0] + new - flow1, L[:, 1] + flow1 - flow2,
                      L[:, 2] + flow2 - flow3], 1) * (1 - kill[:, None])
        H = H + (kill[:, None] * 0).sum(1)
        I = I + flow3 - rem
        R = R + rem
        tot = H + L.sum(1) + I + R
        H, L, I, R = H / tot, L / tot[:, None], I / tot, R / tot
        out["S"][:, t] = I + R; out["Lat"][:, t] = L.sum(1); out["I"][:, t] = I
        out["R"][:, t] = R; out["lai"][:, t] = lai; out["sw"][:, t] = sw / F["awc"]
        out["lwd"][:, t] = lwd; out["spray"][:, t] = spray; out["irr"][:, t] = irr
        out["beta"][:, t] = beta; out["ws"][:, t] = ws
    out["lwd_weather"] = lwd_w
    return out


def counterfactuals(w, F, N, region):
    """Exact CRN replays for every (decision day, action)."""
    res = {}
    for d in CF_DAYS:
        for a in ACTIONS:
            o = simulate(w, F, N, region, override=dict(day=d, action=a))
            res[(d, a)] = o["S"][:, [d + 7, d + 14, d + 21 - 1]]
    return res


if __name__ == "__main__":
    import weather as W
    rng = np.random.default_rng(0)
    for r in W.REGIONS:
        for y in W.REGIONS[r]["years"][:3]:
            w = W.season_weather(r, y)
            F = sample_fields(rng, 16, r); Nn = season_noise(rng, 16)
            o = simulate(w, F, Nn, r)
            S = o["S"]
            print(r, y, "final S mean %.3f q90 %.3f onset frac %.2f sprays %.1f lai %.2f" % (
                S[:, -1].mean(), np.quantile(S[:, -1], .9), (S.max(1) > ONSET).mean(),
                o["spray"].sum(1).mean(), o["lai"].max(1).mean()))
