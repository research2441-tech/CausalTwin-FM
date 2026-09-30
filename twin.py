"""CausalTwin-FM core: compartment-anchored latent twin with
(i) thermal-time cross-modal attention residual, (ii) typed intervention
ports, (iii) particle-filter abduction of field latents from image-derived
severity, and multi-horizon / counterfactual rollouts.

Flags (for ablations):
  attn   : 'thermal' | 'calendar' | 'none'
  ports  : True  -> spray enters only through the residue port pi_t,
                    sprinkler only through the wetness port
           False -> spray/sprinkler enter as generic features of the residual
  abduct : True  -> particle filter over (theta, iota) + state
           False -> latents fixed at prior mean, state filtered only
  images : True  -> image likelihood used in the filter; False -> no images
  iot    : True  -> IoT canopy temperature / wetness used for past days
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax
from common import EPS, HORIZONS

KLAG = 10
NFREQ = 4
DEFAULT = dict(attn="thermal", ports=True, abduct=True, images=True, iot=True, feat="mae", obsop=True)


def softplus(x):
    return jax.nn.softplus(x)


def init_params(key, cfg):
    k = jax.random.split(key, 6)
    dq, dk = 5 + 8 + (0 if cfg["ports"] else 2), 5 + 2 * NFREQ + (0 if cfg["ports"] else 2)
    p = dict(
        topt=jnp.array(0.0),            # Topt = 25 + 5 tanh
        dlo=jnp.array(1.6), dhi=jnp.array(1.6),
        logW0=jnp.log(8.0), logtw=jnp.log(2.0),
        b0=jnp.log(0.3), iota0=jnp.log(3e-3), loglam=jnp.log(30.0),
        logLP=jnp.log(120.0), gam=jnp.array(-2.2),
        kap=jnp.array(1.4), loghl=jnp.log(10.0), rres=jnp.array(-0.8),
        dsprk=jnp.array(2.0), dlai=jnp.array(0.5),
        logso=jnp.log(0.3),
        ob_a=jnp.array(0.0), ob_c=jnp.array(0.0), ob_e=jnp.log(EPS), ob_d=jnp.array(0.0),
        logsi=jnp.log(0.4),
        Wq=jax.random.normal(k[0], (dq, 16)) * 0.3, Wk=jax.random.normal(k[1], (dk, 16)) * 0.3,
        Wv=jax.random.normal(k[2], (dk, 16)) * 0.3, wo=jax.random.normal(k[3], (16,)) * 0.05,
        bo=jnp.array(0.0),
    )
    if cfg["attn"] == "none" and not cfg["ports"]:
        pass
    return p


def temp_resp(p, T):
    topt = 25.0 + 5.0 * jnp.tanh(p["topt"])
    tmin = topt - 4.0 - softplus(p["dlo"]) * 5.0
    tmax = topt + 3.0 + softplus(p["dhi"]) * 3.0
    Tc = jnp.clip(T, tmin + 1e-2, tmax - 1e-2)
    a = (topt - tmin) / (tmax - topt)
    return ((tmax - Tc) / (tmax - topt)) * ((Tc - tmin) / (topt - tmin)) ** a


def efficiency(p, T, W):
    fT = temp_resp(p, T)
    wreq = jnp.minimum(jnp.exp(p["logW0"]) / jnp.maximum(fT, 1e-3), 48.0)
    return fT * jax.nn.sigmoid((W - wreq) / jnp.exp(p["logtw"]))


# ---------------------------------------------------------------- residual
def residual(p, cfg, feat, qimg, dd):
    """feat: (B,T,5) standardized forcing (+2 mgmt if no ports); qimg: (B,T,8);
    dd: (B,T) thermal increments. Returns r: (B,T)."""
    if cfg["attn"] == "none" and cfg["ports"]:
        return jnp.zeros(feat.shape[:2])
    B, T, F = feat.shape
    pad = jnp.concatenate([jnp.zeros((B, KLAG, F)), feat], 1)
    ddp = jnp.concatenate([jnp.zeros((B, KLAG)), dd], 1)
    cdd = jnp.cumsum(ddp, 1)
    idx = jnp.arange(T)[:, None] + KLAG - jnp.arange(1, KLAG + 1)[None, :]   # (T,K) past days
    keys = pad[:, idx]                                                        # (B,T,K,F)
    if cfg["attn"] == "calendar":
        lag = jnp.broadcast_to(jnp.arange(1, KLAG + 1)[None, None] / 7.0, (B, T, KLAG))
    else:
        lag = (cdd[:, KLAG:][:, :, None] - cdd[:, idx]) / 100.0
    fr = 2.0 ** jnp.arange(NFREQ)
    pe = jnp.concatenate([jnp.sin(lag[..., None] * fr), jnp.cos(lag[..., None] * fr)], -1)
    kin = jnp.concatenate([keys, pe], -1)
    q = jnp.concatenate([feat, qimg], -1) @ p["Wq"]
    kk = kin @ p["Wk"]; vv = kin @ p["Wv"]
    if cfg["attn"] == "none":            # no attention, generic mgmt features only
        ctx = vv.mean(2)
    else:
        a = jax.nn.softmax(jnp.einsum("btd,btkd->btk", q, kk) / 4.0, -1)
        ctx = jnp.einsum("btk,btkd->btd", a, vv)
    return 0.7 * jnp.tanh(ctx @ p["wo"] + p["bo"])


def attention_weights(p, cfg, feat, qimg, dd):
    B, T, F = feat.shape
    pad = jnp.concatenate([jnp.zeros((B, KLAG, F)), feat], 1)
    ddp = jnp.concatenate([jnp.zeros((B, KLAG)), dd], 1)
    cdd = jnp.cumsum(ddp, 1)
    idx = jnp.arange(T)[:, None] + KLAG - jnp.arange(1, KLAG + 1)[None, :]
    keys = pad[:, idx]
    lag = (cdd[:, KLAG:][:, :, None] - cdd[:, idx]) / 100.0 if cfg["attn"] == "thermal" else \
        jnp.broadcast_to(jnp.arange(1, KLAG + 1)[None, None] / 7.0, (B, T, KLAG))
    fr = 2.0 ** jnp.arange(NFREQ)
    pe = jnp.concatenate([jnp.sin(lag[..., None] * fr), jnp.cos(lag[..., None] * fr)], -1)
    kin = jnp.concatenate([keys, pe], -1)
    q = jnp.concatenate([feat, qimg], -1) @ p["Wq"]
    return jax.nn.softmax(jnp.einsum("btd,btkd->btk", q, kin @ p["Wk"]) / 4.0, -1), lag


# ---------------------------------------------------------------- dynamics
def step(p, cfg, state, x, theta, iota, xi):
    """state: (...,6) = H,L1,L2,I,R,psi ; x: dict of per-day scalars broadcastable."""
    H, L1, L2, I, R, psi = [state[..., i] for i in range(6)]
    hl = jnp.exp(p["loghl"])
    psi = jnp.minimum(psi * jnp.exp(-jnp.log(2.0) / hl) + x["spray"], 1.0)
    pi = jax.nn.sigmoid(p["kap"]) * psi if cfg["ports"] else 0.0 * psi
    eps = efficiency(p, x["T"], x["W"])
    res = 1.0 - jax.nn.sigmoid(p["rres"]) * x["resist"]
    beta = jnp.exp(p["b0"] + theta + x["r"] + xi) * eps * res * (1.0 - pi)
    inoc = jnp.exp(p["iota0"] + iota) * jnp.exp(-x["t"] / jnp.exp(p["loglam"]))
    new = jnp.minimum(beta * H * (I + inoc), H)
    rate = jnp.clip(2.0 * x["dd10"] / jnp.exp(p["logLP"]), 0.0, 1.0)
    g = jax.nn.sigmoid(p["gam"])
    f1 = L1 * rate; f2 = L2 * rate; rem = g * I
    out = jnp.stack([H - new, L1 + new - f1, L2 + f1 - f2, I + f2 - rem, R + rem, psi], -1)
    return out


def visible(state):
    return state[..., 3] + state[..., 4]


def image_obs(p, cfg, state):
    """Compartment-resolved observation operator: expected image-derived
    log-severity as a learned mixture of infectious and necrotic tissue."""
    if not cfg.get("obsop", True):
        return jnp.log(visible(state) + EPS)
    return p["ob_d"] + jnp.log(jnp.exp(p["ob_a"]) * state[..., 3] + jnp.exp(p["ob_c"]) * state[..., 4]
                               + jnp.exp(p["ob_e"]))


# ---------------------------------------------------------------- inputs
class Inputs:
    """Builds per-field daily twin inputs from the benchmark."""

    def __init__(self, bench, cfg):
        self.b = bench; self.cfg = cfg

    def lai_proxy(self, n):
        b = self.b
        Tm = 0.5 * (b.wx[n, :, 0] + b.wx[n, :, 1])
        gdd = np.cumsum(np.clip(Tm - 15.6, 0, 14.4), -1)
        nl = b.static[n, 1]
        return (3.0 + 0.7 * nl)[..., None] / (1 + np.exp(-(gdd - 450) / 110))

    def past(self, n):
        """Past-mode daily inputs for fields n (array of indices)."""
        b, cfg = self.b, self.cfg
        wx = b.wx[n]; iot = b.iot[n]; miss = b.miss[n] > 0.5
        Tm = 0.5 * (wx[..., 0] + wx[..., 1])
        sprk = b.static[n, 4][:, None] * b.irr[n]
        lai = self.lai_proxy(n)
        if cfg["iot"]:
            T = np.where(miss, Tm, iot[..., 0])
            Wobs = np.where(miss, np.nan, iot[..., 1])
        else:
            T = Tm; Wobs = np.full(Tm.shape, np.nan)
        return dict(T=T, Wobs=Wobs, Ww=wx[..., 4], sprk=sprk, lai=lai, spray=b.spray[n],
                    P=wx[..., 2], dd10=np.clip(T - 10, 0, 25), resist=b.static[n, 2],
                    Tw=Tm)

    def feat(self, d, ports):
        """standardized forcing features for the residual attention."""
        b = self.b
        W = np.where(np.isnan(d["Wobs"]), d["Ww"], d["Wobs"])
        f = np.stack([(d["T"] - 22) / 5, (W - 8) / 5, np.log1p(d["P"]) / 2,
                      (d["P"] > 5).astype(float), d["sprk"]], -1)
        if not ports:
            f = np.concatenate([f, d["spray"][..., None], d["sprk"][..., None]], -1)
        return f.astype(np.float32)

    def qimg(self, n, upto=None):
        b = self.b
        if not self.cfg["images"]:
            E = np.zeros(b.emb[n].shape[:-1] + (8,))
        elif self.cfg["feat"] == "color":
            E = b.colorz[n]
        else:
            E = (b.emb[n] - b.emb_mu) @ b.pca.T / 3.0
        days = b.img_days
        T = b.T
        j = np.clip(np.searchsorted(days, np.arange(T), side="right") - 1, 0, None)
        if upto is not None:
            j = np.minimum(j, list(days).index(upto) if upto in days else
                           np.searchsorted(days, upto, side="right") - 1)
        return E[:, j].astype(np.float32)


def wetness(p, cfg, d, offset):
    """Wetness hours: IoT where observed, otherwise weather wetness + ports + offset."""
    port = d["Ww"] + offset[..., None] + p["dlai"] * jnp.clip(d["lai"] - 2.0, 0, None)
    if cfg["ports"]:
        port = port + p["dsprk"] * d["sprk"]
    port = jnp.clip(port, 0, 24)
    obs = d["Wobs"]
    return jnp.where(jnp.isnan(obs), port, obs), port


# ---------------------------------------------------------------- training
def region_prior(bench, idx_train, lat, idx):
    """Empirical-Bayes prior for field latents: per-region mean/sd of the
    latents fitted on training fields; pooled statistics for unseen regions."""
    th, io = np.asarray(lat["theta"]), np.asarray(lat["iota"])
    reg_tr = bench.region[idx_train]
    pooled = np.array([th.mean(), th.std(), io.mean(), io.std()])
    out = np.zeros((len(idx), 4))
    for i, n in enumerate(idx):
        m = reg_tr == bench.region[n]
        out[i] = [th[m].mean(), th[m].std(), io[m].mean(), io[m].std()] if m.sum() > 20 else pooled
    out[:, 1] = np.maximum(out[:, 1], 0.05); out[:, 3] = np.maximum(out[:, 3], 0.05)
    return out


def train(bench, cfg, idx, steps=1200, seed=0, lr=1e-2, log=None):
    """Fit globals + per-field latents on expert-labelled training fields by
    reverse-mode differentiation through full-season rollouts.  A random
    switch day per field emulates forecast mode (ports + offset) after it."""
    inp = Inputs(bench, cfg)
    d = inp.past(idx)
    B, T = d["T"].shape
    feat = jnp.asarray(inp.feat(d, cfg["ports"]))
    qim = jnp.asarray(inp.qimg(idx))
    days = bench.img_days
    y = jnp.asarray(np.log(bench.S_expert[idx] + EPS))
    yimg = jnp.asarray(bench.hmu[idx] if cfg["feat"] == "mae" else bench.cmu[idx])
    D = {k: jnp.asarray(v, jnp.float32) for k, v in d.items()}
    key = jax.random.PRNGKey(seed)
    p = init_params(key, cfg)
    lat = dict(theta=jnp.zeros(B), iota=jnp.zeros(B))
    params = dict(g=p, l=lat)
    opt = optax.adam(lr); st = opt.init(params)
    tt = jnp.arange(T, dtype=jnp.float32)
    daysj = jnp.asarray(days)

    def loss(params, switch):
        p, l = params["g"], params["l"]
        Wobs = jnp.where(tt[None] > switch[:, None], jnp.nan, D["Wobs"])
        Dm = dict(D); Dm["Wobs"] = Wobs
        diff = jnp.where(jnp.isnan(D["Wobs"]), 0.0,
                         D["Wobs"] - (D["Ww"] + p["dlai"] * jnp.clip(D["lai"] - 2, 0, None)
                                      + (p["dsprk"] * D["sprk"] if cfg["ports"] else 0.0)))
        m = (~jnp.isnan(D["Wobs"])) & (tt[None] <= switch[:, None])
        offset = jnp.sum(jnp.where(m, diff, 0.0), 1) / jnp.maximum(m.sum(1), 1)
        W, _ = wetness(p, cfg, Dm, offset)
        r = residual(p, cfg, feat, qim, D["dd10"])
        s0 = jnp.zeros((B, 6)).at[:, 0].set(1.0)

        def body(s, t):
            x = dict(T=D["T"][:, t], W=W[:, t], spray=D["spray"][:, t], resist=D["resist"],
                     r=r[:, t], t=tt[t], dd10=D["dd10"][:, t])
            s = step(p, cfg, s, x, l["theta"], l["iota"], 0.0)
            return s, s

        _, st_all = jax.lax.scan(body, s0, jnp.arange(T))
        st_img = st_all.transpose(1, 0, 2)[:, daysj]
        vis = visible(st_img)
        so = jnp.exp(p["logso"])
        nll = jnp.mean(0.5 * ((y - jnp.log(vis + EPS)) / so) ** 2 + p["logso"])
        if cfg["images"]:
            si = jnp.exp(p["logsi"])
            nll = nll + jnp.mean(0.5 * ((yimg - image_obs(p, cfg, st_img)) / si) ** 2 + p["logsi"])
        prior = 0.5 * jnp.mean(l["theta"] ** 2 / 0.49 + l["iota"] ** 2) / len(days)
        return nll + prior

    @jax.jit
    def upd(params, st, key):
        switch = jax.random.randint(key, (B,), 20, T).astype(jnp.float32)
        l, g = jax.value_and_grad(loss)(params, switch)
        u, st = opt.update(g, st)
        return optax.apply_updates(params, u), st, l

    hist = []
    for i in range(steps):
        key, k = jax.random.split(key)
        params, st, l = upd(params, st, k)
        if i % 50 == 0 or i == steps - 1:
            hist.append((i, float(l)))
            if log:
                print(cfg, i, float(l), flush=True)
    return params["g"], hist, params["l"]


# ---------------------------------------------------------------- filter
def particle_filter(bench, cfg, p, idx, K=96, sxi=0.25, seed=0, save_days=(), prior=None):
    """Runs the PF over the season in past mode; returns particle states and
    latents saved at the end of each day in save_days, plus the wetness offset."""
    inp = Inputs(bench, cfg)
    d = inp.past(idx)
    B, T = d["T"].shape
    feat = jnp.asarray(inp.feat(d, cfg["ports"]))
    qim = jnp.asarray(inp.qimg(idx))
    D = {k: jnp.asarray(v, jnp.float32) for k, v in d.items()}
    # offset from all past IoT days up to each saved day is approximated by the
    # running mean; computed per day below
    diff = jnp.where(jnp.isnan(D["Wobs"]), 0.0,
                     D["Wobs"] - (D["Ww"] + p["dlai"] * jnp.clip(D["lai"] - 2, 0, None)
                                  + (p["dsprk"] * D["sprk"] if cfg["ports"] else 0.0)))
    cnt = jnp.cumsum(~jnp.isnan(D["Wobs"]), 1)
    run_off = jnp.cumsum(diff, 1) / jnp.maximum(cnt, 1)
    W, _ = wetness(p, cfg, D, run_off[:, -1] * 0.0)       # past mode uses IoT when present
    r = residual(p, cfg, feat, qim, D["dd10"])
    obs = np.full((B, T), np.nan, np.float32); osd = np.ones((B, T), np.float32)
    if cfg["images"]:
        mu = bench.hmu[idx] if cfg["feat"] == "mae" else bench.cmu[idx]
        ls = bench.hls[idx] if cfg["feat"] == "mae" else bench.cls_[idx]
        keep = getattr(bench, "img_keep", None)
        if keep is not None:
            mu = np.where(keep[idx], mu, np.nan)
        obs[:, bench.img_days] = mu; osd[:, bench.img_days] = np.exp(ls)
    obs = jnp.asarray(obs); osd = jnp.asarray(osd)
    so = jnp.exp(p["logso"])
    key = jax.random.PRNGKey(seed)
    k1, k2, key = jax.random.split(key, 3)
    if prior is None:
        prior = np.tile(np.array([0.0, 0.7, 0.0, 1.0]), (B, 1))
    pr = jnp.asarray(prior, jnp.float32)
    if cfg["abduct"]:
        th = pr[:, 0:1] + pr[:, 1:2] * jax.random.normal(k1, (B, K))
        io = pr[:, 2:3] + pr[:, 3:4] * jax.random.normal(k2, (B, K))
    else:
        th = jnp.broadcast_to(pr[:, 0:1], (B, K)); io = jnp.broadcast_to(pr[:, 2:3], (B, K))
    s0 = jnp.zeros((B, K, 6)).at[..., 0].set(1.0)
    logw0 = jnp.zeros((B, K))
    tt = jnp.arange(T, dtype=jnp.float32)

    def body(c, t):
        s, th, io, logw, key = c
        key, kx, kr, kj1, kj2 = jax.random.split(key, 5)
        x = dict(T=D["T"][:, t, None], W=W[:, t, None], spray=D["spray"][:, t, None],
                 resist=D["resist"][:, None], r=r[:, t, None], t=tt[t], dd10=D["dd10"][:, t, None])
        xi = sxi * jax.random.normal(kx, (B, K))
        s = step(p, cfg, s, x, th, io, xi)
        v = image_obs(p, cfg, s)
        o = obs[:, t, None]
        sd = jnp.exp(p["logsi"]) if cfg.get("obsop", True) else jnp.sqrt(osd[:, t, None] ** 2 + so ** 2)
        ll = jnp.where(jnp.isnan(o), 0.0, -0.5 * ((o - v) / sd) ** 2)
        logw = logw + ll
        logw = logw - jax.scipy.special.logsumexp(logw, 1, keepdims=True)
        ess = 1.0 / jnp.sum(jnp.exp(2 * logw), 1)
        # systematic resampling where ESS < K/2
        u = (jax.random.uniform(kr, (B, 1)) + jnp.arange(K)[None]) / K
        cw = jnp.cumsum(jnp.exp(logw), 1)
        ids = jnp.clip(jax.vmap(jnp.searchsorted)(cw, u), 0, K - 1)
        do = (ess < K / 2)[:, None]
        take = lambda a: jnp.take_along_axis(a, ids, 1)
        s = jnp.where(do[..., None], jnp.take_along_axis(s, ids[..., None], 1), s)
        jit_th = 0.05 * jax.random.normal(kj1, (B, K)) if cfg["abduct"] else 0.0
        jit_io = 0.05 * jax.random.normal(kj2, (B, K)) if cfg["abduct"] else 0.0
        th = jnp.where(do, take(th) + jit_th, th)
        io = jnp.where(do, take(io) + jit_io, io)
        logw = jnp.where(do, -jnp.log(K) * jnp.ones_like(logw), logw)
        return (s, th, io, logw, key), (s, th, io, logw)

    _, (S, TH, IO, LW) = jax.lax.scan(body, (s0, th, io, logw0, key), jnp.arange(T))
    sd = np.array(save_days)
    return dict(state=np.asarray(S[sd]).transpose(1, 0, 2, 3),
                theta=np.asarray(TH[sd]).transpose(1, 0, 2),
                iota=np.asarray(IO[sd]).transpose(1, 0, 2),
                logw=np.asarray(LW[sd]).transpose(1, 0, 2),
                offset=np.asarray(run_off)[:, sd])


# ---------------------------------------------------------------- rollout
def rollout(bench, cfg, p, idx, pf, save_days, fut, H, sxi=0.25, seed=1):
    """Roll particles forward H days from each saved day.
    fut: dict of future-mode arrays with shape (B, O, H): Tmin, Tmax, P, Ww(wetness),
    spray, sprk, and 'past_feat' (B,O,KLAG,F) forcing features of the last KLAG days.
    Returns visible severity samples (B,O,H,K) and weights (B,O,K)."""
    inp = Inputs(bench, cfg)
    B, O = len(idx), len(save_days)
    Tm = 0.5 * (fut["Tmin"] + fut["Tmax"])
    lai_all = inp.lai_proxy(idx)
    sdays = np.array(save_days)
    lai = np.stack([lai_all[:, np.clip(sdays + h + 1, 0, bench.T - 1)] for h in range(H)], -1)
    resist = np.broadcast_to(bench.static[idx, 2][:, None], (B, O))
    off = pf["offset"]
    port = fut["Ww"] + off[..., None] + p["dlai"] * np.clip(lai - 2, 0, None)
    if cfg["ports"]:
        port = port + float(p["dsprk"]) * fut["sprk"]
    W = np.clip(port, 0, 24)
    f = np.stack([(Tm - 22) / 5, (W - 8) / 5, np.log1p(fut["P"]) / 2, (fut["P"] > 5).astype(float),
                  fut["sprk"]], -1)
    if not cfg["ports"]:
        f = np.concatenate([f, fut["spray"][..., None], fut["sprk"][..., None]], -1)
    allf = np.concatenate([fut["past_feat"], f], 2).astype(np.float32)          # (B,O,KLAG+H,F)
    qimg = np.repeat(fut["qimg"][:, :, None], KLAG + H, 2).astype(np.float32)
    dd = np.concatenate([fut["past_dd"], np.clip(Tm - 10, 0, 25)], 2).astype(np.float32)
    r = residual(p, cfg, jnp.asarray(allf.reshape(B * O, KLAG + H, -1)),
                 jnp.asarray(qimg.reshape(B * O, KLAG + H, -1)),
                 jnp.asarray(dd.reshape(B * O, KLAG + H)))
    r = np.asarray(r).reshape(B, O, KLAG + H)[:, :, KLAG:]
    s = jnp.asarray(pf["state"]); th = jnp.asarray(pf["theta"]); io = jnp.asarray(pf["iota"])
    K = s.shape[2]
    key = jax.random.PRNGKey(seed)
    outs = []
    for h in range(H):
        key, k = jax.random.split(key)
        x = dict(T=jnp.asarray(Tm[:, :, h, None]), W=jnp.asarray(W[:, :, h, None]),
                 spray=jnp.asarray(fut["spray"][:, :, h, None]), resist=jnp.asarray(resist[..., None]),
                 r=jnp.asarray(r[:, :, h, None]), t=jnp.asarray((sdays + h + 1)[None, :, None], jnp.float32),
                 dd10=jnp.asarray(np.clip(Tm[:, :, h, None] - 10, 0, 25)))
        s = step(p, cfg, s, x, th, io, sxi * jax.random.normal(k, (B, O, K)))
        outs.append(np.asarray(visible(s)))
    return np.stack(outs, 2), np.exp(pf["logw"])


# ---------------------------------------------------------------- shared drivers
def iot_offset(bench, cfg, p, idx, sdays):
    """Running mean of (IoT wetness - weather/port wetness) up to each day in sdays."""
    d = Inputs(bench, cfg).past(idx)
    port = d["Ww"] + float(p["dlai"]) * np.clip(d["lai"] - 2, 0, None)
    if cfg["ports"]:
        port = port + float(p["dsprk"]) * d["sprk"]
    diff = np.where(np.isnan(d["Wobs"]), 0.0, d["Wobs"] - port)
    cnt = np.cumsum(~np.isnan(d["Wobs"]), 1)
    run = np.cumsum(diff, 1) / np.maximum(cnt, 1)
    return run[:, np.asarray(sdays)]


def psi_at(bench, p, idx, sdays):
    hl = float(np.exp(p["loghl"]))
    sp = bench.spray[idx]
    psi = np.zeros(len(idx)); out = np.zeros((len(idx), bench.T))
    for t in range(bench.T):
        psi = np.minimum(psi * np.exp(-np.log(2) / hl) + sp[:, t], 1.0)
        out[:, t] = psi
    return out[:, np.asarray(sdays)]


def drivers(bench, cfg, p, idx, sdays, fut, H, offset):
    inp = Inputs(bench, cfg)
    B, O = len(idx), len(sdays)
    sdays = np.asarray(sdays)
    Tm = 0.5 * (fut["Tmin"] + fut["Tmax"])
    lai_all = inp.lai_proxy(idx)
    lai = np.stack([lai_all[:, np.clip(sdays + h + 1, 0, bench.T - 1)] for h in range(H)], -1)
    port = fut["Ww"] + offset[..., None] + float(p["dlai"]) * np.clip(lai - 2, 0, None)
    if cfg["ports"]:
        port = port + float(p["dsprk"]) * fut["sprk"]
    W = np.clip(port, 0, 24)
    f = np.stack([(Tm - 22) / 5, (W - 8) / 5, np.log1p(fut["P"]) / 2, (fut["P"] > 5).astype(float),
                  fut["sprk"]], -1)
    if not cfg["ports"]:
        f = np.concatenate([f, fut["spray"][..., None], fut["sprk"][..., None]], -1)
    allf = np.concatenate([fut["past_feat"], f], 2).astype(np.float32)
    qimg = np.repeat(fut["qimg"][:, :, None], KLAG + H, 2).astype(np.float32)
    dd = np.concatenate([fut["past_dd"], np.clip(Tm - 10, 0, 25)], 2).astype(np.float32)
    r = residual(p, cfg, jnp.asarray(allf.reshape(B * O, KLAG + H, -1)),
                 jnp.asarray(qimg.reshape(B * O, KLAG + H, -1)),
                 jnp.asarray(dd.reshape(B * O, KLAG + H)))
    r = np.asarray(r).reshape(B, O, KLAG + H)[:, :, KLAG:]
    return dict(T=Tm.astype(np.float32), W=W.astype(np.float32), spray=fut["spray"].astype(np.float32),
                r=r.astype(np.float32), dd10=np.clip(Tm - 10, 0, 25).astype(np.float32),
                resist=np.broadcast_to(bench.static[idx, 2][:, None], (B, O)).astype(np.float32),
                t=np.broadcast_to((sdays[:, None] + np.arange(1, H + 1)[None]), (B, O, H)).astype(np.float32))


def propagate(p, cfg, s, th, io, drv, sxi, key):
    """Differentiable forward propagation. s: (...,K,6); drv arrays (...,H);
    th/io: (...,K). Returns states (H,...,K,6)."""
    H = drv["T"].shape[-1]

    def body(c, h):
        s, key = c
        key, k = jax.random.split(key)
        x = {kk: (drv[kk][..., h, None] if kk != "resist" else drv[kk][..., None]) for kk in
             ["T", "W", "spray", "r", "dd10", "t", "resist"]}
        s = step(p, cfg, s, x, th, io, sxi * jax.random.normal(k, th.shape))
        return (s, key), s

    _, S = jax.lax.scan(body, (s, key), jnp.arange(H))
    return S
