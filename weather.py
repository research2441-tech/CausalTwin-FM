"""Real weather traces and derived microclimate drivers.

Daily Tmin/Tmax/precipitation/ETo come from station files bundled with the
AquaCrop-OSPy package (Hyderabad, India; Cordoba, Argentina; Tunis, Tunisia).
Leaf-wetness duration is not reported there, so a daily wetness model is
calibrated on the hourly TMY3 record of Greensboro, NC (pvlib bundle), where
wetness hours are counted from relative humidity and rain.  The warming
scenario uses delta changes computed from CMIP6 daily output bundled with
AquaCrop (three GCMs, SSP5-8.5 2060-2079 minus historical 1995-2014).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = HERE

# region -> file, sowing (month, day), years usable, hemisphere note
REGIONS = {
    "HYD": dict(file="hyderabad_climate.txt", sow=(6, 20), years=list(range(2000, 2010))),
    "COR": dict(file="cordoba_climate.txt", sow=(11, 5), years=list(range(1991, 2021))),
    "TUN": dict(file="tunis_climate.txt", sow=(4, 25), years=list(range(1979, 2002))),
}
SEASON_DAYS = 140


def load_station(region):
    f = os.path.join(DATA, REGIONS[region]["file"])
    df = pd.read_csv(f, sep=r"\s+")
    df.columns = ["Day", "Month", "Year", "Tmin", "Tmax", "P", "ETo"]
    df["date"] = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day))
    return df.set_index("date")[["Tmin", "Tmax", "P", "ETo"]].astype(float)


def season_weather(region, year):
    df = load_station(region)
    m, d = REGIONS[region]["sow"]
    start = pd.Timestamp(year=year, month=m, day=d)
    w = df.loc[start:start + pd.Timedelta(days=SEASON_DAYS - 1)]
    if len(w) < SEASON_DAYS:
        return None
    w = w.copy()
    w["Tmax"] = np.maximum(w["Tmax"], w["Tmin"] + 0.5)
    return w


# ---------------------------------------------------------------- wetness model
_LWD_COEF = None


def _tmy_daily():
    f = os.path.join(DATA, "723170TYA.CSV")
    df = pd.read_csv(f, skiprows=1)
    rh = df["RHum (%)"].astype(float).values
    t = df["Dry-bulb (C)"].astype(float).values
    pr = df["Lprecip depth (mm)"].astype(float).values
    pr = np.where(pr < 0, 0, pr)
    wet = ((rh >= 87) | (pr > 0.1)).astype(float)
    n = len(rh) // 24
    D = {}
    D["lwd"] = wet[:n * 24].reshape(n, 24).sum(1)
    D["tmin"] = t[:n * 24].reshape(n, 24).min(1)
    D["tmax"] = t[:n * 24].reshape(n, 24).max(1)
    D["p"] = pr[:n * 24].reshape(n, 24).sum(1)
    return pd.DataFrame(D)


def _lwd_features(tmin, tmax, p):
    dtr = tmax - tmin
    return np.stack([np.ones_like(tmin), dtr, tmin, (p > 0.2).astype(float),
                     np.log1p(p), dtr * (p > 0.2)], -1)


def calibrate_lwd():
    """Least-squares daily wetness-hours model fitted on hourly TMY3 data."""
    global _LWD_COEF
    d = _tmy_daily()
    X = _lwd_features(d.tmin.values, d.tmax.values, d.p.values)
    y = d.lwd.values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = np.clip(X @ coef, 0, 24)
    stats = dict(n_days=len(y), mae=float(np.mean(np.abs(pred - y))),
                 r=float(np.corrcoef(pred, y)[0, 1]), coef=coef.tolist())
    _LWD_COEF = coef
    return stats


def leaf_wetness(tmin, tmax, p):
    if _LWD_COEF is None:
        calibrate_lwd()
    return np.clip(_lwd_features(tmin, tmax, p) @ _LWD_COEF, 0, 24)


# ---------------------------------------------------------------- climate deltas
def cmip6_deltas():
    """Monthly-invariant deltas: mean warming (Tmin, Tmax) and precipitation
    ratio, SSP5-8.5 2060-2079 vs historical 1995-2014, averaged over 3 GCMs."""
    out = []
    for g in ["CanESM5", "MIROC6", "NorESM2-LM"]:
        H = pd.read_csv(os.path.join(DATA, f"cmip6_{g}_historical_1995_2014.csv"))
        S = pd.read_csv(os.path.join(DATA, f"cmip6_{g}_ssp585_2060_2079.csv"))
        out.append(dict(gcm=g, dTmin=S.MinTemp.mean() - H.MinTemp.mean(),
                        dTmax=S.MaxTemp.mean() - H.MaxTemp.mean(),
                        pratio=S.Precipitation.mean() / H.Precipitation.mean()))
    df = pd.DataFrame(out)
    return df, dict(dTmin=float(df.dTmin.mean()), dTmax=float(df.dTmax.mean()),
                    pratio=float(df.pratio.mean()))


def apply_delta(w, delta):
    w = w.copy()
    w["Tmin"] += delta["dTmin"]
    w["Tmax"] += delta["dTmax"]
    w["P"] *= delta["pratio"]
    # Hargreaves-style ETo scaling with temperature
    tm0 = (w["Tmin"] - delta["dTmin"] + w["Tmax"] - delta["dTmax"]) / 2
    tm1 = (w["Tmin"] + w["Tmax"]) / 2
    w["ETo"] *= (tm1 + 17.8) / (tm0 + 17.8)
    return w


if __name__ == "__main__":
    print(calibrate_lwd())
    print(cmip6_deltas()[0])
    for r in REGIONS:
        w = season_weather(r, REGIONS[r]["years"][0])
        print(r, w.mean().round(2).to_dict(), leaf_wetness(w.Tmin.values, w.Tmax.values, w.P.values).mean())
