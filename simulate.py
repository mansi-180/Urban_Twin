"""What-if engine. Effects are transparent ASSUMPTIONS (tunable in A), applied on top of model predictions."""
import numpy as np, pandas as pd
from config import *
A = dict(bus_relief=0.025, max_relief=0.30, divert=0.6, mall_cars=12, spill=0.2, stay=0.7)

def apply(base, buses=None, closure=None, mall=None):
    """base: DataFrame[zone_id, hour, vehicles]. buses={zone:n}, closure={zone:frac_capacity_lost}, mall=(zone,size_000sqm)"""
    d = base.copy().reset_index(drop=True)
    d["cap"] = d.zone_id.map(lambda z: ZONES[z]["cap"]); d["cong_base"] = d.vehicles / d.cap; d["parking"] = 0.0
    for z, n in (buses or {}).items():                       # each bus shifts ~2.5% of cars to transit (max 30%)
        d.loc[d.zone_id == z, "vehicles"] *= 1 - min(A["max_relief"], n * A["bus_relief"])
    for z, pct in (closure or {}).items():                   # 60% of blocked traffic diverts to neighbours
        m = d.zone_id == z; moved = (d.loc[m, "vehicles"] * pct * A["divert"]).values
        mv = pd.Series(moved, index=d.loc[m, "hour"].values)
        d.loc[m, "vehicles"] -= moved; d.loc[m, "cap"] *= 1 - pct
        for n_ in ZONES[z]["nbrs"]:
            k = d.zone_id == n_
            d.loc[k, "vehicles"] += d.loc[k, "hour"].map(mv).values / len(ZONES[z]["nbrs"])
    if mall:                                                 # extra trips peak ~6pm; 20% spill to neighbours
        z, size = mall; extra = size * A["mall_cars"] * np.exp(-((d.hour - 18) / 4) ** 2)
        k = d.zone_id == z
        d.loc[k, "vehicles"] += extra[k] * (1 - A["spill"]); d.loc[k, "parking"] = extra[k] * A["stay"]
        for n_ in ZONES[z]["nbrs"]:
            kk = d.zone_id == n_; d.loc[kk, "vehicles"] += extra[kk] * A["spill"] / len(ZONES[z]["nbrs"])
    d["congestion"] = d.vehicles / d.cap
    return d
