"""Generates SIMULATED city traffic data (hourly, 6 zones, 120 days) with realistic patterns,
dirty values (NaN, duplicates, sensor glitches) and a new mall opening on day 60 (city 'evolves')."""
import numpy as np, pandas as pd, os
from config import *
rng = np.random.default_rng(42)
p = lambda h, c, w: np.exp(-((h - c) / w) ** 2)

def prof(z, h, we):
    if z in (0, 1):
        return 0.15 + 0.3 * p(h, 14, 4) if we else 0.2 + 0.6 * p(h, 9, 1.6) + 0.65 * p(h, 18, 1.8) + 0.25 * p(h, 13, 3)
    if z == 2: return (0.2 + 0.4 * p(h, 11, 2) + 0.5 * p(h, 18, 2)) * (1.2 if we else 1)
    if z == 3:
        return 0.1 + 0.1 * p(h, 12, 4) if we else 0.15 + 0.7 * p(h, 8, 1) + 0.4 * p(h, 13, 1.2) + 0.5 * p(h, 17, 1.2)
    if z == 4: return (0.15 + 0.35 * p(h, 13, 3) + 0.55 * p(h, 19, 2.5)) * (1.3 if we else 1)
    return 0.25 + 0.3 * p(h, 11, 3) + 0.3 * p(h, 19, 2) if we else 0.2 + 0.7 * p(h, 8, 1.5) + 0.6 * p(h, 19, 1.8)

def make():
    os.makedirs(DATA, exist_ok=True)
    ts = pd.date_range(START, periods=DAYS * 24, freq="h")
    rainy = rng.random(DAYS) < 0.2
    EV = {d: (0 if d % 2 == 0 else 4) for d in range(5, DAYS, 11)}  # event day -> zone
    rows = []
    for i, t in enumerate(ts):
        d, h = i // 24, t.hour
        we = t.dayofweek >= 5 or str(t.date()) in HOLIDAYS
        rain = float(rng.gamma(2, 2)) if rainy[d] and 12 <= h <= 22 else 0.0
        for z, Z in ZONES.items():
            ev = int(EV.get(d) == z)
            f = 0.75 * prof(z, h, we) * (1 + 0.001 * d) * (1 + 0.03 * min(rain, 5))
            if z == 4 and d >= MALL_OPEN_DAY: f *= 1.45
            if ev and 17 <= h <= 22: f *= 1.3
            v = Z["cap"] * f * rng.lognormal(0, 0.05)
            rows.append(dict(timestamp=t, zone_id=z, sensor_id=f"SENSOR-{z}-01", vehicles=round(v, 1),
                             avg_speed=round(45 * (1 - 0.6 * min(1.2, v / Z["cap"])), 1),
                             rain_mm=round(rain, 1), is_event=ev))
    df = pd.DataFrame(rows)
    cap = df.zone_id.map(lambda z: ZONES[z]["cap"])
    df.loc[rng.choice(len(df), int(.01 * len(df)), replace=False), "vehicles"] = np.nan   # missing
    g = rng.choice(len(df), int(.002 * len(df)), replace=False)
    df.loc[g, "vehicles"] = cap.iloc[g] * 3.5                                              # sensor glitch
    df = pd.concat([df, df.sample(50, random_state=1)])                                    # duplicates
    df.to_csv(RAW, index=False)
    print("raw rows:", len(df))

if __name__ == "__main__": make()
