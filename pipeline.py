"""DE layer: encrypted raw -> clean -> features -> pseudonymize -> SQLite."""
import io, os, json, sqlite3, numpy as np, pandas as pd
from config import *
import privacy

def run():
    privacy.ensure_key()
    if os.path.exists(RAW):
        privacy.encrypt_file(RAW, RAW_ENC); os.remove(RAW)      # raw data never stays in plaintext
    df = pd.read_csv(io.BytesIO(privacy.decrypt_bytes(RAW_ENC)), parse_dates=["timestamp"])
    n0 = len(df)
    df = df.drop_duplicates(["timestamp", "zone_id"]); n1 = len(df)
    cap = df.zone_id.map(lambda z: ZONES[z]["cap"])
    orig_missing = int(df.vehicles.isna().sum())
    glitch = df.vehicles > 2.2 * cap
    df.loc[glitch, "vehicles"] = np.nan
    missing = df.vehicles.isna().sum()
    df = df.sort_values(["zone_id", "timestamp"])
    df["vehicles"] = df.groupby("zone_id")["vehicles"].transform(lambda s: s.interpolate(limit_direction="both"))
    t = df.timestamp
    df["hour"], df["dow"] = t.dt.hour, t.dt.dayofweek
    df["day"] = (t - t.min()).dt.days
    df["is_holiday"] = t.dt.date.astype(str).isin(HOLIDAYS).astype(int)
    df["is_weekend"] = ((df.dow >= 5) | (df.is_holiday == 1)).astype(int)
    df["congestion"] = df.vehicles / df.zone_id.map(lambda z: ZONES[z]["cap"])
    df["sensor_pid"] = df.sensor_id.map(privacy.pseudonymize)
    df = df.drop(columns="sensor_id")
    with sqlite3.connect(DB) as con: df.to_sql("traffic", con, if_exists="replace", index=False)
    report = dict(raw_rows=int(n0), duplicates_removed=int(n0 - n1), clean_rows=int(len(df)),
                  missing_in_raw=orig_missing, glitches_removed=int(glitch.sum()), values_imputed=int(missing),
                  remaining_nulls=int(df[["vehicles", "congestion"]].isna().sum().sum()),
                  zones=int(df.zone_id.nunique()), days=int(df.day.nunique()),
                  start=str(df.timestamp.min().date()), end=str(df.timestamp.max().date()))
    with open(QUALITY, "w") as f: json.dump(report, f, indent=2)
    print(f"rows {n0} -> {n1} after dedup | glitches removed {int(glitch.sum())} | values imputed {int(missing)} | saved {DB}")

if __name__ == "__main__": run()
