"""Simple baseline: average congestion for the same zone, hour and weekday/weekend type."""
import numpy as np
KEYS = ["zone_id", "hour", "is_weekend"]

def fit_baseline(df):
    return df.groupby(KEYS).congestion.mean().rename("pred").reset_index()

def predict_baseline(b, df):
    m = df[KEYS].merge(b, on=KEYS, how="left")
    return m.pred.fillna(b.pred.mean()).values

def mae_baseline(b, df):
    return float(np.abs(predict_baseline(b, df) - df.congestion.values).mean())
