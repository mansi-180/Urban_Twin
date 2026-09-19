"""DL layer: MLP with zone embedding predicts congestion (vehicles/capacity); incremental 'self-evolving' retraining."""
import os, json, sqlite3, numpy as np, pandas as pd, torch, torch.nn as nn
from config import *

def load():
    with sqlite3.connect(DB) as con: return pd.read_sql("select * from traffic", con, parse_dates=["timestamp"])

def make_xy(df):
    h = df["hour"].values / 24 * 2 * np.pi
    cols = [f(k * h) for k in (1, 2, 3) for f in (np.sin, np.cos)]
    cols += [df.is_weekend.values, df.rain_mm.values / 10, df.is_event.values]
    X = torch.tensor(np.stack(cols, 1), dtype=torch.float32)
    z = torch.tensor(df.zone_id.values, dtype=torch.long)
    y = torch.tensor(df.congestion.values, dtype=torch.float32) if "congestion" in df else None
    return X, z, y

class TrafficNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(len(ZONES), 8)
        s.net = nn.Sequential(nn.Linear(9 + 8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(s, X, z): return s.net(torch.cat([X, s.emb(z)], 1)).squeeze(1)

def fit(m, df, epochs=60, lr=3e-3):
    X, z, y = make_xy(df); opt = torch.optim.Adam(m.parameters(), lr=lr); n = len(y)
    m.train()
    for _ in range(epochs):
        p = torch.randperm(n)
        for i in range(0, n, 256):
            b = p[i:i + 256]; opt.zero_grad()
            nn.functional.mse_loss(m(X[b], z[b]), y[b]).backward(); opt.step()
    return m

def predict(m, df):
    X, z, _ = make_xy(df); m.eval()
    with torch.no_grad(): return m(X, z).numpy()

def mae(m, df): return float(np.abs(predict(m, df) - df.congestion.values).mean())

def load_model(name):
    m = TrafficNet(); m.load_state_dict(torch.load(f"{MODELS}/{name}.pt")); return m

def train_initial():
    torch.manual_seed(0); os.makedirs(MODELS, exist_ok=True)
    df = load(); tr, va = df[df.day < 52], df[(df.day >= 52) & (df.day < MALL_OPEN_DAY)]
    m = fit(TrafficNet(), tr, 80)
    from baseline import fit_baseline, mae_baseline
    bl = fit_baseline(tr); sp = va[(va.rain_mm > 0) | (va.is_event == 1)]   # rain / event rows
    res = dict(model_mae=mae(m, va), baseline_mae=mae_baseline(bl, va),
               model_mae_special=mae(m, sp), baseline_mae_special=mae_baseline(bl, sp), special_rows=int(len(sp)))
    with open(METRICS, "w") as f: json.dump(res, f, indent=2)
    print(f"validation MAE (congestion ratio): neural net {res['model_mae']:.4f} vs simple-average baseline {res['baseline_mae']:.4f}")
    print(f"rain/event rows only: neural net {res['model_mae_special']:.4f} vs baseline {res['baseline_mae_special']:.4f}")
    torch.save(m.state_dict(), f"{MODELS}/base.pt")

def evolve():
    """Data arrives in 15-day chunks after the mall opens. Test-then-train: measure error, then update."""
    torch.manual_seed(1); df = load(); base, m = load_model("base"), load_model("base")
    seen, hist = df[df.day < MALL_OPEN_DAY], []
    for c in range(4):
        lo = MALL_OPEN_DAY + 15 * c; ch = df[(df.day >= lo) & (df.day < lo + 15)]; mall = ch[ch.zone_id == 4]
        hist.append(dict(chunk=c + 1, days=f"{lo}-{lo + 14}", mae_frozen=mae(base, ch), mae_evolving=mae(m, ch),
                         mae_mall_frozen=mae(base, mall), mae_mall_evolving=mae(m, mall)))
        fit(m, pd.concat([ch, seen.sample(frac=0.2, random_state=c)]), epochs=25, lr=2e-3)  # new data + replay
        seen = pd.concat([seen, ch])
    pd.DataFrame(hist).round(4).to_csv(HIST, index=False)
    torch.save(m.state_dict(), f"{MODELS}/evolved.pt")
    print(pd.DataFrame(hist).round(4).to_string(index=False))
