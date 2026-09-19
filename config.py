import pandas as pd
DATA, MODELS = "data", "models"
RAW, RAW_ENC = f"{DATA}/raw_traffic.csv", f"{DATA}/raw_traffic.enc"
KEY, DB, HIST = f"{DATA}/secret.key", f"{DATA}/city.db", f"{DATA}/learning_history.csv"
START, DAYS, MALL_OPEN_DAY = "2026-05-04", 120, 60
HOLIDAYS = {str((pd.Timestamp(START) + pd.Timedelta(days=d)).date()) for d in (14, 40, 75, 100)}
# Real Mumbai areas (approx. centre coordinates). Traffic values are SIMULATED using each area's typical
# land-use pattern (id 0-1 office/commuter, 2 market, 3 campus, 4 mall belt, 5 residential).
# cap = assumed road capacity (vehicles/hour), not measured.
ZONES = {
    0: dict(name="BKC (Bandra-Kurla Complex)", lat=19.0665, lon=72.8690, cap=1200, nbrs=[2, 3, 4]),
    1: dict(name="Dadar (Station Hub)", lat=19.0178, lon=72.8478, cap=1000, nbrs=[0, 2]),
    2: dict(name="Bandra West (Linking Rd)", lat=19.0544, lon=72.8402, cap=800, nbrs=[0, 1, 3]),
    3: dict(name="Kalina (Mumbai University)", lat=19.0750, lon=72.8560, cap=700, nbrs=[0, 2, 4]),
    4: dict(name="Kurla (LBS Marg mall belt)", lat=19.0868, lon=72.8890, cap=900, nbrs=[0, 3, 5]),
    5: dict(name="Chembur (Residential)", lat=19.0522, lon=72.8994, cap=850, nbrs=[0, 4]),
}
