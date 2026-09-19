import pandas as pd
DATA, MODELS = "data", "models"
RAW, RAW_ENC = f"{DATA}/raw_traffic.csv", f"{DATA}/raw_traffic.enc"
KEY, DB, HIST = f"{DATA}/secret.key", f"{DATA}/city.db", f"{DATA}/learning_history.csv"
START, DAYS, MALL_OPEN_DAY = "2026-05-04", 120, 60
HOLIDAYS = {str((pd.Timestamp(START) + pd.Timedelta(days=d)).date()) for d in (14, 40, 75, 100)}
# Fictional city: zone name, coordinates, road capacity (vehicles/hour), neighbouring zones
ZONES = {
    0: dict(name="Downtown CBD", lat=19.070, lon=72.870, cap=1200, nbrs=[1, 2]),
    1: dict(name="Tech Park", lat=19.110, lon=72.900, cap=1000, nbrs=[0, 3]),
    2: dict(name="Old Market", lat=19.040, lon=72.850, cap=800, nbrs=[0, 4]),
    3: dict(name="University", lat=19.135, lon=72.915, cap=700, nbrs=[1, 5]),
    4: dict(name="Mall District", lat=19.020, lon=72.880, cap=900, nbrs=[2, 5]),
    5: dict(name="Residential Suburb", lat=19.090, lon=72.940, cap=850, nbrs=[3, 4]),
}
