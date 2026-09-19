# Self-Evolving City Twin (mini project: DL + DSP + DE + DV)

## Run
```
pip install -r requirements.txt
python run_all.py        # generates data -> pipeline -> trains -> evolves (about 1-2 min)
streamlit run app.py     # login: planner/plan123  or  public/pub123
```

## Which file covers which subject
| Subject | Files | What to say |
|---|---|---|
| DE | data_gen.py, pipeline.py | Ingest (encrypted raw) -> dedup, glitch removal, interpolation -> feature engineering -> SQLite |
| DL | model.py | MLP with zone embedding + sin/cos hour features; test-then-train incremental retraining with replay |
| DSP | privacy.py | Hash pseudonyms, Fernet encryption at rest, Laplace differential privacy, k-anonymity, role-based access |
| DV | app.py | Map, heatmap, what-if charts, learning curve, KPI metrics |
| Simulation | simulate.py | Rule-based scenario engine on top of model predictions |

## Be honest in viva
- Data is **simulated** (traffic patterns + a mall opening on day 60). To use real data, replace data_gen.py output with a real CSV in the same columns (timestamp, zone_id, sensor_id, vehicles, rain_mm, is_event).
- What-if effects (bus relief 2.5%/bus, 60% diversion on closure, 12 cars per 1000 sq m for mall) are **assumptions** in `simulate.py`, tunable, not learned. Future work: learn them from real interventions.
- "Self-evolving" = periodic incremental retraining with replay buffer to reduce catastrophic forgetting.

## Likely viva questions
1. Why predict congestion ratio instead of raw vehicles? Generalizes across zones with different capacity.
2. Why replay old data? Avoids forgetting older patterns while adapting to the new mall.
3. What does epsilon mean? Privacy budget; smaller = more noise = stronger privacy, lower accuracy.
4. Why hash sensor IDs if zone is known? Prevents linking records to a specific device; salt stops rainbow-table reversal.
5. Limitation? Simulated data, single-city, rule-based scenarios, hourly granularity.
