import json
import streamlit as st, pandas as pd, numpy as np, altair as alt, pydeck as pdk, os
from config import *
import streamlit.components.v1 as components
import model as M, simulate as S, privacy as P

st.set_page_config(page_title="Self-Evolving City Twin", page_icon="🏙️", layout="wide")
if "role" not in st.session_state: st.session_state.role = None
if not st.session_state.role:
    st.title("🏙️ Self-Evolving City Twin")
    u = st.text_input("Username"); pw = st.text_input("Password", type="password")
    if st.button("Login"):
        r = P.login(u, pw)
        if r: st.session_state.role = r; st.rerun()
        else: st.error("Invalid credentials")
    st.info("Demo logins: planner / plan123 (exact data)  |  public / pub123 (differential-privacy protected data)")
    st.stop()
role = st.session_state.role
names = {z: v["name"] for z, v in ZONES.items()}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

@st.cache_data
def history(): return M.load()
@st.cache_resource
def net(): return M.load_model("evolved")

eps = 1.0
with st.sidebar:
    st.success(f"Role: {role}")
    day = st.selectbox("Day of week", DOW); rain = st.slider("Rain (mm/h)", 0.0, 10.0, 0.0)
    ev = st.selectbox("Event in zone (evening)", [None] + list(ZONES), format_func=lambda z: "None" if z is None else names[z])
    if role == "public": eps = st.slider("Privacy budget ε (lower = more private)", 0.1, 5.0, 1.0)
    if st.button("Logout"): st.session_state.role = None; st.rerun()
view = (lambda v: P.dp_noise(v, eps)) if role == "public" else (lambda v: np.asarray(v, dtype=float))

dow = DOW.index(day)
g = pd.DataFrame([dict(zone_id=z, hour=h, is_weekend=int(dow >= 5), rain_mm=rain, is_event=int(ev == z and h >= 17))
                  for z in ZONES for h in range(24)])
g["congestion"] = M.predict(net(), g).clip(0)
g["vehicles"] = g.congestion * g.zone_id.map(lambda z: ZONES[z]["cap"]); g["zone"] = g.zone_id.map(names)

st.title("🏙️ Self-Evolving City Twin")
t1, t2, t3, t4, t5 = st.tabs(["🗺️ City now", "🧪 What-if simulator", "🧠 Self-evolving model", "🔒 Privacy & security", "🛠️ Data pipeline"])

with t1:
    hr = st.slider("Hour of day", 0, 23, 9)
    s = g[g.hour == hr].copy()
    s["lat"] = s.zone_id.map(lambda z: ZONES[z]["lat"]); s["lon"] = s.zone_id.map(lambda z: ZONES[z]["lon"])
    c = s.congestion.clip(0, 1); s["r"] = (255 * c).astype(int); s["gc"] = (255 * (1 - c)).astype(int)
    s["shown"] = view(s.vehicles.values).round(0)
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("ScatterplotLayer", s, get_position="[lon, lat]", get_radius="700 + 1800 * congestion",
                          get_fill_color="[r, gc, 60, 190]", pickable=True)],
        initial_view_state=pdk.ViewState(latitude=19.055, longitude=72.92, zoom=11.2),
        tooltip={"text": "{zone}\nVehicles/h: {shown}"}))
    hm = g.copy(); hm["shown"] = view(hm.vehicles.values)
    st.altair_chart(alt.Chart(hm).mark_rect().encode(
        x="hour:O", y=alt.Y("zone:N", title=None),
        color=alt.Color("shown:Q", title="Vehicles/h", scale=alt.Scale(scheme="redyellowgreen", reverse=True))),
        use_container_width=True)

with t2:
    st.subheader("🗺️ Road network map: live traffic + construction / road-closure what-if")
    st.caption("Real BKC-Bandra-Kurla-Sion roads (OpenStreetMap). Paste your TomTom key in the map panel for live traffic (needs internet). Scroll down for the demand-based what-if simulator.")
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_traffic_map.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            components.html(f.read(), height=760, scrolling=False)
    else:
        st.warning("live_traffic_map.html not found. Put it in the same folder as app.py.")
    st.divider()
    st.subheader("📊 Demand-based what-if: buses, road closure, new mall")
    c1, c2, c3 = st.columns(3)
    bz = c1.selectbox("Add buses in", list(ZONES), format_func=names.get, key="bz"); nb = c1.slider("Extra buses", 0, 10, 0)
    cz = c2.selectbox("Close road in", list(ZONES), format_func=names.get, key="cz"); cp = c2.slider("Road capacity lost %", 0, 80, 0)
    mz = c3.selectbox("New mall in", list(ZONES), format_func=names.get, key="mz"); ms = c3.slider("Mall size (000 sq m)", 0, 100, 0)
    sim = S.apply(g[["zone_id", "hour", "vehicles"]], {bz: nb} if nb else None, {cz: cp / 100} if cp else None, (mz, ms) if ms else None)
    sim["zone"] = sim.zone_id.map(names)
    ab, a_s = sim.cong_base.mean(), sim.congestion.mean()
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg congestion", f"{a_s:.2f}", f"{(a_s / ab - 1) * 100:+.1f}% vs baseline", delta_color="inverse")
    m2.metric("Worst zone peak", f"{sim.congestion.max():.2f}", f"{sim.congestion.max() - sim.cong_base.max():+.2f}", delta_color="inverse")
    m3.metric("Extra parking demand (cars)", int(view(sim.parking.max())))
    st.caption("Congestion = vehicles / road capacity (>1 means over capacity)")
    st.bar_chart(pd.DataFrame({"Baseline": sim.groupby("zone").cong_base.max(), "Scenario": sim.groupby("zone").congestion.max()}))
    zs = st.selectbox("Hourly detail for zone", list(ZONES), format_func=names.get, key="zs")
    st.line_chart(sim[sim.zone_id == zs].set_index("hour")[["cong_base", "congestion"]].rename(columns={"cong_base": "baseline", "congestion": "scenario"}))
    st.caption("Scenario effects are assumptions (see simulate.py) applied on top of the learned model.")

with t3:
    if os.path.exists(METRICS):
        with open(METRICS) as f: mt = json.load(f)
        k1, k2, k3 = st.columns(3)
        k1.metric("Simple-average baseline (MAE)", f"{mt['baseline_mae']:.4f}")
        k2.metric("Neural network (MAE)", f"{mt['model_mae']:.4f}")
        k3.metric("Error reduction vs baseline", f"{(1 - mt['model_mae'] / mt['baseline_mae']) * 100:.0f}%")
        if "model_mae_special" in mt:
            j1, j2, j3 = st.columns(3)
            j1.metric("Baseline on rain / event hours", f"{mt['baseline_mae_special']:.4f}")
            j2.metric("Neural network on rain / event hours", f"{mt['model_mae_special']:.4f}")
            j3.metric("Error reduction on rain / event hours", f"{(1 - mt['model_mae_special'] / mt['baseline_mae_special']) * 100:.0f}%")
        st.caption("Held-out days 52-59. Baseline = average congestion for the same zone, hour and weekday/weekend type. It ignores rain and events, which is where the neural network helps most.")
        st.divider()
    h = pd.read_csv(HIST)
    st.subheader("Prediction error on newly arriving data (lower is better)")
    st.write(f"A new mall opens in the mall-belt zone ({names[4]}) on day 60 (simulated event). A frozen model never adapts; the evolving model retrains on each new 15-day chunk.")
    st.line_chart(h.set_index("chunk")[["mae_mall_frozen", "mae_mall_evolving"]].rename(columns={"mae_mall_frozen": "Frozen model", "mae_mall_evolving": "Self-evolving model"}))
    st.dataframe(h, hide_index=True)
    d = history(); w = d[d.zone_id == 4].groupby(d.day // 7).congestion.mean().rename("Mall belt weekly congestion")
    st.line_chart(w)

with t4:
    d = history()
    st.markdown(f"""
- **Pseudonymization:** sensor IDs are salted-hashed, e.g. `{', '.join(d.sensor_pid.unique()[:3])}`
- **Encryption at rest:** raw data stored only as Fernet-encrypted file (`raw_traffic.enc` present: **{os.path.exists(RAW_ENC)}**)
- **Access control:** planner sees exact numbers, public sees Laplace-noised counts (ε = {eps if role == 'public' else 'n/a'})
- **k-anonymity:** smallest group size on (zone, hour) = **{P.k_anonymity(d, ['zone_id', 'hour'])}** records
""")
    x = g[g.hour == 9][["zone", "vehicles"]].set_index("zone")
    cmp = pd.DataFrame({"Public view (ε-noised)": P.dp_noise(x.vehicles.values, eps).round(0)}, index=x.index)
    if role == "planner": cmp.insert(0, "Exact (planner only)", x.vehicles.round(0))
    st.table(cmp)


with t5:
    st.subheader("Data engineering: pipeline quality report")
    if os.path.exists(QUALITY):
        with open(QUALITY) as f: q = json.load(f)
        k = st.columns(5)
        k[0].metric("Raw rows", f"{q['raw_rows']:,}"); k[1].metric("Duplicates removed", q["duplicates_removed"])
        k[2].metric("Sensor glitches removed", q["glitches_removed"]); k[3].metric("Gaps interpolated", q["values_imputed"])
        k[4].metric("Clean rows", f"{q['clean_rows']:,}")
        st.bar_chart(pd.DataFrame({"rows": [q["raw_rows"], q["clean_rows"]]}, index=["Raw", "Clean"]))
        st.table(pd.DataFrame({
            "Step": ["1. Decrypt raw file", "2. Remove duplicates", "3. Detect sensor glitches", "4. Fill gaps", "5. Feature engineering", "6. Pseudonymise and store"],
            "What happens": ["Raw data is kept only as a Fernet-encrypted file", f"{q['duplicates_removed']} repeated timestamp-zone rows dropped",
                             f"{q['glitches_removed']} readings above 2.2x road capacity marked invalid", f"{q['values_imputed']} missing/invalid values interpolated per zone (raw file had {q['missing_in_raw']} blanks)",
                             "hour, weekday, weekend/holiday, day index, congestion = vehicles / capacity", "Sensor IDs hashed, saved to SQLite (data/city.db)"]}))
        st.caption(f"Coverage: {q['zones']} zones, {q['days']} days ({q['start']} to {q['end']}). Null values left after cleaning: {q['remaining_nulls']}.")
    else:
        st.info("Report not found. Run: python run_all.py")
