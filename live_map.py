"""
03_live_map.py  (Member 1 - GIS lead)

Builds two self-contained interactive web pages in maps/:
  live_traffic_map.html  live TomTom traffic only (use in the 'City now' tab)
  whatif_map.html        live TomTom + closure what-if panel (needs scenarios.json)

  * Static layers come from Mumbai_Digital_Twin.gpkg (roads, rail, stations,
    bus stops, and congestion_risk if that layer exists with a 'risk_class'
    column containing Low / Moderate / High / Severe).
  * LIVE layers (traffic flow colours + incidents) come from the TomTom
    Traffic API, fetched by the browser each time the page refreshes.
    You paste a TomTom API key into the page. The key is NOT stored in the
    file, so the HTML is safe to share or commit.

Run (same folder as the GPKG):
    python 03_live_map.py
Then double-click maps/live_traffic_map.html

If live layers fail when opened by double-click (browser blocks file:// requests),
serve it instead:
    cd maps
    python -m http.server 8000
    -> open http://localhost:8000/live_traffic_map.html
"""

import json
import os

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping

GPKG = "Mumbai_Digital_Twin.gpkg"
OUT_DIR = "maps"
OUT_HTML = os.path.join(OUT_DIR, "live_traffic_map.html")
MAJOR = ["motorway", "trunk", "primary", "secondary", "tertiary"]
os.makedirs(OUT_DIR, exist_ok=True)


# ------------------------------------------------------------- helpers
def clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def rc(c, p):
    """round nested coordinate lists"""
    if len(c) > 0 and isinstance(c[0], (int, float)):
        return [round(x, p) for x in c]
    return [rc(x, p) for x in c]


def to_fc(gdf, props, prec=5, simplify=None):
    feats = []
    for _, r in gdf.iterrows():
        g = r.geometry
        if g is None or g.is_empty:
            continue
        if simplify:
            g = g.simplify(simplify, preserve_topology=True)
        gj = mapping(g)
        feats.append({
            "type": "Feature",
            "geometry": {"type": gj["type"], "coordinates": rc(gj["coordinates"], prec)},
            "properties": {k: clean(r[k]) for k in props if k in gdf.columns},
        })
    return {"type": "FeatureCollection", "features": feats}


def list_layers(path):
    try:
        import pyogrio
        return [l[0] for l in pyogrio.list_layers(path)]
    except Exception:
        try:
            import fiona
            return fiona.listlayers(path)
        except Exception:
            return []


# ---------------------------------------------------------------- load
study = gpd.read_file(GPKG, layer="study_area").to_crs("EPSG:4326")
roads = gpd.read_file(GPKG, layer="roads").to_crs("EPSG:4326")
rail = gpd.read_file(GPKG, layer="railway_lines").to_crs("EPSG:4326")
stations = gpd.read_file(GPKG, layer="railway_stations").to_crs("EPSG:4326")
bus = gpd.read_file(GPKG, layer="best_stops").to_crs("EPSG:4326")

roads["hw"] = (roads["highway"].fillna("residential")
               .str.split(";").str[0].str.strip()
               .str.replace("_link", "", regex=False))
roads = roads[roads["hw"].isin(MAJOR)].copy()

# two-way roads exist twice in the graph; keep one copy for display
n0 = len(roads)
roads["_k"] = roads.apply(
    lambda r: (min(r["u"], r["v"]), max(r["u"], r["v"]), int(round(r["length"]))), axis=1)
roads = roads.drop_duplicates("_k")
print(f"Roads shown: {len(roads)} (removed {n0 - len(roads)} reverse-direction duplicates)")

risk_fc = None
if "congestion_risk" in list_layers(GPKG):
    risk = gpd.read_file(GPKG, layer="congestion_risk").to_crs("EPSG:4326")
    if "risk_class" in risk.columns:
        score_col = next((c for c in ["risk_index", "risk_score", "index"]
                          if c in risk.columns), None)
        cols = ["road_id", "name", "risk_class"] + ([score_col] if score_col else [])
        if "u" in risk.columns and "v" in risk.columns:
            risk["_k"] = risk.apply(lambda r: (min(r["u"], r["v"]), max(r["u"], r["v"]),
                                               int(round(r.get("length", 0) or 0))), axis=1)
            risk = risk.drop_duplicates("_k")
        risk_fc = to_fc(risk, cols, simplify=0.00002)
        print("congestion_risk layer found and included")
    else:
        print("congestion_risk found but no 'risk_class' column; skipped")
else:
    print("No congestion_risk layer yet (that's fine; it appears once Member 2 adds it)")

roads_fc = to_fc(roads, ["road_id", "name", "hw", "length"], simplify=0.00002)
rail_fc = to_fc(rail, ["name", "railway"], simplify=0.00002)
st_fc = to_fc(stations, ["name"])
bus_fc = to_fc(bus, ["name"])
study_fc = to_fc(study, ["name"])

minx, miny, maxx, maxy = study.total_bounds
bounds = [[float(miny), float(minx)], [float(maxy), float(maxx)]]


def dumps(o):
    return json.dumps(o, separators=(",", ":"))


# ------------------------------------------------------------- template
SCEN_JSON = "data/processed/scenarios.json"
scen_obj = None
if os.path.exists(SCEN_JSON):
    with open(SCEN_JSON, encoding="utf-8") as f:
        scen_obj = json.load(f)
    print(f"What-if scenarios included: {len(scen_obj['scenarios'])}")
else:
    print("No scenarios.json yet: run 04_closure_scenarios.py to add the what-if panel")

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="strict-origin-when-cross-origin">
<title>BKC-Bandra-Kurla-Sion Traffic Digital Twin (prototype)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
html,body{height:100%;margin:0;font-family:system-ui,"Segoe UI",Arial,sans-serif}
#map{position:absolute;inset:0}
.banner{position:absolute;top:10px;left:50%;transform:translateX(-50%);z-index:1000;
  background:rgba(255,255,255,.96);padding:6px 16px;border-radius:8px;
  box-shadow:0 1px 6px rgba(0,0,0,.3);font-weight:600;font-size:15px;text-align:center;pointer-events:none}
.banner small{display:block;font-weight:400;font-size:11px;color:#555}
.panel{position:absolute;top:92px;left:10px;z-index:1000;width:292px;max-height:calc(100% - 110px);
  overflow:auto;background:rgba(255,255,255,.97);padding:10px 12px;border-radius:8px;
  box-shadow:0 1px 6px rgba(0,0,0,.3);font-size:13px;line-height:1.35}
.panel h3{margin:0 0 6px;font-size:14px}
.panel input[type=password],.panel input[type=text]{width:100%;box-sizing:border-box;padding:6px;margin:4px 0;
  border:1px solid #bbb;border-radius:4px}
.panel button{padding:6px 10px;border:0;border-radius:4px;background:#1f5fbf;color:#fff;cursor:pointer;margin-right:4px}
.panel button.sec{background:#666}
#status{margin-top:8px;font-size:12px;color:#155724}
#status.err{color:#b00020}
#inclist{margin:6px 0 0;padding-left:16px;font-size:12px}
#inclist li{cursor:pointer;margin-bottom:3px}
#inclist li:hover{text-decoration:underline}
.legend{position:absolute;bottom:22px;left:10px;z-index:1000;background:rgba(255,255,255,.96);
  padding:8px 10px;border-radius:8px;box-shadow:0 1px 6px rgba(0,0,0,.3);font-size:12px}
.legend div{display:flex;align-items:center;margin:2px 0}
.sw{display:inline-block;width:26px;height:0;border-top:3px solid;margin-right:6px}
.bar{height:10px;width:170px;border-radius:5px;background:linear-gradient(90deg,#7a0000,#e22503,#ffc105,#2ae57b)}
.note{font-size:11px;color:#555;margin-top:8px}
.panel select{width:100%;box-sizing:border-box;padding:5px;margin:4px 0;border:1px solid #bbb;border-radius:4px}
.panel hr{border:0;border-top:1px solid #ddd;margin:10px 0}
#wistats{margin-top:6px;font-size:12px;line-height:1.45}
#wistats .ifline{background:#fff4e5;border-left:3px solid #fd8d3c;padding:4px 6px;margin-bottom:4px}
.tag{display:inline-block;width:18px;height:0;border-top:4px solid;vertical-align:middle;margin-right:4px}
</style>
</head>
<body>
<div id="map"></div>
<div class="banner">BKC&ndash;Bandra&ndash;Kurla&ndash;Sion Urban Traffic Digital Twin
  <small>GIS-based prototype: road network + indicators + live traffic overlay</small></div>

<div class="panel">
  <h3>Live traffic (TomTom)</h3>
  <div>Paste your TomTom API key:</div>
  <input id="key" type="password" placeholder="TomTom API key" autocomplete="off">
  <div>
    <button id="go">Start live traffic</button>
    <button id="stop" class="sec">Stop</button>
  </div>
  <label style="display:block;margin-top:6px"><input id="auto" type="checkbox" checked>
    Auto-refresh every 2 min</label>
  <div id="status">Live layer is off. Static project layers are shown.</div>
  <div id="incbox" style="display:none">
    <b id="inccount"></b>
    <ol id="inclist"></ol>
  </div>
  <div class="note">Live flow/incidents are third-party TomTom data, shown as context.
    They are independent of the project's proxy risk index.</div>

  <div id="whatif" style="display:none">
    <hr>
    <h3>What-if: construction / road closure</h3>
    <div>IF this road segment has construction:</div>
    <select id="scen"></select>
    <div>
      <label><input type="radio" name="mode" value="full" checked> Full closure</label><br>
      <label><input type="radio" name="mode" value="partial"> Lane closure (open but slower)</label>
    </div>
    <div style="margin-top:4px">Show the effect on this trip:</div>
    <select id="od"></select>
    <label style="display:block;margin-top:4px"><input id="showgain" type="checkbox" checked>
      Highlight roads that gain rerouted trips</label>
    <div id="wistats"></div>
    <button id="wiclear" class="sec" style="margin-top:6px">Clear scenario</button>
    <div class="note">Network-level what-if route redistribution using equal-weight sampled trips.
      Results are relative (trips, metres), not vehicle counts or a traffic forecast.</div>
  </div>
</div>

<div class="legend">
  <b>Road class</b>
  <div><span class="sw" style="border-color:#4d004b;border-top-width:5px"></span>Motorway</div>
  <div><span class="sw" style="border-color:#b30000;border-top-width:5px"></span>Trunk</div>
  <div><span class="sw" style="border-color:#e34a33;border-top-width:4px"></span>Primary</div>
  <div><span class="sw" style="border-color:#fc8d59"></span>Secondary</div>
  <div><span class="sw" style="border-color:#2b8cbe;border-top-width:2px"></span>Tertiary</div>
  <div><span class="sw" style="border-color:#222;border-top-style:dashed;border-top-width:2px"></span>Railway</div>
  <div id="legend-live" style="display:none;flex-direction:column;align-items:flex-start;margin-top:6px">
    <b>Live traffic flow</b>
    <span class="bar"></span>
    <span style="display:flex;justify-content:space-between;width:170px"><span>heavy</span><span>free-flow</span></span>
  </div>
  <div id="legend-risk" style="display:none;flex-direction:column;align-items:flex-start;margin-top:6px">
    <b>Risk index (proxy)</b>
    <span><span class="sw" style="border-color:#2ca25f;border-top-width:4px"></span>Low</span>
    <span><span class="sw" style="border-color:#ffd92f;border-top-width:4px"></span>Moderate</span>
    <span><span class="sw" style="border-color:#fd8d3c;border-top-width:4px"></span>High</span>
    <span><span class="sw" style="border-color:#b10026;border-top-width:4px"></span>Severe</span>
  </div>
  <div style="margin-top:6px;color:#555">Data: &copy; OpenStreetMap contributors</div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const DATA = {roads: __ROADS__, rail: __RAIL__, stations: __STATIONS__,
              bus: __BUS__, study: __STUDY__, risk: __RISK__};
const BOUNDS = __BOUNDS__;   // [[south, west], [north, east]]

const map = L.map('map', {preferCanvas: true}).fitBounds(BOUNDS);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  {maxZoom: 19, subdomains: 'abcd',
   attribution: '&copy; OpenStreetMap contributors &copy; CARTO'}).addTo(map);

map.createPane('trafficPane');
map.getPane('trafficPane').style.zIndex = 450;
map.getPane('trafficPane').style.pointerEvents = 'none';
map.createPane('incPane');
map.getPane('incPane').style.zIndex = 500;

const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

// ---------------------------------------------------------- static layers
const COLOR  = {motorway:'#4d004b', trunk:'#b30000', primary:'#e34a33', secondary:'#fc8d59', tertiary:'#2b8cbe'};
const WEIGHT = {motorway:4, trunk:3.5, primary:3, secondary:2.2, tertiary:1.4};
const overlays = {};

['tertiary','secondary','primary','trunk','motorway'].forEach(hw => {
  const feats = DATA.roads.features.filter(f => f.properties.hw === hw);
  overlays['Roads: ' + hw + ' (' + feats.length + ')'] = L.geoJSON(
    {type:'FeatureCollection', features: feats},
    {style: () => ({color: COLOR[hw], weight: WEIGHT[hw], opacity: 0.85}),
     onEachFeature: (f, l) => l.bindTooltip(
       esc(f.properties.name || '(unnamed)') + ' | ' + esc(hw) + ' | ' +
       esc(Math.round(f.properties.length || 0)) + ' m | ' + esc(f.properties.road_id))
    }).addTo(map);
});

overlays['Railway'] = L.geoJSON(DATA.rail, {
  style: () => ({color:'#222', weight:1.6, dashArray:'6 4', opacity:0.9}),
  onEachFeature: (f, l) => l.bindTooltip(esc(f.properties.name || 'Railway'))
}).addTo(map);

overlays['Railway stations'] = L.geoJSON(DATA.stations, {
  pointToLayer: (f, ll) => L.circleMarker(ll, {radius:5, color:'#fff', weight:1.5,
                            fillColor:'#1f1f9e', fillOpacity:1}),
  onEachFeature: (f, l) => l.bindTooltip(esc(f.properties.name || 'Station'))
}).addTo(map);

overlays['Bus stops (OSM)'] = L.geoJSON(DATA.bus, {
  pointToLayer: (f, ll) => L.circleMarker(ll, {radius:2.5, color:'#666', weight:1,
                            fillColor:'#888', fillOpacity:0.8}),
  onEachFeature: (f, l) => l.bindTooltip(esc(f.properties.name || 'Bus stop'))
});

L.geoJSON(DATA.study, {style: () => ({color:'#000', weight:2, fill:false}), interactive:false}).addTo(map);

if (DATA.risk) {
  const RC = {Low:'#2ca25f', Moderate:'#ffd92f', High:'#fd8d3c', Severe:'#b10026'};
  overlays['Congestion/Risk Index (proxy)'] = L.geoJSON(DATA.risk, {
    style: f => ({color: RC[f.properties.risk_class] || '#999', weight: 3.5, opacity: 0.95}),
    onEachFeature: (f, l) => l.bindTooltip(esc(f.properties.name || '(unnamed)') +
      ' | risk: ' + esc(f.properties.risk_class))
  }).addTo(map);
  document.getElementById('legend-risk').style.display = 'flex';
}

L.control.layers(null, overlays, {collapsed: false, position: 'topright'}).addTo(map);

// ------------------------------------------------------------ live layer
const $ = id => document.getElementById(id);
const CAT = {0:'Unknown',1:'Accident',2:'Fog',3:'Dangerous conditions',4:'Rain',5:'Ice',6:'Jam',
             7:'Lane closed',8:'Road closed',9:'Road works',10:'Wind',11:'Flooding',14:'Broken-down vehicle'};
const MAG = {0:'Unknown',1:'Minor',2:'Moderate',3:'Major',4:'Undefined / closure'};
const MAGCOL = {0:'#777',1:'#ffd92f',2:'#fd8d3c',3:'#e31a1c',4:'#7a0177'};

let key = '', flowLayer = null, incLayer = null, timer = null, tileErrs = 0;
try { key = localStorage.getItem('tt_key') || ''; } catch (e) {}
$('key').value = key;

function setStatus(msg, isErr) {
  const s = $('status'); s.textContent = msg; s.className = isErr ? 'err' : '';
}
function flowUrl(k) {
  return 'https://api.tomtom.com/traffic/map/4/tile/flow/relative0/{z}/{x}/{y}.png?key=' +
         encodeURIComponent(k) + '&thickness=8&_=' + Date.now();
}

async function loadIncidents() {
  const fields = '{incidents{type,geometry{type,coordinates},properties{id,iconCategory,' +
                 'magnitudeOfDelay,events{description},from,to,length,delay,roadNumbers}}}';
  const b = BOUNDS;
  const url = 'https://api.tomtom.com/traffic/services/5/incidentDetails?key=' + encodeURIComponent(key) +
    '&bbox=' + [b[0][1], b[0][0], b[1][1], b[1][0]].join(',') +
    '&fields=' + encodeURIComponent(fields) +
    '&language=en-GB&timeValidityFilter=present';
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const j = await r.json();
    const feats = (j.incidents || []);
    if (incLayer) map.removeLayer(incLayer);
    const byId = {};
    incLayer = L.geoJSON({type:'FeatureCollection', features: feats}, {
      pane: 'incPane',
      style: f => ({color: MAGCOL[f.properties.magnitudeOfDelay] || '#777', weight: 6, opacity: 0.9}),
      pointToLayer: (f, ll) => L.circleMarker(ll, {pane:'incPane', radius: 8, color:'#fff', weight:2,
        fillColor: MAGCOL[f.properties.magnitudeOfDelay] || '#777', fillOpacity: 1}),
      onEachFeature: (f, l) => {
        const p = f.properties;
        const desc = (p.events && p.events[0] && p.events[0].description) || '';
        l.bindPopup('<b>' + esc(CAT[p.iconCategory] || 'Incident') + '</b> (' +
          esc(MAG[p.magnitudeOfDelay] || '') + ')<br>' + esc(desc) +
          (p.from ? '<br>From: ' + esc(p.from) : '') + (p.to ? '<br>To: ' + esc(p.to) : '') +
          (p.delay ? '<br>Delay: ~' + Math.round(p.delay / 60) + ' min' : '') +
          (p.length ? '<br>Length: ' + Math.round(p.length) + ' m' : ''));
        byId[p.id] = l;
      }
    }).addTo(map);

    feats.sort((a, b) => (b.properties.magnitudeOfDelay || 0) - (a.properties.magnitudeOfDelay || 0) ||
                         (b.properties.delay || 0) - (a.properties.delay || 0));
    $('incbox').style.display = 'block';
    $('inccount').textContent = feats.length + ' active incident(s) in study area';
    const ol = $('inclist'); ol.innerHTML = '';
    feats.slice(0, 8).forEach(f => {
      const p = f.properties, li = document.createElement('li');
      li.textContent = (CAT[p.iconCategory] || 'Incident') + ' - ' + (p.from || p.to || 'see map') +
                       (p.delay ? ' (~' + Math.round(p.delay / 60) + ' min)' : '');
      li.onclick = () => {
        const l = byId[p.id]; if (!l) return;
        if (l.getBounds) map.fitBounds(l.getBounds().pad(1), {maxZoom: 17});
        else map.setView(l.getLatLng(), 17);
        l.openPopup();
      };
      ol.appendChild(li);
    });
    return true;
  } catch (e) {
    setStatus('Incident feed unavailable (' + e.message + '). Flow tiles may still work.', true);
    return false;
  }
}

async function refresh() {
  const ok = await loadIncidents();
  if (flowLayer) flowLayer.setUrl(flowUrl(key));
  if (ok) setStatus('Live. Last updated ' + new Date().toLocaleTimeString() + '.', false);
}

async function startLive() {
  key = $('key').value.trim();
  if (!key) { setStatus('Enter a TomTom API key first.', true); return; }
  try { localStorage.setItem('tt_key', key); } catch (e) {}
  tileErrs = 0;
  if (flowLayer) map.removeLayer(flowLayer);
  flowLayer = L.tileLayer(flowUrl(key), {pane:'trafficPane', opacity:0.95, maxZoom:19,
                                         attribution:'Live traffic &copy; TomTom'});
  flowLayer.on('tileerror', () => {
    if (++tileErrs === 1) setStatus('Traffic tiles failed to load. Check the key, its permissions and quota.', true);
  });
  flowLayer.addTo(map);
  $('legend-live').style.display = 'flex';
  setStatus('Loading live data...', false);
  await refresh();
  clearInterval(timer);
  if ($('auto').checked) timer = setInterval(refresh, 120000);
}

function stopLive() {
  clearInterval(timer);
  if (flowLayer) { map.removeLayer(flowLayer); flowLayer = null; }
  if (incLayer) { map.removeLayer(incLayer); incLayer = null; }
  $('legend-live').style.display = 'none';
  $('incbox').style.display = 'none';
  setStatus('Live layer is off. Static project layers are shown.', false);
}

$('go').onclick = startLive;
$('stop').onclick = stopLive;

// --------------------------------------------------- what-if scenarios
const SCEN = __SCEN__;
if (SCEN) {
  map.createPane('wiPane');
  map.getPane('wiPane').style.zIndex = 520;
  const wiLayer = L.layerGroup().addTo(map);
  $('whatif').style.display = 'block';

  SCEN.scenarios.forEach((s, i) => {
    const o = document.createElement('option');
    o.value = i; o.textContent = s.id + ': ' + s.label;
    $('scen').appendChild(o);
  });
  SCEN.scenarios[0].od.forEach((t, i) => {
    const o = document.createElement('option');
    o.value = i; o.textContent = t.name;
    $('od').appendChild(o);
  });

  const km = m => (m / 1000).toFixed(2) + ' km';
  const gcol = t => t > 0.66 ? '#b10026' : (t > 0.33 ? '#fd8d3c' : '#ffd92f');
  const curMode = () => document.querySelector('input[name=mode]:checked').value;

  function renderWhatIf(fit) {
    wiLayer.clearLayers();
    const s = SCEN.scenarios[$('scen').selectedIndex];
    const mode = curMode();
    const r = s[mode], st = r.stats;
    const all = [];

    if ($('showgain').checked && r.gainers.length) {
      const mx = Math.max(...r.gainers.map(g => g.delta));
      r.gainers.forEach(g => L.polyline(g.coords,
        {pane: 'wiPane', color: gcol(g.delta / mx), weight: 6, opacity: 0.85})
        .bindTooltip(esc(g.name || '(unnamed)') + ': +' + g.delta + ' rerouted sampled trips')
        .addTo(wiLayer));
    }

    L.polyline(s.closed, {pane: 'wiPane', color: '#000', weight: 10, opacity: 0.9}).addTo(wiLayer);
    L.polyline(s.closed, {pane: 'wiPane', color: mode === 'full' ? '#ff0000' : '#ff9900',
                          weight: 5, dashArray: mode === 'full' ? null : '4 6'})
      .bindTooltip(esc(s.label) + (mode === 'full' ? ' - CLOSED' : ' - lane closure'))
      .addTo(wiLayer);
    s.closed.forEach(c => all.push(c));

    let tripHtml = '';
    const od = s.od[$('od').selectedIndex];
    if (od) {
      L.polyline(od.normal.coords, {pane: 'wiPane', color: '#1f5fbf', weight: 5,
                                    opacity: 0.8, dashArray: '2 9'})
        .bindTooltip('Normal route ' + km(od.normal.len)).addTo(wiLayer);
      od.normal.coords.forEach(c => all.push(c));
      const alt = od[mode];
      if (alt) {
        L.polyline(alt.coords, {pane: 'wiPane', color: '#d0117d', weight: 5, opacity: 0.9})
          .bindTooltip('Route under scenario ' + km(alt.len)).addTo(wiLayer);
        alt.coords.forEach(c => all.push(c));
        const ex = alt.len - od.normal.len;
        tripHtml = '<b>' + esc(od.name) + '</b><br>' +
          '<span class="tag" style="border-color:#1f5fbf"></span>normal ' + km(od.normal.len) + '<br>' +
          '<span class="tag" style="border-color:#d0117d"></span>scenario ' + km(alt.len) +
          ' (' + (ex >= 0 ? '+' : '') + Math.round(ex) + ' m' +
          (alt.same ? ', same route but slower' : '') + ')';
      } else {
        tripHtml = '<b>' + esc(od.name) + '</b><br>No route exists when this road is closed.';
      }
      const a = od.normal.coords[0], b = od.normal.coords[od.normal.coords.length - 1];
      [[a, 'Start'], [b, 'End']].forEach(x =>
        L.circleMarker(x[0], {pane: 'wiPane', radius: 7, color: '#fff', weight: 2,
                              fillColor: '#111', fillOpacity: 1})
          .bindTooltip(x[1]).addTo(wiLayer));
    }

    const cond = mode === 'full' ? 'closed'
      : 'reduced to slow lanes (cost x' + SCEN.meta.partial_factor + ')';
    $('wistats').innerHTML =
      '<div class="ifline"><b>IF</b> ' + esc(s.label) + ' is ' + cond + ' <b>THEN</b>:</div>' +
      '&bull; ' + st.exposed_pct + '% of ' + st.n_pairs + ' sampled through-trips normally use it (' +
        st.exposed + ')<br>' +
      '&bull; ' + st.rerouted + ' trips reroute' +
        (mode === 'full' ? ', ' + st.unreachable + ' have no route'
                         : ', ' + st.stay + ' stay on it (slower)') + '<br>' +
      '&bull; Mean detour: +' + st.mean_extra_m + ' m (median +' + st.median_extra_pct + '%)<br>' +
      (r.gainers.length ? '&bull; Roads gaining most: ' +
        r.gainers.slice(0, 3).map(g => esc(g.name || '(unnamed)') + ' (+' + g.delta + ')').join(', ') +
        '<br>' : '') +
      '<br>' + tripHtml;

    if (fit && all.length) map.fitBounds(L.latLngBounds(all).pad(0.1));
  }

  $('scen').onchange = () => renderWhatIf(true);
  $('od').onchange = () => renderWhatIf(true);
  document.querySelectorAll('input[name=mode]').forEach(r => r.onchange = () => renderWhatIf(false));
  $('showgain').onchange = () => renderWhatIf(false);
  $('wiclear').onclick = () => { wiLayer.clearLayers(); $('wistats').innerHTML = ''; };
  renderWhatIf(false);
}
</script>
</body>
</html>
"""

def render(scen):
    return (TEMPLATE
            .replace("__ROADS__", dumps(roads_fc))
            .replace("__RAIL__", dumps(rail_fc))
            .replace("__STATIONS__", dumps(st_fc))
            .replace("__BUS__", dumps(bus_fc))
            .replace("__STUDY__", dumps(study_fc))
            .replace("__RISK__", dumps(risk_fc) if risk_fc else "null")
            .replace("__BOUNDS__", dumps(bounds))
            .replace("__SCEN__", dumps(scen) if scen else "null"))


# 1) live-only page (for the 'City now' tab): same look, no what-if panel
# 2) what-if page (live TomTom + closure scenarios), only if scenarios.json exists
outputs = [("live_traffic_map.html", None)]
if scen_obj:
    outputs.append(("whatif_map.html", scen_obj))

for fname, scen in outputs:
    path = os.path.join(OUT_DIR, fname)
    html = render(scen)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")

print("Open a page in your browser, paste a TomTom API key, click 'Start live traffic'.")
