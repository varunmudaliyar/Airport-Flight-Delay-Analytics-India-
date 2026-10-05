"""
Airport Flight Delay Analytics (India) - interactive capstone simulator  (UI v2)
Run:  streamlit run app.py
"""
import json
import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Flight Delay Analytics – India", page_icon="✈️", layout="wide")
HERE = Path(__file__).parent

# ----------------------------------------------------------------------------- constants
AIRPORT_NAMES = {
    "AGR": "Agra", "AJL": "Aizawl", "AKD": "Akola", "AMD": "Ahmedabad", "ATQ": "Amritsar", "BBI": "Bhubaneswar",
    "BDQ": "Vadodara", "BHJ": "Bhuj", "BHO": "Bhopal", "BHU": "Bhavnagar", "BKB": "Bikaner", "BLR": "Bengaluru",
    "BOM": "Mumbai", "CCU": "Kolkata", "CJB": "Coimbatore", "CNN": "Kannur", "COK": "Kochi", "DBR": "Darbhanga",
    "DED": "Dehradun", "DEL": "Delhi", "DHM": "Dharamshala", "DIB": "Dibrugarh", "DIU": "Diu", "DMU": "Dimapur",
    "GAU": "Guwahati", "GAY": "Gaya", "GOI": "Goa", "GOP": "Gorakhpur", "GWL": "Gwalior", "HBX": "Hubballi",
    "HYD": "Hyderabad", "IDR": "Indore", "IMF": "Imphal", "ISK": "Nashik", "IXA": "Agartala", "IXB": "Bagdogra",
    "IXC": "Chandigarh", "IXD": "Prayagraj", "IXE": "Mangaluru", "IXG": "Belagavi", "IXJ": "Jammu", "IXM": "Madurai",
    "IXR": "Ranchi", "IXU": "Aurangabad", "IXW": "Jamshedpur", "IXY": "Kandla", "IXZ": "Port Blair", "JAI": "Jaipur",
    "JDH": "Jodhpur", "JGA": "Jamnagar", "JLR": "Jabalpur", "JRH": "Jorhat", "JSA": "Jaisalmer", "KNU": "Kanpur",
    "KUU": "Kullu", "LEH": "Leh", "LKO": "Lucknow", "MAA": "Chennai", "MYQ": "Mysuru", "NAG": "Nagpur",
    "NDC": "Nanded", "PAT": "Patna", "PBD": "Porbandar", "PNQ": "Pune", "RAJ": "Rajkot", "RDP": "Durgapur",
    "REW": "Rewa", "RPR": "Raipur", "SHL": "Shillong", "SLV": "Shimla", "STV": "Surat", "SXR": "Srinagar",
    "SXV": "Salem", "TCR": "Thoothukudi", "TIR": "Tirupati", "TRV": "Thiruvananthapuram", "TRZ": "Tiruchirappalli",
    "UDR": "Udaipur", "VDY": "Vidyanagar", "VGA": "Vijayawada", "VNS": "Varanasi", "VTZ": "Visakhapatnam",
}
AIRLINE_PREFIX = {"IndiGo": "6E", "Air India": "AI", "Air India Express": "IX", "Akasa Air": "QP", "SpiceJet": "SG",
                  "Vistara": "UK", "Star Air": "S5", "Alliance Air": "9I", "Fly91": "IC"}
PEAK_HOURS = set(range(7, 11)) | set(range(17, 22))          # as found in the dataset
WEATHERS = ["Clear/Partly Cloudy", "Rain", "Heavy Rain"]
WEATHER_LABEL = {"Clear/Partly Cloudy": "☀️ Clear", "Rain": "🌧️ Rain", "Heavy Rain": "⛈️ Heavy rain"}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
NAVY, TEAL, RED, AMBER, GREEN, DEEP_RED = "#12385f", "#1a9aa8", "#d64545", "#f0a202", "#2e9d6b", "#a32020"
FEATURE_LABELS = {
    "Origin_Congestion_Index": "Origin congestion", "Turnaround_Risk_Index": "Turnaround risk",
    "Previous_Flight_Delay_Minutes": "Previous-flight delay", "Destination_Airport": "Destination airport",
    "Origin_Airport": "Origin airport", "Peak_Hour": "Peak hour", "Scheduled_Departure_Hour": "Departure hour",
    "Airline": "Airline", "Weather": "Weather type", "Rainfall_mm": "Rainfall", "Temperature_C": "Temperature",
    "Humidity_pct": "Humidity", "Wind_Speed_kmh": "Wind speed", "Visibility_km": "Visibility",
    "Cloud_Cover_pct": "Cloud cover", "Day_of_Week": "Day of week", "Month": "Month", "Is_Weekend": "Weekend",
    "Scheduled_Departure_Minute": "Departure minute",
}
PRESETS = {
    "🌧️ Monsoon evening · Mumbai": dict(f_airline="IndiGo", f_origin="BOM", f_dest="DEL", f_time=dt.time(18, 30),
                                        f_weather="Heavy Rain", prev=18, turn=42),
    "🚦 Delhi morning rush": dict(f_airline="Air India", f_origin="DEL", f_dest="BLR", f_time=dt.time(8, 45),
                                  f_weather="Clear/Partly Cloudy", prev=12, turn=40),
    "🌅 Calm early flight · Jaipur": dict(f_airline="SpiceJet", f_origin="JAI", f_dest="HYD", f_time=dt.time(6, 15),
                                          f_weather="Clear/Partly Cloudy", prev=0, turn=28),
    "⛰️ Quiet afternoon · Shimla": dict(f_airline="Alliance Air", f_origin="SLV", f_dest="DEL", f_time=dt.time(13, 30),
                                        f_weather="Rain", prev=2, turn=32),
}

def apt(code):
    return f"{code} – {AIRPORT_NAMES[code]}" if code in AIRPORT_NAMES else code

# ----------------------------------------------------------------------------- look & feel
pio.templates["fda"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, Roboto, sans-serif", color="#233448", size=13),
    colorway=[TEAL, NAVY, AMBER, RED, GREEN], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="#e9eff6", zeroline=False, linecolor="#d5dfea"),
    yaxis=dict(gridcolor="#e9eff6", zeroline=False, linecolor="#d5dfea"),
    margin=dict(l=8, r=8, t=46, b=8), title=dict(font=dict(size=15, color=NAVY), x=0, xanchor="left"),
    legend=dict(orientation="h", y=1.12, x=0)))
pio.templates.default = "plotly_white+fda"

def show(fig, **kw):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, **kw)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
:root{--navy:#12385f;--teal:#1a9aa8;--ink:#1b2b3c;--muted:#6b7c8f;--line:#e3eaf3;--soft:#f5f8fc;--red:#d64545;--green:#2e9d6b;--amber:#f0a202}
.stApp{font-family:'Inter','Segoe UI',Roboto,sans-serif;color:var(--ink)}
.block-container{padding-top:4.2rem;padding-bottom:3rem;max-width:1400px}
h1,h2,h3,h4{color:var(--navy);letter-spacing:-.01em}
h3{font-size:1.15rem!important;margin-top:.2rem!important}
/* hero */
.hero{display:flex;justify-content:space-between;align-items:center;gap:1.5rem;flex-wrap:wrap;
 background:radial-gradient(1200px 300px at 85% -40%,rgba(255,255,255,.18),transparent 60%),linear-gradient(115deg,#0f2f52 0%,#1b5d8a 60%,#1a9aa8 100%);
 color:#fff;padding:1.3rem 1.7rem;border-radius:18px;margin-bottom:1.1rem;box-shadow:0 10px 30px rgba(18,56,95,.18)}
.hero .eyebrow{font-size:.72rem;letter-spacing:.16em;opacity:.8;font-weight:600}
.hero h1{color:#fff;font-size:1.85rem;font-weight:800;margin:.15rem 0 .25rem 0;padding:0;line-height:1.15}
.hero h1 span{color:#8fe3ec}
.hero p{margin:0;opacity:.9;font-size:.95rem;max-width:560px}
.hero-stats{display:flex;gap:.7rem;flex-wrap:wrap}
.hero-stat{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.22);border-radius:12px;padding:.55rem .9rem;min-width:112px}
.hero-stat b{display:block;font-size:1.25rem;font-weight:800;line-height:1.1}
.hero-stat span{font-size:.72rem;opacity:.85;letter-spacing:.03em}
/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:.15rem;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{padding:.6rem 1rem;font-weight:600;border-radius:10px 10px 0 0}
/* chips + callouts */
.chips{margin:.1rem 0 .4rem 0}
.chip{display:inline-block;padding:.16rem .65rem;border-radius:99px;background:#eaf1f8;color:var(--navy);font-size:.78rem;font-weight:600;margin:0 .3rem .3rem 0}
.chip.hot{background:#fdeaea;color:#a32020}
.callout{background:linear-gradient(90deg,#eef7f8,#f5f8fc);border-left:4px solid var(--teal);border-radius:10px;padding:.7rem 1rem;margin:.4rem 0 1rem 0;font-size:.95rem}
/* cards */
div[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:14px;padding:.7rem 1rem;box-shadow:0 2px 10px rgba(18,56,95,.05)}
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] .card-title){border-radius:16px}
.card-title{font-size:.72rem;font-weight:700;letter-spacing:.12em;color:var(--muted);text-transform:uppercase;margin-bottom:.2rem}
.kpi{background:#fff;border:1px solid var(--line);border-radius:14px;padding:.8rem 1.05rem;box-shadow:0 2px 10px rgba(18,56,95,.05)}
.kpi .kl{font-size:.72rem;color:var(--muted);letter-spacing:.09em;text-transform:uppercase;font-weight:600}
.kpi .kv{font-size:1.75rem;font-weight:800;color:var(--navy);line-height:1.2}
.kpi .ks{font-size:.78rem;color:var(--muted)}
/* sticky result column on the predictor tab */
div[data-testid="stColumn"]:has(.sticky-anchor){position:sticky;top:4rem;align-self:flex-start;max-height:calc(100vh - 4.5rem);overflow-y:auto;padding-right:.2rem}
/* boarding pass */
.pass{background:#fff;border:1px solid var(--line);border-radius:18px;box-shadow:0 8px 26px rgba(18,56,95,.10);overflow:hidden;margin-bottom:.7rem}
.pass-head{display:flex;justify-content:space-between;align-items:center;padding:.6rem 1.2rem;background:var(--navy);color:#fff;font-size:.8rem;letter-spacing:.06em;font-weight:600}
.pass-route{display:flex;align-items:center;justify-content:space-between;padding:1.05rem 1.4rem .3rem}
.pass .code{font-size:2.6rem;font-weight:800;color:var(--navy);line-height:1;letter-spacing:.02em}
.pass .city{color:var(--muted);font-size:.85rem;margin-top:.15rem}
.path{flex:1;margin:0 1rem;position:relative;height:2px;background:repeating-linear-gradient(90deg,#b8c7d8 0 6px,transparent 6px 12px)}
.path:after{content:"✈";position:absolute;left:50%;top:-.8rem;transform:translateX(-50%);background:#fff;padding:0 .4rem;color:var(--teal);font-size:1.15rem}
.pass-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem;padding:.7rem 1.4rem 1rem}
.pass-grid label{display:block;font-size:.66rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;font-weight:600}
.pass-grid b{font-size:1.4rem;font-weight:800;color:var(--ink)}
.perf{border-top:2px dashed #d3deea;margin:0 1.1rem}
.pass-foot{display:flex;align-items:center;gap:1.3rem;padding:1rem 1.4rem 1.15rem}
.stamp{border:3px solid;border-radius:10px;padding:.35rem .8rem;font-weight:800;letter-spacing:.09em;transform:rotate(-4deg);font-size:1.02rem;white-space:nowrap}
.meter-wrap{flex:1}
.meter-top{display:flex;justify-content:space-between;font-size:.78rem;color:var(--muted);font-weight:600;margin-bottom:.4rem}
.meter-top b{color:var(--ink);font-size:.95rem}
.meter{position:relative;height:10px;border-radius:99px;background:linear-gradient(90deg,#2e9d6b 0 35%,#f0a202 35% 65%,#d64545 65% 100%)}
.meter .pin{position:absolute;top:-4px;width:18px;height:18px;border-radius:50%;background:#fff;border:3px solid var(--navy);transform:translateX(-50%);box-shadow:0 2px 6px rgba(0,0,0,.25)}
.meter .thr{position:absolute;top:-6px;width:2px;height:22px;background:#1b2b3c;opacity:.55}
.ticks{display:flex;justify-content:space-between;font-size:.68rem;color:var(--muted);margin-top:.35rem}
/* departures board */
.fids{background:#0a1626;border-radius:14px;max-height:460px;overflow:auto;border:1px solid #16304f}
.fids table{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',Consolas,monospace;font-size:.85rem}
.fids th{position:sticky;top:0;background:#0a1626;color:#7f95ad;text-align:left;font-weight:700;padding:.65rem .8rem;letter-spacing:.09em;font-size:.7rem;text-transform:uppercase;border-bottom:1px solid #16304f}
.fids td{padding:.5rem .8rem;color:#e7eef7;border-bottom:1px solid #0f2238;white-space:nowrap}
.fids tr:hover td{background:#0e2035}
.fids td.t{color:#ffd166;font-weight:700}
.fids td.dim{color:#8ea3ba}
.pill{padding:.14rem .6rem;border-radius:99px;font-size:.72rem;font-weight:700;letter-spacing:.04em}
.pill.ok{background:#12372a;color:#5fe0a2}.pill.warn{background:#3d2f0b;color:#ffc857}.pill.bad{background:#40161a;color:#ff7b82}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading trained models…")
def load_models():
    b = joblib.load(HERE / "flight_delay_models.joblib")
    return b["classifier"], b["regressor"], b["features"]

@st.cache_data
def load_assets():
    return json.loads((HERE / "assets" / "assets.json").read_text())

try:
    clf, reg, FEATURES = load_models()
    A = load_assets()
except Exception as e:                                             # most common cause: scikit-learn version mismatch
    import sklearn
    st.error("Could not load the model or assets.")
    st.code(f"{type(e).__name__}: {e}")
    st.info(f"Installed scikit-learn: {sklearn.__version__}. The model must be loaded with the SAME version used in Colab. "
            "In Colab run `import sklearn; print(sklearn.__version__)` and put that exact version in requirements.txt.")
    st.stop()

AIRPORTS, AIRLINES = A["airports"], A["airlines"]
N_HUBS = sum(1 for v in A["airport_congestion"].values() if v["offpeak"] >= 55)
K, M = A["kpi"], A["metrics"]

st.markdown(
    '<div class="hero"><div><div class="eyebrow">DATA SCIENCE &amp; APPLICATION · CAPSTONE PROJECT</div>'
    '<h1>Airport Flight Delay Analytics <span>India</span></h1>'
    '<p>Predict, explore and simulate departure delays across Indian airports using machine learning.</p></div>'
    f'<div class="hero-stats"><div class="hero-stat"><b>{K["flights"]:,}</b><span>flights analysed</span></div>'
    f'<div class="hero-stat"><b>{K["airports"]}</b><span>origin airports</span></div>'
    f'<div class="hero-stat"><b>{M["roc_auc"]:.2f}</b><span>model ROC-AUC</span></div></div></div>',
    unsafe_allow_html=True)

# ----------------------------------------------------------------------------- model helpers
def predict(df: pd.DataFrame):
    X = df[FEATURES]
    return clf.predict_proba(X)[:, 1], np.clip(reg.predict(X), 0, None)

def one(base: dict, **over):
    p, m = predict(pd.DataFrame([{**base, **over}]))
    return float(p[0]), float(m[0])

def cong_default(code, hour):
    prof = A["airport_congestion"].get(code, {"offpeak": 50, "peak": 70})
    return prof["peak"] if hour in PEAK_HOURS else prof["offpeak"]

def status_of(prob, mins, thr):
    if prob < thr:
        return "ON TIME", GREEN
    return ("SEVERE DELAY", DEEP_RED) if mins >= 60 else ("DELAYED", RED)

def kpi(label, value, sub=""):
    return f'<div class="kpi"><div class="kl">{label}</div><div class="kv">{value}</div><div class="ks">{sub}</div></div>'

# ----------------------------------------------------------------------------- session defaults / presets
DEFAULTS = dict(f_airline="IndiGo", f_origin="BOM", f_dest="DEL", f_time=dt.time(18, 30), f_weather="Clear/Partly Cloudy",
                f_date=dt.date.today(), prev=5, turn=int(round(A["defaults"]["turnaround_mean"])))
for k_, v_ in DEFAULTS.items():
    st.session_state.setdefault(k_, v_)

def apply_preset(p):
    st.session_state.update(p)

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    thr = st.slider("Decision threshold", 0.10, 0.95, 0.50, 0.05,
                    help="A flight is flagged DELAYED when its risk score is at or above this value. "
                         "Higher = fewer false alarms; lower = catch more delays.")
    st.caption("~87 % of flights in the dataset are delayed, so the score is a **relative risk ranking**, not a calibrated probability.")
    st.divider()
    st.markdown("**Models**")
    st.caption("Classifier: Random Forest  \nDelay minutes: Gradient Boosting  \n"
               f"Trained on 2023-24 · tested on 2025 ({M['n_test']:,} flights)")

tab_pred, tab_whatif, tab_day, tab_dash, tab_model = st.tabs(
    ["🎯 Flight predictor", "🎛️ What-if explorer", "🛫 Airport day simulator", "📊 Analytics", "🧠 Model & method"])

# ============================================================================= 1. PREDICTOR
with tab_pred:
    st.caption("Try a ready-made scenario, or set up your own flight below:")
    pcols = st.columns(len(PRESETS))
    for col_, (name_, p_) in zip(pcols, PRESETS.items()):
        col_.button(name_, on_click=apply_preset, args=(p_,), width="stretch", key=f"preset_{name_}")

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="card-title">✈ Flight</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            airline = c1.selectbox("Airline", AIRLINES, key="f_airline")
            origin = c2.selectbox("From", AIRPORTS, format_func=apt, key="f_origin")
            dest_opts = [a for a in AIRPORTS if a != origin]
            if st.session_state.f_dest not in dest_opts:
                st.session_state.f_dest = dest_opts[0]
            destination = c3.selectbox("To", dest_opts, format_func=apt, key="f_dest")
            d1, d2, d3 = st.columns(3)
            flight_date = d1.date_input("Date", key="f_date")
            dep_time = d2.time_input("Departure", key="f_time", step=300)
            weather = d3.selectbox("Weather", WEATHERS, format_func=WEATHER_LABEL.get, key="f_weather")

            hour, minute = dep_time.hour, dep_time.minute
            dow, month = flight_date.weekday(), flight_date.month
            weekend, peak = int(dow >= 5), int(hour in PEAK_HOURS)
            chips = (f'<span class="chip">{DOW[dow]}</span><span class="chip">{"Weekend" if weekend else "Weekday"}</span>'
                     f'<span class="chip {"hot" if peak else ""}">{"Peak hour" if peak else "Off-peak"}</span>')
            if hour < 5:
                chips += '<span class="chip hot">Outside training hours (05–23)</span>'
            st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="card-title">🛠 Airport operations <span style="font-weight:500;text-transform:none;letter-spacing:0">'
                        '· congestion is auto-set from the airport &amp; hour – adjust freely</span></div>', unsafe_allow_html=True)
            o1, o2, o3 = st.columns(3)
            cong = o1.slider("Origin congestion", 0, 100, int(round(cong_default(origin, hour))), key=f"cong_{origin}_{peak}")
            prev = o2.slider("Previous flight delay (min)", 0, 71, key="prev")
            turn = o3.slider("Turnaround risk", 0, 90, key="turn")
            wp, mp = A["weather_profile"][weather], A["month_profile"][str(month)]
            with st.expander("Detailed weather conditions"):
                w1, w2, w3 = st.columns(3)
                temp = w1.slider("Temperature (°C)", 10.0, 45.0, float(np.clip(mp["Temperature_C"], 10, 45)), 0.5, key=f"t{month}")
                hum = w2.slider("Humidity (%)", 20.0, 100.0, float(np.clip(mp["Humidity_pct"], 20, 100)), 1.0, key=f"h{month}")
                wind = w3.slider("Wind (km/h)", 0.0, 35.0, float(np.clip(mp["Wind_Speed_kmh"], 0, 35)), 0.5, key=f"w{month}")
                vis = w1.slider("Visibility (km)", 5.0, 15.0, float(np.clip(wp["Visibility_km"], 5, 15)), 0.1, key=f"v{weather}")
                rain = w2.slider("Rainfall (mm)", 0.0, 30.0, float(np.clip(wp["Rainfall_mm"], 0, 30)), 0.1, key=f"r{weather}")
                cloud = w3.slider("Cloud cover (%)", 0.0, 100.0, float(np.clip(wp["Cloud_Cover_pct"], 0, 100)), 1.0, key=f"c{weather}")

    base = dict(Scheduled_Departure_Hour=hour, Scheduled_Departure_Minute=minute, Day_of_Week=dow, Month=month,
                Is_Weekend=weekend, Peak_Hour=peak, Temperature_C=temp, Humidity_pct=hum, Wind_Speed_kmh=wind,
                Visibility_km=vis, Rainfall_mm=rain, Cloud_Cover_pct=cloud, Origin_Congestion_Index=float(cong),
                Previous_Flight_Delay_Minutes=prev, Turnaround_Risk_Index=float(turn), Airline=airline, Weather=weather,
                Origin_Airport=origin, Destination_Airport=destination)
    prob, mins = one(base)
    label, color = status_of(prob, mins, thr)

    with right:
        st.markdown('<div class="sticky-anchor"></div>', unsafe_allow_html=True)
        eta = dt.datetime.combine(flight_date, dep_time) + dt.timedelta(minutes=round(mins))
        eta_str = f"{eta:%H:%M}" + (" +1d" if eta.date() > flight_date else "")
        pin, tp = max(2.0, min(98.0, prob * 100)), thr * 100
        st.markdown(
            f'<div class="pass"><div class="pass-head"><span>{airline.upper()} · {AIRLINE_PREFIX.get(airline, "")}</span>'
            f'<span>{flight_date:%a %d %b %Y}</span></div>'
            f'<div class="pass-route"><div><div class="code">{origin}</div><div class="city">{AIRPORT_NAMES.get(origin, "")}</div></div>'
            f'<div class="path"></div>'
            f'<div style="text-align:right"><div class="code">{destination}</div><div class="city">{AIRPORT_NAMES.get(destination, "")}</div></div></div>'
            f'<div class="pass-grid"><div><label>Scheduled</label><b>{dep_time:%H:%M}</b></div>'
            f'<div><label>Expected</label><b style="color:{color}">{eta_str}</b></div>'
            f'<div><label>Delay</label><b style="color:{color}">{"+" if mins >= 1 else ""}{mins:.0f} min</b></div></div>'
            f'<div class="perf"></div>'
            f'<div class="pass-foot"><div class="stamp" style="color:{color};border-color:{color}">{label}</div>'
            f'<div class="meter-wrap"><div class="meter-top"><span>Delay risk score</span><b>{prob * 100:.0f}%</b></div>'
            f'<div class="meter"><div class="thr" style="left:{tp}%"></div><div class="pin" style="left:{pin}%"></div></div>'
            f'<div class="ticks"><span>Low</span><span>Moderate</span><span>High</span></div></div></div></div>',
            unsafe_allow_html=True)

        clear = A["weather_profile"]["Clear/Partly Cloudy"]
        cf = {
            "Congestion normalised (40)": dict(Origin_Congestion_Index=40.0),
            "No previous-flight delay": dict(Previous_Flight_Delay_Minutes=0),
            "Smooth turnaround (25)": dict(Turnaround_Risk_Index=25.0),
            "Off-peak time slot": dict(Peak_Hour=0),
            "Clear weather": dict(Weather="Clear/Partly Cloudy", Rainfall_mm=clear["Rainfall_mm"],
                                  Visibility_km=clear["Visibility_km"], Cloud_Cover_pct=clear["Cloud_Cover_pct"]),
        }
        drv = pd.DataFrame([{"Change": k, "Minutes saved": max(0.0, mins - one(base, **v)[1])} for k, v in cf.items()]).sort_values("Minutes saved")
        fig = px.bar(drv, x="Minutes saved", y="Change", orientation="h", text="Minutes saved", height=230,
                     title="What would cut this delay?  (minutes saved)")
        fig.update_traces(marker_color=TEAL, texttemplate="%{text:.0f}", textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis_title=None, xaxis_title=None, margin=dict(l=0, r=30, t=40, b=0))
        with st.container(border=True):
            show(fig)
            top = drv.iloc[-1]
            if top["Minutes saved"] >= 1:
                st.caption(f"Biggest lever: **{top['Change'].lower()}** (≈ {top['Minutes saved']:.0f} min less delay).")
            else:
                st.caption("No single factor adds much delay – conditions are already favourable.")

# ============================================================================= 2. WHAT-IF
with tab_whatif:
    st.markdown("### Sweep one factor")
    st.caption("Everything else stays as set in the *Flight predictor* tab.")
    sweeps = {
        "Origin congestion index": ("Origin_Congestion_Index", np.arange(0, 101, 2.5)),
        "Previous flight delay (min)": ("Previous_Flight_Delay_Minutes", np.arange(0, 72, 2)),
        "Turnaround risk index": ("Turnaround_Risk_Index", np.arange(0, 91, 2.5)),
        "Scheduled departure hour": ("Scheduled_Departure_Hour", np.arange(5, 24, 1)),
        "Rainfall (mm)": ("Rainfall_mm", np.arange(0, 30.5, 1)),
        "Visibility (km)": ("Visibility_km", np.arange(5, 15.1, 0.5)),
    }
    choice = st.selectbox("Factor to vary", list(sweeps))
    col, values = sweeps[choice]
    grid = pd.DataFrame([base] * len(values))
    grid[col] = values
    if col == "Scheduled_Departure_Hour":
        grid["Peak_Hour"] = [int(h in PEAK_HOURS) for h in values]
        grid["Origin_Congestion_Index"] = [cong_default(origin, int(h)) for h in values]
    p_, m_ = predict(grid)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=values, y=p_ * 100, name="Risk score (%)", line=dict(color=RED, width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=values, y=m_, name="Predicted delay (min)", line=dict(color=NAVY, width=3, dash="dot")), secondary_y=True)
    fig.add_vline(x=base[col], line_dash="dash", line_color="#8a99aa", annotation_text="current")
    fig.update_yaxes(title_text="Risk score (%)", secondary_y=False, range=[0, 100])
    fig.update_yaxes(title_text="Delay (min)", secondary_y=True, showgrid=False, range=[0, max(60.0, float(m_.max()) * 1.1)], tickformat=".0f")
    fig.update_layout(height=400, xaxis_title=choice)
    with st.container(border=True):
        show(fig)
    if col == "Scheduled_Departure_Hour":
        st.caption("For the hour sweep, the peak-hour flag and the airport's typical congestion are updated for each hour.")

    st.markdown("### Compare two scenarios")
    sig = f"{origin}{hour}{int(cong)}{prev}{turn}{weather}"
    ca, cb = st.columns(2, gap="large")
    with ca, st.container(border=True):
        st.markdown('<div class="card-title">Scenario A · current settings</div>', unsafe_allow_html=True)
        x1, x2 = st.columns(2)
        x1.metric("Risk score", f"{prob * 100:.0f} %")
        x2.metric("Predicted delay", f"{mins:.0f} min")
        st.caption(f"{apt(origin)} · congestion {int(cong)} · previous delay {prev} min · turnaround {turn} · {WEATHER_LABEL[weather]}")
    with cb, st.container(border=True):
        st.markdown('<div class="card-title">Scenario B · change it</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        b_cong = b1.slider("Congestion", 0, 100, int(cong), key=f"b_cong_{sig}")
        b_prev = b2.slider("Prev. delay", 0, 71, int(prev), key=f"b_prev_{sig}")
        b_turn = b3.slider("Turnaround", 0, 90, int(turn), key=f"b_turn_{sig}")
        b_weather = st.segmented_control("Weather ", WEATHERS, format_func=WEATHER_LABEL.get, default=weather, key=f"b_w_{sig}") or weather
        bw = A["weather_profile"][b_weather]
        over = dict(Origin_Congestion_Index=float(b_cong), Previous_Flight_Delay_Minutes=b_prev, Turnaround_Risk_Index=float(b_turn),
                    Weather=b_weather, Rainfall_mm=bw["Rainfall_mm"], Visibility_km=bw["Visibility_km"], Cloud_Cover_pct=bw["Cloud_Cover_pct"])
        pb, mb = one(base, **over)
        y1, y2 = st.columns(2)
        y1.metric("Risk score", f"{pb * 100:.0f} %", delta=f"{(pb - prob) * 100:+.0f} pts", delta_color="inverse")
        y2.metric("Predicted delay", f"{mb:.0f} min", delta=f"{mb - mins:+.0f} min", delta_color="inverse")

# ============================================================================= 3. AIRPORT DAY SIMULATOR
with tab_day:
    st.markdown("### Simulate a day of departures")
    st.caption("Random flights are generated from the real airline, destination and time-of-day mix in the dataset, then scored "
               "by the trained models. Use the disruption slider to stress-test the airport. "
               f"Note: the {N_HUBS} high-congestion hub airports in this dataset (Delhi, Mumbai, Bengaluru…) are almost always congested, so nearly "
               "every simulated flight there is flagged delayed – smaller airports show the disruption effect more clearly.")
    with st.container(border=True):
        d1, d2, d3, d4 = st.columns(4)
        sim_airport = d1.selectbox("Airport", AIRPORTS, index=AIRPORTS.index("JAI"), format_func=apt, key="sim_apt")
        sim_date = d2.date_input("Date", dt.date.today(), key="sim_date")
        sim_weather = d3.selectbox("Weather", WEATHERS, format_func=WEATHER_LABEL.get, key="sim_w")
        n_fl = d4.slider("Departures", 20, 200, 80, 10)
        e1, e2 = st.columns([3, 1])
        extra = e1.slider("🚧 Disruption – extra congestion (runway works, ATC restriction…)", 0, 40, 0)
        if "seed" not in st.session_state:
            st.session_state.seed = 7
        e2.write("")
        if e2.button("🎲 New random day", width="stretch"):
            st.session_state.seed += 1

    rng = np.random.default_rng(st.session_state.seed)
    hrs = np.array(sorted(map(int, A["hour_share"])))
    hp = np.array([A["hour_share"][str(h)] for h in hrs]); hp = hp / hp.sum()
    al = np.array(list(A["airline_share"])); alp = np.array(list(A["airline_share"].values())); alp = alp / alp.sum()
    dests = [d for d in A["dest_share"] if d != sim_airport]
    dp = np.array([A["dest_share"][d] for d in dests]); dp = dp / dp.sum()
    wp2, mp2 = A["weather_profile"][sim_weather], A["month_profile"][str(sim_date.month)]
    hour_s = rng.choice(hrs, n_fl, p=hp)
    peak_s = np.array([int(h in PEAK_HOURS) for h in hour_s])
    cong_s = np.array([cong_default(sim_airport, int(h)) for h in hour_s]) + rng.normal(0, 8, n_fl) + extra
    sim = pd.DataFrame({
        "Scheduled_Departure_Hour": hour_s, "Scheduled_Departure_Minute": rng.integers(0, 60, n_fl),
        "Day_of_Week": sim_date.weekday(), "Month": sim_date.month, "Is_Weekend": int(sim_date.weekday() >= 5), "Peak_Hour": peak_s,
        "Temperature_C": np.clip(rng.normal(mp2["Temperature_C"], 2, n_fl), 10, 45),
        "Humidity_pct": np.clip(rng.normal(mp2["Humidity_pct"], 6, n_fl), 20, 100),
        "Wind_Speed_kmh": np.clip(rng.normal(mp2["Wind_Speed_kmh"], 2, n_fl), 0, 35),
        "Visibility_km": np.clip(rng.normal(wp2["Visibility_km"], 0.8, n_fl), 5, 15),
        "Rainfall_mm": np.clip(rng.normal(wp2["Rainfall_mm"], max(wp2["Rainfall_mm"] * .3, .3), n_fl), 0, 30),
        "Cloud_Cover_pct": np.clip(rng.normal(wp2["Cloud_Cover_pct"], 10, n_fl), 0, 100),
        "Origin_Congestion_Index": np.clip(cong_s, 0, 100),
        "Previous_Flight_Delay_Minutes": np.clip(rng.exponential(A["defaults"]["prev_delay_mean"], n_fl).round(), 0, 71).astype(int),
        "Turnaround_Risk_Index": np.clip(rng.normal(A["defaults"]["turnaround_mean"], A["defaults"]["turnaround_std"], n_fl), 0.4, 88),
        "Airline": rng.choice(al, n_fl, p=alp), "Weather": sim_weather,
        "Origin_Airport": sim_airport, "Destination_Airport": rng.choice(dests, n_fl, p=dp)})
    ps, ms = predict(sim)
    sim["risk"], sim["mins"] = ps, ms
    sim["delayed"] = ps >= thr
    sim["status"] = np.where(~sim.delayed, "On time", np.where(ms >= 60, "Severe delay", "Delayed"))

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Departures", n_fl, apt(sim_airport)), unsafe_allow_html=True)
    k2.markdown(kpi("Flagged delayed", f"{sim.delayed.mean() * 100:.0f}%", f"threshold {thr:.2f}"), unsafe_allow_html=True)
    k3.markdown(kpi("Average delay", f"{sim.mins.mean():.0f} min", "predicted"), unsafe_allow_html=True)
    k4.markdown(kpi("Severe delays", int((sim.status == "Severe delay").sum()), "60 min or more"), unsafe_allow_html=True)
    st.write("")

    hourly = sim.groupby("Scheduled_Departure_Hour").agg(avg=("mins", "mean"), flights=("mins", "size")).reset_index()
    fig = px.bar(hourly, x="Scheduled_Departure_Hour", y="avg", color="avg", color_continuous_scale=["#9ad7c1", AMBER, RED],
                 hover_data=["flights"], height=300, labels={"Scheduled_Departure_Hour": "Hour of day", "avg": "Avg predicted delay (min)"},
                 title=f"Predicted delay through the day – {apt(sim_airport)}")
    fig.update_layout(coloraxis_showscale=False)
    with st.container(border=True):
        show(fig)

    board = sim.sort_values(["Scheduled_Departure_Hour", "Scheduled_Departure_Minute"]).copy()
    board["Flight"] = [f"{AIRLINE_PREFIX.get(a, 'XX')} {rng.integers(100, 9999)}" for a in board.Airline]
    board["Scheduled"] = [f"{h:02d}:{m:02d}" for h, m in zip(board.Scheduled_Departure_Hour, board.Scheduled_Departure_Minute)]
    st.markdown("### Departures board")
    st.caption("Flight numbers are illustrative.")
    rows = []
    for r in board.itertuples():
        exp = (dt.datetime(2000, 1, 1, r.Scheduled_Departure_Hour, r.Scheduled_Departure_Minute) + dt.timedelta(minutes=round(r.mins)))
        cls = {"On time": "ok", "Delayed": "warn", "Severe delay": "bad"}[r.status]
        rows.append(f"<tr><td class='t'>{r.Scheduled}</td><td class='t'>{exp:%H:%M}</td><td>{r.Flight}</td>"
                    f"<td>{apt(r.Destination_Airport)}</td><td class='dim'>{r.Airline}</td>"
                    f"<td class='dim'>{r.risk * 100:.0f}%</td><td>+{r.mins:.0f} min</td><td><span class='pill {cls}'>{r.status.upper()}</span></td></tr>")
    st.markdown("<div class='fids'><table><thead><tr><th>Sched.</th><th>Expected</th><th>Flight</th><th>Destination</th><th>Airline</th>"
                "<th>Risk</th><th>Delay</th><th>Status</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>",
                unsafe_allow_html=True)

# ============================================================================= 4. ANALYTICS
with tab_dash:
    st.markdown("### What the historical data says")
    st.caption(f"{K['date_from']} → {K['date_to']}")
    a, b, c, d = st.columns(4)
    a.markdown(kpi("Flights analysed", f"{K['flights']:,}", f"{K['airports']} airports · {K['airlines']} airlines"), unsafe_allow_html=True)
    b.markdown(kpi("Share delayed", f"{K['delay_rate'] * 100:.1f}%", "labelled delayed in the data"), unsafe_allow_html=True)
    c.markdown(kpi("Avg delay · all flights", f"{K['avg_delay_all']:.1f} min"), unsafe_allow_html=True)
    d.markdown(kpi("Avg delay · delayed only", f"{K['avg_delay_when_delayed']:.1f} min"), unsafe_allow_html=True)

    cgs = A["by_congestion"]
    st.markdown(f"<div class='callout'>💡 <b>Key insight:</b> departures from very congested airports run "
                f"<b>{cgs[-1]['avg_delay'] - cgs[0]['avg_delay']:.0f} minutes later</b> on average than from low-congestion airports "
                f"({cgs[-1]['avg_delay']:.0f} vs {cgs[0]['avg_delay']:.0f} min). Airline differences are tiny and weather effects are smaller.</div>", unsafe_allow_html=True)

    def bar(data, x, y, title, color=TEAL, h=310, **kw):
        f = px.bar(pd.DataFrame(data), x=x, y=y, title=title, color_discrete_sequence=[color], height=h, **kw)
        return f

    r1, r2 = st.columns(2)
    with r1, st.container(border=True):
        al_df = pd.DataFrame(A["by_airline"]).sort_values("avg_delay")
        show(bar(al_df, "avg_delay", "key", "Average delay by airline (min)", NAVY, orientation="h", labels={"key": "", "avg_delay": "min"}))
    with r2, st.container(border=True):
        show(bar(pd.DataFrame(cgs), "key", "avg_delay", "Delay rises sharply with congestion", RED, labels={"key": "Congestion band", "avg_delay": "Avg delay (min)"}))

    r3, r4 = st.columns(2)
    with r3, st.container(border=True):
        hr = pd.DataFrame(A["by_hour"]).sort_values("key")
        f = px.line(hr, x="key", y="avg_delay", markers=True, title="Average delay by scheduled hour (peak hours shaded)", height=310)
        for x0, x1 in [(6.5, 10.5), (16.5, 21.5)]:
            f.add_vrect(x0=x0, x1=x1, fillcolor=AMBER, opacity=.12, line_width=0)
        f.update_layout(xaxis_title="Hour", yaxis_title="min")
        show(f)
    with r4, st.container(border=True):
        dw = pd.DataFrame(A["by_dow"]).sort_values("key"); dw["Day"] = [DOW[int(k)] for k in dw.key]
        show(bar(dw, "Day", "avg_delay", "Average delay by day of week", AMBER, labels={"avg_delay": "min"}))

    r5, r6 = st.columns(2)
    with r5, st.container(border=True):
        show(bar(pd.DataFrame(A["by_weather"]), "key", "avg_delay", "Average delay by weather", NAVY, labels={"key": "", "avg_delay": "min"}))
    with r6, st.container(border=True):
        f = px.line(pd.DataFrame(A["monthly_trend"]), x="key", y="avg_delay", title="Monthly average delay, 2023-2025", height=310,
                    color_discrete_sequence=[RED])
        f.update_layout(xaxis_title=None, yaxis_title="min")
        show(f)

    st.markdown("### Airport league table")
    ap = pd.DataFrame(A["by_airport"]).rename(columns={"key": "Airport"})
    ap["Name"] = ap.Airport.map(lambda c_: AIRPORT_NAMES.get(c_, ""))
    ap = ap.sort_values("avg_delay", ascending=False)
    view = pd.DataFrame({"Airport": ap.Airport, "Name": ap.Name, "Flights": ap.flights,
                         "Delay rate": (ap.delay_rate * 100).round(1), "Avg delay (min)": ap.avg_delay.round(1),
                         "Avg congestion": ap.congestion.round(1)})
    st.dataframe(view, width="stretch", hide_index=True, height=360, column_config={
        "Delay rate": st.column_config.ProgressColumn("Delay rate", format="%.1f%%", min_value=0, max_value=100),
        "Avg delay (min)": st.column_config.ProgressColumn("Avg delay (min)", format="%.1f", min_value=0, max_value=70),
        "Avg congestion": st.column_config.ProgressColumn("Avg congestion", format="%.0f", min_value=0, max_value=100)})

# ============================================================================= 5. MODEL
with tab_model:
    st.markdown("### How well does the model work?")
    st.caption(f"Time-based hold-out: trained on 2023-2024 flights, evaluated on {M['n_test']:,} unseen flights from 2025.")
    a, b, c, d, e = st.columns(5)
    a.markdown(kpi("Accuracy", f"{M['accuracy'] * 100:.1f}%", f"baseline {M['baseline_accuracy'] * 100:.1f}%"), unsafe_allow_html=True)
    b.markdown(kpi("Precision", f"{M['precision'] * 100:.1f}%"), unsafe_allow_html=True)
    c.markdown(kpi("Recall", f"{M['recall'] * 100:.1f}%"), unsafe_allow_html=True)
    d.markdown(kpi("F1 score", f"{M['f1']:.3f}"), unsafe_allow_html=True)
    e.markdown(kpi("ROC-AUC", f"{M['roc_auc']:.3f}"), unsafe_allow_html=True)
    st.write("")
    f1_, f2_ = st.columns(2)
    roc = go.Figure()
    roc.add_trace(go.Scatter(x=M["roc"]["fpr"], y=M["roc"]["tpr"], name="Random Forest", line=dict(color=TEAL, width=3), fill="tozeroy", fillcolor="rgba(26,154,168,.10)"))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(color="#8a99aa", dash="dash")))
    roc.update_layout(title="ROC curve", height=340, xaxis_title="False positive rate", yaxis_title="True positive rate", legend=dict(x=0.62, y=0.12, orientation="v"))
    with f1_, st.container(border=True):
        show(roc)
    cm = np.array(M["confusion"])
    hm = px.imshow(cm, text_auto=",", x=["Pred. on time", "Pred. delayed"], y=["Actual on time", "Actual delayed"],
                   color_continuous_scale="Blues", title="Confusion matrix (threshold 0.5)", height=340)
    hm.update_layout(coloraxis_showscale=False)
    with f2_, st.container(border=True):
        show(hm)

    g1, g2 = st.columns([1.5, 1])
    imp = pd.Series(A["importance"]).sort_values().tail(10)
    imp.index = [FEATURE_LABELS.get(i, i) for i in imp.index]
    fi = px.bar(x=imp.values, y=imp.index, orientation="h", title="What drives delays? (permutation importance)",
                color_discrete_sequence=[NAVY], height=380, labels={"x": "drop in ROC-AUC", "y": ""})
    with g1, st.container(border=True):
        show(fi)
    with g2:
        st.markdown(kpi("Delay-minutes model · MAE", f"{M['reg_mae']:.1f} min", "Gradient Boosting"), unsafe_allow_html=True)
        st.write("")
        st.markdown(kpi("Delay-minutes model · R²", f"{M['reg_r2']:.2f}", "share of variance explained"), unsafe_allow_html=True)

    st.markdown("""
#### Method in brief
1. **Data** – 150,000 departures, Jan 2023 – Dec 2025: airline, route, schedule, weather, airport congestion, knock-on delay, turnaround risk.
2. **Targets** – `Delay_Target` (delayed / on time) for classification and `Departure_Delay` (minutes) for regression. `Departure_Delay` is *never* an input to the classifier (that would leak the answer).
3. **Features** – one-hot for airline & weather; target-encoding for origin/destination airports; numeric features as-is.
4. **Evaluation** – time-based split (train 2023-24, test 2025), so the score reflects predicting the future, not shuffled history.
5. **Simulation** – this app feeds user-chosen or randomly generated flights into the trained models to explore what-if scenarios.

#### Limitations (please read)
- The dataset appears **synthetic/simulated**: ~87 % of flights are delayed and delays are either 0 or ≥ 19 min, far from real Indian on-time data. Findings describe this dataset, not real-world aviation.
- The risk score comes from a class-balanced Random Forest and is **not a calibrated probability**.
- Weather has little effect in this dataset, and the model cannot see events outside its features (strikes, ATC flow control, aircraft faults).
- The airport-day simulator draws inputs at random from typical distributions; it illustrates the idea and is not a forecast.
""")
