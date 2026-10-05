"""
Build assets/assets.json for the Streamlit app.
Run ONCE (and again if you retrain):   python prepare_assets.py
Needs: india_flight_delay_model_input.csv + flight_delay_models.joblib in this folder.
The app itself does NOT need the big CSV afterwards.
"""
import json, warnings
import joblib, numpy as np, pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, roc_curve,
                             mean_absolute_error, mean_squared_error, r2_score)
warnings.filterwarnings("ignore")

CSV, MODEL = "india_flight_delay_model_input.csv", "flight_delay_models.joblib"
df = pd.read_csv(CSV, parse_dates=["Flight_Date"])
df["Weather"] = df.Weather.replace({"Cloudy": "Clear/Partly Cloudy"})
df["Year"] = df.Flight_Date.dt.year
bundle = joblib.load(MODEL)
FEATURES, clf, reg = bundle["features"], bundle["classifier"], bundle["regressor"]

A = {}
# ---------- profiles used to auto-fill the simulator ----------
prof = df.groupby(["Origin_Airport", "Peak_Hour"]).Origin_Congestion_Index.mean().unstack().round(1)
A["airport_congestion"] = {a: {"offpeak": float(r.get(0, np.nan)), "peak": float(r.get(1, np.nan))} for a, r in prof.iterrows()}
A["weather_profile"] = df.groupby("Weather")[["Rainfall_mm", "Visibility_km", "Cloud_Cover_pct"]].mean().round(1).to_dict("index")
m = df.groupby("Month")[["Temperature_C", "Humidity_pct", "Wind_Speed_kmh"]].mean().round(1)
A["month_profile"] = {int(k): v for k, v in m.to_dict("index").items()}
A["airline_share"] = df.Airline.value_counts(normalize=True).round(4).to_dict()
A["dest_share"] = df.Destination_Airport.value_counts(normalize=True).round(5).to_dict()
A["hour_share"] = df.Scheduled_Departure_Hour.value_counts(normalize=True).sort_index().round(4).to_dict()
A["defaults"] = {"prev_delay_mean": float(df.Previous_Flight_Delay_Minutes.mean()),
                 "turnaround_mean": float(df.Turnaround_Risk_Index.mean()),
                 "turnaround_std": float(df.Turnaround_Risk_Index.std())}

# ---------- analytics ----------
A["kpi"] = {"flights": int(len(df)), "delay_rate": float(df.Delay_Target.mean()),
            "avg_delay_all": float(df.Departure_Delay.mean()),
            "avg_delay_when_delayed": float(df[df.Delay_Target == 1].Departure_Delay.mean()),
            "airports": int(df.Origin_Airport.nunique()), "airlines": int(df.Airline.nunique()),
            "date_from": str(df.Flight_Date.min().date()), "date_to": str(df.Flight_Date.max().date())}
def agg(col):
    g = df.groupby(col).agg(flights=("Delay_Target", "size"), delay_rate=("Delay_Target", "mean"),
                            avg_delay=("Departure_Delay", "mean"), congestion=("Origin_Congestion_Index", "mean")).round(4)
    return [{"key": (k if isinstance(k, str) else int(k)), **v} for k, v in g.to_dict("index").items()]
for name, col in [("by_airline", "Airline"), ("by_airport", "Origin_Airport"), ("by_hour", "Scheduled_Departure_Hour"),
                  ("by_month", "Month"), ("by_dow", "Day_of_Week"), ("by_weather", "Weather")]:
    A[name] = agg(col)
df["cong_band"] = pd.cut(df.Origin_Congestion_Index, [-1, 40, 60, 80, 100], labels=["Low (≤40)", "Medium (40–60)", "High (60–80)", "Very high (80+)"])
g = df.groupby("cong_band", observed=True).agg(flights=("Delay_Target", "size"), delay_rate=("Delay_Target", "mean"), avg_delay=("Departure_Delay", "mean")).round(4)
A["by_congestion"] = [{"key": str(k), **v} for k, v in g.to_dict("index").items()]
ym = df.groupby(df.Flight_Date.dt.to_period("M").astype(str)).agg(delay_rate=("Delay_Target", "mean"), avg_delay=("Departure_Delay", "mean")).round(4)
A["monthly_trend"] = [{"key": k, **v} for k, v in ym.to_dict("index").items()]

# ---------- model performance on the 2025 hold-out (same split as the notebook) ----------
test = df[df.Year == 2025]
X, y, yr = test[FEATURES], test.Delay_Target, test.Departure_Delay
p = clf.predict_proba(X)[:, 1]; pred = (p >= .5).astype(int)
fpr, tpr, _ = roc_curve(y, p); idx = np.linspace(0, len(fpr) - 1, 120).astype(int)
pr = np.clip(reg.predict(X), 0, None)
A["metrics"] = {"n_test": int(len(test)), "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
                "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p),
                "pr_auc": average_precision_score(y, p), "baseline_accuracy": float(y.mean()),
                "confusion": confusion_matrix(y, pred).tolist(),
                "roc": {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()},
                "reg_mae": mean_absolute_error(yr, pr), "reg_rmse": float(np.sqrt(mean_squared_error(yr, pr))), "reg_r2": r2_score(yr, pr)}
s = X.sample(15000, random_state=42)
imp = permutation_importance(clf, s, y.loc[s.index], scoring="roc_auc", n_repeats=3, random_state=42, n_jobs=-1)
A["importance"] = dict(sorted(zip(FEATURES, imp.importances_mean.round(5).tolist()), key=lambda t: -t[1]))
A["airports"] = sorted(df.Origin_Airport.unique().tolist())
A["airlines"] = sorted(df.Airline.unique().tolist())

json.dump(A, open("assets/assets.json", "w"), indent=1, default=float)
print("assets/assets.json written |", {k: round(v, 4) for k, v in A["metrics"].items() if isinstance(v, float)})
