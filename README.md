# ✈️ Airport Flight Delay Analytics (India) – Simulator

Streamlit web app for the Data Science & Application capstone. It loads your trained models
(`flight_delay_models.joblib`) and lets you predict, explore and simulate flight delays.

## Run locally
Windows: use Python 3.12 or 3.13 (not 3.14) inside a virtual environment:
```bash
py -3.12 -m venv venv
venv\Scripts\activate
```
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501

## IMPORTANT – scikit-learn version
The model file was saved in Colab with scikit-learn **1.6.1**. It must be loaded with the same version,
which is pinned in `requirements.txt`. If Colab later gives you a different version
(`import sklearn; print(sklearn.__version__)`), change the pin to match, or retrain and re-export.

## Deploy free (Streamlit Community Cloud)
1. Create a GitHub repo and upload everything in this folder (the .joblib is ~42 MB, which is allowed).
2. Go to share.streamlit.io → New app → pick the repo, main file `app.py` → Deploy.

## Files
| File | Purpose |
|---|---|
| `app.py` | The web app (5 tabs) |
| `flight_delay_models.joblib` | Your trained classifier + regressor |
| `assets/assets.json` | Pre-computed dashboard stats, model metrics, feature importance |
| `prepare_assets.py` | Regenerates `assets.json` from the CSV + model (only needed if you retrain) |
| `requirements.txt`, `.streamlit/config.toml` | Dependencies and theme |

## What's in the UI
Boarding-pass result card that stays visible while you scroll, one-click scenario presets, an airport-style
departure board in the day simulator, and a light theme with an in-app decision-threshold control.

## Tabs
1. **Flight predictor** – enter a flight, get delay risk, predicted minutes and "what would cut this delay".
2. **What-if explorer** – sweep congestion / hour / rain etc., and compare two scenarios.
3. **Airport day simulator** – simulate a day of departures at any airport, inject a disruption, see a departures board.
4. **Analytics dashboard** – historical patterns by airline, airport, hour, weather, congestion.
5. **Model & method** – metrics, ROC, confusion matrix, feature importance, limitations.

## Regenerate assets after retraining
Put `india_flight_delay_model_input.csv` next to `prepare_assets.py`, then `python prepare_assets.py`.
