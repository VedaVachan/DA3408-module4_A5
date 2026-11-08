"""eta_data.py -- shared data + model training for the Module 4 assignment (delivery-time ETA regression).
Run:  python eta_data.py   -> data/eta_train.csv, data/eta_fresh.csv, data/eta_v1.joblib (Ridge), data/eta_v2.joblib (GradientBoosting)"""
import numpy as np, pandas as pd, joblib, os
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

FEATURES = ["distance_km", "n_stops", "hour", "weekday", "traffic_index", "rain", "weight_kg"]

def make_data(n=5000, seed=1):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"distance_km": r.gamma(2.0, 12, n).clip(0.1, 200).round(2), "n_stops": r.integers(1, 25, n),
                      "hour": r.integers(0, 24, n), "weekday": r.integers(0, 7, n), "traffic_index": r.uniform(0, 1, n).round(3),
                      "rain": r.integers(0, 2, n), "weight_kg": r.gamma(2.0, 8, n).clip(0.1, 200).round(2)})
    rush = d.hour.isin([7, 8, 9, 16, 17, 18, 19]).astype(float)
    d["eta_minutes"] = (8 + 1.6 * d.distance_km * (1 + 0.9 * d.traffic_index * (1 + rush)) + 3.5 * d.n_stops
                        + 7 * d.rain * (d.distance_km > 20) + 0.04 * d.weight_kg * d.n_stops + 12 * rush * (d.n_stops > 10)
                        + r.normal(0, 4, n)).round(1)
    return d

def main(out="data"):
    os.makedirs(out, exist_ok=True)
    tr, fresh = make_data(5000, 1), make_data(2000, 99)
    tr.to_csv(f"{out}/eta_train.csv", index=False); fresh.to_csv(f"{out}/eta_fresh.csv", index=False)
    models = {"v1": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
              "v2": GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=0)}
    for v, m in models.items():
        m.fit(tr[FEATURES], tr.eta_minutes)
        mae = mean_absolute_error(fresh.eta_minutes, m.predict(fresh[FEATURES]))
        joblib.dump({"model": m, "feature_columns": FEATURES, "version": v}, f"{out}/eta_{v}.joblib")
        print(f"{v}: {type(m).__name__:28s} fresh-data MAE = {mae:.2f} min")

if __name__ == "__main__":
    main()
