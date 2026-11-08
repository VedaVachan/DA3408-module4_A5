"""Required Part B interfaces and CLI. Importing this module performs no I/O.

Run from the project root: python -m partB.eta_mlflow register|gates|registry
decide() uses only the standard library; MLflow is imported only for real work.
The pyfunc wrapper class is defined here too (see eta_bundle_model_class), so the
logged model needs no extra code files: cloudpickle stores the class by value.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
from importlib.metadata import version
import json
import math
from numbers import Real
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "eta-model"
METRIC_KEYS = ("mae", "mae_rush", "mae_rain", "p95_ms")


def _valid_number(value):
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except (OverflowError, ValueError):
        return False


def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05):
    """Pure, inclusive gate. Slice tolerance is interpreted as RELATIVE regression.

    Gain=(prod MAE-candidate MAE)/prod MAE. A slice passes when candidate MAE
    <= prod slice MAE*(1+max_slice_regression). Decimal(str(value)) keeps stated
    decimal boundaries inclusive without accepting slightly outside thresholds.
    Invalid/missing/nonfinite/negative metrics or thresholds fail closed.
    Zero overall baseline permits only zero candidate MAE with min_rel_gain=0;
    a zero-error slice permits only a zero-error candidate slice.
    """
    criteria = dict(valid_metrics=False, overall_gain=False, latency=False,
                    rush_slice=False, rain_slice=False)
    thresholds = dict(min_rel_gain=min_rel_gain, max_p95_ms=max_p95_ms,
                      max_slice_regression=max_slice_regression)
    valid = (isinstance(prod, dict) and isinstance(cand, dict)
             and all(_valid_number(value) for value in thresholds.values())
             and all(key in metrics and _valid_number(metrics[key])
                     for metrics in (prod, cand) for key in METRIC_KEYS))
    if not valid:
        return {"criteria": criteria, "promote": False, "reason": "invalid_metrics_or_thresholds"}
    number = lambda value: Decimal(str(value))
    p = {key: number(prod[key]) for key in METRIC_KEYS}
    c = {key: number(cand[key]) for key in METRIC_KEYS}
    gain = number(min_rel_gain)
    regression = number(max_slice_regression)
    criteria["valid_metrics"] = True
    if p["mae"] == 0:
        criteria["overall_gain"] = c["mae"] == 0 and gain == 0
    else:
        criteria["overall_gain"] = c["mae"] <= p["mae"] * (1 - gain)
    criteria["latency"] = c["p95_ms"] <= number(max_p95_ms)
    criteria["rush_slice"] = c["mae_rush"] <= p["mae_rush"] * (1 + regression)
    criteria["rain_slice"] = c["mae_rain"] <= p["mae_rain"] * (1 + regression)
    return {"criteria": criteria, "promote": all(criteria.values())}


_BUNDLE_MODEL_CLASS = None


def eta_bundle_model_class():
    """Return the portable ETA pyfunc class (built once, MLflow imported lazily).

    Defining the class inside a function keeps `import partB.eta_mlflow` free of
    MLflow, and makes cloudpickle serialise the class by value into
    python_model.pkl, so `mlflow models serve` can load it without this file.
    All model data arrives through the MLflow "bundle" artifact.
    """
    global _BUNDLE_MODEL_CLASS
    if _BUNDLE_MODEL_CLASS is not None:
        return _BUNDLE_MODEL_CLASS
    import mlflow.pyfunc

    class EtaBundleModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import joblib
            bundle = joblib.load(context.artifacts["bundle"])
            self.model = bundle["model"]
            self.feature_columns = list(bundle["feature_columns"])
            self.model_version = str(bundle["version"])

        def predict(self, context, model_input, params=None):
            import numpy as np
            import pandas as pd
            if not isinstance(model_input, pd.DataFrame):
                raise TypeError("ETA pyfunc expects a pandas DataFrame")
            frame = model_input.loc[:, self.feature_columns]
            values = np.asarray(self.model.predict(frame), dtype=float)
            if values.shape != (len(frame),) or not np.isfinite(values).all():
                raise ValueError("Model returned invalid predictions")
            return pd.DataFrame({
                "eta_minutes": np.round(values, 1),
                "model_version": [self.model_version] * len(frame),
            }, index=frame.index)

    _BUNDLE_MODEL_CLASS = EtaBundleModel
    return EtaBundleModel


def configure_tracking():
    import mlflow
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    if not uri.startswith("sqlite:///"):
        raise ValueError("This assignment requires a SQLite-backed MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return uri


def registry_table():
    import mlflow
    from mlflow.tracking import MlflowClient
    configure_tracking()
    client = MlflowClient()
    resolved = {alias: str(client.get_model_version_by_alias(MODEL_NAME, alias).version)
                for alias in ("prod", "candidate")}
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    table = [{"registry_version": str(item.version),
              "bundle_version": item.tags.get("bundle_version", "unknown"),
              "aliases": [alias for alias, number in resolved.items() if number == str(item.version)],
              "run_id": item.run_id, "source": item.source}
             for item in sorted(versions, key=lambda item: int(item.version))]
    return {"model_name": MODEL_NAME, "tracking_uri": mlflow.get_tracking_uri(),
            "aliases": resolved, "versions": table}


def register():
    import joblib
    import mlflow
    import mlflow.pyfunc
    import pandas as pd
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient
    from types import SimpleNamespace
    from eta_data import FEATURES

    EtaBundleModel = eta_bundle_model_class()
    configure_tracking()
    paths = {label: ROOT / "data" / f"eta_{label}.joblib" for label in ("v1", "v2")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Generate the supplied models first: missing {path}")
    example = pd.read_csv(ROOT / "data/eta_train.csv").loc[:, FEATURES].head(5)
    pins = [f"{name}=={version(name)}" for name in
            ("mlflow", "cloudpickle", "numpy", "pandas", "scikit-learn", "scipy", "joblib")]
    client = MlflowClient()
    experiment = client.get_experiment_by_name("SwiftParcel-ETA")
    if experiment is None:
        experiment_id = client.create_experiment(
            "SwiftParcel-ETA", artifact_location=(ROOT / "mlartifacts").as_uri())
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    created = {}
    for label, path in paths.items():
        bundle = joblib.load(path)
        if bundle["version"] != label or list(bundle["feature_columns"]) != list(FEATURES):
            raise ValueError(f"Unexpected bundle label or feature order: {path}")
        wrapper = EtaBundleModel()
        # The wrapper uses only this documented context attribute.
        wrapper.load_context(SimpleNamespace(artifacts={"bundle": str(path)}))
        output = wrapper.predict(None, example)
        signature = infer_signature(example, output)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with mlflow.start_run(run_name=f"eta-{label}"):
            mlflow.log_params({"bundle_version": label, "bundle_sha256": digest})
            info = mlflow.pyfunc.log_model(
                name=f"eta_{label}", python_model=EtaBundleModel(),
                artifacts={"bundle": str(path)},
                signature=signature, input_example=example, pip_requirements=pins,
                metadata={"bundle_version": label, "bundle_sha256": digest},
            )
            registered = mlflow.register_model(info.model_uri, MODEL_NAME)
            created[label] = str(registered.version)
            client.set_model_version_tag(MODEL_NAME, registered.version, "bundle_version", label)
            client.set_model_version_tag(MODEL_NAME, registered.version, "bundle_sha256", digest)
    # Assign aliases only after both versions were created successfully.
    client.set_registered_model_alias(MODEL_NAME, "prod", created["v1"])
    client.set_registered_model_alias(MODEL_NAME, "candidate", created["v2"])
    table = registry_table()
    assert table["aliases"] == {"prod": created["v1"], "candidate": created["v2"]}
    table["pinned_pip_requirements"] = pins
    return table


def _evaluate(model, fresh):
    import numpy as np
    from eta_data import FEATURES
    x = fresh.loc[:, FEATURES]
    output = model.predict(x)
    prediction = output["eta_minutes"].to_numpy(dtype=float)
    if prediction.shape != (len(fresh),) or not np.isfinite(prediction).all():
        raise ValueError("Invalid evaluation predictions")
    error = np.abs(prediction - fresh["eta_minutes"].to_numpy(dtype=float))
    rush = fresh["hour"].isin([7, 8, 9, 16, 17, 18, 19]).to_numpy()
    rain = fresh["rain"].eq(1).to_numpy()
    single_rows = [x.iloc[[index]].copy() for index in range(200)]
    for row in single_rows[:10]:
        model.predict(row)  # Excluded warmup; frames are also built outside timing.
    durations = []
    for row in single_rows:
        start = time.perf_counter()
        model.predict(row)
        durations.append((time.perf_counter() - start) * 1000)
    return {"mae": float(error.mean()), "mae_rush": float(error[rush].mean()),
            "mae_rain": float(error[rain].mean()),
            "p95_ms": float(np.percentile(durations, 95, method="linear")),
            "timing_samples_ms": durations,
            "bundle_version": str(output["model_version"].iloc[0])}


def promotion_gate(**thresholds):
    import mlflow.pyfunc
    from mlflow.tracking import MlflowClient
    from eta_data import make_data
    configure_tracking()
    client = MlflowClient()
    aliases = {alias: str(client.get_model_version_by_alias(MODEL_NAME, alias).version)
               for alias in ("prod", "candidate")}
    prod = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@prod")
    candidate = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@candidate")
    fresh = make_data(2000, seed=99)
    metrics = {"prod": _evaluate(prod, fresh), "candidate": _evaluate(candidate, fresh)}
    # Detect an external alias change during evaluation instead of silently mixing versions.
    after = {alias: str(client.get_model_version_by_alias(MODEL_NAME, alias).version)
             for alias in aliases}
    if after != aliases:
        raise RuntimeError("Registry aliases changed during evaluation; repeat against stable aliases")
    result = decide(metrics["prod"], metrics["candidate"], **thresholds)
    applied = dict(min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05)
    applied.update(thresholds)
    result.update(metrics=metrics, thresholds=applied, registry_aliases=aliases,
                  evaluation_rows=len(fresh), rush_rows=int(fresh.hour.isin([7,8,9,16,17,18,19]).sum()),
                  rain_rows=int(fresh.rain.eq(1).sum()), timed_predictions_per_model=200,
                  warmup_predictions_per_model=10,
                  slice_regression_interpretation="relative",
                  latency_method="in-process pyfunc.predict, prebuilt single-row DataFrames; linear p95",
                  aliases_changed=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["register", "registry", "gates"])
    args = parser.parse_args()
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[name] = "1"
    if args.action in {"register", "registry"}:
        result = register() if args.action == "register" else registry_table()
        print(json.dumps(result, indent=2), flush=True)
    else:
        for label, thresholds in [("default", {}), ("strict_gain_60_percent", {"min_rel_gain": 0.60})]:
            result = promotion_gate(**thresholds)
            # Raw timing samples are omitted so the console output stays readable.
            summary = {key: value for key, value in result.items() if key != "metrics"}
            summary["metrics"] = {arm: {key: value for key, value in values.items()
                                       if key != "timing_samples_ms"}
                                  for arm, values in result["metrics"].items()}
            print(f"\nGATE: {label}\n{json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
