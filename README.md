# SwiftParcel ETA Serving — DA3408 Assignment 5 (DA24B029)

This project serves SwiftParcel's delivery-time (ETA) models in three ways:
- **Part A:** a FastAPI REST service, with load tests.
- **Part B:** the MLflow model registry, with a promotion gate and a shadow test.
- **Part C:** a sticky canary on Ray Serve.

Part D adds a release smoke test and the canary abort rules. Results and
answers are in **REPORT.pdf**.

## Folder structure

```text
DA3408_DA24B029_5/
├── README.md
├── REPORT.pdf
├── AI_DISCLOSURE.md
├── requirements.txt
├── eta_data.py
├── load_test.py
├── data/
│   ├── eta_train.csv
│   ├── eta_fresh.csv
│   └── eta_sample.csv
├── partA/
│   ├── eta_app.py
│   ├── test_eta_app.py
│   ├── loadtest_results.json
│   ├── batch1_results.json
│   ├── batch100_results.json
│   ├── two_workers_results.json
│   ├── cpu.txt
│   └── memory.txt
├── partB/
│   ├── eta_mlflow.py
│   ├── test_gate.py
│   ├── shadow_compare.py
│   ├── registry_output.txt
│   ├── gate_output.txt
│   └── shadow_output.txt
├── partC/
│   ├── eta_ray.py
│   ├── verify_ray.py
│   └── verify_output.txt
├── partD/
│   ├── smoke_eta.py
│   └── verify_output.txt
└── screenshots/
    ├── 01_partA_tests.png
    ├── 02_partA_health_prediction.png
    ├── 03_partA_validation.png
    ├── 04_partA_metrics.png
    ├── 05_partA_baseline_load.png
    ├── 06_partA_batch_comparison.png
    ├── 07_partA_two_workers_startup.png
    ├── 08_partA_two_workers_load.png
    ├── 09_partB_tests.png
    ├── 10_partB_registry.png
    ├── 11_partB_default_gate.png
    ├── 12_partB_strict_gate.png
    ├── 13_partB_shadow.png
    ├── 14_partC_live_verification.png
    ├── 15_partD_smoke_verification.png
    └── 16_partB_mlflow_ui.png
```

Model files (`*.joblib`), `mlflow.db` and `mlartifacts/` are not included.
`eta_data.py` recreates the models.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python eta_data.py
python -c "import pandas as pd; from eta_data import FEATURES; pd.read_csv('data/eta_fresh.csv')[FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"
```

Run all commands from the project folder with `.venv` activated.

## Tests

```bash
python -m pytest partA/test_eta_app.py -q     # 64 tests
python -m pytest partB/test_gate.py -q        # 105 tests
```

## Part A — REST service

Start the server (leave it running):
```bash
ETA_MODEL=data/eta_v1.joblib python -m uvicorn partA.eta_app:app --port 8000 --workers 1 --no-access-log
```

In a second terminal:
```bash
curl -s http://127.0.0.1:8000/healthz
python load_test.py --url http://127.0.0.1:8000/v1/models/eta:predict --format rest \
  --data data/eta_sample.csv --concurrency 1 2 4 8 16 --duration 4 --batch 1 \
  --out partA/loadtest_results.json
```

For batching, rerun `load_test.py` with `--concurrency 1 --batch 1` and again
with `--batch 100`.

For two workers, restart the server with `--workers 2`, and set
`PROMETHEUS_MULTIPROC_DIR` to an empty folder so the two workers share metrics.

## Part B — MLflow

```bash
python -m partB.eta_mlflow register    # v1 = prod, v2 = candidate
python -m partB.eta_mlflow gates       # default and strict promotion gates
python -m partB.shadow_compare         # same 200 requests sent to both versions
```

## Part C — Ray Serve canary

Port 8000 must be free:
```bash
python -m partC.verify_ray
```

## Part D — Smoke test

With a Part A server running:
```bash
python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1 --max-p95-ms 80
echo $?    # 0 = pass, 1 = fail
```

The canary abort criteria are in REPORT.pdf.
