"""Real MLflow CLI servers and 200 paired, single-row HTTP comparisons.

Run from project root after registration:
    python -m partB.shadow_compare | tee partB/shadow_output.txt
Only subprocess groups created by this helper are stopped. Existing listeners
are never killed. No registry alias is moved by this experiment. Server logs go
to a temporary directory and are printed only if a server fails.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

from partB.eta_mlflow import ROOT, MODEL_NAME, configure_tracking


def stop_server(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait(timeout=10)
            return
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def wait_ready(process, session, base):
    import requests
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"MLflow serving exited during startup for {base}")
        try:
            if session.get(base + "/ping", timeout=1).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(.2)
    raise RuntimeError(f"MLflow serving readiness timed out for {base}")


def invoke(session, base, record, expected_version):
    start = time.perf_counter()
    response = session.post(base + "/invocations", json={"dataframe_records": [record]}, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Prediction failed at {base}: HTTP {response.status_code}: {response.text[:1200]}")
    body = response.json()
    elapsed_ms = (time.perf_counter() - start) * 1000
    rows = body.get("predictions") if isinstance(body, dict) else None
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(f"Unexpected MLflow prediction response: {body}")
    prediction = rows[0]
    if prediction.get("model_version") != expected_version:
        raise ValueError(f"Expected bundle {expected_version}, received {prediction.get('model_version')}")
    value = float(prediction["eta_minutes"])
    if not math.isfinite(value):
        raise ValueError("Non-finite shadow prediction")
    return value, elapsed_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prod-port", type=int, default=5101)
    parser.add_argument("--candidate-port", type=int, default=5102)
    args = parser.parse_args()
    if args.prod_port == args.candidate_port or any(not 1 <= p <= 65535 for p in (args.prod_port, args.candidate_port)):
        parser.error("Use two distinct ports between 1 and 65535")
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[name] = "1"
    import numpy as np
    import pandas as pd
    import requests
    from mlflow.tracking import MlflowClient
    from eta_data import FEATURES

    uri = configure_tracking()
    client = MlflowClient()
    versions = {alias: str(client.get_model_version_by_alias(MODEL_NAME, alias).version)
                for alias in ("prod", "candidate")}
    data = pd.read_csv(ROOT / "data/eta_fresh.csv").loc[:, FEATURES].head(200)
    if len(data) != 200:
        raise ValueError("The shadow experiment needs exactly 200 fresh-data rows")
    # Probe all ports before starting either process; never stop an existing listener.
    probes = []
    try:
        for port in (args.prod_port, args.candidate_port):
            probe = socket.socket()
            probes.append(probe)
            probe.bind(("127.0.0.1", port))
    finally:
        for probe in probes:
            probe.close()
    logs = Path(tempfile.mkdtemp(prefix="eta-shadow-logs-"))
    environment = os.environ.copy()
    environment["MLFLOW_TRACKING_URI"] = environment["MLFLOW_REGISTRY_URI"] = uri
    # `mlflow models serve` launches its server with the first `python` on PATH.
    # Put this interpreter's folder first so the server uses the same environment
    # (the one with MLflow installed), even if the venv was not activated.
    environment["PATH"] = os.path.dirname(sys.executable) + os.pathsep + environment.get("PATH", "")
    records = data.to_dict(orient="records")
    processes, measured = {}, []
    failed = False
    try:
        with ExitStack() as stack:
            sessions, bases = {}, {}
            for alias, port in [("prod", args.prod_port), ("candidate", args.candidate_port)]:
                bases[alias] = f"http://127.0.0.1:{port}"
                sessions[alias] = stack.enter_context(requests.Session())
                sessions[alias].trust_env = False
                log = stack.enter_context((logs / f"mlflow_{alias}_serve.log").open("w"))
                command = [sys.executable, "-m", "mlflow", "models", "serve", "-m",
                           f"models:/{MODEL_NAME}@{alias}", "--host", "127.0.0.1",
                           "--port", str(port), "--workers", "1", "--env-manager", "local"]
                print(f"Starting {alias} on port {port}, registry version {versions[alias]}", flush=True)
                processes[alias] = subprocess.Popen(command, cwd=ROOT, env=environment,
                                                     stdout=log, stderr=subprocess.STDOUT,
                                                     start_new_session=True)
            try:
                for alias in sessions:
                    wait_ready(processes[alias], sessions[alias], bases[alias])
                for index in range(5):
                    invoke(sessions["prod"], bases["prod"], records[index], "v1")
                    invoke(sessions["candidate"], bases["candidate"], records[index], "v2")
                for index, record in enumerate(records):
                    result = {"row_index": index}
                    order = ["prod", "candidate"] if index % 2 == 0 else ["candidate", "prod"]
                    for alias in order:
                        label = "v1" if alias == "prod" else "v2"
                        value, latency = invoke(sessions[alias], bases[alias], record, label)
                        result[f"{alias}_eta_minutes"] = value
                        result[f"{alias}_latency_ms"] = latency
                    difference = abs(Decimal(str(result["prod_eta_minutes"])) -
                                     Decimal(str(result["candidate_eta_minutes"])))
                    result["absolute_difference_minutes"] = float(difference)
                    result["difference_gt_10_minutes"] = difference > Decimal("10")
                    measured.append(result)
                after = {alias: str(client.get_model_version_by_alias(MODEL_NAME, alias).version)
                         for alias in versions}
                if versions != after:
                    raise RuntimeError("Registry aliases changed during shadow comparison")
                frame = pd.DataFrame(measured)
                summary = {
                    "paired_requests": len(frame),
                    "mean_absolute_difference_minutes": float(frame.absolute_difference_minutes.mean()),
                    "percent_difference_gt_10_minutes": float(frame.difference_gt_10_minutes.mean() * 100),
                    "prod_p95_ms": float(np.percentile(frame.prod_latency_ms, 95, method="linear")),
                    "candidate_p95_ms": float(np.percentile(frame.candidate_latency_ms, 95, method="linear")),
                }
                print("\nSHADOW SUMMARY\n" + json.dumps(summary, indent=2), flush=True)
            finally:
                for process in processes.values():
                    stop_server(process)
    except BaseException:
        failed = True
        for log in sorted(logs.glob("*.log")):
            print(f"\n--- {log.name} (last 40 lines) ---", file=sys.stderr)
            print("\n".join(log.read_text(errors="replace").splitlines()[-40:]), file=sys.stderr)
        raise
    finally:
        for process in processes.values():
            stop_server(process)
        if not failed:
            for log in logs.glob("*.log"):
                log.unlink()
            logs.rmdir()


if __name__ == "__main__":
    main()
