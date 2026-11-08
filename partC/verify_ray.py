"""Deploy and verify the real sticky canary on an isolated two-CPU Ray instance.

Run from project root: python -m partC.verify_ray 2>&1 | tee partC/verify_output.txt
Pass --out FILE to also save a detailed JSON record of the run.
No mocks, local-testing mode, existing cluster attachment, or global ray-stop.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Ray status dictionaries contain enum keys; normalize before JSON encoding.
    def plain(item):
        if isinstance(item, Enum):
            return item.name
        if isinstance(item, dict):
            return {str(plain(key)): plain(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(entry) for entry in item]
        return item
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def enum_value(value):
    return getattr(value, "name", str(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--out", default=None, help="Optional JSON results file")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.clients <= 32:
        parser.error("Use a valid port and 1-32 HTTP clients")
    if not __debug__:
        raise RuntimeError("Verification assertions must be enabled; do not run Python with -O")
    output = (ROOT / args.out).resolve() if args.out else None
    if output is not None:
        if output.exists():
            raise FileExistsError(f"Preserving {output}; choose a different --out for a deliberate rerun")
        output.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))  # Refuse an occupied port; never kill its listener.
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[name] = "1"
    import joblib
    import numpy as np
    import pandas as pd
    import ray
    from ray import serve
    import requests
    from eta_data import FEATURES
    from partC.eta_ray import APP_NAME, CPU_PER_REPLICA, AUTOSCALING, THREAD_ENV, build

    if ray.is_initialized():
        raise RuntimeError("Run this verifier in a fresh Python process; it owns its own local Ray instance")
    required = ["data/eta_fresh.csv", "data/eta_v1.joblib", "data/eta_v2.joblib",
                "partC/eta_ray.py"]
    for name in required:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f"Missing {name}; generate the supplied datasets/models first")
    rows = pd.read_csv(ROOT / "data/eta_fresh.csv").loc[:, FEATURES].head(3)
    if len(rows) != 3:
        raise ValueError("Expected at least three fresh-data rows")
    instances = rows.to_dict(orient="records")
    # These IDs and this cohort size are fixed before observing the distribution.
    customers = [f"customer-{index:04d}" for index in range(1000)]
    base = f"http://127.0.0.1:{args.port}"
    document = {
        "completed": False, "started_utc": datetime.now(timezone.utc).isoformat(),
        "ray_version": ray.__version__, "python_version": sys.version,
        "package_versions": {name: version(name) for name in
                             ["ray", "fastapi", "starlette", "pydantic", "scikit-learn", "pandas", "numpy"]},
        "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in required},
        "customers": customers, "http_clients": args.clients, "http_port": args.port,
        "configured_autoscaling": AUTOSCALING,
        "cpu_plan": {"prod": 2 * CPU_PER_REPLICA, "candidate_max": 3 * CPU_PER_REPLICA,
                     "ingress": CPU_PER_REPLICA, "maximum_deployment_reservations": 6 * CPU_PER_REPLICA,
                     "note": "Ray logical reservations, not OS affinity or physical CPU caps"},
        "checks": {}, "snapshots": {}, "observations": {},
    }
    def checkpoint():
        if output is not None:
            write_json(output, document)

    checkpoint()

    def wait_ready(weight):
        deadline = time.monotonic() + 120
        with requests.Session() as session:
            session.trust_env = False
            while time.monotonic() < deadline:
                status = serve.status()
                application = status.applications.get(APP_NAME)
                if application and enum_value(application.status) in {"DEPLOY_FAILED", "UNHEALTHY"}:
                    raise RuntimeError(f"Serve deployment unhealthy: {application}")
                try:
                    response = session.get(base + "/healthz", timeout=2)
                    if response.status_code == 200:
                        body = response.json()
                        if (body.get("status") == "ok" and body.get("candidate_weight") == weight
                                and application and enum_value(application.status) == "RUNNING"):
                            return body
                except requests.RequestException:
                    pass
                time.sleep(.2)
        raise RuntimeError(f"Timed out waiting for weight {weight}; inspect Serve status and Ray logs")

    def snapshot(label):
        status = serve.status()
        application = status.applications[APP_NAME]
        deployments = application.deployments
        counts = {}
        for name in ["eta-prod", "eta-candidate", "eta-router"]:
            states = {enum_value(state): count for state, count in deployments[name].replica_states.items()}
            counts[name] = states.get("RUNNING", 0)
        assert counts["eta-prod"] == 2, counts
        assert counts["eta-router"] == 1, counts
        assert 1 <= counts["eta-candidate"] <= 3, counts
        assert sum(counts.values()) * CPU_PER_REPLICA <= 2
        document["snapshots"][label] = {
            "serve_status": asdict(status), "running_replicas": counts,
            "cluster_resources": ray.cluster_resources(), "available_resources": ray.available_resources(),
        }
        checkpoint()
        print(f"Resources [{label}]: running={counts}; cluster CPUs={ray.cluster_resources().get('CPU')}", flush=True)

    def collect(ids, beta=False, batch=None):
        local = threading.local()
        sessions = []
        lock = threading.Lock()
        payload_rows = instances[:1] if batch is None else batch

        def query(customer):
            if not hasattr(local, "session"):
                local.session = requests.Session()
                local.session.trust_env = False
                with lock:
                    sessions.append(local.session)
            response = local.session.post(
                base + "/v1/models/eta:predict",
                json={"customer_id": customer, "instances": payload_rows},
                headers={"X-Beta": "1"} if beta else {}, timeout=30,
            )
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
            body = response.json()
            assert body["arm"] in {"v1", "v2"}, body
            assert body["model_version"] == body["arm"], body
            assert body["replica_cpus"] == CPU_PER_REPLICA, body
            assert body["router_cpus"] == CPU_PER_REPLICA, body
            assert body["request_id"] == response.headers["X-Request-ID"], body
            assert len(body["predictions"]) == len(payload_rows), body
            assert all(isinstance(value, (float, int)) and math.isfinite(value) for value in body["predictions"]), body
            return customer, body

        try:
            with ThreadPoolExecutor(max_workers=args.clients) as executor:
                result = dict(executor.map(query, ids))
        finally:
            for session in sessions:
                session.close()
        return result

    def arms(responses):
        return {customer: body["arm"] for customer, body in responses.items()}

    try:
        print("Starting an isolated local Ray instance: 2 CPUs, 0 GPUs, 128 MiB object store", flush=True)
        context = ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False,
                           object_store_memory=128 * 1024 * 1024,
                           namespace="swiftparcel-a5",
                           runtime_env={"env_vars": {**THREAD_ENV, "PYTHONPATH": str(ROOT)}})
        document["ray_session_dir"] = context.address_info.get("session_dir")
        resources = ray.cluster_resources()
        assert resources.get("CPU") == 2, resources
        assert resources.get("GPU", 0) == 0, resources
        checkpoint()
        serve.start(proxy_location="HeadOnly", http_options={"host": "127.0.0.1", "port": args.port})
        print("Deploying weight=0.2", flush=True)
        serve.run(build(weight=.2), name=APP_NAME, route_prefix="/")
        initial_health = wait_ready(.2)
        snapshot("weight_0.2_initial")

        # Fail on an ingress/body-binding error before scheduling a whole cohort.
        document["observations"]["ingress_probe"] = collect(["ingress-preflight"])
        checkpoint()
        print("[PASS] Ingress probe: JSON prediction returned HTTP 200", flush=True)

        first_responses = collect(customers)
        baseline = arms(first_responses)
        document["observations"]["initial_responses"] = first_responses
        candidate_count = sum(arm == "v2" for arm in baseline.values())
        share = candidate_count / len(customers)
        assert .16 <= share <= .24, f"Candidate share outside tolerance: {share}"
        document["checks"]["a_candidate_share"] = {"passed": True, "candidate_customers": candidate_count,
                                                   "distinct_customers": 1000, "share": share}
        checkpoint()
        print(f"[PASS] A: candidate share={candidate_count}/1000 ({share:.1%}), within [16%,24%]", flush=True)

        repeated = arms(collect(customers[:200]))
        assert all(repeated[customer] == baseline[customer] for customer in repeated)
        document["observations"]["repeat_200"] = repeated
        document["checks"]["b_repeat_stickiness"] = {"passed": True, "customers": 200}
        checkpoint()
        print("[PASS] B: 200/200 customers retained their arm on a second call", flush=True)

        # Replace the app, rather than relying on an idempotent no-op run call.
        print("Replacing the app with the identical weight=0.2 definition", flush=True)
        serve.delete(APP_NAME)
        serve.run(build(weight=.2), name=APP_NAME, route_prefix="/")
        replaced_health = wait_ready(.2)
        assert initial_health["router_instance_id"] != replaced_health["router_instance_id"]
        snapshot("weight_0.2_redeployed")
        redeployed = arms(collect(customers[:200]))
        assert all(redeployed[customer] == baseline[customer] for customer in redeployed)
        document["observations"]["redeployed_200"] = redeployed
        document["checks"]["c_redeployment_stickiness"] = {
            "passed": True, "customers": 200, "old_router": initial_health["router_instance_id"],
            "new_router": replaced_health["router_instance_id"], "router_replaced": True,
        }
        checkpoint()
        print("[PASS] C: new router instance; 200/200 customers retained their arm after redeployment", flush=True)

        print("Increasing candidate weight to 0.5", flush=True)
        serve.run(build(weight=.5), name=APP_NAME, route_prefix="/")
        wait_ready(.5)
        snapshot("weight_0.5")
        raised_responses = collect(customers)
        raised = arms(raised_responses)
        returned = [customer for customer in customers if baseline[customer] == "v2" and raised[customer] != "v2"]
        assert not returned, returned
        document["observations"]["raised_weight_responses"] = raised_responses
        document["checks"]["d_monotonic_increase"] = {
            "passed": True, "previous_candidates": candidate_count, "returned_to_prod": len(returned),
            "candidate_customers_at_0.5": sum(arm == "v2" for arm in raised.values()),
        }
        checkpoint()
        print(f"[PASS] D: 0/{candidate_count} previous candidates returned to production at weight=0.5", flush=True)

        beta_customers = [customer for customer in customers if raised[customer] == "v1"][:50]
        assert len(beta_customers) == 50
        forced = arms(collect(beta_customers, beta=True))
        assert all(arm == "v2" for arm in forced.values())
        reset = arms(collect(beta_customers))
        assert all(arm == "v1" for arm in reset.values())
        document["observations"].update(beta_forced_50=forced, beta_reset_50=reset)
        document["checks"]["e_beta_override"] = {"passed": True, "forced_candidate": 50,
                                                  "natural_prod_restored": 50}
        checkpoint()
        print("[PASS] E: X-Beta: 1 forced candidate for 50 natural-production customers; normal routing restored", flush=True)

        # Verify real batch inference against the unchanged native bundles in both arms.
        for label, customer, beta in [("v1", beta_customers[0], False), ("v2", beta_customers[0], True)]:
            body = collect([customer], beta=beta, batch=instances)[customer]
            bundle = joblib.load(ROOT / "data" / f"eta_{label}.joblib")
            expected = np.round(bundle["model"].predict(rows.loc[:, bundle["feature_columns"]]), 1).tolist()
            assert body["arm"] == label and body["predictions"] == expected, body
        document["checks"]["batch_predictions"] = {"passed": True, "rows_per_batch": 3, "arms": ["v1", "v2"]}
        snapshot("final")
        document["completed"] = True
        print("[PASS] Batch predictions match the native v1 and v2 bundles", flush=True)
        print("ALL FIVE REQUIRED LIVE CHECKS PASSED", flush=True)
    except BaseException as error:
        document["error"] = f"{type(error).__name__}: {error}"
        if ray.is_initialized():
            try:
                document["failure_serve_status"] = asdict(serve.status())
            except Exception as status_error:
                document["failure_status_error"] = str(status_error)
        raise
    finally:
        document["cleanup"] = {}
        try:
            if ray.is_initialized():
                serve.shutdown()  # Only the fresh cluster created by this script.
        except Exception as error:
            document["cleanup"]["serve_shutdown_error"] = str(error)
        finally:
            ray.shutdown()
            document["cleanup"]["ray_initialized_after_shutdown"] = ray.is_initialized()
            document["finished_utc"] = datetime.now(timezone.utc).isoformat()
            checkpoint()
            if output is not None:
                print(f"Actual evidence saved: {output}", flush=True)
            print("Ray initialized after shutdown:", ray.is_initialized(), flush=True)
    return 1 if document["cleanup"].get("serve_shutdown_error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
