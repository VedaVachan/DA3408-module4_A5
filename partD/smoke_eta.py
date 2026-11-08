"""Release gate for the Part A ETA REST service; exit 0 for pass, 1 for failure.

Example: python partD/smoke_eta.py --url http://127.0.0.1:8000 \
    --expect-version v1 --max-p95-ms 80

No server, model or cluster is created by this program. All checks use real HTTP.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import build_opener, HTTPRedirectHandler, ProxyHandler, Request
from uuid import uuid4

PREDICT_PATH = "/v1/models/eta:predict"
PROBES = [
    dict(distance_km=10.0, n_stops=4, hour=12, weekday=2,
         traffic_index=0.2, rain=0, weight_kg=8.0),
    dict(distance_km=20.0, n_stops=8, hour=8, weekday=4,
         traffic_index=0.4, rain=1, weight_kg=15.0),
    dict(distance_km=40.0, n_stops=2, hour=18, weekday=5,
         traffic_index=0.6, rain=0, weight_kg=5.0),
]


class SmokeFailure(Exception):
    """An observable release condition did not hold."""


def require(condition, message):
    # Deliberately not `assert`: release checks must also run under python -O.
    if not condition:
        raise SmokeFailure(message)


def finite_positive(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be a finite positive number") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def sample_count(text):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer >= 20") from None
    if value < 20:
        raise argparse.ArgumentTypeError("must be an integer >= 20")
    return value


def endpoint_urls(url):
    parts = urlsplit(url)
    if (parts.scheme not in {"http", "https"} or not parts.netloc
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment):
        raise ValueError("URL must be HTTP(S), without credentials, query or fragment")
    # Validate the port here, before making network calls.
    _ = parts.port
    path = parts.path.rstrip("/")
    if not path:
        prefix = ""
    elif path.endswith(PREDICT_PATH):
        prefix = path[:-len(PREDICT_PATH)]
    else:
        raise ValueError(f"Use a server root URL or an endpoint ending in {PREDICT_PATH}")
    make = lambda suffix: urlunsplit((parts.scheme, parts.netloc, prefix + suffix, "", ""))
    return make(PREDICT_PATH), make("/healthz")


def percentile95(samples):
    require(bool(samples), "No latency samples")
    ordered = sorted(samples)
    position = .95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # A redirect must not conceal an incorrect deployment URL.


@dataclass
class Reply:
    status: int
    body: object
    request_ids: list[str]
    sent_request_id: str
    elapsed_ms: float


class Client:
    def __init__(self, timeout):
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}), NoRedirects())

    def request(self, url, payload=None):
        request_id = "smoke-" + str(uuid4())
        headers = {"Accept": "application/json", "X-Request-ID": request_id}
        content = None
        if payload is not None:
            content = json.dumps(payload, allow_nan=False).encode()
            headers["Content-Type"] = "application/json"
        request = Request(url, data=content, headers=headers,
                          method="POST" if payload is not None else "GET")
        started = time.perf_counter()
        try:
            try:
                response = self.opener.open(request, timeout=self.timeout)
            except HTTPError as exc:
                response = exc  # Preserve expected 422s, and report every other status.
            with response:
                raw = response.read()
                status = response.status
                response_ids = response.headers.get_all("X-Request-ID", [])
            body = json.loads(raw, parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"Nonfinite JSON constant: {token}")))
        except (URLError, OSError, ValueError) as exc:
            raise SmokeFailure(f"HTTP/JSON failure: {type(exc).__name__}: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        return Reply(status, body, response_ids, request_id, elapsed_ms)


def prediction(reply, count, expected_version):
    require(reply.status == 200, f"Expected HTTP 200, received {reply.status}")
    body = reply.body
    require(isinstance(body, dict), "Prediction response must be a JSON object")
    require(body.get("model_version") == expected_version,
            f"Expected model_version={expected_version!r}; received {body.get('model_version')!r}")
    require(body.get("request_id") == reply.sent_request_id,
            "JSON request_id did not echo the submitted ID")
    require(reply.request_ids == [reply.sent_request_id],
            f"Expected one matching X-Request-ID header, received {reply.request_ids!r}")
    values = body.get("predictions")
    require(isinstance(values, list) and len(values) == count,
            f"Expected {count} predictions")
    for value in values:
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value), "Predictions must be finite JSON numbers")
        require(value >= 0, "Smoke probe produced a negative ETA")
        require(math.isclose(value * 10, round(value * 10), rel_tol=0, abs_tol=1e-6),
                "Predictions must be rounded to one decimal")
    return values


def run_gate(url, expected_version, max_p95_ms=80.0, samples=30, timeout=5.0):
    endpoint, health_url = endpoint_urls(url)
    client = Client(timeout)
    document = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "url": endpoint, "expected_version": expected_version,
        "max_p95_ms": max_p95_ms, "sample_count": samples,
        "warmup_requests": 5,
        "latency_method": "sequential HTTP round trip including JSON decode; linear p95; no retries",
        "scope": "Part A REST contract; finite smoke probes, not a global monotonicity proof",
        "checks": {}, "passed": False,
    }

    def check(name, function):
        try:
            detail = function()
            result = {"passed": True, "detail": detail}
        except Exception as exc:
            result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        document["checks"][name] = result
        print(f"[{'PASS' if result['passed'] else 'FAIL'}] {name}: "
              + json.dumps(result.get("detail", result.get("error")), allow_nan=False), flush=True)
        return result["passed"]

    def predict(rows):
        return prediction(client.request(endpoint, {"instances": rows}), len(rows), expected_version)

    def health():
        reply = client.request(health_url)
        require(reply.status == 200, f"Health returned HTTP {reply.status}")
        require(isinstance(reply.body, dict) and reply.body.get("status") == "ok", "Health is not ok")
        require(reply.body.get("model_version") == expected_version, "Health reports the wrong model version")
        return {"status": "ok", "model_version": reply.body["model_version"]}

    if not check("health_and_version", health):
        document["blocked_reason"] = "Health/version failed; dependent checks were not attempted"
        return document

    check("valid_prediction_version_and_request_id", lambda: {"predictions": predict(PROBES[:1])})

    def batch_contract():
        singles = [predict([row])[0] for row in PROBES]
        batch = predict(PROBES)
        require(batch == singles, f"Batch differs from ordered individual results: {batch} vs {singles}")
        return {"individual_predictions": singles, "batch_predictions": batch}

    check("batch_order_and_consistency", batch_contract)

    def monotonic(feature, changes):
        pairs = []
        for row, changed in zip(PROBES, changes):
            low = predict([row])[0]
            high = predict([{**row, feature: changed}])[0]
            require(high >= low, f"Increasing {feature} shortened ETA: {low} -> {high}")
            pairs.append({"from": row[feature], "to": changed,
                          "baseline_eta_minutes": low, "changed_eta_minutes": high})
        require(any(pair["changed_eta_minutes"] > pair["baseline_eta_minutes"] for pair in pairs),
                f"All {feature} probes were flat; possible constant/insensitive model")
        return {"feature": feature, "pairs": pairs}

    check("sanity_distance", lambda: monotonic("distance_km", [25.0, 40.0, 60.0]))
    check("sanity_traffic", lambda: monotonic("traffic_index", [.8, .85, .9]))

    def invalid_inputs():
        cases = {
            "negative_distance": {"instances": [{**PROBES[0], "distance_km": -1}]},
            "empty_batch": {"instances": []},
            "infeasible": {"instances": [{**PROBES[0], "distance_km": 160, "n_stops": 21}]},
        }
        result = {}
        for name, payload in cases.items():
            reply = client.request(endpoint, payload)
            require(reply.status == 422, f"Invalid case {name}: expected 422, received {reply.status}")
            result[name] = reply.status
        return result

    check("invalid_inputs_return_422", invalid_inputs)

    def latency():
        for i in range(5):
            predict([PROBES[i % len(PROBES)]])
        measurements = []
        for i in range(samples):
            reply = client.request(endpoint, {"instances": [PROBES[i % len(PROBES)]]})
            prediction(reply, 1, expected_version)  # Wrong/failed responses never qualify as fast success.
            measurements.append(reply.elapsed_ms)
        p95 = percentile95(measurements)
        document["latency"] = {"successful_requests": len(measurements), "p95_ms": p95,
                               "threshold_ms": max_p95_ms, "samples_ms": measurements}
        require(p95 <= max_p95_ms, f"p95 {p95:.6f} ms exceeds {max_p95_ms:.6f} ms")
        return {"successful_requests": samples, "p95_ms": p95, "threshold_ms": max_p95_ms}

    check("latency_p95", latency)
    document["passed"] = all(item["passed"] for item in document["checks"].values())
    return document


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def main(argv=None):
    parser = Parser(description=__doc__)
    parser.add_argument("--url", required=True, help="Part A server root or full prediction URL")
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-p95-ms", type=finite_positive, default=80.0)
    parser.add_argument("--samples", type=sample_count, default=30)
    parser.add_argument("--timeout", type=finite_positive, default=5.0)
    parser.add_argument("--out", type=Path, help="Save JSON evidence; refuses to overwrite an existing file")
    args = parser.parse_args(argv)
    try:
        if not args.expect_version.strip():
            raise ValueError("--expect-version must not be blank")
        if args.out and args.out.exists():
            raise ValueError(f"Evidence already exists: {args.out}; choose a new --out")
        result = run_gate(args.url, args.expect_version, args.max_p95_ms, args.samples, args.timeout)
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result["exit_code"] = 0 if result["passed"] else 1
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x") as stream:
                stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(f"Evidence saved: {args.out}", flush=True)
        print("SMOKE GATE: " + ("PASS" if result["passed"] else "FAIL"), flush=True)
        return result["exit_code"]
    except Exception as exc:
        print(f"SMOKE GATE: FAIL ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
