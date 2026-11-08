"""Contract tests. Generate the supplied models before running pytest from the root."""
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import UUID

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from partA import eta_app as service

URL = "/v1/models/eta:predict"
VALID = dict(distance_km=20.0, n_stops=4, hour=12, weekday=2,
             traffic_index=0.4, rain=0, weight_kg=8.0)


@pytest.fixture
def client():
    with TestClient(service.app) as connection:
        yield connection


def assert_id(response, expected=None):
    request_id = response.headers["X-Request-ID"]
    assert response.json()["request_id"] == request_id
    if expected is not None:
        assert request_id == expected
    else:
        assert UUID(request_id).version == 4


def assert_validation(response, field=None):
    assert response.status_code == 422
    body = response.json()
    assert set(body) == {"error", "details", "request_id"}
    assert body["error"] == "validation_failed"
    assert body["details"]
    assert all(set(detail) == {"field", "msg"} for detail in body["details"])
    if field:
        assert any(field in detail["field"] for detail in body["details"])


def test_valid_request_uses_real_model_and_rounds(client):
    response = client.post(URL, json={"instances": [VALID]})
    assert response.status_code == 200
    expected = service.model.predict(pd.DataFrame([VALID], columns=service.FEATURE_COLUMNS))[0]
    body = response.json()
    assert body["predictions"] == [float(np.round(expected, 1))]
    assert body["model_version"] == service.MODEL_VERSION
    assert_id(response)


def test_batch_order_and_feature_order(client, monkeypatch):
    def sentinel(frame):
        assert list(frame.columns) == [
            "distance_km", "n_stops", "hour", "weekday", "traffic_index", "rain", "weight_kg"
        ]
        assert frame["distance_km"].tolist() == [30.0, 10.0, 20.0]
        return np.array([301.24, 101.26, 201.27])
    monkeypatch.setattr(service.model, "predict", sentinel)
    response = client.post(URL, json={"instances": [
        {**VALID, "distance_km": distance} for distance in [30.0, 10.0, 20.0]
    ]})
    assert response.status_code == 200
    assert response.json()["predictions"] == [301.2, 101.3, 201.3]


@pytest.mark.parametrize("field,value", [
    ("distance_km", 0.09), ("distance_km", 500.01),
    ("n_stops", 0), ("n_stops", 61),
    ("hour", -1), ("hour", 24), ("weekday", -1), ("weekday", 7),
    ("traffic_index", -0.01), ("traffic_index", 1.01),
    ("rain", -1), ("rain", 2), ("weight_kg", 0), ("weight_kg", 1000.01),
])
def test_out_of_range(client, field, value):
    response = client.post(URL, json={"instances": [{**VALID, field: value}]})
    assert_validation(response, field)


@pytest.mark.parametrize("field,value", [
    ("distance_km", 0.1), ("distance_km", 500),
    ("n_stops", 1), ("n_stops", 60), ("hour", 0), ("hour", 23),
    ("weekday", 0), ("weekday", 6), ("traffic_index", 0), ("traffic_index", 1),
    ("rain", 0), ("rain", 1), ("weight_kg", 0.001), ("weight_kg", 1000),
])
def test_valid_field_boundaries(client, field, value):
    assert client.post(URL, json={"instances": [{**VALID, field: value}]}).status_code == 200


def test_missing_feature(client):
    row = VALID.copy()
    del row["hour"]
    assert_validation(client.post(URL, json={"instances": [row]}), "hour")


@pytest.mark.parametrize("field,value", [
    ("distance_km", "twenty"), ("distance_km", "20"),
    ("hour", 1.5), ("rain", True), ("weight_kg", None),
])
def test_wrong_types(client, field, value):
    assert_validation(client.post(URL, json={"instances": [{**VALID, field: value}]}), field)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_numbers(client, token):
    body = json.dumps({"instances": [{**VALID, "distance_km": "TOKEN"}]})
    response = client.post(URL, content=body.replace('"TOKEN"', token),
                           headers={"Content-Type": "application/json"})
    assert_validation(response)
    assert_id(response)


def test_infeasible_cross_field_rule(client):
    response = client.post(URL, json={"instances": [{**VALID, "distance_km": 151, "n_stops": 21}]})
    assert_validation(response)
    assert "infeasible" in response.text.lower()


@pytest.mark.parametrize("distance,stops", [(150, 21), (151, 20)])
def test_cross_field_rule_requires_both_strict_inequalities(client, distance, stops):
    response = client.post(URL, json={"instances": [{**VALID, "distance_km": distance, "n_stops": stops}]})
    assert response.status_code == 200


@pytest.mark.parametrize("size", [1, 500])
def test_valid_batch_size_limits(client, size):
    response = client.post(URL, json={"instances": [VALID] * size})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == size


@pytest.mark.parametrize("size", [0, 501])
def test_invalid_batch_size_limits(client, size):
    assert_validation(client.post(URL, json={"instances": [VALID] * size}), "instances")


@pytest.mark.parametrize("body", [{}, {"instances": None}, {"instances": VALID}])
def test_invalid_body_shape(client, body):
    assert_validation(client.post(URL, json=body), "instances")


def test_malformed_json_has_error_envelope_and_id(client):
    response = client.post(URL, content='{"instances":', headers={"Content-Type": "application/json"})
    assert_validation(response)
    assert_id(response)


@pytest.mark.parametrize("valid", [True, False])
def test_request_id_echo_on_success_and_error(client, valid):
    response = client.post(URL, json={"instances": [VALID] if valid else []},
                           headers={"X-Request-ID": "student-demo-123"})
    assert response.status_code == (200 if valid else 422)
    assert_id(response, "student-demo-123")


def test_generated_ids_are_unique_including_error(client):
    responses = [client.post(URL, json={"instances": rows}) for rows in [[VALID], []]]
    for response in responses:
        assert_id(response)
    assert responses[0].headers["X-Request-ID"] != responses[1].headers["X-Request-ID"]


def test_health(client):
    response = client.get("/healthz", headers={"X-Request-ID": "health-check"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_version"] == service.MODEL_VERSION
    assert_id(response, "health-check")


@pytest.mark.parametrize("method,path,status", [("get", "/missing", 404), ("get", URL, 405)])
def test_http_errors_preserve_ids(client, method, path, status):
    response = getattr(client, method)(path, headers={"X-Request-ID": "http-error"})
    assert response.status_code == status
    assert_id(response, "http-error")


@pytest.mark.parametrize("request_id", [None, "failure-check"])
def test_inference_failure_is_generic(client, monkeypatch, request_id):
    def broken_predict(frame):
        raise RuntimeError("secret-internal-path-and-password")
    monkeypatch.setattr(service.model, "predict", broken_predict)
    headers = {"X-Request-ID": request_id} if request_id is not None else {}
    response = client.post(URL, json={"instances": [VALID]}, headers=headers)
    assert response.status_code == 500
    assert response.json()["error"] == "internal_server_error"
    assert "secret" not in response.text and "RuntimeError" not in response.text
    assert_id(response, request_id)


@pytest.mark.parametrize("bad_predictions", [[float("nan")], [float("inf")], [1, 2]])
def test_invalid_model_output_fails_safely(client, monkeypatch, bad_predictions):
    monkeypatch.setattr(service.model, "predict", lambda frame: bad_predictions)
    response = client.post(URL, json={"instances": [VALID]})
    assert response.status_code == 500
    assert response.json()["error"] == "internal_server_error"
    assert_id(response)


def test_metrics_count_statuses_and_exclude_scrapes(client, monkeypatch):
    # Counter deltas make the test independent of test order or prior requests.
    def count(status):
        return service.REGISTRY.get_sample_value(
            "eta_requests_total", {"path": URL, "status": str(status)}) or 0
    before = {status: count(status) for status in [200, 422, 500]}
    client.post(URL, json={"instances": [VALID]})
    client.post(URL, json={"instances": []})
    def fail(frame):
        raise RuntimeError("metrics test")
    monkeypatch.setattr(service.model, "predict", fail)
    client.post(URL, json={"instances": [VALID]})
    assert all(count(status) == before[status] + 1 for status in before)
    total_before = service.REGISTRY.get_sample_value("eta_request_latency_seconds_count")
    first = client.get("/metrics")
    second = client.get("/metrics")
    assert first.status_code == second.status_code == 200
    assert "text/plain" in second.headers["content-type"]
    assert "X-Request-ID" in second.headers
    for text in ["eta_requests_total{", "eta_request_latency_seconds_bucket{",
                 'le="+Inf"', "eta_request_latency_seconds_sum", "eta_request_latency_seconds_count"]:
        assert text in second.text
    assert 'path="/metrics"' not in second.text
    assert service.REGISTRY.get_sample_value("eta_request_latency_seconds_count") == total_before


def test_model_is_not_reloaded_per_request(client, monkeypatch):
    def forbidden_load(*args, **kwargs):
        raise AssertionError("joblib.load called during request")
    monkeypatch.setattr(joblib, "load", forbidden_load)
    for _ in range(2):
        assert client.post(URL, json={"instances": [VALID]}).status_code == 200


def test_eta_model_environment_override(tmp_path):
    alternate = tmp_path / "alternate.joblib"
    joblib.dump({"model": service.model, "feature_columns": service.FEATURE_COLUMNS,
                 "version": "environment-check"}, alternate)
    environment = os.environ.copy()
    environment["ETA_MODEL"] = str(alternate)
    result = subprocess.run(
        [sys.executable, "-c", "from partA.eta_app import MODEL_VERSION; print(MODEL_VERSION)"],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "environment-check"
