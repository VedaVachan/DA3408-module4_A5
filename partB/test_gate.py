"""Synthetic policy tests: no tracking server, registry, model, or network needed."""
from copy import deepcopy
import pytest
from partB.eta_mlflow import decide


def metrics(mae=10.0, rush=20.0, rain=30.0, latency=5.0):
    return dict(mae=mae, mae_rush=rush, mae_rain=rain, p95_ms=latency)


def test_all_criteria_pass():
    result = decide(metrics(), metrics(mae=9.0, latency=9.0))
    assert result["promote"] is True
    assert all(result["criteria"].values())


@pytest.mark.parametrize("candidate_mae,passed", [(9.699999, True), (9.7, True), (9.700001, False)])
def test_overall_gain_boundary(candidate_mae, passed):
    result = decide(metrics(), metrics(mae=candidate_mae))
    assert result["criteria"]["overall_gain"] is passed
    assert result["promote"] is passed


@pytest.mark.parametrize("latency,passed", [(9.999999, True), (10.0, True), (10.000001, False)])
def test_latency_boundary(latency, passed):
    result = decide(metrics(), metrics(mae=9.0, latency=latency))
    assert result["criteria"]["latency"] is passed
    assert result["promote"] is passed


@pytest.mark.parametrize("rush,passed", [(20.999999, True), (21.0, True), (21.000001, False)])
def test_rush_slice_boundary(rush, passed):
    result = decide(metrics(), metrics(mae=9.0, rush=rush))
    assert result["criteria"]["rush_slice"] is passed
    assert result["promote"] is passed


@pytest.mark.parametrize("rain,passed", [(31.499999, True), (31.5, True), (31.500001, False)])
def test_rain_slice_boundary(rain, passed):
    result = decide(metrics(), metrics(mae=9.0, rain=rain))
    assert result["criteria"]["rain_slice"] is passed
    assert result["promote"] is passed


@pytest.mark.parametrize("key,value,failed", [
    ("mae", 9.8, "overall_gain"), ("p95_ms", 10.1, "latency"),
    ("mae_rush", 21.1, "rush_slice"), ("mae_rain", 31.6, "rain_slice"),
])
def test_exactly_one_failed_criterion_blocks_promotion(key, value, failed):
    candidate = metrics(mae=9.0)
    candidate[key] = value
    result = decide(metrics(), candidate)
    assert result["promote"] is False
    assert result["criteria"][failed] is False
    assert all(passed for name, passed in result["criteria"].items() if name != failed)


def test_overall_gain_is_relative():
    result = decide(metrics(mae=100), metrics(mae=98))
    assert result["criteria"]["overall_gain"] is False


def test_slice_regression_is_relative():
    result = decide(metrics(), metrics(mae=9, rush=20.5, rain=31))
    assert result["promote"] is True


@pytest.mark.parametrize("candidate_mae,passed", [(3.999999, True), (4.0, True), (4.000001, False)])
def test_strict_sixty_percent_gain_boundary(candidate_mae, passed):
    assert decide(metrics(), metrics(mae=candidate_mae), min_rel_gain=.60)["promote"] is passed


def test_latency_is_absolute_candidate_limit():
    assert decide(metrics(latency=1000), metrics(mae=9, latency=10))["promote"] is True


def test_zero_slice_tolerance_accepts_equal_and_rejects_regression():
    assert decide(metrics(), metrics(mae=9), max_slice_regression=0)["promote"] is True
    assert decide(metrics(), metrics(mae=9, rush=20.000001), max_slice_regression=0)["promote"] is False


def test_inputs_are_not_modified():
    prod, candidate = metrics(), metrics(mae=9)
    original = deepcopy((prod, candidate))
    decide(prod, candidate)
    assert (prod, candidate) == original


@pytest.mark.parametrize("arm", ["prod", "candidate"])
@pytest.mark.parametrize("key", ["mae", "mae_rush", "mae_rain", "p95_ms"])
def test_missing_metrics_fail_closed(arm, key):
    values = {"prod": metrics(), "candidate": metrics(mae=9)}
    del values[arm][key]
    result = decide(values["prod"], values["candidate"])
    assert result["promote"] is False
    assert result["criteria"]["valid_metrics"] is False


@pytest.mark.parametrize("arm", ["prod", "candidate"])
@pytest.mark.parametrize("key", ["mae", "mae_rush", "mae_rain", "p95_ms"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, None, "1", True])
def test_invalid_metrics_fail_closed(arm, key, bad):
    values = {"prod": metrics(), "candidate": metrics(mae=9)}
    values[arm][key] = bad
    assert decide(values["prod"], values["candidate"])["promote"] is False


@pytest.mark.parametrize("name", ["min_rel_gain", "max_p95_ms", "max_slice_regression"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, None, "0.03", True])
def test_invalid_thresholds_fail_closed(name, bad):
    assert decide(metrics(), metrics(mae=9), **{name: bad})["promote"] is False


@pytest.mark.parametrize("bad", [None, [], 1])
def test_invalid_metric_containers_fail_closed(bad):
    assert decide(bad, metrics(mae=9))["promote"] is False
    assert decide(metrics(), bad)["promote"] is False


def test_zero_overall_baseline_requires_zero_gain_and_zero_candidate_error():
    assert decide(metrics(mae=0), metrics(mae=0))["promote"] is False
    assert decide(metrics(mae=0), metrics(mae=0), min_rel_gain=0)["promote"] is True
    assert decide(metrics(mae=0), metrics(mae=.000001), min_rel_gain=0)["promote"] is False


@pytest.mark.parametrize("key", ["mae_rush", "mae_rain"])
def test_zero_slice_baseline_allows_only_zero_error(key):
    prod, candidate = metrics(), metrics(mae=9)
    prod[key] = candidate[key] = 0
    assert decide(prod, candidate)["promote"] is True
    candidate[key] = .000001
    assert decide(prod, candidate)["promote"] is False
