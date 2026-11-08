"""SwiftParcel REST contract. Run from project root: python -m uvicorn partA.eta_app:app.

Generate bundles first with the unchanged eta_data.py. Pydantic 2 is required.
The model is loaded once per process, not once per request.
"""
from __future__ import annotations

import logging
import os
import time
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)
MODEL_PATH = os.environ.get("ETA_MODEL", "data/eta_v1.joblib")
_bundle = joblib.load(MODEL_PATH)
model = _bundle["model"]
FEATURE_COLUMNS = list(_bundle["feature_columns"])
MODEL_VERSION = str(_bundle["version"])

# A private registry prevents duplicate registration during isolated app imports.
REGISTRY = CollectorRegistry()
REQUESTS = Counter(
    "eta_requests_total", "Completed HTTP requests, excluding metrics scrapes.",
    ["path", "status"], registry=REGISTRY,
)
LATENCY = Histogram(
    "eta_request_latency_seconds", "Server request latency in seconds, excluding scrapes.",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.08, 0.1, 0.25, 0.5, 1, 2.5, 5),
    registry=REGISTRY,
)


class Delivery(BaseModel):
    # Numbers must be finite JSON numbers; integer fields reject booleans/fractions.
    # Integer JSON values are accepted for the continuous fields, e.g. distance=20.
    model_config = ConfigDict(strict=True, allow_inf_nan=False, extra="forbid")

    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def check_feasibility(self) -> Delivery:
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible: distance exceeds 150 km and stops exceed 20")
        return self


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[Delivery] = Field(min_length=1, max_length=500)


class RequestContextMiddleware:
    """Pure ASGI middleware: IDs, safe unexpected errors, and HTTP instrumentation."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        supplied = Headers(scope=scope).get("x-request-id")
        request_id = supplied if supplied is not None else str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status = 500
        response_started = False

        async def send_with_id(message: Message):
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            logger.exception("Unhandled request failure; request_id=%s", request_id)
            if response_started:
                raise
            response = JSONResponse(
                status_code=500,
                content={"error": "internal_server_error", "request_id": request_id},
            )
            await response(scope, receive, send_with_id)
        finally:
            if scope.get("path") != "/metrics":
                # Use route templates; arbitrary 404 URLs cannot create unbounded labels.
                route = scope.get("route")
                path = getattr(route, "path", "__unmatched__")
                REQUESTS.labels(path=path, status=str(status)).inc()
                LATENCY.observe(time.perf_counter() - started)


# Only the specified routes are exposed. This also avoids unrelated JSON schema
# endpoints whose output cannot contain a per-request ID without changing OpenAPI.
app = FastAPI(
    title="SwiftParcel ETA", docs_url=None, redoc_url=None, openapi_url=None,
    redirect_slashes=False,
)
app.add_middleware(RequestContextMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Do not serialize Pydantic's ctx directly: it may contain a ValueError object.
    details = []
    for error in exc.errors():
        location = [str(item) for item in error["loc"] if item != "body"]
        details.append({"field": ".".join(location) or "body", "msg": error["msg"]})
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_failed", "details": details,
            "request_id": request.state.request_id,
        },
    )


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail), "request_id": request.state.request_id},
        headers=exc.headers,
    )


@app.post("/v1/models/eta:predict")
def predict(payload: PredictionRequest, request: Request):
    rows = [delivery.model_dump() for delivery in payload.instances]
    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    # A synchronous route keeps CPU inference off the ASGI event loop. Middleware
    # handles model exceptions and sends a generic 500 without leaking internals.
    predictions = np.asarray(model.predict(frame), dtype=float)
    if predictions.shape != (len(rows),) or not np.isfinite(predictions).all():
        raise ValueError("Model returned invalid predictions")
    return {
        "predictions": np.round(predictions, 1).tolist(),
        "model_version": MODEL_VERSION,
        "request_id": request.state.request_id,
    }


@app.get("/healthz")
def health(request: Request):
    # The PDF's global JSON request-ID rule also applies to health responses.
    return {"status": "ok", "model_version": MODEL_VERSION,
            "request_id": request.state.request_id}


@app.get("/metrics")
def metrics():
    registry = REGISTRY
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        # This environment variable must be set BEFORE starting Python. Use a
        # new, empty directory for each server run; see RUN_PART_A.md.
        from prometheus_client import multiprocess
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
    return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})
