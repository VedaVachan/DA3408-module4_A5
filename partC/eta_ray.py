"""SwiftParcel Ray Serve app. build(weight=0.1, extra_work_ms=0.0) -> Application.

No Ray cluster is started and no model is loaded on import. Each EtaModel replica
loads its bundle once. The optional extra CPU work applies only to the candidate.
"""
# Keep concrete endpoint types: Ray's class-based-view/serialization path must
# preserve FastAPI's body, Request and Response injection annotations.

import asyncio
from fractions import Fraction
import hashlib
import logging
import math
from numbers import Real
import os
from pathlib import Path
import time
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import ray
from ray import serve
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "swiftparcel-eta"
CPU_PER_REPLICA = 0.25
THREAD_ENV = {name: "1" for name in
              ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
ACTOR_OPTIONS = {"num_cpus": CPU_PER_REPLICA, "num_gpus": 0,
                 "runtime_env": {"env_vars": THREAD_ENV}}
AUTOSCALING = dict(min_replicas=1, max_replicas=3, target_ongoing_requests=2,
                   upscale_delay_s=5, downscale_delay_s=30)
logger = logging.getLogger(__name__)


# --- Sticky routing (pure; no Ray, model, random seed or process state) ---

def weight_fraction(weight):
    if isinstance(weight, bool) or not isinstance(weight, Real) or not 0 <= weight <= 1:
        raise ValueError("weight must be a finite number between 0 and 1")
    return Fraction(str(weight))


def choose_arm(customer_id, weight=0.1, force_candidate=False):
    if not isinstance(customer_id, str) or not customer_id:
        raise ValueError("customer_id must be a nonempty string")
    fraction = weight_fraction(weight)
    if force_candidate:
        return "v2"
    bucket = int.from_bytes(hashlib.sha256(customer_id.encode("utf-8")).digest(), "big")
    # Exact integer comparison avoids float rounding at weights 0 and 1.
    # Weight changes only the threshold; the customer hash never changes.
    return "v2" if bucket * fraction.denominator < fraction.numerator * (1 << 256) else "v1"


class Delivery(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False, extra="forbid")
    distance_km: float = Field(ge=.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def feasible(self) -> "Delivery":
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery")
        return self


class CustomerRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    customer_id: str = Field(min_length=1)
    instances: list[Delivery] = Field(min_length=1, max_length=500)


@serve.deployment(ray_actor_options=ACTOR_OPTIONS, max_ongoing_requests=8,
                  max_constructor_retry_count=1)
class EtaModel:
    def __init__(self, bundle_path, extra_work_ms=0.0):
        bundle = joblib.load(bundle_path)
        self.model = bundle["model"]
        self.feature_columns = list(bundle["feature_columns"])
        self.version = str(bundle["version"])
        self.extra_work_ms = extra_work_ms
        self.served_by = f"{self.version}:{os.getpid()}"
        self.assigned_cpus = ray.get_runtime_context().get_assigned_resources().get("CPU", 0)

    def _predict(self, rows):
        frame = pd.DataFrame(rows, columns=self.feature_columns)
        if self.extra_work_ms:
            deadline = time.perf_counter() + self.extra_work_ms / 1000
            while time.perf_counter() < deadline:
                pass  # Optional CPU work per candidate batch; disabled by default.
        values = np.asarray(self.model.predict(frame), dtype=float)
        if values.shape != (len(rows),) or not np.isfinite(values).all():
            raise ValueError("Invalid model output")
        return {"predictions": np.round(values, 1).tolist(), "model_version": self.version,
                "served_by": self.served_by, "replica_cpus": self.assigned_cpus}

    async def predict(self, rows):
        # Keep the actor's asyncio loop responsive while CPU inference runs.
        return await asyncio.to_thread(self._predict, rows)


http_app = FastAPI(title="SwiftParcel sticky canary", docs_url=None, redoc_url=None,
                   openapi_url=None, redirect_slashes=False)


@serve.deployment(name="eta-router", num_replicas=1, ray_actor_options=ACTOR_OPTIONS,
                  max_ongoing_requests=32, max_constructor_retry_count=1)
@serve.ingress(http_app)
class EtaRouter:
    def __init__(self, production, candidate, weight):
        self.production = production
        self.candidate = candidate
        self.weight = weight
        self.instance_id = str(uuid4())
        self.assigned_cpus = ray.get_runtime_context().get_assigned_resources().get("CPU", 0)

    @http_app.get("/healthz")
    def health(self):
        return {"status": "ok", "candidate_weight": self.weight,
                "router_instance_id": self.instance_id, "router_cpus": self.assigned_cpus}

    @http_app.post("/v1/models/eta:predict")
    async def predict(self, request: Request,
                      payload: CustomerRequest = Body(...)):
        supplied = request.headers.get("X-Request-ID")
        request_id = supplied if supplied is not None else str(uuid4())
        # Ray owns the response request-ID header.
        arm = choose_arm(payload.customer_id, self.weight,
                         force_candidate=request.headers.get("X-Beta") == "1")
        handle = self.candidate if arm == "v2" else self.production
        try:
            result = await handle.predict.remote([row.model_dump() for row in payload.instances])
        except Exception:
            logger.exception("Ray inference failed; request_id=%s", request_id)
            return JSONResponse(status_code=500,
                                content={"error": "internal_server_error", "request_id": request_id})
        result.update(arm=arm, request_id=request_id, router_cpus=self.assigned_cpus)
        return result


def build(weight=0.1, extra_work_ms=0.0):
    weight_fraction(weight)
    if (isinstance(extra_work_ms, bool) or not isinstance(extra_work_ms, Real)
            or not math.isfinite(extra_work_ms) or extra_work_ms < 0):
        raise ValueError("extra_work_ms must be a finite nonnegative number")
    paths = {label: ROOT / "data" / f"eta_{label}.joblib" for label in ("v1", "v2")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Generate the supplied models first: {path}")
    prod = EtaModel.options(name="eta-prod", num_replicas=2).bind(str(paths["v1"]), 0.0)
    cand = EtaModel.options(name="eta-candidate", num_replicas="auto",
                            autoscaling_config=dict(AUTOSCALING)).bind(str(paths["v2"]), float(extra_work_ms))
    return EtaRouter.bind(prod, cand, float(weight))
