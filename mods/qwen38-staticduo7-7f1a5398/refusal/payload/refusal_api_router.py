# SPDX-License-Identifier: Apache-2.0
"""Global refusal dial with a drained, cache-invalidating transition.

POST supports one AsyncMP API client and DP=1. TP worker count is verified.
Uncertain pause/reset or partial worker mutation leaves the control blocked;
restart the candidate server before using the dial again. Per-request salts
continue to select lambda independently and do not call this endpoint.
"""

import asyncio
import math

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vllm.engine.protocol import EngineClient
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.entrypoints.serve.utils.api_utils import validate_json_request
from vllm.logger import init_logger

logger = init_logger(__name__)
router = APIRouter()

# Cota amplia a proposito. 0 = base intacta (proyeccion identidad bit a bit);
# 1.5 = el punto calibrado de ESTE checkpoint (lambda_eff medido 1.4994-1.5008
# sobre 7 tensores, spread 0.09%); >1.5 sobredispara e invierte la componente;
# <0 AMPLIFICA la direccion en vez de eliminarla. La curva de refusal NO es
# monotona: a 2.5 el refusal vuelve a subir, asi que el maximo no es un capricho.
LAMBDA_MIN = -1.0
LAMBDA_MAX = 4.0


class RefusalLambdaRequest(BaseModel):
    lambda_: float = Field(..., alias="lambda", ge=LAMBDA_MIN, le=LAMBDA_MAX)

    model_config = {"populate_by_name": True}


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


async def _rpc(raw_request: Request, method: str, args: tuple = ()):
    return await engine_client(raw_request).collective_rpc(method, args=args)


class _Control:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.blocked = False


def _control(raw_request):
    state = raw_request.app.state
    if not hasattr(state, "refusal_lambda_control"):
        state.refusal_lambda_control = _Control()
    return state.refusal_lambda_control


def _all_ranks(results, count, value):
    return (
        isinstance(results, (list, tuple))
        and len(results) == count
        and all(
            isinstance(r, (float, int)) and not isinstance(r, bool)
            and math.isfinite(r) and abs(r - value) <= 1e-9
            for r in results
        )
    )


def _supported_workers(engine):
    # A frontend-local lock cannot serialize multiple API clients or DP engines.
    if not isinstance(getattr(engine, "engine_core", None), AsyncMPClient):
        return None
    if getattr(engine, "_client_count", None) != 1:
        return None
    parallel = engine.vllm_config.parallel_config
    if parallel.data_parallel_size != 1:
        return None
    count = parallel.world_size
    if not isinstance(count, int) or count <= 0:
        return None
    return count


@router.post("/admin/refusal_lambda", dependencies=[Depends(validate_json_request)])
async def set_refusal_lambda(request: RefusalLambdaRequest, raw_request: Request):
    value = float(request.lambda_)
    if not math.isfinite(value):
        return JSONResponse(status_code=422, content={"error": "lambda must be finite"})
    engine = engine_client(raw_request)
    count = _supported_workers(engine)
    if count is None:
        return JSONResponse(status_code=503, content={
            "error": "global dial requires one AsyncMP API client and DP=1",
        })
    control = _control(raw_request)
    async with control.lock:
        if control.blocked:
            return JSONResponse(status_code=503, content={
                "error": "global dial blocked after an uncertain transition; restart server",
            })
        if await engine.is_paused():
            return JSONResponse(status_code=409, content={
                "error": "engine already paused; refusal dial does not own this pause",
            })
        stage = "pause_and_clear_cache"
        pause_complete = False
        mutation_started = False
        try:
            # EngineCoreProc drains running work, synchronizes workers, and raises
            # if either local or connector prefix-cache reset returns false.
            await engine.pause_generation(mode="wait", clear_cache=True)
            pause_complete = True
            if not await engine.is_paused():
                raise RuntimeError("engine did not retain the requested pause")
            # PAUSED_NEW leaves waiting/preempted requests queued. Never change
            # the global default while one might contain output from the old one.
            if engine.get_num_unfinished_requests() != 0:
                await engine.resume_generation()
                return JSONResponse(status_code=409, content={
                    "error": "pending requests remain; lambda unchanged; retry when idle",
                })
            stage = "set_workers"
            mutation_started = True
            results = await engine.collective_rpc("set_refusal_lambda", args=(value,))
            if not _all_ranks(results, count, value):
                raise RuntimeError("worker setter results are missing or inconsistent")
            stage = "verify_workers"
            observed = await engine.collective_rpc("get_refusal_lambda")
            if not _all_ranks(observed, count, value):
                raise RuntimeError("worker readback is missing or inconsistent")
            stage = "resume"
            await engine.resume_generation()
        except asyncio.CancelledError:
            control.blocked = True
            logger.error("refusal transition cancelled at %s; restart required", stage)
            raise
        except Exception as exc:
            control.blocked = True
            # Never resume after an uncertain pause/reset or a partial RPC. A
            # resume here would permit stale KV or inconsistent TP ranks to run.
            logger.exception("refusal transition failed at %s", stage)
            return JSONResponse(status_code=503, content={
                "error": "global transition failed; restart server before resuming",
                "stage": stage,
                "pause_complete": pause_complete,
                "mutation_started": mutation_started,
                "detail": str(exc),
            })
        logger.info("refusal lambda changed to %s after cache reset on %d ranks", value, count)
        return JSONResponse(content={"lambda": value, "ranks": count, "cache_reset": True})


@router.get("/admin/refusal_lambda")
async def get_refusal_lambda(raw_request: Request):
    control = _control(raw_request)
    async with control.lock:
        engine = engine_client(raw_request)
        results = await _rpc(raw_request, "get_refusal_lambda")
        count = _supported_workers(engine)
        candidate = results[0] if results else None
        consistent = (
            count is not None and isinstance(candidate, (float, int))
            and not isinstance(candidate, bool) and math.isfinite(candidate)
            and _all_ranks(results, count, candidate)
        )
        return JSONResponse(content={
            "lambda": candidate if consistent else None,
            "consistent": consistent,
            "per_rank": [
                repr(r) if isinstance(r, float) and not math.isfinite(r) else r
                for r in results
            ],
            "blocked": control.blocked,
        })


def attach_router(app: FastAPI):
    import os

    # Se gatea por VARIABLE DE ENTORNO, no por is_enabled().
    #
    # El estado de refusal_projection nace en init_from_env, y eso ocurre en UN
    # solo sitio: el constructor del modelo (models/qwen4_exp/nvidia/
    # model.py). Con mp + TP=2 el modelo vive en los procesos worker, mientras
    # que este router se monta en el proceso del servidor API, que no construye
    # modelo nunca. Alli is_enabled() es False SIEMPRE, la ruta no se montaba, y
    # /admin/refusal_lambda daba 404 permanente aunque la proyeccion estuviera
    # activa en los workers. La variable si la ven los dos procesos, porque es
    # del contenedor.
    #
    # Es el patron de serve/lora/api_router.py, que gatea por
    # envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING y no por estado en proceso.
    if not (os.environ.get("VLLM_REFUSAL_DIRECTION")
            or os.environ.get("SGLANG_REFUSAL_DIRECTION")):
        # Sin direccion no hay hook que controlar: no se monta. Un endpoint que
        # devuelve 503 en cada llamada es peor que no tenerlo, porque el panel
        # lo ofreceria como dial vivo.
        return
    logger.warning(
        "Proyeccion rank-1 ACTIVA: /admin/refusal_lambda montado. Esta ruta "
        "cambia el comportamiento del modelo en caliente y NO debe salir por "
        "el ingress publico."
    )
    app.include_router(router)
