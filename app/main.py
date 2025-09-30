# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from loguru import logger
from .logging_setup import setup_logging
from pydantic import BaseModel
import asyncio
from typing import List
from .infer import predict_bio
from .settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info(
        "Starting API with settings: use_queue={}, batch_max_size={}, timeout_ms={}",
        settings.use_queue,
        settings.batch_max_size,
        settings.batch_timeout_ms,
    )
    # лёгкий прогрев графа и токенизатора; не критично медленный даже на CPU
    sample = ["тестовый прогрев"] * 4
    try:
        logger.debug("Warmup start")
        predict_bio(sample)
        for _ in range(max(1, settings.warmup_requests // 4) - 1):
            predict_bio(sample)
        logger.debug("Warmup done")
    except Exception as e:
        logger.warning("Warmup warning: {}", repr(e))
    # Стартуем воркер после запуска приложения и старта event loop
    if settings.use_queue:
        asyncio.create_task(_batch_worker())
        logger.info("Batch worker started")
    yield


api = FastAPI(lifespan=lifespan)


class PredictOut(BaseModel):
    entities: List[List[tuple[int, int, str]]]


class SinglePredictIn(BaseModel):
    input: str


class SingleEntityOut(BaseModel):
    start_index: int
    end_index: int
    entity: str


_queue = asyncio.Queue()
BATCH_MAX_SIZE = settings.batch_max_size
BATCH_TIMEOUT_MS = settings.batch_timeout_ms


async def _batch_worker():
    while True:
        first = await _queue.get()
        batch = [first]
        try:
            while len(batch) < BATCH_MAX_SIZE:
                item = await asyncio.wait_for(_queue.get(), timeout=BATCH_TIMEOUT_MS / 1000)
                batch.append(item)
        except asyncio.TimeoutError:
            logger.debug("Batch timeout, processing {} items", len(batch))

        try:
            texts = sum([x["texts"] for x in batch], [])
            logger.debug("Processing batch of {} texts", len(texts))
            result = predict_bio(texts)
            p = 0
            for x in batch:
                n = len(x["texts"])
                x["future"].set_result(result[p : p + n])
                p += n
            logger.debug("Batch processed")
        except Exception as e:
            logger.exception("Batch error: {}", repr(e))
            # гарантируем завершение всех ожиданий при ошибке
            for x in batch:
                if not x["future"].done():
                    x["future"].set_exception(e)


# Воркер запускается в _warmup(), чтобы гарантировать наличие запущенного event loop


@api.get("/healthz")
async def healthz():
    return {"status": "ok"}


@api.post("/api/predict", response_model=List[SingleEntityOut])
async def predict_single(inp: SinglePredictIn):
    try:
        spans_batch = predict_bio([inp.input])
        spans = spans_batch[0]
        return [{"start_index": int(s), "end_index": int(e), "entity": str(lab)} for (s, e, lab) in spans]
    except Exception as e:
        logger.exception("/api/predict error: {}", repr(e))
        raise
