# app/main.py
from fastapi import FastAPI, HTTPException
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


api = FastAPI(title="NER Hackathon Stub (Async, rule-based)", lifespan=lifespan)


class PredictIn(BaseModel):
    input: str


class SpanOut(BaseModel):
    start_index: int
    end_index: int
    entity: str


_queue = asyncio.Queue(maxsize=settings.queue_maxsize)
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
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, predict_bio, texts)
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


@api.get("/health")
async def health():
    return {"status": "ok"}


@api.post("/api/predict", response_model=List[SpanOut])
async def predict(inp: PredictIn):
    try:
        if settings.use_queue:
            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            try:
                await asyncio.wait_for(
                    _queue.put({"texts": [inp.input], "future": fut}),
                    timeout=settings.batch_timeout_ms / 1000,
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=503, detail="Queue is busy")
            spans_batch = await asyncio.wait_for(fut, timeout=max(0.001, settings.request_timeout_ms / 1000))
        else:
            # fallback: вычисление в пуле потоков, чтобы не блокировать event loop
            loop = asyncio.get_running_loop()
            spans_batch = await loop.run_in_executor(None, predict_bio, [inp.input])
        spans = spans_batch[0]
        result = [SpanOut(start_index=int(s), end_index=int(e), entity=str(lab)) for (s, e, lab) in spans]
        if settings.return_debug:
            logger.debug(
                "API response: {}",
                [{"start_index": r.start_index, "end_index": r.end_index, "entity": r.entity} for r in result],
            )
        return result
    except Exception as e:
        logger.exception("/api/predict error: {}", repr(e))
        raise
