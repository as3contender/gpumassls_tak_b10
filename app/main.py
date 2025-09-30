# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import asyncio
from typing import List
from .infer import predict_bio
from .settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # лёгкий прогрев графа и токенизатора; не критично медленный даже на CPU
    sample = ["тестовый прогрев"] * 4
    try:
        predict_bio(sample)
        for _ in range(max(1, settings.warmup_requests // 4) - 1):
            predict_bio(sample)
    except Exception as e:
        print("Warmup warning:", e)
    # Стартуем воркер после запуска приложения и старта event loop
    asyncio.create_task(_batch_worker())
    yield


api = FastAPI(lifespan=lifespan)


class PredictIn(BaseModel):
    texts: List[str]


class Entity(BaseModel):
    label: str
    start: int
    end: int
    text: str


class PredictOut(BaseModel):
    entities: List[List[Entity]]


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
            pass

        try:
            texts = sum([x["texts"] for x in batch], [])
            result = predict_bio(texts)
            p = 0
            for x in batch:
                n = len(x["texts"])
                x["future"].set_result(result[p : p + n])
                p += n
        except Exception as e:
            # гарантируем завершение всех ожиданий при ошибке
            for x in batch:
                if not x["future"].done():
                    x["future"].set_exception(e)


# Воркер запускается в _warmup(), чтобы гарантировать наличие запущенного event loop


@api.get("/healthz")
async def healthz():
    return {"status": "ok"}


@api.post("/predict", response_model=PredictOut)
async def predict(inp: PredictIn):
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    await _queue.put({"texts": inp.texts, "future": fut})
    out = await fut
    return {"entities": out}
