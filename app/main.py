# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from typing import List
from .model import ONNXNER
from .settings import settings

api = FastAPI()
ner = ONNXNER()


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


@api.on_event("startup")
async def _warmup():
    # лёгкий прогрев графа и токенизатора; не критично медленный даже на CPU
    sample = ["тестовый прогрев"] * 4
    try:
        ner.predict(sample)
        for _ in range(max(1, settings.warmup_requests // 4) - 1):
            ner.predict(sample)
    except Exception as e:
        print("Warmup warning:", e)


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

        texts = sum([x["texts"] for x in batch], [])
        result = ner.predict(texts)

        p = 0
        for x in batch:
            n = len(x["texts"])
            x["future"].set_result(result[p : p + n])
            p += n


asyncio.get_event_loop().create_task(_batch_worker())


@api.get("/healthz")
async def healthz():
    return {"status": "ok"}


@api.post("/predict", response_model=PredictOut)
async def predict(inp: PredictIn):
    fut = asyncio.get_event_loop().create_future()
    await _queue.put({"texts": inp.texts, "future": fut})
    out = await fut
    return {"entities": out}
