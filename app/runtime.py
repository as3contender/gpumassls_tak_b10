# app/runtime.py
import os
from typing import List, Dict, Any
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from .settings import settings
from loguru import logger


def load_labels(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


def _pick_providers() -> list:
    """
    Возвращает список провайдеров для ORT.
    - Если ORT_FORCE_CPU=1 → только CPUExecutionProvider.
    - Если доступен CUDAExecutionProvider → CUDA + CPU.
    - Иначе → CPU.
    """
    if settings.ort_force_cpu or os.getenv("ORT_FORCE_CPU") == "1":
        return ["CPUExecutionProvider"]

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class ONNXNER:
    def __init__(self):
        self.labels = load_labels(settings.labels_path)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_dir, use_fast=True)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Threading & execution mode tuning
        try:
            so.intra_op_num_threads = int(settings.ort_intra_op_num_threads)
        except Exception:
            pass
        try:
            so.inter_op_num_threads = int(settings.ort_inter_op_num_threads)
        except Exception:
            pass
        if getattr(settings, "ort_execution_mode_parallel", True):
            try:
                so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            except Exception:
                pass

        model_path = os.path.join(settings.model_dir, "model.onnx")
        providers = _pick_providers()
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)

        # Log provider details to verify CUDA usage in production
        try:
            logger.info("ORT available providers: {}", ort.get_available_providers())
            logger.info("ORT session providers: {}", self.session.get_providers())
        except Exception:
            pass

        # Имена входов/выходов
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _encode(self, texts: List[str]):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.max_seq_len,
            return_offsets_mapping=True,
            return_tensors="np",
            is_split_into_words=False,
        )
        feed = {}
        # подставляем по существующим именам
        if "input_ids" in self.input_names:
            feed["input_ids"] = enc["input_ids"].astype(np.int64)
        else:
            # на случай странных имён входов типа input.1
            for n in self.input_names:
                if n.endswith("input_ids"):
                    feed[n] = enc["input_ids"].astype(np.int64)

        if "attention_mask" in self.input_names:
            feed["attention_mask"] = enc["attention_mask"].astype(np.int64)
        else:
            for n in self.input_names:
                if n.endswith("attention_mask"):
                    feed[n] = enc["attention_mask"].astype(np.int64)

        if "token_type_ids" in enc and "token_type_ids" in self.input_names:
            feed["token_type_ids"] = enc["token_type_ids"].astype(np.int64)

        offsets = enc["offset_mapping"]  # [B,T,2]
        return feed, offsets, texts

    def _bio_to_spans(self, text: str, ids_row: np.ndarray, offsets_row: np.ndarray) -> list[Dict[str, Any]]:
        ents, cur = [], None  # cur = (label, start, end)

        def flush():
            nonlocal cur
            if cur and cur[2] > cur[1]:
                label, s, e = cur
                ents.append({"label": label, "start": int(s), "end": int(e), "text": text[s:e]})
            cur = None

        for i, lid in enumerate(ids_row.tolist()):
            s, e = offsets_row[i]
            if s == 0 and e == 0:  # спецтокены/паддинг
                flush()
                continue
            lab = self.labels[lid]
            parts = lab.split("-", 1)
            prefix, ent = (parts[0], parts[1]) if len(parts) == 2 else ("O", "O")

            if prefix == "B":
                flush()
                cur = (ent, s, e)
            elif prefix == "I" and cur and cur[0] == ent:
                cur = (cur[0], cur[1], e)
            else:
                flush()
                if prefix == "I":  # «висячее» I → начнём новую
                    cur = (ent, s, e)
        flush()
        return ents

    def predict(self, texts: List[str]) -> list[list[Dict[str, Any]]]:
        feed, offsets, orig = self._encode(texts)
        outs = self.session.run(self.output_names, feed)
        # ищем логиты [B,T,C]
        logits = next((o for o in outs if isinstance(o, np.ndarray) and o.ndim == 3), None)
        if logits is None:
            raise RuntimeError("Не найден выход [B,T,C] с логитами в ONNX-графе.")
        pred_ids = logits.argmax(axis=-1)  # [B,T]

        batch = []
        for b in range(pred_ids.shape[0]):
            batch.append(self._bio_to_spans(orig[b], pred_ids[b], offsets[b]))
        return batch

    def predict_bio(self, texts: List[str]) -> list[list[Dict[str, Any]]]:
        """
        Совместимый с API метод: возвращает сущности на уровне span, как и predict().
        Оставлен алиасом, потому что в API используется имя predict_bio.
        """
        return self.predict(texts)

    def predict_raw(self, texts: List[str]) -> list[Dict[str, Any]]:
        """
        Возвращает "сырые" ответы модели по каждому элементу батча:
        - tokens: токены токенизатора
        - offsets: смещения символов для каждого токена [start, end]
        - pred_ids: id предсказанных меток для каждого токена
        - pred_labels: строковые метки для каждого токена (BIO)
        - logits: логиты модели для каждого токена (np.ndarray формы [T, C])

        ВНИМАНИЕ: этот метод предназначен для интроспекции в Python.
        Возвращаемые np.ndarray не сериализуются в JSON без дополнительной обработки.
        """
        # Переиспользуем логику кодирования, но забираем полный enc для токенов
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.max_seq_len,
            return_offsets_mapping=True,
            return_tensors="np",
            is_split_into_words=False,
        )

        # Сформируем feed под имена входов графа
        feed: Dict[str, Any] = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = enc["input_ids"].astype(np.int64)
        else:
            for n in self.input_names:
                if n.endswith("input_ids"):
                    feed[n] = enc["input_ids"].astype(np.int64)

        if "attention_mask" in self.input_names:
            feed["attention_mask"] = enc["attention_mask"].astype(np.int64)
        else:
            for n in self.input_names:
                if n.endswith("attention_mask"):
                    feed[n] = enc["attention_mask"].astype(np.int64)

        if "token_type_ids" in enc and "token_type_ids" in self.input_names:
            feed["token_type_ids"] = enc["token_type_ids"].astype(np.int64)

        offsets = enc["offset_mapping"]  # [B,T,2]

        outs = self.session.run(self.output_names, feed)
        logits = next((o for o in outs if isinstance(o, np.ndarray) and o.ndim == 3), None)
        if logits is None:
            raise RuntimeError("Не найден выход [B,T,C] с логитами в ONNX-графе.")

        pred_ids = logits.argmax(axis=-1)  # [B,T]

        batch_raw: list[Dict[str, Any]] = []
        for b in range(pred_ids.shape[0]):
            token_ids = enc["input_ids"][b].tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            pred_ids_row = pred_ids[b].tolist()
            pred_labels_row = [self.labels[i] for i in pred_ids_row]

            item = {
                "text": texts[b],
                "tokens": tokens,
                "offsets": offsets[b].tolist(),
                "pred_ids": pred_ids_row,
                "pred_labels": pred_labels_row,
                "logits": logits[b],  # np.ndarray [T, C]
            }
            batch_raw.append(item)

        return batch_raw
