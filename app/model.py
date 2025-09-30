# app/model.py
import os
from typing import List, Dict, Any
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from .settings import settings
from .preprocess import preprocess_query, mask_text_keep_length, spans_overlap


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
        return [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": 1,
                },
            ),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]


class ONNXNER:
    def __init__(self):
        self.labels = load_labels(settings.labels_path)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_dir, use_fast=True)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        model_path = os.path.join(settings.model_dir, "model.onnx")
        providers = _pick_providers()
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)

        # Имена входов/выходов
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.preprocess_enabled = settings.preprocess_enabled
        self.return_debug = settings.return_debug

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

    def predict(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        # 1) предобработка числовых сущностей
        if self.preprocess_enabled:
            pre_entities_all = [preprocess_query(t) for t in texts]
        else:
            pre_entities_all = [[] for _ in texts]

        # 2) маскируем найденные числовые сущности — длина строки сохраняется
        masked_texts = [mask_text_keep_length(t, pre_entities_all[i], fill_char=" ") for i, t in enumerate(texts)]

        # 3) обычный NER по маске
        feed, offsets, _ = self._encode(masked_texts)  # NB: offsets соответствуют masked_texts
        outs = self.session.run(self.output_names, feed)
        logits = next((o for o in outs if isinstance(o, np.ndarray) and o.ndim == 3), None)
        if logits is None:
            raise RuntimeError("Не найден выход [B,T,C] с логитами в ONNX-графе.")
        pred_ids = logits.argmax(axis=-1)  # [B,T]

        # 4) конвертация BIO→span на masked_texts (индексы валидны для original, т.к. длина сохранена)
        ner_entities_all: List[List[Dict[str, Any]]] = []
        for b in range(pred_ids.shape[0]):
            ner_entities_all.append(self._bio_to_spans(texts[b], pred_ids[b], offsets[b]))

        # 5) слияние: предобработка выигрывает для {PERCENT,VOLUME}, NER — для остальных
        merged_all: List[List[Dict[str, Any]]] = []
        num_labels = {"PERCENT", "VOLUME"}

        for i, (pre, ner) in enumerate(zip(pre_entities_all, ner_entities_all)):
            # pre: List[Span] -> в dict-формат
            pre_as_dicts = [
                {"label": lab.replace("B-", "").replace("I-", ""), "start": s, "end": e, "text": texts[i][s:e]}
                for (s, e, lab) in pre
            ]

            # оставляем из NER только нечисловые метки и те, кто НЕ пересекается с pre
            ner_filtered = []
            for ent in ner:
                label = ent["label"].split("-", 1)[-1] if "-" in ent["label"] else ent["label"]
                if label in num_labels:
                    continue
                conflict = any(spans_overlap((ent["start"], ent["end"], label), (s, e, l)) for (s, e, l) in pre)
                if not conflict:
                    ner_filtered.append({"label": label, **{k: ent[k] for k in ("start", "end", "text")}})

            merged = pre_as_dicts + ner_filtered
            # (опционально) отсортировать по позиции
            merged.sort(key=lambda x: (x["start"], x["end"]))
            merged_all.append(merged)

        # (опционально) вернуть отладку
        if self.return_debug:
            # завернём вместе с pre/ner
            return [
                {"merged": m, "pre": p, "ner": n} for m, p, n in zip(merged_all, pre_entities_all, ner_entities_all)
            ]

        return merged_all
