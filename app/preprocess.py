# app/preprocess.py
from __future__ import annotations
from typing import List, Tuple
import regex as re

# --- Regex-шаблоны ---
WORD_RE = re.compile(r"\p{L}[\p{L}\p{N}-]*", re.UNICODE)
RE_PERCENT_SIGN = re.compile(r"(?<!\d)\d{1,2}(?:[.,]\d)?\s*%")
RE_PERCENT_WORD = re.compile(r"\b\d{1,2}(?:[.,]\d)?\s*(?:проц|процент(?:а|ов)?)\b", re.IGNORECASE)
UNITS = ["мл", "ml", "l", "л", "г", "гр", "kg", "кг", "шт", "уп", "пак", "ш", "к"]
RE_VOLUME = re.compile(rf"\b\d+(?:[.,]\d+)?\s*(?:{'|'.join(UNITS)})\b", re.IGNORECASE)
RE_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")

PACK_WORDS = ["бутыл", "банка", "пакет", "упаков", "рулон", "лист", "пачк", "флакон"]

# базовый лексикон «жирных» продуктов, если не подгружаем из train
DEFAULT_FATTY_WORDS = ["молоко", "кефир", "сливки", "сметана", "творог", "сыр"]

Span = Tuple[int, int, str]  # (start, end, label)


def extract_explicit_numeric(text: str) -> List[Span]:
    ents: List[Span] = []
    for rx in (RE_PERCENT_SIGN, RE_PERCENT_WORD):
        for m in rx.finditer(text):
            ents.append((m.start(), m.end(), "B-PERCENT"))
    for m in RE_VOLUME.finditer(text):
        ents.append((m.start(), m.end(), "B-VOLUME"))
    return sorted(ents)


def infer_implicit_numeric(text: str, fatty_words=DEFAULT_FATTY_WORDS) -> List[Span]:
    ents: List[Span] = []
    for m in RE_NUMBER.finditer(text):
        s, e = m.span()
        raw = text[s:e]
        # если это уже «15%» — пропускаем (явный процент поймает extract_explicit_numeric)
        if raw.endswith("%"):
            continue
        try:
            val = float(raw.replace(",", "."))
        except Exception:
            continue
        ctx = text[max(0, s - 20) : min(len(text), e + 20)].lower()
        if val == 0:
            ents.append((s, e, "B-PERCENT"))
        elif 1 <= val <= 99 and any(w in ctx for w in fatty_words):
            ents.append((s, e, "B-PERCENT"))
        elif val >= 100 or any(w in ctx for w in PACK_WORDS):
            ents.append((s, e, "B-VOLUME"))
    return ents


def merge_overlapping_entities(entities: List[Span]) -> List[Span]:
    if not entities:
        return []
    entities = sorted(entities)
    result: List[Span] = []
    for start, end, label in entities:
        if result:
            ls, le, ll = result[-1]
            # перекрытие одного типа
            if label == ll and not (end <= ls or start >= le):
                # вложенность
                if start >= ls and end <= le:
                    continue
                elif ls >= start and le <= end:
                    result[-1] = (start, end, label)
                    continue
                # частичное перекрытие
                else:
                    result[-1] = (min(start, ls), max(end, le), label)
                    continue
        result.append((start, end, label))
    return result


def preprocess_query(text: str, fatty_words=DEFAULT_FATTY_WORDS) -> List[Span]:
    explicit = extract_explicit_numeric(text)
    implicit = infer_implicit_numeric(text, fatty_words=fatty_words)
    return merge_overlapping_entities(explicit + implicit)


def mask_text_keep_length(text: str, spans: List[Span], fill_char: str = " ") -> str:
    """
    Возвращает строку той же длины, где указанные диапазоны заменены на fill_char.
    Это важно: индексы токенайзера в offset_mapping останутся валидны относительно оригинала.
    """
    if not spans:
        return text
    chars = list(text)
    for s, e, _ in spans:
        for i in range(s, e):
            chars[i] = fill_char
    return "".join(chars)


def spans_overlap(a: Span, b: Span) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])
