#!/usr/bin/env python3
"""
Batch inference CLI for the ONNX NER model.

- Reads texts from a CSV file (column `sample`)
- Runs batched inference using `ONNXNER`
- Optionally applies regex-based postprocessing for VOLUME and PERCENT
- Writes predictions to CSV with columns: sample;annotation

Example:
    python -m repo.app.infer \
        --csv-in train/data/submission.csv \
        --csv-out submission_pred.csv \
        --batch-size 16
"""

import argparse
import csv
import os
from typing import List, Dict, Any

from .runtime import ONNXNER
from .postprocess import (
    inject_regex_entities,
    inject_volume_keywords_levenshtein,
    ensure_leading_word_O,
    ensure_all_words_covered_with_spans,
    snap_spans_to_word_boundaries,
    split_words_with_offsets,
    derive_word_bio_from_token_spans,
    apply_word_level_rules,
    word_bio_to_spans,
    merge_subtokens_to_wordlevel_bio,
    nullify_entities_after_prepositions,
    nullify_if_starts_with_all,
    Span,
)


def read_texts_from_csv(csv_path: str) -> List[str]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        texts: List[str] = []
        for row in reader:
            sample = (row.get("sample") or "").strip()
            if sample:
                texts.append(sample)
        return texts


def write_predictions_to_csv(texts: List[str], spans_batch: List[List[Span]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["sample", "annotation"])
        for text, spans in zip(texts, spans_batch):
            writer.writerow([text, str(spans)])


_NER: ONNXNER | None = None


def _get_ner() -> ONNXNER:
    global _NER
    if _NER is None:
        _NER = ONNXNER()
    return _NER


def predict_bio(texts: List[str], apply_regex_postprocess: bool = True) -> List[List[Span]]:
    """
    Module-level prediction function used by the API:
    - Reuses a singleton ONNXNER session
    - Returns token-level spans (start, end, BIO label), optionally enriched by regex postprocessing
    """
    ner = _get_ner()
    raw_items = ner.predict_raw(texts)

    # Build token-level BIO spans from raw offsets and labels
    spans_batch: List[List[Span]] = []
    for item in raw_items:
        spans: List[Span] = []
        for (s, e), lab in zip(item["offsets"], item["pred_labels"]):
            if int(s) == 0 and int(e) == 0:
                continue
            if lab == "O":
                continue
            spans.append((int(s), int(e), str(lab)))
        spans_batch.append(spans)

    if apply_regex_postprocess:
        spans_batch = [inject_regex_entities(t, s) for t, s in zip(texts, spans_batch)]
        spans_batch = [inject_volume_keywords_levenshtein(t, s) for t, s in zip(texts, spans_batch)]
        # Nullify rules (after prepositions; entire phrase if starts with "все/всё")
        spans_batch = [nullify_entities_after_prepositions(t, s) for t, s in zip(texts, spans_batch)]
        spans_batch = [nullify_if_starts_with_all(t, s) for t, s in zip(texts, spans_batch)]
        # Ensure first word has explicit O if none provided
        spans_batch = [ensure_leading_word_O(t, s) for t, s in zip(texts, spans_batch)]

    # Word-first pipeline: tokenize to words, map token BIO to words, apply word-level rules, then emit spans
    merged_batch = []
    for text, token_spans in zip(texts, spans_batch):
        words = split_words_with_offsets(text)
        word_bio = derive_word_bio_from_token_spans(words, token_spans)
        word_bio = apply_word_level_rules(text, words, word_bio)
        spans = word_bio_to_spans(words, word_bio)
        merged_batch.append(spans)

    # Return tuples directly for API: [(start, end, label), ...]
    return merged_batch  # type: ignore


def run(csv_in: str, csv_out: str, batch_size: int, apply_regex_postprocess: bool) -> None:
    ner = ONNXNER()
    texts = read_texts_from_csv(csv_in)
    n = len(texts)
    print(f"Total samples: {n}")

    all_spans: List[List[Span]] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = texts[start:end]
        raw_items = ner.predict_raw(batch_texts)
        spans_batch: List[List[Span]] = []
        for t, item in zip(batch_texts, raw_items):
            token_spans: List[Span] = []
            for (s, e), lab in zip(item["offsets"], item["pred_labels"]):
                if int(s) == 0 and int(e) == 0:
                    continue
                # Keep O as well; downstream word rules may rely on it
                token_spans.append((int(s), int(e), str(lab)))
            words = split_words_with_offsets(t)
            word_bio = derive_word_bio_from_token_spans(words, token_spans)
            if apply_regex_postprocess:
                word_bio = apply_word_level_rules(t, words, word_bio)
            merged = word_bio_to_spans(words, word_bio)
            spans_batch.append(merged)
        all_spans.extend(spans_batch)
        if start % 500 == 0:
            print(f"Processed {end}/{n}")

    write_predictions_to_csv(texts, all_spans, csv_out)
    print(f"✅ Saved predictions to {csv_out}")


def main():
    parser = argparse.ArgumentParser(description="Batch inference for ONNX NER model")
    parser.add_argument("--csv-in", default="train/data/submission.csv", help="Input CSV with a 'sample' column")
    parser.add_argument("--csv-out", default="submission_pred.csv", help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--no-regex-postprocess", action="store_true", help="Disable regex-based postprocessing")

    args = parser.parse_args()
    run(
        csv_in=args.__dict__["csv-in"],
        csv_out=args.__dict__["csv-out"],
        batch_size=args.__dict__["batch-size"],
        apply_regex_postprocess=not args.__dict__["no-regex-postprocess"],
    )


if __name__ == "__main__":
    main()
