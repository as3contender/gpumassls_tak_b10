import re
from typing import List, Tuple
import string

Span = Tuple[int, int, str]

# --------------------------
# Regexes for VOLUME and PERCENT detection
# --------------------------
NUM = r"\d+(?:[.,]\d+)?"
UNIT = (
    r"(?:"
    r"л(?:\.)?|литр[а-я]*|мл(?:\.)?|"
    r"кг(?:\.)?|г(?:рамм)?|гр(?:амм)?|"
    r"шт(?:ук)?|рулон[а-я]*|"
    r"пакет(?:ик)?[а-я]*|десяток(?:\s*(?:шт|штук))?"
    r")"
)
VOLUME_RE = re.compile(rf"(?<!\w)(?:{NUM}\s*{UNIT}|{UNIT}\s*{NUM})(?!\w)", re.IGNORECASE)

PWORD = r"(?:%|проц(?:ент)?[а-я]*)"
PERCENT_RE = re.compile(rf"(?<!\w)(?:{NUM}\s*{PWORD}|{PWORD}\s*{NUM})(?!\w)", re.IGNORECASE)

DEFAULT_PREPOSITIONS = {
    # Simple and common Russian prepositions (lowercase)
    "в",
    "во",
    "на",
    "к",
    "ко",
    "от",
    "до",
    "из",
    "изо",
    "с",
    "со",
    "у",
    "за",
    "для",
    "по",
    "о",
    "об",
    "обо",
    "при",
    "через",
    "над",
    "под",
    "перед",
    "между",
    "про",
    "без",
    "около",
    "вокруг",
    "после",
    "среди",
    "вне",
    "кроме",
    "ради",
    "согласно",
    "насчёт",
    "насчет",
    "вместо",
    "вроде",
    "наперекор",
    "вопреки",
    "сквозь",
    "путём",
    "путем",
    "благодаря",
    "из-за",
    "изза",
    "из-под",
    "изпод",
    "вслед",
    "навстречу",
    "мимо",
    "вдоль",
    "поперёк",
    "поперек",
    "вглубь",
    "вширь",
    "вокрест",
    "попросту",
    "доя",
    "мытья",
    "дл",
}


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(0, min(a_end, b_end) - max(a_start, b_start)) > 0


def sanitize_bio(spans: List[Span]) -> List[Span]:
    """Ensure I-X without prior B-X becomes B-X."""
    spans_sorted = sorted(spans, key=lambda x: (x[0], x[1]))
    seen_started = set()
    sanitized: List[Span] = []
    for start, end, label in spans_sorted:
        if label.startswith("I-"):
            ent_type = label[2:]
            if ent_type not in seen_started:
                label = f"B-{ent_type}"
                seen_started.add(ent_type)
            sanitized.append((start, end, label))
        elif label.startswith("B-"):
            seen_started.add(label[2:])
            sanitized.append((start, end, label))
        else:
            sanitized.append((start, end, label))
    return sanitized


def inject_regex_entities(text: str, base_spans: List[Span]) -> List[Span]:
    """Inject VOLUME/PERCENT spans using regex if no overlap with existing labels."""
    result: List[Span] = list(base_spans)

    def add_entities(regex: re.Pattern, ent_type: str):
        last_end_for_type = None
        for m in regex.finditer(text):
            s, e = m.start(), m.end()
            overlaps_existing = any(spans_overlap(s, e, bs, be) for bs, be, _ in result)
            if overlaps_existing:
                continue
            prefix = "I-" if (last_end_for_type is not None and text[last_end_for_type:s].strip() == "") else "B-"
            result.append((s, e, f"{prefix}{ent_type}"))
            last_end_for_type = e

    add_entities(VOLUME_RE, "VOLUME")
    add_entities(PERCENT_RE, "PERCENT")

    return sanitize_bio(result)


def merge_bio_token_spans_to_entities(text: str, bio_spans: List[Span]) -> List[Span]:
    """
    Merge token-level BIO spans into contiguous entity spans by entity type.
    Spaces between tokens are allowed inside a single entity (e.g., "930 мл").

    Input spans are (start, end, label) where label may include BIO prefix.
    Output spans are (start, end, base_label) without BIO prefix.
    """
    # Sort by start, then end
    spans_sorted = sorted(bio_spans, key=lambda x: (x[0], x[1]))
    merged: List[Span] = []
    current_label: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    def flush():
        nonlocal current_label, current_start, current_end
        if current_label is not None and current_start is not None and current_end is not None:
            merged.append((current_start, current_end, current_label))
        current_label = None
        current_start = None
        current_end = None

    for s, e, lab in spans_sorted:
        base = lab.split("-", 1)[-1] if "-" in lab else lab
        if base == "O":
            flush()
            continue
        if current_label is None:
            current_label, current_start, current_end = base, s, e
            continue
        # If same label and only whitespace between segments, merge
        gap = text[current_end:s]
        if current_label == base and gap.strip() == "":
            current_end = e
        else:
            flush()
            current_label, current_start, current_end = base, s, e

    flush()
    return merged


def merge_subtokens_to_wordlevel_bio(text: str, bio_spans: List[Span]) -> List[Span]:
    """
    Merge only contiguous sub-token spans within a single word (no spaces between).
    Keep BIO prefixes, and DO NOT merge across spaces. If two consecutive word-level
    spans have the same base label, the later one keeps its own prefix (typically I-)
    as produced by the model or regex injector.
    """
    spans_sorted = sorted(bio_spans, key=lambda x: (x[0], x[1]))
    merged: List[Span] = []

    current_start: int | None = None
    current_end: int | None = None
    current_label: str | None = None
    current_base: str | None = None

    def flush():
        nonlocal current_start, current_end, current_label, current_base
        if current_start is not None and current_end is not None and current_label is not None:
            merged.append((current_start, current_end, current_label))
        current_start = None
        current_end = None
        current_label = None
        current_base = None

    for s, e, lab in spans_sorted:
        base = lab.split("-", 1)[-1] if "-" in lab else lab
        if current_start is None:
            current_start, current_end, current_label, current_base = s, e, lab, base
            continue
        # Only merge if same base label AND no spaces in the gap (sub-token continuation)
        gap = text[current_end:s]
        if current_base == base and gap == "":
            # Extend current word-level span; keep the original prefix of the first sub-token
            current_end = e
        else:
            flush()
            current_start, current_end, current_label, current_base = s, e, lab, base

    flush()
    return merged


# --------------------------
# Additional post-processing rules
# --------------------------

NONSPACE = re.compile(r"\S+")


def split_words_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in NONSPACE.finditer(text)]


def _normalize_token(token: str) -> str:
    return token.strip(string.punctuation + "«»\"'()[]{}\u00ab\u00bb").lower()


def nullify_entities_after_prepositions(
    text: str,
    spans: List[Span],
    prepositions: set[str] = DEFAULT_PREPOSITIONS,
    nullify_count: int = 2,
) -> List[Span]:
    """
    After any preposition word in the original text, drop the next `nullify_count` entities.
    Prepositions are detected from the full text, not from entities.
    """
    if not spans:
        return spans

    words = split_words_with_offsets(text)
    # Map each entity span to the index of the word covering its start
    ent_word_idx: List[int] = []
    for s, e, _ in spans:
        idx = None
        for wi, (_, ws, we) in enumerate(words):
            if ws <= s < we:
                idx = wi
                break
        if idx is None:
            for wi, (_, ws, we) in enumerate(words):
                if ws >= s:
                    idx = wi
                    break
        ent_word_idx.append(-1 if idx is None else idx)

    # Preposition indices
    prep_indices: List[int] = [wi for wi, (tok, _, _) in enumerate(words) if _normalize_token(tok) in prepositions]

    # Build word_idx -> list of entity span indices
    word_to_span_idxs: dict[int, List[int]] = {}
    for ei, widx in enumerate(ent_word_idx):
        if widx < 0:
            continue
        word_to_span_idxs.setdefault(widx, []).append(ei)

    to_drop_span_idxs: set[int] = set()
    # For each preposition, drop spans on the preposition word itself and then
    # drop all spans belonging to the next `nullify_count` distinct entity words
    sorted_entity_word_idxs = sorted(word_to_span_idxs.keys())
    for pidx in prep_indices:
        # 1) nullify spans that overlap the preposition word itself
        if pidx in word_to_span_idxs:
            for ei in word_to_span_idxs[pidx]:
                to_drop_span_idxs.add(ei)
        # 2) nullify next N entity words after the preposition
        dropped_words = 0
        for widx in sorted_entity_word_idxs:
            if widx <= pidx:
                continue
            # skip if this word already dropped via earlier preposition
            if any(i in to_drop_span_idxs for i in word_to_span_idxs[widx]):
                continue
            for ei in word_to_span_idxs[widx]:
                to_drop_span_idxs.add(ei)
            dropped_words += 1
            if dropped_words >= nullify_count:
                break

    if not to_drop_span_idxs and not prep_indices:
        return spans
    # Mark targeted spans as 'O' instead of removing
    new_spans: List[Span] = []
    for i, (s, e, lab) in enumerate(spans):
        if i in to_drop_span_idxs:
            new_spans.append((s, e, "O"))
        else:
            new_spans.append((s, e, lab))

    # Ensure preposition words appear as explicit 'O' spans even if they had no spans
    # Collect current coverage intervals to avoid duplicates
    def covered(ws: int, we: int) -> bool:
        for cs, ce, _ in new_spans:
            if ws >= cs and we <= ce:
                return True
        return False

    for pidx in prep_indices:
        if 0 <= pidx < len(words):
            _, ws, we = words[pidx]
            if not covered(ws, we):
                new_spans.append((ws, we, "O"))
            # also force O for the next `nullify_count` words regardless of having spans
            for k in range(1, nullify_count + 1):
                widx = pidx + k
                if 0 <= widx < len(words):
                    _, nws, nwe = words[widx]
                    if not covered(nws, nwe):
                        new_spans.append((nws, nwe, "O"))

    # Keep order
    new_spans.sort(key=lambda x: (x[0], x[1]))
    return new_spans


def nullify_if_starts_with_all(text: str, spans: List[Span]) -> List[Span]:
    """If the first word is 'все' or 'всё', drop all entities."""
    words = split_words_with_offsets(text)
    if not words:
        return spans
    first_norm = _normalize_token(words[0][0])
    if first_norm in {"все", "всё"}:
        # Nullify all spans by converting their labels to 'O'
        return [(s, e, "O") for (s, e, _) in spans]
    return spans


# --------------------------
# Levenshtein-based VOLUME keyword injection
# --------------------------


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))  # insertion  # deletion  # substitution
        prev = cur
    return prev[-1]


def _norm_for_fuzzy(token: str) -> str:
    t = _normalize_token(token)
    t = t.replace("ё", "е").replace("ъ", "")
    return t


def _is_adj_big(token: str) -> bool:
    t = _norm_for_fuzzy(token)
    if t.startswith("больш"):
        return True
    return _levenshtein(t, "большой") <= 1


def _is_noun_volume(token: str) -> bool:
    t = _norm_for_fuzzy(token)
    targets = ["объем", "обьем", "обем"]  # normalized variations
    return any(_levenshtein(t, tgt) <= 1 for tgt in targets)


def inject_volume_keywords_levenshtein(text: str, base_spans: List[Span]) -> List[Span]:
    """
    Detect bigram phrases like "большой объем" (including small misspellings) and
    inject them as B-VOLUME (first word) and I-VOLUME (second word), if they don't
    overlap with existing labeled spans.
    """
    result: List[Span] = list(base_spans)
    words = split_words_with_offsets(text)

    def overlaps(s: int, e: int) -> bool:
        return any(spans_overlap(s, e, ws, we) for ws, we, _ in result)

    for i in range(len(words) - 1):
        w1, s1, e1 = words[i]
        w2, s2, e2 = words[i + 1]
        if _is_adj_big(w1) and _is_noun_volume(w2):
            if not overlaps(s1, e1) and not overlaps(s2, e2):
                result.append((s1, e1, "B-VOLUME"))
                result.append((s2, e2, "I-VOLUME"))

    # Keep BIO consistency afterwards
    return sanitize_bio(result)


def ensure_leading_word_O(text: str, spans: List[Span]) -> List[Span]:
    """
    Ensure the first word appears as an explicit 'O' span if no span overlaps it.
    """
    words = split_words_with_offsets(text)
    if not words:
        return spans
    _, ws, we = words[0]
    has_overlap = any(not (e <= ws or s >= we) for s, e, _ in spans)
    if has_overlap:
        return spans
    out = list(spans)
    out.append((ws, we, "O"))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def snap_spans_to_word_boundaries(text: str, spans: List[Span]) -> List[Span]:
    """
    Snap every span to exact word boundaries (based on non-space tokenization).
    Assumes each span should belong to a single word.
    """
    words = split_words_with_offsets(text)
    if not words or not spans:
        return spans
    snapped: List[Span] = []
    for s, e, lab in spans:
        # find word covering start
        ws, we = None, None
        for _, w_s, w_e in words:
            if w_s <= s < w_e:
                ws, we = w_s, w_e
                break
        if ws is None:
            # fallback: nearest word starting after s
            for _, w_s, w_e in words:
                if w_s >= s:
                    ws, we = w_s, w_e
                    break
        if ws is None:
            # keep original if no word found
            ws, we = s, e
        snapped.append((ws, we, lab))
    # deduplicate while preserving order
    seen = set()
    result: List[Span] = []
    for span in sorted(snapped, key=lambda x: (x[0], x[1])):
        key = (span[0], span[1], span[2])
        if key in seen:
            continue
        seen.add(key)
        result.append(span)
    return result


def ensure_all_words_covered_with_spans(text: str, spans: List[Span]) -> List[Span]:
    """
    Ensure that every word in the text has an explicit span. If a word has no
    overlapping span, add an 'O' span covering exactly that word.
    """
    words = split_words_with_offsets(text)
    if not words:
        return spans
    out = list(spans)

    def has_overlap(ws: int, we: int) -> bool:
        for s, e, _ in out:
            if not (e <= ws or s >= we):
                return True
        return False

    for _, ws, we in words:
        if not has_overlap(ws, we):
            out.append((ws, we, "O"))

    out.sort(key=lambda x: (x[0], x[1]))
    # Deduplicate identical spans
    dedup: List[Span] = []
    seen = set()
    for s, e, lab in out:
        key = (s, e, lab)
        if key in seen:
            continue
        seen.add(key)
        dedup.append((s, e, lab))
    return dedup


# --------------------------
# Word-level pipeline utilities
# --------------------------


def derive_word_bio_from_token_spans(
    words: List[Tuple[str, int, int]],
    token_spans: List[Span],
) -> List[str]:
    """
    Map token-level BIO spans to word-level BIO labels.
    Rule: word gets the base label that appears first among its tokens (non-O).
    BIO is then recomputed across words so consecutive same-type words get I-.
    """
    word_base: List[str] = ["O"] * len(words)
    for s, e, lab in sorted(token_spans, key=lambda x: (x[0], x[1])):
        if lab == "O":
            continue
        base = lab.split("-", 1)[-1] if "-" in lab else lab
        # find word covering token start
        for wi, (_, ws, we) in enumerate(words):
            if ws <= s < we:
                if word_base[wi] == "O":
                    word_base[wi] = base
                break
    # recompute BIO across words
    bio: List[str] = []
    prev_type: str | None = None
    for t in word_base:
        if t == "O":
            bio.append("O")
            prev_type = None
        else:
            if prev_type == t:
                bio.append(f"I-{t}")
            else:
                bio.append(f"B-{t}")
            prev_type = t
    return bio


def apply_word_level_rules(
    text: str,
    words: List[Tuple[str, int, int]],
    bio: List[str],
    nullify_count_after_prep: int = 2,
) -> List[str]:
    """
    Apply postprocessing on word sequence:
    - inject VOLUME/PERCENT via regex
    - inject bigram 'большой объем' (fuzzy)
    - nullify after prepositions and if starts with 'все/всё'
    Recompute BIO after base-type updates.
    """
    # Base labels
    base = [b.split("-", 1)[-1] if b != "O" else "O" for b in bio]

    # Regex injections → set base labels on overlapped words
    for m in VOLUME_RE.finditer(text):
        s, e = m.start(), m.end()
        for i, (_, ws, we) in enumerate(words):
            if not (we <= s or ws >= e):
                base[i] = "VOLUME"
    for m in PERCENT_RE.finditer(text):
        s, e = m.start(), m.end()
        for i, (_, ws, we) in enumerate(words):
            if not (we <= s or ws >= e):
                base[i] = "PERCENT"

    # Bigram 'большой объем' fuzzy
    for i in range(len(words) - 1):
        w1 = _norm_for_fuzzy(words[i][0])
        w2 = _norm_for_fuzzy(words[i + 1][0])
        if _is_adj_big(w1) and _is_noun_volume(w2):
            base[i] = "VOLUME"
            base[i + 1] = "VOLUME"

    # Nullify if starts with all
    if words:
        first_norm = _normalize_token(words[0][0])
        if first_norm in {"все", "всё"}:
            base = ["O"] * len(words)

    # Nullify after prepositions (word-level)
    for i, (tok, _, _) in enumerate(words):
        if _normalize_token(tok) in DEFAULT_PREPOSITIONS:
            base[i] = "O"
            for k in range(1, nullify_count_after_prep + 1):
                j = i + k
                if 0 <= j < len(base):
                    base[j] = "O"

    # Recompute BIO
    out_bio: List[str] = []
    prev_type: str | None = None
    for t in base:
        if t == "O":
            out_bio.append("O")
            prev_type = None
        else:
            if prev_type == t:
                out_bio.append(f"I-{t}")
            else:
                out_bio.append(f"B-{t}")
            prev_type = t
    return out_bio


def word_bio_to_spans(words: List[Tuple[str, int, int]], bio: List[str]) -> List[Span]:
    spans: List[Span] = []
    for (tok, s, e), lab in zip(words, bio):
        spans.append((s, e, lab))
    return spans
