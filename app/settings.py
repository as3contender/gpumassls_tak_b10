from pydantic_settings import BaseSettings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../gpumassls_task_b10


class Settings(BaseSettings):
    model_dir: str = str(PROJECT_ROOT / "models/ner_xlmr_wordlevel_entity")
    labels_path: str = str(PROJECT_ROOT / "models/ner_xlmr_wordlevel_entity/labels.txt")
    max_seq_len: int = 256
    batch_max_size: int = 16
    batch_timeout_ms: int = 10
    queue_maxsize: int = 5000
    request_timeout_ms: int = 2000
    warmup_requests: int = 50
    use_queue: bool = True

    preprocess_enabled: bool = False
    mask_numeric_before_ner: bool = True
    return_debug: bool = False

    ort_force_cpu: bool = False
    ort_intra_op_num_threads: int = 8
    ort_inter_op_num_threads: int = 1
    ort_execution_mode_parallel: bool = True

    # Postprocess controls
    pp_token_inject_regex: bool = False
    pp_token_inject_volume_levenshtein: bool = True
    pp_token_nullify_after_prepositions: bool = True
    pp_token_nullify_if_starts_with_all: bool = True
    pp_token_ensure_leading_word_o: bool = True

    pp_word_rules_enabled: bool = True
    pp_word_nullify_count_after_prep: int = 2


settings = Settings()
