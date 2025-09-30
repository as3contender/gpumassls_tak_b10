from pydantic_settings import BaseSettings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../gpumassls_task_b10


class Settings(BaseSettings):
    model_dir: str = str(PROJECT_ROOT / "models/ner_xlmr_wordlevel_entity")
    labels_path: str = str(PROJECT_ROOT / "models/ner_xlmr_wordlevel_entity/labels.txt")
    max_seq_len: int = 256
    batch_max_size: int = 64
    batch_timeout_ms: int = 10
    warmup_requests: int = 50
    use_queue: bool = False

    preprocess_enabled: bool = False
    mask_numeric_before_ner: bool = True
    return_debug: bool = False

    ort_force_cpu: bool = False


settings = Settings()
