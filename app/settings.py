from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_dir: str = "./models/ner_xlmr_wordlevel_entity"
    labels_path: str = "./models/ner_xlmr_wordlevel_entity/labels.txt"
    max_seq_len: int = 256
    batch_max_size: int = 64
    batch_timeout_ms: int = 10
    warmup_requests: int = 50

    preprocess_enabled: bool = False
    mask_numeric_before_ner: bool = True
    return_debug: bool = False

    ort_force_cpu: bool = False


settings = Settings()
