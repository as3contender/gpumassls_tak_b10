import os
import sys
from loguru import logger


def setup_logging() -> None:
    """
    Configure loguru logger:
    - Single stdout sink
    - Level from LOG_LEVEL env (default: INFO)
    - Include time, level, message, and exception traces
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        backtrace=True,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
    )

    # Reduce uvicorn/access noise unless DEBUG
    if level not in {"DEBUG"}:
        logger.disable("uvicorn.access")
