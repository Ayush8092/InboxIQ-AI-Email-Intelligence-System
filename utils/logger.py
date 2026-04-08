import logging
import os
from config.constants import LOG_FILE, LOG_LEVEL

def setup_logger(name: str = "aeoa") -> logging.Logger:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    level  = getattr(logging, LOG_LEVEL.upper(), logging.DEBUG)
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger