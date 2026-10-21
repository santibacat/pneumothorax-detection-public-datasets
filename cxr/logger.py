import logging
import sys
from logging.handlers import RotatingFileHandler

import os


def in_jupyter() -> bool:
    try:
        from IPython import get_ipython

        return "IPKernelApp" in get_ipython().config
    except Exception:
        return False


def setup_logging(log_file: str = "cxr.log", log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 🧹 In Jupyter: remove root handlers to avoid duplicate logs
    if in_jupyter():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # ✅ Add StreamHandler (console logs) only if it's not already there
    if not any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
        for h in logger.handlers
    ):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # ✅ Add RotatingFileHandler only if it's not already added
    if not any(
        isinstance(h, RotatingFileHandler)
        and getattr(h, "baseFilename", "") == os.path.abspath(log_file)
        for h in logger.handlers
    ):
        file_handler = RotatingFileHandler(
            log_file, maxBytes=20 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # more verbose to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug("Logging setup complete.")


# Call the setup function when the module is imported
setup_logging()
