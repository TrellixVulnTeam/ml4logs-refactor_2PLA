# ===== IMPORTS =====
# === Standard library ===
import logging


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def setup_logging(level=logging.INFO):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    LOG_FORMAT = '[{asctime}][{levelname}][{name}] {message}'
    formatter = logging.Formatter(LOG_FORMAT, style='{')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    logger.info('Root logger is configured')
