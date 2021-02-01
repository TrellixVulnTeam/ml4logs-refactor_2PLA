# ===== IMPORTS =====
# === Standard library ===
import logging

# === Thirdparty ===
import requests

# === Local ===
import ml4logs


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def download(name: str, force: bool = False):
    tarpath = ml4logs.DATASETS[name]['tarpath']
    if not force and tarpath.exists():
        logger.info('File %s already exists', tarpath)
        return

    logger.info('Start downloading %s', name)
    response = requests.get(ml4logs.DATASETS[name]['url'])
    tarpath.write_bytes(response.content)
    logger.info('Saved into %s', tarpath)
