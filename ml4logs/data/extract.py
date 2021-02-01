# ===== IMPORTS =====
# === Standard library ===
import logging
import tarfile

# === Local ===
import ml4logs


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def extract(name: str, force: bool = False):
    logger.info('Start extracting %s', name)
    tarpath = ml4logs.DATASETS[name]['tarpath']
    xdir = ml4logs.DATASETS[name]['xdir']
    with tarfile.open(tarpath, 'r:gz') as tar:
        members = tar.getmembers()
        if not force:
            members = list(filter(
                lambda m: not (xdir / m.name).exists(),
                members
            ))
        logger.info('Extracting %d files from %s', len(members), tarpath)
        tar.extractall(xdir, members=members)
    logger.info('Saved into %s', xdir)
