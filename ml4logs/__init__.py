# ===== IMPORTS =====
# === Standard library ===
import pathlib
import os
# import json

# === Thirdparty ===
# === Local ===
from ml4logs import data
from ml4logs import utils


# ===== GLOBALS =====
PROJECT_DIR = pathlib.Path(os.getenv('PROJECT_DIR'))
DATASETS = {
    'HDFS_1': {
        'url': 'https://zenodo.org/record/3227177/files/HDFS_1.tar.gz',
        'tarpath': PROJECT_DIR / 'data/raw/HDFS_1.tar.gz',
        'xdir': PROJECT_DIR / 'data/raw/HDFS_1',
        'logs_path': PROJECT_DIR / 'data/raw/HDFS_1/logs.txt',
        'labels_path': PROJECT_DIR / 'data/raw/HDFS_1/labels.csv',
        'prepare_f': data.prepare_hdfs_1,
        'ftdir': PROJECT_DIR / 'models/embeddings/fasttext/HDFS_1'
    },
    'HDFS_2': {
        'url': 'https://zenodo.org/record/3227177/files/HDFS_2.tar.gz',
        'tarpath': PROJECT_DIR / 'data/raw/HDFS_2.tar.gz',
        'xdir': PROJECT_DIR / 'data/raw/HDFS_2',
        'logs_path': PROJECT_DIR / 'data/raw/HDFS_2/logs.txt',
        'labels_path': None,
        'prepare_f': data.prepare_hdfs_2,
        'ftdir': PROJECT_DIR / 'models/embeddings/fasttext/HDFS_2'
    },
    'BGL': {
        'url': 'https://zenodo.org/record/3227177/files/BGL.tar.gz',
        'tarpath': PROJECT_DIR / 'data/raw/BGL.tar.gz',
        'xdir': PROJECT_DIR / 'data/raw/BGL',
        'logs_path': PROJECT_DIR / 'data/raw/BGL/logs.txt',
        'labels_path': PROJECT_DIR / 'data/raw/BGL/labels.csv',
        'prepare_f': data.prepare_bgl,
        'ftdir': PROJECT_DIR / 'models/embeddings/fasttext/BGL'
    },
    'DUMMY': {
        'url': None,
        'tarpath': None,
        'xdir': PROJECT_DIR / 'data/raw/DUMMY',
        'logs_path': PROJECT_DIR / 'data/raw/DUMMY/logs.txt',
        'labels_path': None,
        'prepare_f': None,
        'ftdir': PROJECT_DIR / 'models/embeddings/fasttext/DUMMY'
    }
}
