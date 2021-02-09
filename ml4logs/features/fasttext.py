# ===== IMPORTS =====
# === Standard library ===
import logging
import pathlib

# === Thirdparty ===
import numpy as np
import fasttext


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def train_fasttext(args):
    logs_path = pathlib.Path(args['logs_path'])
    model_path = pathlib.Path(args['model_path'])

    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info('Train fasttext model on \'%s\'', logs_path)
    model = fasttext.train_unsupervised(str(logs_path), **args['model_args'])
    logger.info('Save fasttext model into \'%s\'', model_path)
    model.save_model(str(model_path))


def preprocess_fasttext(args):
    logs_path = pathlib.Path(args['logs_path'])
    model_path = pathlib.Path(args['model_path'])
    embeddings_path = pathlib.Path(args['embeddings_path'])

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info('Read logs from \'%s\'', logs_path)
    logs = logs_path.read_text().strip().split('\n')
    logger.info('Load fasttext model from \'%s\'', model_path)
    model = fasttext.load_model(str(model_path))
    logger.info('Obtain embeddings for logs')
    embeddings = np.stack(tuple(map(model.get_sentence_vector, logs)))
    logger.info('Embeddings shape is %s', embeddings.shape)
    logger.info('Save embeddings into \'%s\'', embeddings_path)
    np.save(embeddings_path, embeddings)
