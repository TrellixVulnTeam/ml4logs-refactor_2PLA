# ===== IMPORTS =====
# === Standard library ===
import logging
import pathlib
import re
import functools as ftools

# === Thirdparty ===
import numpy as np
import pandas as pd

# === Local ===
import ml4logs


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def aggregate(args):
    HANDLERS = {
        'HDFS1': aggregate_hdfs,
        'BGL': aggregate_merge_XY,
        'Thunderbird': aggregate_merge_XY
    }

    HANDLERS[args['dataset']](args)


def aggregate_hdfs(args):
    METHODS = {
        'sum': sum_per_block,
        'average': average_per_block,
        'max': max_per_block,
        'min': min_per_block
    }

    logs_path = pathlib.Path(args['logs_path'])
    labels_path = pathlib.Path(args['labels_path'])
    embeddings_path = pathlib.Path(args['embeddings_path'])
    out_path = pathlib.Path(args['out_path'])

    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info('Read logs from \'%s\'', logs_path)
    logs = logs_path.read_text().strip().split('\n')
    logger.info('Load embeddings from \'%s\'', embeddings_path)
    embeddings = np.load(embeddings_path)
    logger.info('Read labels from \'%s\'', labels_path)
    labels = pd.read_csv(labels_path).set_index('BlockId')
    logger.info('Map labels to 0/1 (normal/anomaly)')
    labels['Label'] = labels['Label'].map({'Normal': 0, 'Anomaly': 1})

    logger.info('Extract blocks from log lines')
    blocks = tuple(map(ftools.partial(re.findall, r'(blk_-?\d+)'), logs))
    logger.info('Aggregate embeddings by block id using \'%s\' method',
                args['method'])
    embeddings_per_block = METHODS[args['method']](embeddings, blocks)
    blks, X = tuple(zip(*embeddings_per_block.items()))
    X = np.stack(X)
    Y = labels.loc[list(blks)]['Label'].to_numpy()

    logger.info('X = %s, Y = %s', X.shape, Y.shape)
    logger.info('Save dataset into \'%s\'', out_path)
    np.savez(out_path, X=X, Y=Y)


def sum_per_block(embeddings, blocks):
    ZEROS = np.zeros(embeddings.shape[-1])
    result = {}
    for embedding, blocks in zip(embeddings, blocks):
        for bid in blocks:
            result[bid] = result.get(bid, ZEROS) + embedding
    return result


def average_per_block(embeddings, blocks):
    ZEROS = np.zeros(embeddings.shape[-1])
    result = {}
    counter = {}
    for embedding, blocks in zip(embeddings, blocks):
        for bid in blocks:
            result[bid] = result.get(bid, ZEROS) + embedding
            counter[bid] = counter.get(bid, 0) + 1
    for bid in counter:
        result[bid] /= counter[bid]
    return result


def max_per_block(embeddings, blocks):
    result = {}
    for embedding, blocks in zip(embeddings, blocks):
        for bid in blocks:
            result[bid] = np.fmax(result.get(bid, embedding), embedding)
    return result


def min_per_block(embeddings, blocks):
    result = {}
    for embedding, blocks in zip(embeddings, blocks):
        for bid in blocks:
            result[bid] = np.fmin(result.get(bid, embedding), embedding)
    return result


def aggregate_merge_XY(args):
    labels_path = pathlib.Path(args['labels_path'])
    embeddings_path = pathlib.Path(args['embeddings_path'])
    out_path = pathlib.Path(args['out_path'])

    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info('Load embeddings from \'%s\'', embeddings_path)
    X = np.load(embeddings_path)
    logger.info('Read labels from \'%s\'', labels_path)
    labels = np.load(labels_path)
    Y = labels[:len(X)]
    logger.info('X = %s, Y = %s', X.shape, Y.shape)
    logger.info('Save dataset into \'%s\'', out_path)
    np.savez(out_path, X=X, Y=Y)
