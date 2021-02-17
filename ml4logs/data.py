# ===== IMPORTS =====
# === Standard library ===
import logging
import pathlib
import tarfile
import re
import itertools as itools

# === Thirdparty ===
import requests
import numpy as np
import pandas as pd

# === Local ===
import ml4logs


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def download(args):
    path = pathlib.Path(args['path'])

    ml4logs.utils.mkdirs(files=[path])

    logger.info('Download \'%s\'', args['url'])
    response = requests.get(args['url'])
    logger.info('Save into \'%s\'', path)
    path.write_bytes(response.content)


def extract(args):
    in_path = pathlib.Path(args['in_path'])
    out_dir = pathlib.Path(args['out_dir'])

    ml4logs.utils.mkdirs(folders=[out_dir])

    logger.info('Open \'%s\' as tarfile', in_path)
    with tarfile.open(in_path, 'r:gz') as tar:
        members = tar.getmembers()
        logger.info('Extract %d files', len(members))
        tar.extractall(out_dir, members=members)


def prepare(args):
    HANDLERS = {
        'HDFS1': prepare_hdfs_1,
        'HDFS2': prepare_hdfs_2,
        'BGL': prepare_bgl,
        'Thunderbird': prepare_thunderbird
    }

    HANDLERS[args['dataset']](args)


def prepare_hdfs_1(args):
    in_dir = pathlib.Path(args['in_dir'])
    logs_path = pathlib.Path(args['logs_path'])
    blocks_path = pathlib.Path(args['blocks_path'])
    labels_path = pathlib.Path(args['labels_path'])

    ml4logs.utils.mkdirs(files=[logs_path, labels_path, blocks_path])

    in_labels_path = in_dir / 'anomaly_label.csv'
    logger.info('Read labels from \'%s\'', in_labels_path)
    labels_df = pd.read_csv(in_labels_path)
    logger.info('Map labels to 0/1 (normal/anomaly)')
    labels = labels_df['Label'].map({'Normal': 0, 'Anomaly': 1}).to_numpy()
    logger.info('Save mapped labels into \'%s\'', labels_path)
    np.save(labels_path, labels)

    in_logs_path = in_dir / 'HDFS.log'
    logger.info('Move \'%s\' to \'%s\'', in_logs_path, logs_path)
    in_logs_path.replace(logs_path)

    blk_mapping = dict(zip(labels_df['BlockId'], labels_df.index))
    pattern = re.compile(r'(blk_-?\d+)')
    blocks = []
    logger.info('Extract blocks from log lines')
    with logs_path.open() as logs_in_f:
        for line in logs_in_f:
            match = pattern.search(line.strip())
            blocks.append(blk_mapping[match.group()])
    logger.info('Save blocks into \'%s\'', blocks_path)
    np.save(blocks_path, np.array(blocks))


def prepare_hdfs_2(args):
    raise NotImplementedError('Prepare is not implemented for HDFS2 dataset')


def prepare_bgl(args):
    in_dir = pathlib.Path(args['in_dir'])
    split_labels(args, in_dir / 'BGL.log', '-')


def prepare_thunderbird(args):
    in_dir = pathlib.Path(args['in_dir'])
    split_labels(args, in_dir / 'Thunderbird.log', '-')


def split_labels(args, in_path, normal_label):
    logs_path = pathlib.Path(args['logs_path'])
    labels_path = pathlib.Path(args['labels_path'])

    ml4logs.utils.mkdirs(files=[logs_path, labels_path])

    n_lines = ml4logs.utils.count_file_lines(in_path)
    step = n_lines // 10
    logger.info('Start splitting labels and log messages')
    labels = []
    with in_path.open(encoding='utf8') as in_f, \
            logs_path.open('w', encoding='utf8') as logs_out_f:
        for i, line in enumerate(in_f):
            label, raw_log = tuple(line.strip().split(maxsplit=1))
            logs_out_f.write(f'{raw_log}\n')
            labels.append(0 if label == normal_label else 1)
            if i % step <= 0:
                logger.info('Processed %d / %d lines', i, n_lines)
    logger.info('Save labels into \'%s\'', labels_path)
    np.save(labels_path, np.array(labels))


def head(args):
    logs_path = pathlib.Path(args['logs_path'])
    logs_head_path = pathlib.Path(args['logs_head_path'])

    ml4logs.utils.mkdirs(files=[logs_head_path])

    logger.info('Read first %d lines from \'%s\'', args['n_rows'], logs_path)
    with logs_path.open() as in_f:
        logs_head = tuple(map(
            lambda l: l.strip(), itools.islice(in_f, args['n_rows'])))
    logger.info('Save them into \'%s\'', logs_head_path)
    logs_head_path.write_text('\n'.join(logs_head))
