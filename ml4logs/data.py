# ===== IMPORTS =====
# === Standard library ===
import logging
import pathlib
import tarfile
import itertools as itools

# === Thirdparty ===
import requests
import numpy as np


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def download(args):
    path = pathlib.Path(args['path'])

    if not args['force'] and path.exists():
        logger.info('File \'%s\' exists', path.name)
        logger.info('Argument \'force\' is false')
        logger.info('Skip download step')
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info('Download \'%s\'', args['url'])
    response = requests.get(args['url'])
    logger.info('Save into \'%s\'', path)
    path.write_bytes(response.content)


def extract(args):
    in_path = pathlib.Path(args['in_path'])
    out_dir = pathlib.Path(args['out_dir'])

    if not in_path.exists():
        logger.error('File \'%s\' does not exist', in_path)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info('Open \'%s\' as tarfile', in_path)
    with tarfile.open(in_path, 'r:gz') as tar:
        members = tar.getmembers()
        if not args['force']:
            logger.info('Filter out existing files')
            members = list(filter(
                lambda m: not (out_dir / m.name).exists(),
                members
            ))
        logger.info('Extract %d files', len(members))
        tar.extractall(out_dir, members=members)


def prepare(args):
    HANDLERS = {
        'HDFS1': prepare_hdfs_1,
        'HDFS2': prepare_hdfs_2,
        'BGL': prepare_bgl,
        'Thunderbird': prepare_thunderbird
    }

    if args['dataset'] not in HANDLERS:
        logger.error('Unknown dataset \'%s\'', args['dataset'])
        return

    HANDLERS[args['dataset']](args)


def prepare_hdfs_1(args):
    in_dir = pathlib.Path(args['in_dir'])
    out_logs_path = pathlib.Path(args['logs_path'])
    out_labels_path = pathlib.Path(args['labels_path'])

    FOLDERS_TO_CREATE = [out_logs_path.parent, out_labels_path.parent]
    for folder in FOLDERS_TO_CREATE:
        folder.mkdir(parents=True, exist_ok=True)

    FILES_TO_RENAME = [
        (in_dir / 'HDFS.log', out_logs_path),
        (in_dir / 'anomaly_label.csv', out_labels_path)
    ]
    for in_path, out_path in FILES_TO_RENAME:
        if not args['force'] and out_path.exists():
            logger.info('File \'%s\' already exists and \'force\' is false',
                        out_path)
            continue
        if not in_path.exists():
            logger.error('File \'%s\' does not exist', in_path)
            continue
        logger.info('Rename \'%s\' with \'%s\'', in_path, out_path)
        in_path.replace(out_path)


def prepare_hdfs_2(args):
    logger.error('Prepare is not implemented for \'%s\'', args['dataset'])


def prepare_bgl(args):
    in_dir = pathlib.Path(args['in_dir'])
    split_labels(args, in_dir / 'BGL.log', '-')


def prepare_thunderbird(args):
    in_dir = pathlib.Path(args['in_dir'])
    split_labels(args, in_dir / 'Thunderbird.log', '-')


def split_labels(args, in_path, normal_label):
    out_logs_path = pathlib.Path(args['logs_path'])
    out_labels_path = pathlib.Path(args['labels_path'])

    if not args['force'] and out_logs_path.exists() \
            and out_labels_path.exists():
        logger.info('Output files already exists')
        return
    if not in_path.exists():
        logger.error('File \'%s\' does not exist', in_path)
        return
    FOLDERS_TO_CREATE = [out_logs_path.parent, out_labels_path.parent]
    for folder in FOLDERS_TO_CREATE:
        folder.mkdir(parents=True, exist_ok=True)

    logger.info('Read logs from \'%s\'', in_path)
    logs = in_path.read_text(encoding='utf8').strip().split('\n')
    logger.info('Split them into labels and raw logs')
    labels, raw_logs = tuple(zip(*map(
        lambda line: line.split(maxsplit=1), logs)))
    logger.info('Save raw logs into \'%s\'', out_logs_path)
    out_logs_path.write_text('\n'.join(raw_logs), encoding='utf8')
    logger.info('Map labels to 0/1 (normal/anomaly)')
    labels = np.array(tuple(map(
        lambda l: 0 if l == normal_label else 1, labels)))
    logger.info('Save labels into \'%s\'', out_labels_path)
    np.save(out_labels_path, labels)


def head(args):
    logs_path = pathlib.Path(args['logs_path'])
    logs_head_path = pathlib.Path(args['logs_head_path'])

    if not args['force'] and logs_head_path.exists():
        logger.info('File \'%s\' already exists and \'force\' is false',
                    logs_head_path)
        return
    if not logs_path.exists():
        logger.error('File \'%s\' does not exist', logs_path)
        return
    logs_head_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info('Read first %d lines from \'%s\'', args['n_rows'], logs_path)
    with logs_path.open() as in_f:
        logs_head = tuple(map(
            lambda l: l.strip(), itools.islice(in_f, args['n_rows'])))
    logger.info('Save them into \'%s\'', logs_head_path)
    logs_head_path.write_text('\n'.join(logs_head))
