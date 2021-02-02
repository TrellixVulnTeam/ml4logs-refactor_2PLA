# ===== IMPORTS =====
# === Standard library ===
import logging
import pathlib
import tarfile
import itertools as itools

# === Thirdparty ===
import requests


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def download(args):
    path = pathlib.Path(args['path'])

    if not args['force'] and path.exists():
        logger.info('File \'%s\' already exists and \'force\' is false',
                    path)
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
        'BGL': prepare_bgl
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
    logger.error('Prepare is not implemented for \'%s\'', args['dataset'])


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
        logs_head = tuple(itools.islice(in_f, args['n_rows']))
    logger.info('Save them into \'%s\'', logs_head_path)
    logs_head_path.write_text('\n'.join(logs_head))
