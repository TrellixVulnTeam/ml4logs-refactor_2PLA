# ===== IMPORTS =====
# === Standard library ===
import argparse
import logging

# === Local ===
import ml4logs


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def download(args: argparse.Namespace):
    for name in set(args.datasets):
        logger.info('Download %s', name)
        tardir = ml4logs.DATASETS[name]['tarpath'].parent
        if not tardir.exists():
            if not args.mkdir:
                logger.error('%s does not exist and --mkdir is false', tardir)
                continue
            tardir.mkdir(parents=True)
        ml4logs.data.download(name, args.force)


def extract(args: argparse.Namespace):
    for name in set(args.datasets):
        logger.info('Extract %s', name)
        tarpath = ml4logs.DATASETS[name]['tarpath']
        if not tarpath.exists():
            logger.error('%s does not exist', tarpath)
            continue
        xdir = ml4logs.DATASETS[name]['xdir']
        if not xdir.exists():
            if not args.mkdir:
                logger.error('%s does not exist and --mkdir is false', xdir)
                continue
            xdir.mkdir(parents=True)
        ml4logs.data.extract(name, args.force)


def prepare(args: argparse.Namespace):
    for name in set(args.datasets):
        logger.info('Prepare %s', name)
        xdir = ml4logs.DATASETS[name]['xdir']
        if not xdir.exists():
            logger.error('%s does not exist', xdir)
            continue
        ml4logs.DATASETS[name]['prepare_f'](name)


# ===== MAIN =====
def main():
    ml4logs.utils.setup_logging()

    COMMANDS = {
        'download': download,
        'extract': extract,
        'prepare': prepare
    }

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Download parser
    d_parser = subparsers.add_parser('download')
    d_parser.add_argument(
        'datasets', nargs='+',
        choices=list(ml4logs.DATASETS.keys())
    )
    d_parser.add_argument('--mkdir', action='store_true')
    d_parser.add_argument('--force', action='store_true')

    # Extract parser
    e_parser = subparsers.add_parser('extract')
    e_parser.add_argument(
        'datasets', nargs='+',
        choices=list(ml4logs.DATASETS.keys())
    )
    e_parser.add_argument('--mkdir', action='store_true')
    e_parser.add_argument('--force', action='store_true')

    # Prepare parser
    p_parser = subparsers.add_parser('prepare')
    p_parser.add_argument(
        'datasets', nargs='+',
        choices=list(ml4logs.DATASETS.keys())
    )

    # Call handler
    args = parser.parse_args()
    COMMANDS[args.command](args)


if __name__ == '__main__':
    main()
