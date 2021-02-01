# ===== IMPORTS =====
# === Standard library ===
import argparse
import logging
import pathlib

# === Thirdparty ===
import fasttext

# === Local ===
import ml4logs


# ===== GLOBALS =====
logger = logging.getLogger(__name__)
FASTTEXT_KWARGS = {
    'dim': {'type': int, 'default': 100},
    'minCount': {'type': int, 'default': 10**4},
    'minn': {'type': int, 'default': 1},
    'maxn': {'type': int, 'default': 1},
    'thread': {'type': int, 'default': 1},
    'verbose': {'default': 2}
}


# ===== FUNCTIONS =====
def train_fasttext(args: argparse.Namespace):
    logger.info('Train fasttext on %s', args.dataset)

    args_dict = vars(args)
    ft_keys = tuple(FASTTEXT_KWARGS.keys())
    ft_kwargs = {key: args_dict[key] for key in ft_keys}

    logs_path = ml4logs.DATASETS[args.dataset]['logs_path']
    if not logs_path.exists():
        logger.error('%s does not exist', logs_path)
        return

    if args.model_path is None:
        ft_dir = ml4logs.DATASETS[args.dataset]['ftdir']
        setattr(args, 'model_path', ft_dir / 'skipgram.bin')
    ft_dir = args.model_path.parent
    if not ft_dir.exists():
        if not args.mkdir:
            logger.error('%s does not exist and --mkdir is false', ft_dir)
            return
        ft_dir.mkdir(parents=True)

    model = fasttext.train_unsupervised(str(logs_path), **ft_kwargs)

    logger.info("Saving model into %s...", args.model_path)
    model.save_model(str(args.model_path))
    logger.info("Model is saved")


# ===== MAIN =====
def main():
    ml4logs.utils.setup_logging()

    COMMANDS = {
        'train_fasttext': train_fasttext
    }

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    # FastText parser
    ft_parser = subparsers.add_parser('train_fasttext')
    ft_parser.add_argument('dataset', choices=list(ml4logs.DATASETS.keys()))
    ft_parser.add_argument('--mkdir', action='store_true')
    ft_parser.add_argument('--model_path', type=pathlib.Path)
    for key, kwargs in FASTTEXT_KWARGS.items():
        ft_parser.add_argument('--{}'.format(key), **kwargs)

    # Call handler
    args = parser.parse_args()
    COMMANDS[args.command](args)


if __name__ == '__main__':
    main()
