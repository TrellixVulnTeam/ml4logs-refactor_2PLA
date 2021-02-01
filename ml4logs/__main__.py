# ===== IMPORTS =====
# === Standard library ===
import argparse
import logging
import pathlib
import json

# === Local ===
import ml4logs


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== MAIN =====
def main():
    ml4logs.utils.setup_logging()

    COMMANDS = {
        'download': ml4logs.data.download,
        'extract': ml4logs.data.extract,
        'prepare': ml4logs.data.prepare,
        'train_fasttext': ml4logs.features.fasttext.train_fasttext,
        'preprocess_fasttext': ml4logs.features.fasttext.preprocess_fasttext,
        'aggregate': ml4logs.features.aggregate.aggregate,
        'split': ml4logs.features.utils.split,
        'scale': ml4logs.features.utils.scale,
        'train_test_model': ml4logs.models.train_test.train_test_model
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=pathlib.Path)
    args = parser.parse_args()

    if not args.config_path.exists():
        logger.error('File \'%s\' does not exist', args.config_path)
        return

    logger.info('Read config file \'%s\'', args.config_path)
    config = json.loads(args.config_path.read_text())

    logger.info('Execute pipeline')
    for step in config['pipeline']:
        if step['skip']:
            logger.info('Skip \'%s\' step', step['action'])
            continue
        logger.info('Perform \'%s\' step', step['action'])
        COMMANDS[step['action']](step)


if __name__ == '__main__':
    main()
