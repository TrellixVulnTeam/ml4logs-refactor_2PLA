# ===== IMPORTS =====
# === Standard library ===
import logging
import pathlib

# === Thirdparty ===
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === Local ===
import ml4logs


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def split(args):
    in_path = pathlib.Path(args['in_path'])
    train_path = pathlib.Path(args['train_path'])
    test_path = pathlib.Path(args['test_path'])

    ml4logs.utils.mkdirs(files=[train_path, test_path])

    logger.info('Load dataset from \'%s\'', in_path)
    npzfile = np.load(in_path)
    logger.info('Split with \'train size\' = %.2f and \'random seed\' = %d',
                args['train_size'], args['seed'])
    x_train, x_test, y_train, y_test = train_test_split(
        npzfile['X'], npzfile['Y'],
        train_size=args['train_size'], stratify=npzfile['Y'],
        random_state=args['seed']
    )

    logger.info('Save train dataset into \'%s\'', train_path)
    np.savez(train_path, X=x_train, Y=y_train)
    logger.info('Save test dataset into \'%s\'', test_path)
    np.savez(test_path, X=x_test, Y=y_test)


def scale(args):
    train_path = pathlib.Path(args['train_path'])
    test_path = pathlib.Path(args['test_path'])
    train_scaled_path = pathlib.Path(args['train_scaled_path'])
    test_scaled_path = pathlib.Path(args['test_scaled_path'])

    ml4logs.utils.mkdirs(files=[train_scaled_path, test_scaled_path])

    scaler = StandardScaler()
    logger.info('Load train dataset from \'%s\'', train_path)
    npzfile_train = np.load(train_path)
    logger.info('Scale train dataset using sklearn StandardScaler')
    x_train_scaled = scaler.fit_transform(npzfile_train['X'])
    logger.info('Load test dataset from \'%s\'', test_path)
    npzfile_test = np.load(test_path)
    logger.info('Scale test dataset using fitted sklearn StandardScaler')
    x_test_scaled = scaler.transform(npzfile_test['X'])

    logger.info('Save train scaled dataset into \'%s\'', train_scaled_path)
    np.savez(train_scaled_path, X=x_train_scaled, Y=npzfile_train['Y'])
    logger.info('Save test scaled dataset into \'%s\'', test_scaled_path)
    np.savez(test_scaled_path, X=x_test_scaled, Y=npzfile_test['Y'])


def onehot(args):
    in_path = pathlib.Path(args['in_path'])
    out_path = pathlib.Path(args['out_path'])

    ml4logs.utils.mkdirs(files=[out_path])

    logger.info('Load index array')
    array = np.load(in_path)
    logger.info('Create onehot encoded array')
    onehot_array = np.eye(array.max() + 1)[array]
    logger.info('Save onehot encoded array')
    np.save(out_path, onehot_array)
