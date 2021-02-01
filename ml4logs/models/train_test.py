# ===== IMPORTS =====
# === Standard library ===
import logging
import pathlib

# === Thirdparty ===
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score,
                             average_precision_score,
                             precision_recall_fscore_support)


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== CLASSES =====
class LinearSVCWrapper:
    def __init__(self, **kwargs):
        self._linear_svc = LinearSVC(**kwargs)
        self._clf = CalibratedClassifierCV(self._linear_svc)

    def fit(self, X, Y):
        self._clf.fit(X, Y)
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)


# ===== FUNCTIONS =====
def train_test_model(args):
    MODEL_CLASSES = {
        'logistic_regression': LogisticRegression,
        'decision_tree': DecisionTreeClassifier,
        'linear_svc': LinearSVCWrapper
    }

    train_path = pathlib.Path(args['train_path'])
    test_path = pathlib.Path(args['test_path'])
    # stats_dir = pathlib.Path(args['stats_dir'])

    # if not args['force'] and stats_dir.exists():
    #     logger.info('Folder \'%s\' already exists and \'force\' is false',
    #                 stats_dir)
    #     return
    FILES_TO_CHECK = [train_path, test_path]
    for file_path in FILES_TO_CHECK:
        if not file_path.exists():
            logger.error('File \'%s\' does not exist', file_path)
            return

    logger.info('Load train dataset from \'%s\'', train_path)
    npzfile_train = np.load(train_path)
    logger.info('Load test dataset from \'%s\'', test_path)
    npzfile_test = np.load(test_path)

    logger.info('Initialize model \'%s\'', args['model'])
    model = MODEL_CLASSES[args['model']](**args['model_args'])
    logger.info('Fit train data to model')
    model.fit(npzfile_train['X'], npzfile_train['Y'])
    logger.info('Predict classes and probability on test data')
    c_pred = model.predict(npzfile_test['X'])
    y_pred = model.predict_proba(npzfile_test['X'])[:, 1]

    logger.info('Compute couple of metrics on test data')
    auc = roc_auc_score(npzfile_test['Y'], y_pred)
    ap = average_precision_score(npzfile_test['Y'], y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        npzfile_test['Y'], c_pred, average='binary', zero_division=0
    )
    logger.info('AUC = %.2f, AP = %.2f', auc, ap)
    logger.info('Precision = %.2f, Recall = %.2f, F1-score = %.2f',
                precision, recall, f1)
