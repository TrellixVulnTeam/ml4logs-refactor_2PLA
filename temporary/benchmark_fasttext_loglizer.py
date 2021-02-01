#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===== IMPORTS =====
# === Standard library ===
import pathlib
import argparse
import logging

# === Thirdparty ===
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve,
                             roc_auc_score,
                             average_precision_score,
                             precision_recall_fscore_support)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.svm import LinearSVC

# ==== LOGGING ====
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
LOG_FORMAT = "[{asctime}][{levelname}][{name}] {message}"
formatter = logging.Formatter(LOG_FORMAT, style="{")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# ===== MAIN =====
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=pathlib.Path)
    parser.add_argument('report_dir', type=pathlib.Path)
    args = parser.parse_args()

    assert(args.dataset.exists() and args.dataset.is_file())
    assert(args.report_dir.exists() and args.report_dir.is_dir())

    # Load and preprocess data
    logger.info(f"Loading dataset {args.dataset}...")
    npzfile = np.load(args.dataset)
    logger.info("Dataset is loaded")

    # DEBUG
    # N = 10**4
    # npzfile = {'X': npzfile['X'][:N], 'Y': npzfile['Y'][:N]}
    # END DEBUG

    logger.info("Start preprocessing data...")
    x_train, x_test, y_train, y_test = train_test_split(
        npzfile['X'], npzfile['Y'],
        train_size=0.5, stratify=npzfile['Y']
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    logger.info("Data are preprocessed")

    # Define models
    models = {
        'Tree': DecisionTreeClassifier(),
        'LogisticRegression': LogisticRegression(
            C=100, tol=1e-2, max_iter=10**3),
        'LinearSVC': LinearSVC(penalty='l1', tol=0.1, dual=False),
        'IsolationForest': IsolationForest(
            random_state=2019, max_samples=0.9999,
            contamination=0.03, n_jobs=4),
    }

    roc = []
    score = []

    for name, model in models.items():
        logger.info(f"===== Processing '{name}' =====")
        model.fit(x_train_scaled, y_train)
        c_pred = model.predict(x_test_scaled)
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(x_test_scaled)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_pred = model.decision_function(x_test_scaled)
            if (name in {"IsolationForest"}):
                y_pred = -y_pred
                c_pred[c_pred == 1] = 0
                c_pred[c_pred == -1] = 1
        else:
            raise NotImplementedError()
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, c_pred, average='binary', zero_division=0)

        stats = {
            'Model': name,
            'AUC': auc,
            'AP': ap,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
        logger.info(f"Stats: {stats}")
        roc.append(pd.DataFrame({'Model': name, 'FPR': fpr, 'TPR': tpr}))
        score.append(pd.DataFrame([stats]))

    # Save results
    logger.info("Saving results...")
    pd.concat(roc).to_csv(
        args.report_dir / f'{args.dataset.stem}-loglizer-roc.csv',
        index=False
    )
    pd.concat(score).to_csv(
        args.report_dir / f'{args.dataset.stem}-loglizer-score.csv',
        index=False
    )
    logger.info("Saving results... Done")
