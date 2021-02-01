#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve,
                             roc_auc_score,
                             average_precision_score,
                             precision_recall_fscore_support)
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "[{asctime}][{levelname}][{name}] {message}", style="{")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

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
        'AutoEncoder': AutoEncoder(verbose=2),
        'VAE': VAE(verbosity=2)
    }

    # DEBUG
    # models['AutoEncoder'].set_params(epochs=10)
    # models['VAE'].set_params(epochs=10)
    # END DEBUG

    roc = []
    score = []

    for name, model in models.items():
        logger.info(f"===== Processing '{name}' =====")
        model.fit(x_train_scaled, y_train)
        c_pred = model.predict(x_test_scaled)
        y_pred = model.predict_proba(x_test_scaled)[:, 1]

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
    pd.concat(roc).to_csv(args.report_dir /
                          f'{args.dataset.stem}-pyod-roc.csv', index=False)
    pd.concat(score).to_csv(args.report_dir /
                            f'{args.dataset.stem}-pyod-score.csv', index=False)
    logger.info("Saving results... Done")
