#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Feb 08 13:51 2022

Preprocessing of MCCV folds using kNN imputation and MRMR feature selection

@author: cspielvogel
"""

import os
import math

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pymrmr


def main():
    mccv_path = "/home/cspielvogel/DataStorage/SNN/Data/Tresorit"

    for dataset in os.listdir(mccv_path):
        mccv_path = "/home/cspielvogel/DataStorage/SNN/Data/Tresorit"
        dataset_path = os.path.join(mccv_path, dataset)

        try:
            mccv_path = os.path.join(dataset_path, "03-Monte Carlo split")
        except FileNotFoundError as e:
            print(e)
            continue

        for fold in os.listdir(mccv_path):

            # Skip non-fold directories
            if not fold.startswith("Fold-"):
                continue

            fold_path = os.path.join(mccv_path, fold)

            # Load precreated MCCV folds
            x_train = pd.read_csv(os.path.join(fold_path, "TDS.csv"), sep=";", index_col=0)
            y_train = pd.read_csv(os.path.join(fold_path, "TLD.csv"), sep=";", index_col=0)
            x_test = pd.read_csv(os.path.join(fold_path, "VDS.csv"), sep=";", index_col=0)
            y_test = pd.read_csv(os.path.join(fold_path, "VLD.csv"), sep=";", index_col=0)

            # Imputation of missing values
            imputer = KNNImputer(
                n_neighbors=math.ceil((len(y_train) + len(y_test)) / 20),   # 5% of total samples rounded up
                weights="distance"
            )
            imputer.fit(x_train)
            x_train[:] = imputer.transform(x_train)
            x_test[:] = imputer.transform(x_test)

            # Save imputed folds
            path_stem = fold_path.replace("/Tresorit/", "/Folds_imputed/").replace("03-Monte Carlo split", "04-Imputed")
            if not os.path.exists(path_stem):
                os.makedirs(path_stem)
            x_train.to_csv(os.path.join(path_stem, "TDS.csv"), sep=";")
            y_train.to_csv(os.path.join(path_stem, "TLD.csv"), sep=";")
            x_test.to_csv(os.path.join(path_stem, "VDS.csv"), sep=";")
            y_test.to_csv(os.path.join(path_stem, "VLD.csv"), sep=";")

            # Standardize
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train[:] = scaler.transform(x_train)
            x_test[:] = scaler.transform(x_test)

            # Save standardized folds
            path_stem = fold_path.replace("/Tresorit/", "/Folds_standardized/").replace("03-Monte Carlo split", "05-Standardized")
            if not os.path.exists(path_stem):
                os.makedirs(path_stem)
            x_train.to_csv(os.path.join(path_stem, "TDS.csv"), sep=";")
            y_train.to_csv(os.path.join(path_stem, "TLD.csv"), sep=";")
            x_test.to_csv(os.path.join(path_stem, "VDS.csv"), sep=";")
            y_test.to_csv(os.path.join(path_stem, "VLD.csv"), sep=";")

            # Perform feature selection
            max_feats = int(np.round(np.sqrt(len(y_train) + len(y_test)), 0))
            if x_train.shape[1] > max_feats:
                keep_feats = pymrmr.mRMR(x_train, "MIQ", max_feats)
                x_train = x_train[keep_feats]
                x_test = x_test[keep_feats]

            # Save fold with selected features
            path_stem = fold_path.replace("/Tresorit/", "/Folds_feature-selected/").replace("03-Monte Carlo split", "06-Feature-selected")
            if not os.path.exists(path_stem):
                os.makedirs(path_stem)
            x_train.to_csv(os.path.join(path_stem, "TDS.csv"), sep=";")
            y_train.to_csv(os.path.join(path_stem, "TLD.csv"), sep=";")
            x_test.to_csv(os.path.join(path_stem, "VDS.csv"), sep=";")
            y_test.to_csv(os.path.join(path_stem, "VLD.csv"), sep=";")


if __name__ == "__main__":
    main()
