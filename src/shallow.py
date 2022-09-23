import pickle

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import optuna
import numpy as np
import argparse


def svm_opt(trial, X, y):
    use_std = trial.suggest_categorical('std', [True, False])
    n_components = trial.suggest_int('n_components', 10, 888)
    C = trial.suggest_float('svc_c', 1e-10, 1e10, log=True)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

    std = StandardScaler()
    pca = PCA(n_components=n_components)
    clf = SVC(C=C, gamma=gamma, cache_size=2048)

    if use_std:
        pipe = Pipeline(steps=[('std', std), ('pca', pca), ('clf', clf)], memory='sklearn_tmp')
    else:
        pipe = Pipeline(steps=[('pca', pca), ('clf', clf)], memory='sklearn_tmp')
    scores = cross_val_score(clf, X, y, cv=5, verbose=3)
    return np.mean(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'both'],
        default='val'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1
    )

    args = parser.parse_args()
    split = args.split
    n_jobs = args.n_jobs

    if split == 'val' or 'both':
        with open('features/val_features_1_6gf.pickle', 'rb') as fp:
            val_features = pickle.load(fp)
        x_val = [x for x, y in val_features]
        y_val = [y for x, y in val_features]
        X_val = np.array(x_val)
        y_val = np.array(y_val)
        X, y = X_val, y_val
    if split == 'train' or 'both':
        with open('features/train_features_1_6gf.pickle', 'rb') as fp:
            train_features = pickle.load(fp)
        x_train = [x for x, y in train_features]
        y_train = [y for x, y in train_features]
        X_train = np.array(x_train)
        y_train = np.array(y_train)
        X, y = X_train, y_train
    if split == 'both':
        X = np.concatenate((X_train, X_val))
        y = np.concatenate((y_train, y_val))

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: svm_opt(trial, X, y), n_trials=100, n_jobs=n_jobs)


if __name__ == '__main__':
    main()
