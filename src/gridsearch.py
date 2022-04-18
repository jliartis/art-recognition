import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA

import pickle


def log_reg_1():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)

    std = StandardScaler()
    pca = PCA()
    clf = LogisticRegression(max_iter=1000)
    param_grid = [{
        'pca__n_components': range(784, 100, -150),
        'clf__solver': ['liblinear'],
        'clf__penalty': ['l2', 'l1'],
        'clf__C': [0.1, 1.0, 10.0]
        }, {
        'pca__n_components': range(784, 100, -150),
        'clf__solver': ['lbfgs'],
        'clf__penalty': ['l2'],
        'clf__C': [0.1, 1.0, 10.0]
        }, {
        'pca__n_components': range(784, 100, -150),
        'clf__solver': ['lbfgs'],
        'clf__penalty': ['none']
        }, {
        'pca__n_components': range(784, 100, -150),
        'clf__solver': ['sag'],
        'clf__penalty': ['l2'],
        'clf__C': [0.1, 1.0, 10.0]
        }, {
        'pca__n_components': range(784, 100, -150),
        'clf__solver': ['sag'],
        'clf__penalty': ['none']
        }, {
        'pca__n_components': range(784, 100, -150),
        'clf__solver': ['saga'],
        'clf__penalty': ['l2', 'l1'],
        'clf__C': [0.1, 1.0, 10.0]
        }, {
        'pca__n_components': range(784, 100, -150),
        'clf__solver': ['saga'],
        'clf__penalty': ['none']
        }, {
        'pca__n_components': range(784, 100, -150),
        'clf__solver': ['saga'],
        'clf__penalty': ['elasticnet'],
        'clf__C': [0.1, 1.0, 10.0],
        'clf__l1_ratio': [0.2, 0.5, 0.8]
        }
    ]


    pipe = Pipeline(steps=[('pca', pca), ('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=5, scoring=['accuracy', 'f1_macro'],
                             verbose=10, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    lr_results = pd.DataFrame(estimator.cv_results_)

    lr_results.rename(axis=1, inplace=True, mapper={
        'param_clf__solver': 'solver',
        'param_clf__penalty': 'penalty',
        'param_clf__C': 'C',
        'param_pca__n_components': 'pca'
    })

    with open('lr_results_1.pickle', 'wb') as f:
        pickle.dump(lr_results, f)

    print(lr_results[
        ['mean_fit_time', 'solver', 'penalty', 'C', 'mean_test_accuracy', 'std_test_accuracy',
        'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).head(16))
    print(lr_results[
        ['mean_fit_time', 'solver', 'penalty', 'C', 'mean_test_f1_macro', 'std_test_f1_macro',
        'rank_test_f1_macro']
    ].sort_values(by=['rank_test_f1_macro']).head(16))
    

def log_reg_2():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)

    std = StandardScaler()
    pca = PCA()
    clf = LogisticRegression(max_iter=1000)
    param_grid = [{
        'pca__n_components': range(350, 49, -50),
        'clf__solver': ['liblinear'],
        'clf__penalty': ['l1'],
        'clf__C': [0.05, 0.1, 0.2],
        }, {
        'pca__n_components': range(350, 49, -50),
        'clf__solver': ['saga'],
        'clf__penalty': ['elasticnet'],
        'clf__C': [0.05, 0.1, 0.2],
        'clf__l1_ratio': [0.6, 0.8, 0.9]
        }
    ]


    pipe = Pipeline(steps=[('pca', pca), ('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=5, scoring=['accuracy', 'f1_macro'],
                             verbose=2, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    lr_results = pd.DataFrame(estimator.cv_results_)

    lr_results.rename(axis=1, inplace=True, mapper={
        'param_clf__solver': 'solver',
        'param_clf__penalty': 'penalty',
        'param_clf__C': 'C',
        'param_pca__n_components': 'pca'
    })

    with open('lr_results_2.pickle', 'wb') as f:
        pickle.dump(lr_results, f)

    print(lr_results[
        ['mean_fit_time', 'solver', 'C', 'pca', 'mean_test_accuracy', 'std_test_accuracy',
        'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).head(16))
    print(lr_results[
        ['mean_fit_time', 'solver', 'C', 'pca', 'mean_test_f1_macro', 'std_test_f1_macro',
        'rank_test_f1_macro']
    ].sort_values(by=['rank_test_f1_macro']).head(16))

    
    
def naive_bayes_1():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)

    std = StandardScaler()
    pca = PCA()
    qt = QuantileTransformer()
    clf = GaussianNB()
    param_grid = [{
        'pca__n_components': range(784, 100, -150),
        }
    ]


    pipe = Pipeline(steps=[('pca', pca), ('qt', qt), ('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=5, scoring=['accuracy', 'f1_macro'],
                             verbose=10, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    nb_results = pd.DataFrame(estimator.cv_results_)

    nb_results.rename(axis=1, inplace=True, mapper={
        'param_pca__n_components': 'pca'
    })

    with open('nb_results_1.pickle', 'wb') as f:
        pickle.dump(nb_results, f)

    print(nb_results[
        ['mean_fit_time','pca', 'mean_test_accuracy', 'std_test_accuracy',
        'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).head(16))
    print(nb_results[
        ['mean_fit_time', 'pca', 'mean_test_f1_macro', 'std_test_f1_macro',
        'rank_test_f1_macro']
    ].sort_values(by=['rank_test_f1_macro']).head(16))
    

def naive_bayes_2():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)

    std = StandardScaler()
    pca = PCA()
    qt = QuantileTransformer()
    clf = GaussianNB()
    param_grid = [{
        'pca__n_components': range(200, 49, -25),
        }
    ]


    pipe = Pipeline(steps=[('pca', pca), ('qt', qt), ('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=5, scoring=['accuracy', 'f1_macro'],
                             verbose=10, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    nb_results = pd.DataFrame(estimator.cv_results_)

    nb_results.rename(axis=1, inplace=True, mapper={
        'param_pca__n_components': 'pca'
    })

    with open('nb_results_2.pickle', 'wb') as f:
        pickle.dump(nb_results, f)

    print(nb_results[
        ['mean_fit_time','pca', 'mean_test_accuracy', 'std_test_accuracy',
        'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).head(16))
    print(nb_results[
        ['mean_fit_time', 'pca', 'mean_test_f1_macro', 'std_test_f1_macro',
        'rank_test_f1_macro']
    ].sort_values(by=['rank_test_f1_macro']).head(16))

    
def knn_1():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    std = StandardScaler()
    pca = PCA()
    clf = KNeighborsClassifier()
    param_grid = [{
        'pca__n_components': range(784, 20, -60),
        'clf__n_neighbors': [1, 5, 9],
        'clf__metric': ['euclidean', 'manhattan', 'chebyshev', 'seuclidean']
        }
    ]


    pipe = Pipeline(steps=[('std', std), ('pca', pca), ('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=5, scoring=['accuracy', 'f1_macro'],
                             verbose=10, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    knn_results = pd.DataFrame(estimator.cv_results_)

    knn_results.rename(axis=1, inplace=True, mapper={
        'param_pca__n_components': 'pca',
        'param_clf__n_neighbors': 'neighbors',
        'param_clf__metric': 'metric'
    })

    with open('knn_results_1.pickle', 'wb') as f:
        pickle.dump(knn_results, f)

    print(knn_results[
        ['mean_fit_time', 'metric', 'neighbors', 'pca', 'mean_test_accuracy', 'std_test_accuracy',
        'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).groupby('metric').head(4))
    print(knn_results[
        ['mean_fit_time', 'metric', 'neighbors', 'pca', 'mean_test_f1_macro', 'std_test_f1_macro',
        'rank_test_f1_macro']
    ].sort_values(by=['rank_test_f1_macro']).groupby('metric').head(4))
    

def knn_2():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    std = StandardScaler()
    pca = PCA()
    clf = KNeighborsClassifier()
    param_grid = [{
        'pca__n_components': range(784, 20, -60),
        'clf__n_neighbors': [1, 5, 9],
        'clf__metric': ['euclidean', 'manhattan', 'chebyshev', 'seuclidean']
        }
    ]


    pipe = Pipeline(steps=[('pca', pca), ('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=5, scoring=['accuracy', 'f1_macro'],
                             verbose=10, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    knn_results = pd.DataFrame(estimator.cv_results_)

    knn_results.rename(axis=1, inplace=True, mapper={
        'param_pca__n_components': 'pca',
        'param_clf__n_neighbors': 'neighbors',
        'param_clf__metric': 'metric'
    })

    with open('knn_results_2.pickle', 'wb') as f:
        pickle.dump(knn_results, f)

    print(knn_results[
        ['mean_fit_time', 'metric', 'neighbors', 'pca', 'mean_test_accuracy', 'std_test_accuracy',
        'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).groupby('metric').head(4))
    print(knn_results[
        ['mean_fit_time', 'metric', 'neighbors', 'pca', 'mean_test_f1_macro', 'std_test_f1_macro',
        'rank_test_f1_macro']
    ].sort_values(by=['rank_test_f1_macro']).groupby('metric').head(4))

    
def knn_3():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    std = StandardScaler()
    pca = PCA()
    clf = KNeighborsClassifier()
    param_grid = [{
        'clf__n_neighbors': [1, 5, 9],
        'clf__metric': ['euclidean', 'manhattan', 'chebyshev', 'seuclidean']
        }
    ]


    pipe = Pipeline(steps=[('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=5, scoring=['accuracy', 'f1_macro'],
                             verbose=10, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    knn_results = pd.DataFrame(estimator.cv_results_)

    knn_results.rename(axis=1, inplace=True, mapper={
        'param_pca__n_components': 'pca',
        'param_clf__n_neighbors': 'neighbors',
        'param_clf__metric': 'metric'
    })

    with open('knn_results_3.pickle', 'wb') as f:
        pickle.dump(knn_results, f)

    print(knn_results[
        ['mean_fit_time', 'metric', 'neighbors', 'mean_test_accuracy', 'std_test_accuracy',
        'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).groupby('metric').head(4))
    print(knn_results[
        ['mean_fit_time', 'metric', 'neighbors', 'mean_test_f1_macro', 'std_test_f1_macro',
        'rank_test_f1_macro']
    ].sort_values(by=['rank_test_f1_macro']).groupby('metric').head(4))
    
    
    
def knn_4():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    std = StandardScaler()
    pca = PCA()
    clf = KNeighborsClassifier()
    param_grid = [{
        'clf__n_neighbors': [1, 5, 9],
        'clf__metric': ['euclidean', 'manhattan', 'chebyshev', 'seuclidean']
        }
    ]


    pipe = Pipeline(steps=[('std', std), ('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=5, scoring=['accuracy', 'f1_macro'],
                             verbose=10, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    knn_results = pd.DataFrame(estimator.cv_results_)

    knn_results.rename(axis=1, inplace=True, mapper={
        'param_pca__n_components': 'pca',
        'param_clf__n_neighbors': 'neighbors',
        'param_clf__metric': 'metric'
    })

    with open('knn_results_4.pickle', 'wb') as f:
        pickle.dump(knn_results, f)

    print(knn_results[
        ['mean_fit_time', 'metric', 'neighbors', 'mean_test_accuracy', 'std_test_accuracy',
        'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).groupby('metric').head(4))
    print(knn_results[
        ['mean_fit_time', 'metric', 'neighbors', 'mean_test_f1_macro', 'std_test_f1_macro',
        'rank_test_f1_macro']
    ].sort_values(by=['rank_test_f1_macro']).groupby('metric').head(4))


def knn_5():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    std = StandardScaler()
    pca = PCA()
    clf = KNeighborsClassifier()
    param_grid = [{
        'pca__n_components': range(100, 19, -10),
        'clf__n_neighbors': [13, 15, 17],
        'clf__metric': ['euclidean']
        }
    ]


    pipe = Pipeline(steps=[('std', std), ('pca', pca), ('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=5, scoring=['accuracy', 'f1_macro'],
                             verbose=10, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    knn_results = pd.DataFrame(estimator.cv_results_)

    knn_results.rename(axis=1, inplace=True, mapper={
        'param_pca__n_components': 'pca',
        'param_clf__n_neighbors': 'neighbors',
        'param_clf__metric': 'metric'
    })

    with open('knn_results_5.pickle', 'wb') as f:
        pickle.dump(knn_results, f)

    print(knn_results[
        ['mean_fit_time', 'metric', 'neighbors', 'pca', 'mean_test_accuracy', 'std_test_accuracy',
        'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).groupby('neighbors').head(4))
    print(knn_results[
        ['mean_fit_time', 'metric', 'neighbors', 'pca', 'mean_test_f1_macro', 'std_test_f1_macro',
        'rank_test_f1_macro']
    ].sort_values(by=['rank_test_f1_macro']).groupby('neighbors').head(4))
    
    
def tree_1():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    std = StandardScaler()
    pca = PCA()
    clf = DecisionTreeClassifier()
    param_grid = [{
        'pca__n_components': range(784, 20, -60),
        'clf__criterion': ['gini', 'entropy']
        }
    ]


    pipe = Pipeline(steps=[('std', std), ('pca', pca), ('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=5, scoring=['accuracy', 'f1_macro'],
                             verbose=10, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    tree_results = pd.DataFrame(estimator.cv_results_)

    tree_results.rename(axis=1, inplace=True, mapper={
        'param_pca__n_components': 'pca',
        'param_clf__criterion': 'criterion',
    })

    with open('tree_results_1.pickle', 'wb') as f:
        pickle.dump(tree_results, f)

    print(tree_results[
        ['mean_fit_time', 'criterion', 'pca', 'mean_test_accuracy', 'std_test_accuracy',
        'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).groupby('criterion').head(8))
    print(tree_results[
        ['mean_fit_time', 'criterion', 'pca', 'mean_test_f1_macro', 'std_test_f1_macro',
        'rank_test_f1_macro']
    ].sort_values(by=['rank_test_f1_macro']).groupby('criterion').head(8))
    
    
def tree_2():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    std = StandardScaler()
    pca = PCA()
    clf = DecisionTreeClassifier()
    param_grid = [{
        'pca__n_components': range(150, 20, -10),
        'clf__criterion': ['gini', 'entropy']
        }
    ]


    pipe = Pipeline(steps=[('std', std), ('pca', pca), ('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=5, scoring=['accuracy', 'f1_macro'],
                             verbose=10, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    tree_results = pd.DataFrame(estimator.cv_results_)

    tree_results.rename(axis=1, inplace=True, mapper={
        'param_pca__n_components': 'pca',
        'param_clf__criterion': 'criterion',
    })

    with open('tree_results_2.pickle', 'wb') as f:
        pickle.dump(tree_results, f)

    print(tree_results[
        ['mean_fit_time', 'criterion', 'pca', 'mean_test_accuracy', 'std_test_accuracy',
        'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).groupby('criterion').head(8))
    print(tree_results[
        ['mean_fit_time', 'criterion', 'pca', 'mean_test_f1_macro', 'std_test_f1_macro',
        'rank_test_f1_macro']
    ].sort_values(by=['rank_test_f1_macro']).groupby('criterion').head(8))
    
    
def ada_boost_1():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    std = StandardScaler()
    pca = PCA()
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))
    param_grid = [{
        'pca__n_components': range(120, 19, -20),
        'clf__n_estimators': [25, 50, 100],
        'clf__learning_rate': [0.5, 1.0, 2.0]
        }
    ]


    pipe = Pipeline(steps=[('std', std), ('pca', pca), ('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=5, scoring=['accuracy', 'f1_macro'],
                             verbose=10, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    tree_results = pd.DataFrame(estimator.cv_results_)

    tree_results.rename(axis=1, inplace=True, mapper={
        'param_pca__n_components': 'pca',
        'param_clf__n_estimators': 'estimators',
        'param_clf__learning_rate': 'lr'
    })

    with open('adaboost_results_1.pickle', 'wb') as f:
        pickle.dump(tree_results, f)

    print(tree_results[
        ['mean_fit_time', 'estimators', 'lr', 'pca', 'mean_test_accuracy', 'std_test_accuracy',
        'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).head(8))
    print(tree_results[
        ['mean_fit_time', 'estimators', 'lr', 'pca', 'mean_test_f1_macro', 'std_test_f1_macro',
        'rank_test_f1_macro']
    ].sort_values(by=['rank_test_f1_macro']).head(8))
    
    
def ada_boost_2():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    std = StandardScaler()
    pca = PCA()
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))
    param_grid = [{
        'pca__n_components': range(140, 19, -20),
        'clf__n_estimators': [100, 200, 400],
        'clf__learning_rate': [0.5, 1.0, 2.0]
        }
    ]


    pipe = Pipeline(steps=[('std', std), ('pca', pca), ('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=5, scoring=['accuracy', 'f1_macro'],
                             verbose=10, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    tree_results = pd.DataFrame(estimator.cv_results_)

    tree_results.rename(axis=1, inplace=True, mapper={
        'param_pca__n_components': 'pca',
        'param_clf__n_estimators': 'estimators',
        'param_clf__learning_rate': 'lr'
    })

    with open('adaboost_results_2.pickle', 'wb') as f:
        pickle.dump(tree_results, f)

    print(tree_results[
        ['mean_fit_time', 'estimators', 'lr', 'pca', 'mean_test_accuracy', 'std_test_accuracy',
        'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).head(8))
    print(tree_results[
        ['mean_fit_time', 'estimators', 'lr', 'pca', 'mean_test_f1_macro', 'std_test_f1_macro',
        'rank_test_f1_macro']
    ].sort_values(by=['rank_test_f1_macro']).head(8))

def ada_boost_3():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    std = StandardScaler()
    pca = PCA()
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))
    param_grid = [{
        'pca__n_components': range(90, 9, -10),
        'clf__n_estimators': [400, 600, 800],
        'clf__learning_rate': [1.0, 2.0, 4.0, 8.0]
        }
    ]


    pipe = Pipeline(steps=[('std', std), ('pca', pca), ('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=5, scoring=['accuracy', 'f1_macro'],
                             verbose=10, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    tree_results = pd.DataFrame(estimator.cv_results_)

    tree_results.rename(axis=1, inplace=True, mapper={
        'param_pca__n_components': 'pca',
        'param_clf__n_estimators': 'estimators',
        'param_clf__learning_rate': 'lr'
    })

    with open('adaboost_results_3.pickle', 'wb') as f:
        pickle.dump(tree_results, f)

    print(tree_results[
        ['mean_fit_time', 'estimators', 'lr', 'pca', 'mean_test_accuracy', 'std_test_accuracy',
        'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).head(8))
    print(tree_results[
        ['mean_fit_time', 'estimators', 'lr', 'pca', 'mean_test_f1_macro', 'std_test_f1_macro',
        'rank_test_f1_macro']
    ].sort_values(by=['rank_test_f1_macro']).head(8))
    
    
def ada_boost_4():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    std = StandardScaler()
    pca = PCA()
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))
    param_grid = [{
        'pca__n_components': range(40, 9, -5),
        'clf__n_estimators': [800, 1200, 1400],
        'clf__learning_rate': [0.25, 0.5, 2.0, 2.5]
        }
    ]


    pipe = Pipeline(steps=[('std', std), ('pca', pca), ('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=10, scoring=['accuracy', 'f1_macro'],
                             verbose=10, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    tree_results = pd.DataFrame(estimator.cv_results_)

    tree_results.rename(axis=1, inplace=True, mapper={
        'param_pca__n_components': 'pca',
        'param_clf__n_estimators': 'estimators',
        'param_clf__learning_rate': 'lr'
    })

    with open('adaboost_results_4.pickle', 'wb') as f:
        pickle.dump(tree_results, f)

    print(tree_results[
        ['mean_fit_time', 'estimators', 'lr', 'pca', 'mean_test_accuracy', 'std_test_accuracy',
        'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).head(8))
    print(tree_results[
        ['mean_fit_time', 'estimators', 'lr', 'pca', 'mean_test_f1_macro', 'std_test_f1_macro',
        'rank_test_f1_macro']
    ].sort_values(by=['rank_test_f1_macro']).head(8))
    
    
def ada_boost_5():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    std = StandardScaler()
    pca = PCA()
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))
    param_grid = [{
        'pca__n_components': range(30, 4, -5),
        'clf__n_estimators': [1400, 1800, 2000],
        'clf__learning_rate': [1.8, 2.0]
        }
    ]


    pipe = Pipeline(steps=[('std', std), ('pca', pca), ('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=10, scoring=['accuracy', 'f1_macro'],
                             verbose=10, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    tree_results = pd.DataFrame(estimator.cv_results_)

    tree_results.rename(axis=1, inplace=True, mapper={
        'param_pca__n_components': 'pca',
        'param_clf__n_estimators': 'estimators',
        'param_clf__learning_rate': 'lr'
    })

    with open('adaboost_results_5.pickle', 'wb') as f:
        pickle.dump(tree_results, f)

    print(tree_results[
        ['mean_fit_time', 'estimators', 'lr', 'pca', 'mean_test_accuracy', 'std_test_accuracy',
        'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).head(8))
    print(tree_results[
        ['mean_fit_time', 'estimators', 'lr', 'pca', 'mean_test_f1_macro', 'std_test_f1_macro',
        'rank_test_f1_macro']
    ].sort_values(by=['rank_test_f1_macro']).head(8))
    
def svm_1():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    std = StandardScaler()
    pca = PCA()
    clf = SVC(cache_size=8000)
    param_grid = [{
        'pca__n_components': range(784, 20, -60),
        'clf__kernel': ['linear']
        }, {
        'pca__n_components': range(784, 20, -60),
        'clf__kernel': ['rbf'],
        'clf__gamma': ['auto', 'scale']
        }, {
        'pca__n_components': range(784, 20, -60),
        'clf__kernel': ['sigmoid'],
        'clf__gamma': ['auto', 'scale'],
        'clf__coef0': [0, 1]
        }, {
        'pca__n_components': range(784, 20, -60),
        'clf__kernel': ['poly'],
        'clf__gamma': ['auto', 'scale'],
        'clf__coef0': [0, 1],
        'clf__degree': [2, 3, 4]
        }
    ]


    pipe = Pipeline(steps=[('std', std), ('pca', pca), ('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=5, scoring=['accuracy'],
                             verbose=10, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    results = pd.DataFrame(estimator.cv_results_)

    results.rename(axis=1, inplace=True, mapper={
        'param_pca__n_components': 'pca #components',
        'param_clf__kernel': 'kernel',
        'param_clf__degree': 'poly degree',
        'param_clf__gamma': 'gamma',
        'param_clf__coef0': 'coef0'
    })

    with open('svm_results_1.pickle', 'wb') as f:
        pickle.dump(results, f)

    print(results[
        ['mean_fit_time', 'pca #components', 'kernel', 'poly degree', 'gamma', 'coef0',
         'mean_test_accuracy', 'std_test_accuracy', 'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).groupby('kernel').head(5))
    

def svm_2():
    with open('train_features.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    std = StandardScaler()
    pca = PCA()
    clf = SVC(cache_size=8000)
    param_grid = [{
        'pca__n_components': range(784, 399, -40),
        'clf__kernel': ['rbf'],
        'clf__gamma': [1e-2, 1e-3, 1e-4],
        'clf__C': [0.5, 1, 2]
        }, {
        'pca__n_components': range(500, 249, -25),
        'clf__kernel': ['poly'],
        'clf__gamma': [1e-2, 1e-3, 1e-4],
        'clf__coef0': [0.5, 1, 2],
        'clf__degree': [4, 5],
        'clf__C': [0.5, 1, 2]
        }
    ]


    pipe = Pipeline(steps=[('std', std), ('pca', pca), ('clf', clf)], memory='sklearn_tmp')
    estimator = GridSearchCV(pipe, param_grid, cv=5, scoring=['accuracy'],
                             verbose=10, error_score="raise", n_jobs=-1, refit=False)
    estimator.fit(X_train, y_train)

    results = pd.DataFrame(estimator.cv_results_)

    results.rename(axis=1, inplace=True, mapper={
        'param_pca__n_components': 'pca #components',
        'param_clf__kernel': 'kernel',
        'param_clf__degree': 'poly degree',
        'param_clf__gamma': 'gamma',
        'param_clf__coef0': 'coef0',
        'param_clf__C': 'C'
    })

    with open('svm_results_2.pickle', 'wb') as f:
        pickle.dump(results, f)

    print(results[
        ['mean_fit_time', 'pca #components', 'kernel', 'poly degree', 'gamma', 'coef0', 'C',
         'mean_test_accuracy', 'std_test_accuracy', 'rank_test_accuracy']
    ].sort_values(by=['rank_test_accuracy']).groupby('kernel').head(5))