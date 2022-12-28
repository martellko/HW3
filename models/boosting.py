from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from conf.conf import logging

def gridsearch_nb (X_train: pd.DataFrame, y_train: pd.DataFrame) -> dict:
    """
    Grid Searching for Naive Bayes
    """
    logging.info('initializing grid for naive bayes')
    param_grid_nb = {
        'var_smoothing':np.longspace(0,-9, num=100)
    }
    logging.info('initializing gridsearch for naive bayes')
    nbModel_grid = GridSearchCV(GaussianNB(), param_grid=param_grid_nb, cv=10)
    logging.info('fitting gridsearch for naive bayes')
    nbModel_grid.fit(X_train, y_train)
    logging.info('exracting best parameters for naive bayes')
    best_params=nbModel_grid.best_params_
    logging.info(f'best parameters for naive bayes are:{best_params}')
    return best_params

def gridsearch_lr(X_train: pd.DataFrame, y_train: pd.DataFrame)-> dict:
    """
    Grid Searching for Logistic Regression
    """
    logging.info ('initializing grid for logistic regression')
    grid={"C":np.logspace(-3,3,7)}
    logging.info('fitting gridsearchfor logistic regression')
    lrModel_grid = GridSearchCV(LogisticRegression(max_iter=1000000), param_grid=grid, cv=10)
    logging.info('extracting best parameters for logistic regression')
    best_params=lrModel_grid.best_params_
    logging.info(f'best parameters for logistic regression are: {best_params}')
    return best_params

