import pandas as pd
from sklearn.model_selection import train_test_split

from connector.pg_connector import get_data
from conf.conf import logging, settings
from util.util import save_model, load_model
import dynaconf
import numpy as np
import warnings

warnings.filterwarnings("ingore")

def split(df: pd.DataFrame)-> pd.DataFrame:
    """
    Spliting DataSet
    """
    logging.info('splitting the df to X and y')
    X = df.iloc[:, :-1]
    y = df['target']
    logging.info('spitting the data to train and test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 51)
    return X_train, X_test, y_train, y_test

def training_model(ModelClass, X_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs)-> any:
    """
    Training chosen model
    """
    settings.load_file(path="conf/settings.toml")
    logging.info('initializing model')
    if ModelClass.__name__ == 'LogisticRegression':
        clf = ModelClass(**kwargs, max_iter=1000000)
    else:
        clf = ModelClass(**kwargs)
    logging.info('training model')
    clf.fit(X_train, y_train)
    score = clf.score(X_train, y_train)
    logging.info(f"Model score is {clf.score(X_train, y_train)}")
    save_model((settings.Dir.dir+str(ModelClass.__name__)+'.pkl'), clf)
    return clf, score

def predict(ModelPath, X_train)-> any:
    """
    Predicting with a chosen model
    """
    settings.load_file(path="conf/settings.toml")
    clf = load_model((ModelPath))
    logging.info('predicting outout or given data')
    prediction = clf.predict(X_train)
    logging.info('Model predicts:')
    return print (prediction)