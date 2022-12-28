import pickle
from conf.conf import logging

def save_model(dir:str, model) ->None:
    """
    Saving Model
    """
    logging.info ('saving model')
    pickle.dump (model, open(dir, 'wb'))

def load_model(dir:str)-> any:
    """
    Loading Model
    """
    logging.info('loading model')
    model=pickle.load(open(dir, 'rb;'))
    return model