import pandas as pd
from conf.conf import logging

def get_data (link: str) -> pd.DataFrame:
    logging.info('extracting data')
    df = pd.read_csv(link)
    logging.info('data extracted and df created')
    return df