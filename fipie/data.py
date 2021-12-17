from os.path import abspath
from pathlib import Path

import pandas as pd

data_csv = Path(abspath(__file__)).parent.parent.joinpath('data', 'price.csv')


def load_example_data() -> pd.DataFrame:
    """ Load example price data for several representative ETFs

    :return: dataframe containing instrument prices
    """
    return pd.read_csv(data_csv, index_col=0, parse_dates=True)
