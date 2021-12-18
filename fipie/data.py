import pandas as pd

csv_url = 'https://raw.githubusercontent.com/thoriuchi0531/fipie/main/data/price.csv'


def load_example_data() -> pd.DataFrame:
    """ Load example price data for several representative ETFs

    :return: dataframe containing instrument prices
    """
    return pd.read_csv(csv_url, index_col=0, parse_dates=True)
