import pandas as pd

from fipie.data import load_example_data


def test_load_example_data():
    data = load_example_data()
    assert isinstance(data, pd.DataFrame)


def test_instrument_size():
    data = load_example_data()

    # 7 instruments
    assert data.shape[1] == 7
