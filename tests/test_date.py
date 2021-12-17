from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from fipie.data import load_example_data
from fipie.date import infer_ts_frequency, infer_ann_factor


def create_hourly():
    # roughly hourly, representing weekends and holidays.
    ts = pd.Series(index=[
        datetime(2021, 1, 1, 0),
        datetime(2021, 1, 1, 1),
        datetime(2021, 1, 1, 2),
        datetime(2021, 1, 1, 3),
        datetime(2021, 1, 2, 0),
    ], dtype=np.float64)
    return ts


def create_daily():
    # roughly daily, representing weekends and holidays.
    ts = pd.Series(index=[
        datetime(2021, 1, 1),
        datetime(2021, 1, 2),
        datetime(2021, 1, 3),
        datetime(2021, 1, 4),
        datetime(2021, 1, 7),
    ], dtype=np.float64)
    return ts


def create_weekly():
    ts = load_example_data()
    ts = ts.asfreq('w', method='pad')
    return ts


def create_monthly():
    ts = load_example_data()
    ts = ts.asfreq('m', method='pad')
    return ts


def test_infer_ts_frequency_hourly():
    ts = create_hourly()

    assert infer_ts_frequency(ts) == 'h'
    assert infer_ann_factor(ts) == 252 * 24


def test_infer_ts_frequency_daily():
    ts = create_daily()

    assert infer_ts_frequency(ts) == 'd'
    assert infer_ann_factor(ts) == 252


def test_infer_ts_frequency_weekly():
    ts = create_weekly()

    assert infer_ts_frequency(ts) == 'w'
    assert infer_ann_factor(ts) == 52


def test_infer_ts_frequency_monthly():
    ts = create_monthly()

    assert infer_ts_frequency(ts) == 'm'
    assert infer_ann_factor(ts) == 12


@pytest.mark.xfail
def test_infer_ts_frequency_unknown_freq():
    ts = create_daily()
    ts = ts.asfreq('2w', method='pad')

    infer_ann_factor(ts)
