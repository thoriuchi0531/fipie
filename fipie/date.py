from typing import Union
import pandas as pd


def infer_ts_frequency(data: Union[pd.Series, pd.DataFrame]) -> str:
    """ Infer the frequency of the given time-series

    :param data: time-series whose sample frequency is inferred.
    :return: offset string such as 'd'
    """
    # infer frequency from the median of `timedelta`. Best effort to infer to remove the effect of weekends etc.
    total_seconds = data.index.to_series().diff().dt.total_seconds().median()
    if total_seconds == 3600:
        # hourly data
        return 'h'
    elif total_seconds == 3600 * 24:
        # daily data
        return 'd'
    elif total_seconds == 3600 * 24 * 7:
        # weekly data
        return 'w'
    elif total_seconds in [3600 * 24 * 30, 3600 * 24 * 31]:
        # monthly data
        return 'm'
    else:
        raise ValueError(f'Frequency cannot be inferred. Index is {data.index}')


def infer_ann_factor(data: Union[pd.Series, pd.DataFrame]) -> float:
    """ Infer the annualisation factor

    :param data: time-series whose annualisation factor is inferred
    :return: an annualisacion constant number. For daily series this returns 252.
    """
    freq = infer_ts_frequency(data)

    freq_map = {
        'h': 252 * 24,
        'd': 252,
        'w': 52,
        'm': 12,
    }
    return freq_map[freq]
