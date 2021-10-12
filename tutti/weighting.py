import abc
from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from tutti.common import ReprMixin
from tutti.date import infer_ann_factor


class Weighting(ReprMixin, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def optimise(self, ret: pd.DataFrame, *args, **kwargs):
        """ Calculate weights for instruments """


class MeanVariance(Weighting):
    """ Weights are determined by the mean-variance approach and maximising the Sharpe ratio.
    Expected returns and risk are estimated by historical means and covariance matrix """

    def optimise(self, ret: pd.DataFrame, *args, **kwargs):
        initial_weights = np.ones(len(ret.columns)) / len(ret.columns)
        mu = ret.mean()
        sigma = ret.cov()

        result = minimize(
            negative_sharpe_ratio,
            initial_weights,
            (mu, sigma),
            method='SLSQP',
        )
        weights = result['x']
        return pd.Series(weights, index=ret.columns)


class EqualWeight(Weighting):
    """ Equal nominal weighting across instruments. For instance, if there are 4 instruments in a portfolio,
    25% of capital is allocated to each one. """

    def __init__(self):
        pass

    def optimise(self, ret: pd.DataFrame, *args, **kwargs) -> pd.Series:
        instruments = ret.columns
        w = 1 / len(instruments)
        return pd.Series(w, index=instruments)


class VolatilityParity(Weighting):
    """ Volatility parity weighting across instruments. Allocate capital so that each instrument has a volatility
    equal to the target volatility. As a result instruments with lower volatility gets a relatively higher nominal
    weight. This method ignores the correlation between assets. """

    def __init__(self, target_vol: float = 0.1):
        self.target_vol = target_vol

    def optimise(self, ret: pd.DataFrame, *args, **kwargs) -> pd.Series:
        ann_factor = infer_ann_factor(ret)
        vol = ret.std().mul(ann_factor ** 0.5)
        scaling = self.target_vol / vol

        instruments = ret.columns
        return pd.Series(scaling / len(instruments), index=instruments)


def negative_sharpe_ratio(weights: List[float], mu: np.array, sigma: np.array) -> float:
    """ Calculate negative Sharpe ratio

    :param weights: instrument weights
    :param mu: expected return of instruments
    :param sigma: covariance matrix of instruments
    :return: negative Sharpe ratio
    """
    weights = np.array(weights)

    port_ret = mu.dot(weights)
    port_vol = weights.dot(sigma).dot(weights) ** 0.5
    port_sharpe = port_ret / port_vol
    return -1 * port_sharpe
