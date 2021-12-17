import abc
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from fipie.common import ReprMixin
from fipie.date import infer_ann_factor


class Weighting(ReprMixin, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def optimise(self, ret: pd.DataFrame, *args, **kwargs) -> pd.Series:
        """ Calculate weights for instruments

        :param ret: return time-series
        :type ret: pd.DataFrame
        :return: weights for each instrument
        :rtype: pd.Series
        """


class MeanVariance(Weighting):
    """ Weights are determined by the mean-variance approach and maximising the Sharpe ratio.
    Expected returns and risk are estimated by historical means and covariance matrix """

    def __init__(self, fully_invested: bool = True, bounds: Tuple[float, None] = (0, None)):
        """

        :param fully_invested: If True, weights are rescaled so that they add up to 100%. By default the optimal weights
            are rescaled to add up to 100% as the Sharpe ratio is scale-invariant with respect to the weight.
        :type fully_invested: bool, default True
        :param bounds: Lower and upper bounds of weights. If None, weights are unbounded, i.e., ``(0, None)`` means
            it only allows long positions.
        :type bounds: tuple, list-like

        .. note::
           With default parameters, this class produces a long-only fully-invested portfolio.
        """
        self.fully_invested = fully_invested
        self.bounds = bounds

    def optimise(self, ret: pd.DataFrame, *args, **kwargs) -> pd.Series:
        initial_weights = np.ones(len(ret.columns)) / len(ret.columns)
        mu = ret.mean()
        sigma = ret.cov()
        bounds = [self.bounds] * len(ret.columns)
        const = create_const(self.fully_invested)

        result = minimize(
            negative_sharpe_ratio,
            initial_weights,
            (mu, sigma),
            method='SLSQP',
            bounds=bounds,
            constraints=const,
        )
        weights = result['x']
        weights = pd.Series(weights, index=ret.columns)
        return weights


class MinimumVariance(Weighting):
    """ Create a portfolio by minimising its variance.  """

    def __init__(self, fully_invested: bool = True, bounds: Tuple[float, None] = (0, None)):
        """

        :param fully_invested: If True, weights are rescaled so that they add up to 100%. By default the optimal weights
            are rescaled to add up to 100% as the Sharpe ratio is scale-invariant with respect to the weight.
        :type fully_invested: bool, default True
        :param bounds: Lower and upper bounds of weights. If None, weights are unbounded, i.e., ``(0, None)`` means
            it only allows long positions.
        :type bounds: tuple, list-like

        .. note::
           With default parameters, this class produces a long-only fully-invested portfolio.
        """
        self.fully_invested = fully_invested
        self.bounds = bounds

    def optimise(self, ret: pd.DataFrame, *args, **kwargs) -> pd.Series:
        initial_weights = np.ones(len(ret.columns)) / len(ret.columns)
        sigma = ret.cov()
        bounds = [self.bounds] * len(ret.columns)
        const = create_const(self.fully_invested)

        result = minimize(
            portfolio_variance,
            initial_weights,
            (sigma,),
            method='SLSQP',
            bounds=bounds,
            constraints=const,
            tol=1e-9,
        )
        weights = result['x']
        weights = pd.Series(weights, index=ret.columns)
        return weights


class MaximumDiversification(Weighting):
    r""" Create a portfolio which maximises the diversification factor

    .. math::
        \frac{ w^T \cdot \sigma }{ \sqrt{w^T \cdot \Sigma \cdot w} }

    where :math:`w` is the weight vector of each instrument, :math:`\sigma` is the volatility vector of each instrument,
    :math:`\Sigma` is the covariance matrix.
    The numerator of the diversification factor is a weighted average of instrument volatility whereas
    the denominator is the portfolio volatility after diversification.
    """

    def __init__(self, fully_invested: bool = True, bounds: Tuple[float, None] = (0, None)):
        """

        :param fully_invested: If True, weights are rescaled so that they add up to 100%. By default the optimal weights
            are rescaled to add up to 100% as the Sharpe ratio is scale-invariant with respect to the weight.
        :type fully_invested: bool, default True
        :param bounds: Lower and upper bounds of weights. If None, weights are unbounded, i.e., ``(0, None)`` means
            it only allows long positions.
        :type bounds: tuple, list-like

        .. note::
           With default parameters, this class produces a long-only fully-invested portfolio.
        """
        self.fully_invested = fully_invested
        self.bounds = bounds

    def optimise(self, ret: pd.DataFrame, *args, **kwargs) -> pd.Series:
        initial_weights = np.ones(len(ret.columns)) / len(ret.columns)
        sigma = ret.cov()
        bounds = [self.bounds] * len(ret.columns)
        const = create_const(self.fully_invested)

        result = minimize(
            negative_diversification_factor,
            initial_weights,
            (sigma,),
            method='SLSQP',
            bounds=bounds,
            constraints=const,
        )
        weights = result['x']
        weights = pd.Series(weights, index=ret.columns)
        return weights


class EqualRiskContribution(Weighting):
    r""" Create a portfolio with equal risk contribution (ERC, aka risk parity) such that each instrument contributes
    the same amount of risk to the portfolio.
    More formally, let :math:`\sigma \left( w \right)` be the volatility of portfolio and :math:`w` be the weight
    for each instrument. The volatility of portfolio can be decomposed to the following:

    .. math::
        \sigma \left( w \right) =
        \sum_i \sigma_i \left( x \right) =
        \sum_i w_i \frac{ {\partial} \sigma \left( x \right) }{ {\partial} w_i }

    where :math:`\sigma_i \left( x \right)` is the total risk contribution of instrument :math:`i`,
    :math:`\frac{ {\partial} \sigma \left( x \right) }{ {\partial} w_i }` is the marginal risk contribution.

    The ERC portfolio is derived such that all instruments have the same amount of total risk contribution.


    **Reference**

    -   Maillard, S., Roncalli, T. and Teïletche, J., 2010. The properties of equally weighted risk contribution portfolios. The Journal of Portfolio Management, 36(4), pp.60-70.
    """

    def __init__(self, fully_invested: bool = True, bounds: Tuple[float, None] = (0, None)):
        """

        :param fully_invested: If True, weights are rescaled so that they add up to 100%. By default the optimal weights
            are rescaled to add up to 100% as the Sharpe ratio is scale-invariant with respect to the weight.
        :type fully_invested: bool, default True
        :param bounds: Lower and upper bounds of weights. If None, weights are unbounded, i.e., ``(0, None)`` means
            it only allows long positions.
        :type bounds: tuple, list-like

        .. note::
           With default parameters, this class produces a long-only fully-invested portfolio.
        """
        self.fully_invested = fully_invested
        self.bounds = bounds

    def optimise(self, ret: pd.DataFrame, *args, **kwargs) -> pd.Series:
        initial_weights = np.ones(len(ret.columns)) / len(ret.columns)
        sigma = ret.cov()
        bounds = [self.bounds] * len(ret.columns)
        const = create_const(self.fully_invested)

        result = minimize(
            total_risk_contribution_error,
            initial_weights,
            (sigma,),
            method='SLSQP',
            bounds=bounds,
            constraints=const,
            tol=1e-9,
        )
        weights = result['x']
        weights = pd.Series(weights, index=ret.columns)
        return weights


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

    def __init__(self, target_vol: float = 0.1, fully_invested: bool = False):
        """

        :param target_vol: Annualised target volatility of each instrument
        :param fully_invested: If True, weights are rescaled so that they add up to 100%.
        """
        self.target_vol = target_vol
        self.fully_invested = fully_invested

    def optimise(self, ret: pd.DataFrame, *args, **kwargs) -> pd.Series:
        ann_factor = infer_ann_factor(ret)
        vol = ret.std().mul(ann_factor ** 0.5)
        scaling = self.target_vol / vol

        instruments = ret.columns
        weights = pd.Series(scaling / len(instruments), index=instruments)

        weights = post_process(weights, self.fully_invested)

        return weights


def portfolio_variance(weights: List[float], sigma: np.array) -> float:
    """ Calculate portfolio variance from the given weights and covariance matrix

    :param weights: instrument weights
    :param sigma: covariance matrix of instruments
    :return: portfolio variance
    :rtype: float
    """
    weights = np.array(weights)
    return weights.dot(sigma).dot(weights)


def negative_sharpe_ratio(weights: List[float], mu: np.array, sigma: np.array) -> float:
    """ Calculate negative Sharpe ratio

    :param weights: instrument weights
    :param mu: expected return of instruments
    :param sigma: covariance matrix of instruments
    :return: negative Sharpe ratio
    :rtype: float
    """
    port_ret = mu.dot(np.array(weights))
    port_vol = portfolio_variance(weights, sigma) ** 0.5

    port_sharpe = port_ret / port_vol
    return -1 * port_sharpe


def diversification_factor(weights: List[float], sigma: np.array) -> float:
    """ Calculate the negative diversification factor which is the ratio between the weighted average of instrument volatility
    and resulting portfolio volatility

    :param weights: instrument weights
    :param sigma: covariance matrix of instruments
    :return: diversification factor
    :rtype: float
    """
    vol_vector = np.diag(sigma) ** 0.5
    weighted_average = vol_vector.dot(weights)
    port_vol = portfolio_variance(weights, sigma) ** 0.5

    return weighted_average / port_vol


def negative_diversification_factor(weights: List[float], sigma: np.array) -> float:
    """ Compute the negative diversification factor for minimisation """
    return -1 * diversification_factor(weights, sigma)


def total_risk_contribution_error(weights: List[float], sigma: np.array) -> float:
    """ Compute the sum of squared error between all total risk contributions """
    marginal = sigma.dot(weights)
    total = marginal * weights

    error = sum([((i - total[0]) ** 2) ** 0.5 for i in total])

    # multiply with some large number as `error` being the squared diff in total risk is usually a small number
    c = 1e5
    return error * c


def post_process(weights: pd.Series, fully_invested: bool) -> pd.Series:
    """ Post-process of weighting

    :param weights: raw weights
    :type weights: pd.Series
    :param fully_invested: If True, weights are rescaled so that they add up to 100%. By default the optimal weights
        are rescaled to add up to 100% as the Sharpe ratio is scale-invariant with respect to the weight.
    :type fully_invested: bool, default True
    :return: processed weights
    :rtype: pd.Series
    """

    if fully_invested:
        weights /= weights.sum()

    return weights


def create_const(fully_invested: bool = False) -> List:
    const = []
    if fully_invested:
        fun = lambda x: x.sum() - 1
        const.append({
            'type': 'eq',
            'fun': fun
        })

    return const
