import numpy as np
import pytest
from scipy.optimize import minimize

from tutti import Portfolio, NoCluster, VolatilityParity, MeanVariance
from tutti.data import load_example_data
from tutti.weighting import negative_sharpe_ratio


def test_volatility_parity():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    portfolio = Portfolio(ret)
    weight = portfolio.weight_latest(
        VolatilityParity(target_vol=0.1),
        NoCluster(),
    )

    scaled = ret * weight
    scaled_vol = scaled.std() * (52 ** 0.5)

    # equal vol weighted
    assert pytest.approx(scaled_vol - 0.1 / 7) == 0


def test_mean_variance():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    portfolio = Portfolio(ret)
    weight = portfolio.weight_latest(
        MeanVariance(),
        NoCluster(),
    )

    initial_weights = np.ones(len(ret.columns)) / len(ret.columns)
    mu = ret.mean()
    sigma = ret.cov()
    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        (mu, sigma),
        method='SLSQP',
    )
    optimal = result['fun'] * -1

    # set `min_count` > 0 to make sure the first row gets NaNs
    portfolio_return = (ret * weight).sum(axis=1, min_count=1)
    portfolio_sharpe = portfolio_return.mean() / portfolio_return.std()  # daily sharpe

    assert optimal == pytest.approx(portfolio_sharpe)
