import numpy as np
import pytest
from scipy.optimize import minimize

from tutti import (Portfolio, NoCluster, VolatilityParity, MeanVariance, MinimumVariance, MaximumDiversification,
                   EqualRiskContribution)
from tutti.data import load_example_data
from tutti.weighting import negative_sharpe_ratio


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
        bounds=[(0, None) for _ in range(len(ret.columns))],
    )
    optimal = result['fun'] * -1

    # set `min_count` > 0 to make sure the first row gets NaNs
    portfolio_return = (ret * weight).sum(axis=1, min_count=1)
    portfolio_sharpe = portfolio_return.mean() / portfolio_return.std()  # daily sharpe

    assert optimal == pytest.approx(portfolio_sharpe)


def test_mean_variance_fully_invested():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    portfolio = Portfolio(ret)
    weight = portfolio.weight_latest(
        MeanVariance(fully_invested=True),
        NoCluster(),
    )
    assert pytest.approx(weight.sum()) == 1


def test_mean_variance_long_only():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    portfolio = Portfolio(ret)
    weight = portfolio.weight_latest(
        MeanVariance(fully_invested=True, bounds=(0, None)),
        NoCluster(),
    )

    assert pytest.approx(weight.min()) == 0
    assert pytest.approx(weight.sum()) == 1


def test_minimum_variance():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    portfolio = Portfolio(ret)
    weight = portfolio.weight_latest(
        MinimumVariance(),
        NoCluster(),
    )
    assert pytest.approx(weight.sum()) == 1

    weight = portfolio.weight_latest(
        MinimumVariance(bounds=(0, None)),
        NoCluster(),
    )
    assert pytest.approx(weight.min()) == 0
    assert pytest.approx(weight.sum()) == 1


def test_max_diversification():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    portfolio = Portfolio(ret)
    weight = portfolio.weight_latest(
        MaximumDiversification(),
        NoCluster(),
    )
    assert pytest.approx(weight.sum()) == 1


def test_erc():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    portfolio = Portfolio(ret)
    weight = portfolio.weight_latest(
        EqualRiskContribution(),
        NoCluster(),
    )
    assert pytest.approx(weight.sum()) == 1


def test_volatility_parity():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    portfolio = Portfolio(ret)
    weight = portfolio.weight_latest(
        VolatilityParity(target_vol=0.1, fully_invested=False),
        NoCluster(),
    )

    scaled = ret * weight
    scaled_vol = scaled.std() * (52 ** 0.5)

    # equal vol weighted
    assert pytest.approx(scaled_vol - 0.1 / 7) == 0

    weight = portfolio.weight_latest(
        VolatilityParity(target_vol=0.1, fully_invested=True),
        NoCluster(),
    )
    assert pytest.approx(weight.sum()) == 1
