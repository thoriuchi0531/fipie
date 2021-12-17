import numpy as np
import pandas as pd
import pytest

from fipie import Portfolio, NoCluster, CorrMatrixDistance, EqualWeight
from fipie.data import load_example_data
from fipie.tree import Tree


def test_portfolio_creation():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    portfolio = Portfolio(ret)

    assert isinstance(portfolio.ret, pd.DataFrame)
    assert str(portfolio) == 'Portfolio(7 assets)'


def test_portfolio_creation_single_asset():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()['SPY']

    portfolio = Portfolio(ret)

    assert isinstance(portfolio.ret, pd.DataFrame)
    assert str(portfolio) == 'Portfolio(1 asset)'


@pytest.mark.xfail
def test_portfolio_creation_not_supported():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    _ = Portfolio(ret.values)


def test_create_tree():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    portfolio = Portfolio(ret)
    tree = portfolio.create_tree(NoCluster())
    assert isinstance(tree, Tree)

    tree = portfolio.create_tree(NoCluster(), ret.tail(10))
    assert isinstance(tree, Tree)


def test_weight_latest():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    portfolio = Portfolio(ret)
    weight = portfolio.weight_latest(
        EqualWeight(),
        NoCluster(),
    )

    assert isinstance(weight, pd.Series)
    assert len(weight) == ret.shape[1]  # number of instruments
    assert weight.name == 'weight'
    assert weight.iat[0] == pytest.approx(1 / ret.shape[1])
    assert pytest.approx(weight - weight.iat[0]) == 0  # all instruments have the same weight


def test_local_weight():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    portfolio = Portfolio(ret)
    weight = portfolio.weight_latest(
        EqualWeight(),
        CorrMatrixDistance(max_clusters=3),
        final_weight=False
    )
    """
    Node(root)
    Node(cluster_0)
        Node(cluster_1)
            Node(SPY)
            Node(IWM)
            Node(MDY)
        Node(QQQ)
    Node(cluster_2)
        Node(GLD)
        Node(USO)
    Node(TLT)
    """
    assert isinstance(weight, pd.Series)
    assert len(weight) == ret.shape[1]  # number of instruments + root
    assert weight.name == 'local_weight'
    assert weight['GLD'] == pytest.approx(0.5)
    assert weight['TLT'] == pytest.approx(1 / 3)


def test_weight_historical():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    portfolio = Portfolio(ret)
    weight = portfolio.weight_historical(
        EqualWeight(),
        CorrMatrixDistance(max_clusters=3),
        freq='m',
        lookback=52 * 2,  # 2 years
    )

    rebalance_dates = ret.asfreq('m', method='pad').index
    assert isinstance(weight, pd.DataFrame)
    assert len(weight) == len(rebalance_dates)
    assert weight.index[0] == rebalance_dates[0]

    portfolio2 = Portfolio(ret.tail(52 * 2))
    weight2 = portfolio2.weight_latest(
        EqualWeight(),
        CorrMatrixDistance(max_clusters=3)
    )

    assert pytest.approx(weight2 - weight.iloc[-1]) == 0


def test_weight_latest_redundant_instrument():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()
    ret['SPY'] = 0

    portfolio = Portfolio(ret)
    weight = portfolio.weight_latest(
        EqualWeight(),
        CorrMatrixDistance(max_clusters=3),
    )
    assert np.isnan(weight['SPY'])


def test_weight_historical_redundant_instrument():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()
    ret.loc[:'2020', 'SPY'] = 0.0  # instrument not contributing until some time

    portfolio = Portfolio(ret)
    weight = portfolio.weight_historical(
        EqualWeight(),
        CorrMatrixDistance(max_clusters=3),
    )
    assert np.isnan(weight.loc['2020-12-31', 'SPY'])
    assert not np.isnan(weight.loc['2021-01-31', 'SPY'])  # start having weights once SPY has some volatility
