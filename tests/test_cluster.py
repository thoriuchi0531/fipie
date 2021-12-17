from fipie import NoCluster, CorrMatrixDistance
from fipie.data import load_example_data


def test_no_cluster():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    cluster = NoCluster()
    idx = cluster.create_clusters(ret)

    assert len(idx) == 7
    assert all(idx == 1)


def test_corr_matrix_distance():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    cluster = CorrMatrixDistance(max_clusters=3)
    idx = cluster.create_clusters(ret)
    idx_map = {k: v for k, v in zip(ret.columns, idx)}

    assert len(idx) == 7
    assert len(set(idx)) == 3
    assert idx_map['SPY'] == 1
    assert idx_map['SPY'] != idx_map['USO']
    assert idx_map['SPY'] != idx_map['TLT']
