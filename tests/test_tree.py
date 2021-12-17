from fipie import NoCluster, EqualWeight
from fipie.data import load_example_data
from fipie.tree import Tree, create_tree


def test_create_tree():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    tree = create_tree(ret, NoCluster())

    assert len(tree.nodes) == ret.shape[1] + 1

    root = tree.nodes[0]
    node = tree.nodes[1]

    assert str(root) == 'Node(root)'
    assert str(node) == 'Node(SPY)'
    assert str(tree) == 'Tree'

    assert not root.is_leaf
    assert node.is_leaf
    assert root.level == 0
    assert node.level == 1


def test_tree_show():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    tree = create_tree(ret, NoCluster())
    tree.show()


def test_init_weight():
    price = load_example_data()
    ret = price.asfreq('w', method='pad').pct_change()

    tree = create_tree(ret, NoCluster())
    tree.set_local_weights(EqualWeight())

    node = tree.nodes[1]

    assert node.is_weight_set()

    tree.init_weights()

    assert not node.is_weight_set()
