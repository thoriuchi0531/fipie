from enum import Enum, unique
from typing import List, Optional

import numpy as np
import pandas as pd

from fipie.cluster import ClusterAlgo, NoCluster
from fipie.weighting import Weighting


@unique
class SpecialNode(Enum):
    ROOT = 'root'
    CLUSTER = 'cluster'


special_nodes = [i.value for i in SpecialNode]


class Node:
    """ Node instance in a tree. This can be the root, intermediate (cluster node) or leaf (bottom node) """
    INITIAL_WEIGHT = np.nan

    def __init__(self, node_id: str, parent=None):
        self.node_id = node_id
        self.parent = parent

        self._node_return = None
        self.local_weight = None
        self.init_weight()
        if self.node_id in special_nodes:
            self.is_leaf = False
        else:
            self.is_leaf = True

    def __repr__(self):
        return f'Node({self.node_id})'

    @property
    def level(self) -> int:
        """ The level of a node in the tree. Starting from 0 for the root. """
        if self.parent is None:
            # root
            return 0
        else:
            return self.parent.level + 1

    @property
    def weight(self) -> float:
        """ The final portfolio weight for a node.
        `weight` multiplied by its return is the contribution to the portfolio """
        if self.parent is None:
            return 1
        else:
            return self.parent.weight * self.local_weight

    def init_weight(self):
        """ Initialise the weight """
        self.local_weight = self.INITIAL_WEIGHT

    def is_weight_set(self) -> bool:
        """ Check if the weight is set already """
        return not np.isnan(self.local_weight)


class Tree:
    """ Tree class which has a collection of nodes and construct the tree structure depending on
    the applied clustering algorithm """

    def __init__(self):
        self.nodes = []  # type: List[Node]
        self._node_map = dict()
        self._cluster_count = 0

    def __repr__(self):
        return f'Tree'

    def add_node(self, node):
        """ Add a node to the tree """
        if node.node_id == SpecialNode.CLUSTER.value:
            # create a unique id in case intermediate
            node.node_id = f'{SpecialNode.CLUSTER.value}_{self._cluster_count}'
            self._cluster_count += 1

        self.nodes.append(node)
        self._node_map[node.node_id] = node

    def show(self):
        """ Visualise the tree structure with padding reflecting the node levels """
        for i in self.nodes:
            space = '    ' * i.level
            print(f'{space}{i}')

    def get_siblings(self, node: Node) -> List[Node]:
        """ Return a list of sibling nodes in the same cluster

        :param node: target node
        :return: list of sibling nodes
        """
        return [i for i in self.nodes if i.parent is node.parent]

    def get_children(self, node: Node) -> List[Node]:
        """ Return a list of nodes which are directly linked to the given node. Does not search recursively.

        :param node: target node
        :return: list of child nodes
        """
        return [i for i in self.nodes if i.parent is node]

    def assign_node_returns(self, ret: pd.DataFrame):
        """ Assign returns to the bottom nodes

        :param ret: dataframe containing instruments returns which are assigned to the individual nodes
        :return: None
        """
        for i in self.nodes:
            if i.node_id in ret.columns:
                i._node_return = ret[i.node_id]

    def init_weights(self):
        """ Initialise weights of all nodes """
        for i in self.nodes:
            i.init_weight()

    def node_return(self, node: Node, weighting: Weighting) -> pd.Series:
        """ Get the node's return. If the node is located at the bottom of tree, this just returns its pre-assigned
        return. Otherwise its return is computed recursively from bottom with children's local weights.

        :param node: target node
        :param weighting: If the `node` argument is not at the bottom of tree, this weighting method is applied to
        first determine the local weights for the children nodes. The target node's return is then computed as
        the weighted sum of children returns.
        :return: node's return series
        """
        if node._node_return is not None:
            # bottom leaf node
            return node._node_return.rename(node.node_id)
        else:
            # upper nodes
            nodes = self.get_children(node)
            self._set_local_weights(nodes, weighting)

            ret = pd.concat([self.node_return(i, weighting) * i.local_weight
                             for i in nodes], axis=1)
            return ret.sum(axis=1).rename(node.node_id)

    def _set_local_weights(self, nodes: List[Node], weighting: Weighting):
        """ Set local weights for the provided list of nodes.

        :param nodes: list of target nodes
        :param weighting: weighting instance to determine local weights
        :return: None
        """
        ret = pd.concat([self.node_return(i, weighting) for i in nodes], axis=1)
        weights = weighting.optimise(ret)
        for k, w in weights.items():
            self._node_map[k].local_weight = w

    def set_local_weights(self, weighting: Weighting):
        """ Set local weights for all nodes in the tree

        :param weighting: weighting instance to determine local weights
        :return: None
        """
        # make sure all weights are initialised in case this function gets called multiple times in a row
        self.init_weights()

        for i in self.nodes:
            if i.is_weight_set():
                continue

            siblings = self.get_siblings(i)
            self._set_local_weights(siblings, weighting)


def _create_tree_inner(data: pd.DataFrame,
                       cluster_algo: ClusterAlgo,
                       tree: Optional[Tree] = None,
                       parent: Optional[Node] = None) -> Tree:
    """ Private function to construct a tree instance """
    cluster_idx = cluster_algo.create_clusters(data)
    unique_ids = list(set(cluster_idx))
    max_clusters = cluster_algo.max_clusters

    if tree is None:
        tree = Tree()
        parent = Node(SpecialNode.ROOT.value)
        tree.add_node(parent)

    for i in unique_ids:
        sub_data = data.loc[:, cluster_idx == i]
        n_instr = sub_data.shape[1]

        if max_clusters is None:
            # no cluster. all linked to the parent
            for idx, j in enumerate(sub_data):
                tree.add_node(Node(j, parent=parent))
        elif n_instr > max_clusters:
            # `max_node_size` is defined and the number of instruments is greater than that
            node = Node(SpecialNode.CLUSTER.value, parent=parent)
            tree.add_node(node)
            _ = _create_tree_inner(sub_data, cluster_algo, tree=tree, parent=node)
        elif n_instr == 1:
            # omit the intermediate cluster node and link directly to the parent
            node = Node(sub_data.columns[0], parent=parent)
            tree.add_node(node)
        else:
            node = Node(SpecialNode.CLUSTER.value, parent=parent)
            tree.add_node(node)
            for idx, j in enumerate(sub_data):
                tree.add_node(Node(j, parent=node))

    return tree


def create_tree(data: pd.DataFrame, cluster_algo: ClusterAlgo = NoCluster()) -> Tree:
    """ Construct a tree with the given return time series and clustering algorithm
    By default, there is no cluster and all instruments are put into a single group.

    :param data: return series which are used to create nodes
    :param cluster_algo:
    :return: tree of nodes
    """
    data = cluster_algo.pre_process(data)
    tree = _create_tree_inner(data, cluster_algo)
    tree.assign_node_returns(data)
    return tree
