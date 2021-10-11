import abc

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from tutti.common import ReprMixin


class ClusterAlgo(ReprMixin, metaclass=abc.ABCMeta):

    def __init__(self, max_clusters):
        self.max_clusters = max_clusters

    @abc.abstractmethod
    def create_clusters(self, data: pd.DataFrame) -> np.array:
        """Return a list of cluster indices for each instrument

        :param data: instrument returns which the clustering method is based on
        :return: array of cluster indices
        """
        pass


class NoCluster(ClusterAlgo):
    """ Create no cluster and put all instruments into one group
    which is the classical way of constructing a portfolio """

    def __init__(self):
        super().__init__(max_clusters=None)

    def create_clusters(self, data: pd.DataFrame) -> np.array:
        return np.array([1] * data.shape[1])


class CorrMatrixDistance(ClusterAlgo):
    """ Create clusters based on the distance in correlation matrix. The `max_node_size` parameter controls
    the maximum number of instruments in a cluster.

    cf. https://github.com/TheLoneNut/CorrelationMatrixClustering/blob/master/CorrelationMatrixClustering.ipynb
    """

    def create_clusters(self, data: pd.DataFrame) -> np.array:
        corr = data.corr().values
        distance = sch.distance.pdist(corr)
        linkage = sch.linkage(distance, method='complete')
        # idx starting from 1
        cluster_idx = sch.fcluster(linkage, self.max_clusters, criterion='maxclust')
        return cluster_idx
