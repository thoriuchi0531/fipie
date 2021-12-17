import abc

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from fipie.common import ReprMixin


class ClusterAlgo(ReprMixin, metaclass=abc.ABCMeta):

    def __init__(self, max_clusters):
        self.max_clusters = max_clusters

    def pre_process(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Pre-process the data time-series before feeding to the clustering algorithm.  """
        return data

    @abc.abstractmethod
    def create_clusters(self, data: pd.DataFrame) -> np.array:
        """Return a list of cluster indices for each instrument

        :param data: instrument returns which the clustering method is based on
        :return: array of cluster indices
        :rtype: np.array
        """


class NoCluster(ClusterAlgo):
    """ Create no cluster and put all instruments into one group
    which is the classical way of constructing a portfolio """

    def __init__(self):
        """ Instantiate the clustering algorithm. No extra parameter is needed. """
        super().__init__(max_clusters=None)

    def create_clusters(self, data: pd.DataFrame) -> np.array:
        return np.array([1] * data.shape[1])


class CorrMatrixDistance(ClusterAlgo):
    """ Create clusters based on the distance in correlation matrix. The ``max_node_size`` parameter controls
    the maximum number of instruments in a cluster.

    cf. https://github.com/TheLoneNut/CorrelationMatrixClustering/blob/master/CorrelationMatrixClustering.ipynb
    """

    def __init__(self, max_clusters: int):
        """ Instantiate the clustering algorithm

        :param max_clusters: maximum number of clusters (or instruments) one cluster can have.
        :type max_clusters: int
        """
        super().__init__(max_clusters=max_clusters)

    def pre_process(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Pre-process the data time-series before feeding to the clustering algorithm.
        Remove instruments whose return standard deviation is zero in which case the correlation matrix can not be
        computed.

        :param data: instrument returns which the clustering method is based on
        :type data: pd.DataFrame
        :return: pre-processed data
        :rtype: pd.DataFrame
        """
        std = data.std()
        return data.loc[:, std > 0]

    def create_clusters(self, data: pd.DataFrame) -> np.array:
        corr = data.corr().values
        distance = sch.distance.pdist(corr)
        linkage = sch.linkage(distance, method='complete')
        # idx starting from 1
        cluster_idx = sch.fcluster(linkage, self.max_clusters, criterion='maxclust')
        return cluster_idx
