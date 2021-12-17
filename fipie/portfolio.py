from typing import Optional

import numpy as np
import pandas as pd

from fipie import tree
from fipie.cluster import ClusterAlgo, NoCluster
from fipie.weighting import Weighting


class Portfolio:
    """ A portfolio of instrument returns """

    def __init__(self, ret: pd.DataFrame):
        """ Create a ``Portfolio`` instance

        :param ret: time-series of instrument returns
        :type ret: pd.DataFrame

        .. note::
            ``ret`` is frequency agnostic -- i.e., it can be daily, weekly or any other frequency as long as
            ``fipie.date.infer_ts_frequency`` can infer its frequency.
        """
        ret = self._preprocess_returns(ret)
        self.ret = ret

    def __repr__(self):
        n_asset = self.ret.shape[1]
        if n_asset == 1:
            return f'Portfolio({n_asset} asset)'
        else:
            return f'Portfolio({n_asset} assets)'

    def _preprocess_returns(self, ret) -> pd.DataFrame:
        if isinstance(ret, pd.DataFrame):
            # No need to prerocess
            return ret
        elif isinstance(ret, pd.Series):
            return ret.to_frame()
        else:
            raise ValueError(f'Unsupported data type for returns. Got {ret}')

    def create_tree(self,
                    cluster: ClusterAlgo,
                    ret: Optional[pd.DataFrame] = None) -> tree.Tree:
        """ Create a tree out of the return data frame

        :param cluster: clustering algorithm instance
        :type cluster: ClusterAlgo
        :param ret: portfolio returns to use to create a tree. If not provided, use the returns provided upon
            instantiation. If provided, this parameter will be used to create a tree instead.
        :type ret: pd.DataFrame, optional
        :return: ``Tree`` instance which groups instruments into clusters
        """
        if ret is None:
            ret = self.ret
        return tree.create_tree(ret, cluster)

    def _calculate_weight(self,
                          ret: pd.DataFrame,
                          weighting: Weighting,
                          cluster: ClusterAlgo,
                          instrument_only: bool = True,
                          final_weight: bool = True) -> pd.Series:
        """ An inner function to compute the latest portfolio weights given the return, weighting scheme and clustering
        algorithm.

        :param ret: portfolio returns
        :param weighting: weighting scheme instance
        :param cluster: clustering algorithm instance
        :param instrument_only: If True only weights for instruments are shown and ones for intermediate are omitted
        :param final_weight: If True return the final weights for each instruments are returned.
        :return: weights for each node
        """
        tree = self.create_tree(cluster, ret)
        tree.set_local_weights(weighting)
        result = [(i.node_id, i.local_weight, i.weight) for i in tree.nodes]
        result = pd.DataFrame(result, columns=['node_id', 'local_weight', 'weight'])
        result = result.set_index('node_id')

        if instrument_only:
            # only select rows that are in the original return time-series
            instruments = ret.columns.tolist()
            result = result.reindex(index=instruments)

        if final_weight:
            result = result['weight']
        else:
            result = result['local_weight']

        return result

    def weight_latest(self,
                      weighting: Weighting,
                      cluster: ClusterAlgo = NoCluster(),
                      instrument_only: bool = True,
                      final_weight: bool = True) -> pd.Series:
        r""" Compute the latest portfolio weights using the full return time-series.

        :param weighting: weighting scheme instance
        :type weighting: Weighting
        :param cluster: clustering algorithm instance
        :type cluster: ClusterAlgo
        :param instrument_only: If True only weights for instruments are shown and ones for intermediate are omitted
        :type instrument_only: bool, default True
        :param final_weight: If True return the final weights for each instruments are returned. The portfolio return
            :math:`r` can then be calculated as follows:

            .. math::
                r = \sum_i w_i \cdot r_i

            where :math:`i` is the index for each instrument, :math:`w_i` is the final weight for instrument :math:`i`,
            and :math:`r_i` is the return for instrument :math:`i`.

        :type final_weight: bool, default True
        :return: weights for each node
        :rtype: pd.Series
        """
        result = self._calculate_weight(self.ret, weighting, cluster,
                                        instrument_only=instrument_only,
                                        final_weight=final_weight)

        return result

    def weight_historical(self,
                          weighting: Weighting,
                          cluster: ClusterAlgo = NoCluster(),
                          instrument_only: bool = True,
                          final_weight: bool = True,
                          freq: str = 'm',
                          lookback: int = 52 * 2) -> pd.DataFrame:
        """ Compute the historical portfolio weights by applying the calculation on a rolling basis

        :param weighting: weighting scheme instance
        :type weighting: Weighting
        :param cluster: clustering algorithm instance
        :type cluster: ClusterAlgo
        :param instrument_only: If True only weights for instruments are shown and ones for intermediate are omitted
        :type instrument_only: bool, default True
        :param final_weight: If True return the final weights for each instruments are returned.
        :type final_weight: bool, default True
        :param freq: frequency to update the portfolio weights.
        :type freq: str, default 'm'
        :param lookback: the number of return samples (lookback horizon) to compute the portfolio weights
        :type lookback: int, default 52 * 2 (2 years with weekly observations)
        :return: historical weights for each node
        :rtype: pd.DataFrame
        """
        # rebalance dates
        dates = self.ret.asfreq(freq, method='pad').index

        result = []
        for i in dates:
            ret = self.ret.loc[:i].tail(lookback)

            if len(ret) == lookback:
                weight = self._calculate_weight(ret, weighting, cluster,
                                                instrument_only=instrument_only,
                                                final_weight=final_weight)
                weight = weight.to_frame(i).T
            else:
                weight = pd.Series(np.nan, index=ret.columns).to_frame(i).T
            result.append(weight)

        result = pd.concat(result)
        return result
