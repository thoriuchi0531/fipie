from typing import Optional

import numpy as np
import pandas as pd

from tutti import tree
from tutti.cluster import ClusterAlgo, NoCluster
from tutti.weighting import Weighting


class Portfolio:
    """ A portfolio of instrument returns """

    def __init__(self, ret: pd.DataFrame):
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
        :param ret: portfolio returns
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
            result = result.loc[instruments, :]

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
        """ Compute the latest portfolio weights using the full return time-series.

        :param weighting: weighting scheme instance
        :param cluster: clustering algorithm instance
        :param instrument_only: If True only weights for instruments are shown and ones for intermediate are omitted
        :param final_weight: If True return the final weights for each instruments are returned.
        :return: weights for each node
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
        :param cluster: clustering algorithm instance
        :param instrument_only: If True only weights for instruments are shown and ones for intermediate are omitted
        :param final_weight: If True return the final weights for each instruments are returned.
        :param freq: frequency to update the portfolio weights
        :param lookback: the number of return samples (lookback horizon) to compute the portfolio weights
        :return: historical weights for each node
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
