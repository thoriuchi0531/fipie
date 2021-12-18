from fipie.portfolio import Portfolio
from fipie.weighting import (EqualWeight, VolatilityParity, MeanVariance, MinimumVariance, MaximumDiversification,
                             EqualRiskContribution)
from fipie.cluster import NoCluster, CorrMatrixDistance
from pkg_resources import get_distribution, DistributionNotFound

try:
    _version = get_distribution(__name__).version
except DistributionNotFound:
    from .version import version as _version

__version__ = _version
