"""
Multivariate Hawkes for Modelling Order Flow Imbalance 

An implementation of multivariate Hawkes processes for cryptocurrency
market making, based on the paper "Modelling Order Flow Asymmetries with Hawkes Processes".
"""

__version__ = "1.0.0"
__author__ = "Israel Bergenstein"

from .univariate_hawkes import UnivariateHawkesCalibrator, UnivariateHawkesMarketMaker
from .multivariate_hawkes import MultivariateHawkesCalibrator, HawkesMarketMaker
from .avellaneda_stoikov import AvellanedaStoikovMM
from .calibration import EMCalibrator
from .strategies import BacktestEngine
from .metrics import PerformanceMetrics
from .visualization import Visualizer

__all__ = [
    'UnivariateHawkesCalibrator',
    'UnivariateHawkesMarketMaker',
    'MultivariateHawkesCalibrator',
    'HawkesMarketMaker',
    'AvellanedaStoikovMM',
    'EMCalibrator',
    'BacktestEngine',
    'PerformanceMetrics',
    'Visualizer'
]
