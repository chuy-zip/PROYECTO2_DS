"""
Modulo de cargadores de modelos.
Cada modelo tiene su propio archivo loader.
"""

from .lstm1_loader import predict_with_lstm1, get_lstm1_info
from .lstm2_loader import predict_with_lstm2, get_lstm2_info
from .lstm3_loader import predict_with_lstm3, get_lstm3_info

__all__ = [
    'predict_with_lstm1',
    'get_lstm1_info',
    'predict_with_lstm2',
    'get_lstm2_info',
    'predict_with_lstm3',
    'get_lstm3_info',
]
