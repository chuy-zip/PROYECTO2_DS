"""
Modulo de cargadores de modelos.
Cada modelo tiene su propio archivo loader.
"""

from .lstm1_loader import predict_with_lstm1, get_lstm1_info
# from .lstm2_loader import predict_with_lstm2, get_lstm2_info  # TODO: Implementar
# from .transformer_loader import predict_with_transformer, get_transformer_info  # TODO: Implementar

__all__ = [
    'predict_with_lstm1',
    'get_lstm1_info',
    # 'predict_with_lstm2',
    # 'get_lstm2_info',
    # 'predict_with_transformer',
    # 'get_transformer_info',
]
