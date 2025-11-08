
#Modulo principal de carga de modelos.
#Este archivo importa las funciones de los modulos especificos para cada modelo.


# Importar funciones de LSTM 1
from model_loders.lstm1_loader import predict_with_lstm1, get_lstm1_info

# Importar funciones de LSTM 2
from model_loders.lstm2_loader import predict_with_lstm2, get_lstm2_info

# Importar funciones de LSTM 3
from model_loders.lstm3_loader import predict_with_lstm3, get_lstm3_info

__all__ = [
    'predict_with_lstm1',
    'get_lstm1_info',
    'predict_with_lstm2',
    'get_lstm2_info',
    'predict_with_lstm3',
    'get_lstm3_info',
]
