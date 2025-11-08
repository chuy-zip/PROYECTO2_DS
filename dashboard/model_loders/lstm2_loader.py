#Modulo para cargar y ejecutar predicciones con el modelo LSTM 2 (Bidireccional).

from typing import Dict


def predict_with_lstm2(text: str) -> Dict[str, any]:
    """
    Realiza una prediccion usando el modelo LSTM 2.

    Args:
        text: Texto del discurso a clasificar

    Returns:
        Diccionario con los resultados de la prediccion

    Raises:
        NotImplementedError: Este modelo aun no esta implementado
    """
    raise NotImplementedError("El modelo LSTM 2 aun no esta implementado")


def get_lstm2_info() -> Dict[str, any]:
    """
    Obtiene informacion sobre el modelo LSTM 2.

    Returns:
        Diccionario con informacion del modelo

    Raises:
        NotImplementedError: Este modelo aun no esta implementado
    """
    raise NotImplementedError("El modelo LSTM 2 aun no esta implementado")
