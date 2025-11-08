#Modulo para cargar y ejecutar predicciones con el modelo Transformer.


from typing import Dict


def predict_with_transformer(text: str) -> Dict[str, any]:
    """
    Realiza una prediccion usando el modelo Transformer.

    Args:
        text: Texto del discurso a clasificar

    Returns:
        Diccionario con los resultados de la prediccion

    Raises:
        NotImplementedError: Este modelo aun no esta implementado
    """
    raise NotImplementedError("El modelo Transformer aun no esta implementado")


def get_transformer_info() -> Dict[str, any]:
    """
    Obtiene informacion sobre el modelo Transformer.

    Returns:
        Diccionario con informacion del modelo

    Raises:
        NotImplementedError: Este modelo aun no esta implementado
    """
    raise NotImplementedError("El modelo Transformer aun no esta implementado")
