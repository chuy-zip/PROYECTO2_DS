# modulo para poder cargar lsa metricas de los 3 distintos modelos
import json
from pathlib import Path
from typing import Dict, Any


def get_metrics_path(filename: str) -> Path:
    """
    Obtiene la ruta al archivo de metricas.

    Args:
        filename: Nombre del archivo JSON de metricas

    Returns:
        Path: Ruta completa al archivo
    """
    current_dir = Path(__file__).parent
    metrics_dir = current_dir / "model_metrics"
    return metrics_dir / filename


def load_lstm1_metrics() -> Dict[str, Any]:
    """
    Carga las metricas del modelo LSTM_1.

    Returns:
        Dict: Diccionario con las metricas del modelo incluyendo:
            - model_name: Nombre del modelo
            - model_type: Tipo de modelo
            - final_results: Accuracy, F1-macro, F1-weighted
            - class_metrics: metricas por clase (precision, recall, f1_score, support)
            - confusion_matrix: Matriz de confusion con labels
            - training_info: Informacion del dataset y clases

    """
    metrics_path = get_metrics_path("lstm1_metrics.json")

    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    return metrics


def load_lstm2_metrics() -> Dict[str, Any]:
    """
    Carga las metricas del modelo LSTM_2.

    Returns:
        Dict: Diccionario con las metricas del modelo incluyendo:
            - model_name: Nombre del modelo
            - model_type: Tipo de modelo
            - final_results: Test loss, Accuracy, F1-macro, F1-weighted
            - class_metrics: metricas por clase (precision, recall, f1_score, support)
            - confusion_matrix: Matriz de confusion con labels
            - training_info: Informacion del dataset y clases

    """
    metrics_path = get_metrics_path("lstm2_metrics.json")

    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    return metrics


def load_transformer1_metrics() -> Dict[str, Any]:
    """
    Carga las metricas del modelo Transformer_1.

    Returns:
        Dict: Diccionario con las metricas del modelo incluyendo:
            - model_name: Nombre del modelo
            - model_type: Tipo de modelo
            - final_results: Accuracy, Precision, Recall, F1-score
            - class_metrics: metricas por clase (precision, recall, f1_score, support)
            - confusion_matrix: Matriz de confusion con labels
            - training_info: Informacion del dataset y clases

    """
    metrics_path = get_metrics_path("transformer1_metrics.json")

    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    return metrics


def load_all_metrics() -> Dict[str, Dict[str, Any]]:
    """
    Carga las metricas de todos los modelos.

    Returns:
        Dict: Diccionario con las metricas de todos los modelos:
            - 'LSTM_1': metricas del modelo LSTM_1
            - 'LSTM_2': metricas del modelo LSTM_2
            - 'Transformer_1': metricas del modelo Transformer_1

    """
    return {
        'LSTM_1': load_lstm1_metrics(),
        'LSTM_2': load_lstm2_metrics(),
        'Transformer_1': load_transformer1_metrics()
    }