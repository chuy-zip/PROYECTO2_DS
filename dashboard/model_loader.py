
#Modulo para cargar modelos entrenados y realizar predicciones.


import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


# ARQUITECTURA DEL MODELO LSTM 1

class AdvancedLSTMClassifier(nn.Module):
    """Arquitectura del modelo LSTM avanzado con atencion."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.5):
        super(AdvancedLSTMClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM bidireccional
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)

        # Mecanismo de atencion
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Transformacion de estados ocultos
        self.hidden_transform = nn.Linear(hidden_dim * 2 * num_layers, hidden_dim * 2)

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Mecanismo de atencion
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * lstm_out, dim=1)

        # Estados ocultos
        hidden_combined = hidden.permute(1, 0, 2).contiguous().view(hidden.size(1), -1)
        hidden_features = self.hidden_transform(hidden_combined)

        # Combinar caracteristicas
        combined_features = torch.cat([context_vector, hidden_features], dim=1)

        # Clasificacion
        output = self.classifier(combined_features)
        return output


# FUNCIONES DE PREPROCESAMIENTO

def tokenizer(text: str) -> List[str]:
    """
    Tokeniza el texto siguiendo el mismo proceso del entrenamiento.

    """
    text = str(text).lower()
    text = text.replace('<br />', ' ').replace('\n', ' ').replace('\t', ' ')
    tokens = text.split()
    return tokens


def tokens_to_indices(tokens: List[str], vocab: Dict[str, int], max_length: int = 200) -> torch.Tensor:
    """
    Convierte tokens a indices numericos usando el vocabulario.

    Args:
        tokens: Lista de tokens
        vocab: Diccionario del vocabulario
        max_length: Longitud maxima de la secuencia

    Returns:
        Tensor con los indices
    """
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    indices = indices[:max_length]

    # Padding si es necesario
    if len(indices) < max_length:
        indices = indices + [vocab['<pad>']] * (max_length - len(indices))

    return torch.tensor([indices], dtype=torch.long)


def preprocess_text(text: str, vocab: Dict[str, int], max_length: int = 200) -> torch.Tensor:
    """
    Preprocesa el texto completo para ser usado por el modelo.

    Args:
        text: Texto a preprocesar
        vocab: Diccionario del vocabulario
        max_length: Longitud maxima de la secuencia

    Returns:
        Tensor listo para el modelo
    """
    tokens = tokenizer(text)
    indices = tokens_to_indices(tokens, vocab, max_length)
    return indices


# CARGA Y PREDICCION DEL MODELO LSTM 1

# Cache global para el modelo LSTM 1
_lstm1_model = None
_lstm1_vocab = None
_lstm1_label_classes = None
_lstm1_device = None


def load_lstm1_model(force_reload: bool = False) -> Tuple[nn.Module, Dict[str, int], List[str], torch.device]:
    """
    Carga el modelo LSTM 1 desde el archivo guardado.
    Implementa caching para no recargar el modelo en cada prediccion.

    Args:
        force_reload: Si True, fuerza la recarga del modelo aunque este en cache

    Returns:
        Tupla con (modelo, vocabulario, clases, device)
    """
    global _lstm1_model, _lstm1_vocab, _lstm1_label_classes, _lstm1_device

    # Si ya esta cargado y no se fuerza recarga, devolver cache
    if not force_reload and _lstm1_model is not None:
        return _lstm1_model, _lstm1_vocab, _lstm1_label_classes, _lstm1_device

    # Ruta al modelo
    current_dir = Path(__file__).parent.parent
    model_path = current_dir / "snapshots" / "lstm" / "lstm_model_complete.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo en: {model_path}")

    # Determinar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extraer configuracion
    model_config = checkpoint['model_config']
    vocab = checkpoint['vocab']
    label_classes = checkpoint['label_encoder_classes']

    # Crear modelo con la arquitectura
    model = AdvancedLSTMClassifier(
        vocab_size=model_config['vocab_size'],
        embedding_dim=model_config['embedding_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        num_classes=model_config['num_classes'],
        dropout=model_config['dropout']
    )

    # Cargar pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Guardar en cache
    _lstm1_model = model
    _lstm1_vocab = vocab
    _lstm1_label_classes = label_classes
    _lstm1_device = device

    return model, vocab, label_classes, device


def predict_with_lstm1(text: str) -> Dict[str, any]:
    """
    Realiza una prediccion usando el modelo LSTM 1.

    Args:
        text: Texto del discurso a clasificar

    Returns:
        Diccionario con:
            - predicted_class: Clase predicha
            - probabilities: Probabilidades para cada clase
            - confidence: Confianza de la prediccion (probabilidad maxima)
            - all_classes: Lista de todas las clases
    """
    # Cargar modelo (usa cache si ya esta cargado)
    model, vocab, label_classes, device = load_lstm1_model()

    # Preprocesar texto
    input_tensor = preprocess_text(text, vocab, max_length=200)
    input_tensor = input_tensor.to(device)

    # Hacer prediccion
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()

    # Preparar resultados
    predicted_class = label_classes[predicted_idx]

    # Crear diccionario de probabilidades por clase
    probs_dict = {}
    for idx, class_name in enumerate(label_classes):
        probs_dict[class_name] = float(probabilities[0][idx].item())

    return {
        'predicted_class': predicted_class,
        'probabilities': probs_dict,
        'confidence': float(confidence),
        'all_classes': label_classes
    }


# FUNCION DE UTILIDAD PARA OBTENER INFO DEL MODELO

def get_lstm1_info() -> Dict[str, any]:
    """
    Obtiene informacion sobre el modelo LSTM 1 sin cargarlo completamente.

    Returns:
        Diccionario con informacion del modelo
    """
    current_dir = Path(__file__).parent.parent
    model_path = current_dir / "snapshots" / "lstm" / "lstm_model_complete.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo en: {model_path}")

    # Cargar solo metadata
    checkpoint = torch.load(model_path, map_location='cpu')

    return {
        'model_name': 'LSTM_1',
        'model_type': 'LSTM Avanzado con Atencion',
        'classes': checkpoint['label_encoder_classes'],
        'test_accuracy': checkpoint.get('test_accuracy', None),
        'f1_macro': checkpoint.get('f1_macro', None),
        'vocab_size': checkpoint['model_config']['vocab_size'],
        'embedding_dim': checkpoint['model_config']['embedding_dim'],
        'hidden_dim': checkpoint['model_config']['hidden_dim'],
        'num_layers': checkpoint['model_config']['num_layers']
    }
