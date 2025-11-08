
#Modulo para cargar y ejecutar predicciones con el modelo LSTM 2 (Bidireccional con Atencion).


import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ARQUITECTURA DEL MODELO LSTM 2

class AttentionPooling(nn.Module):
    """Mecanismo de atencion para pooling sobre salidas LSTM."""

    def __init__(self, hidden_dim, attn_dim=128):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h, mask):
        # h: (B, L, H)
        # mask: (B, L) True para tokens validos
        energy = torch.tanh(self.linear(h))
        scores = self.v(energy).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-1e9'))
        alpha = torch.softmax(scores, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return context, alpha


class LSTMWithAttention(nn.Module):
    """LSTM Bidireccional con Atencion."""

    def __init__(self, vocab_size, embed_dim=256, hidden_size=320,
                 num_layers=2, num_classes=3, dropout=0.35,
                 bidirectional=True, padding_idx=0, use_aux=False, aux_dim=0, attn_dim=128):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.dropout_in = nn.Dropout(dropout*0.5)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout if num_layers>1 else 0.0)
        factor = 2 if bidirectional else 1
        self.attention = AttentionPooling(hidden_dim=hidden_size*factor, attn_dim=attn_dim)
        self.use_aux = use_aux
        fc_input = hidden_size*factor + (aux_dim if use_aux else 0)
        self.fc = nn.Sequential(
            nn.Linear(fc_input, fc_input//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input//2, num_classes)
        )

    def forward(self, x, lengths, aux=None):
        # x: (B, L)
        mask = (x != self.padding_idx)  # (B, L) boolean
        emb = self.embedding(x)
        emb = self.dropout_in(emb)

        # pack
        lengths_cpu = lengths.cpu()
        packed = pack_padded_sequence(emb, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out_unpacked, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))

        # Attention pooling
        context, attn_weights = self.attention(out_unpacked, mask)

        if self.use_aux and aux is not None:
            context = torch.cat([context, aux], dim=1)

        logits = self.fc(context)
        return logits, attn_weights


# FUNCIONES DE PREPROCESAMIENTO

def tokenizer(text: str) -> List[str]:
    """
    Tokeniza el texto siguiendo el mismo proceso del entrenamiento.

    Args:
        text: Texto a tokenizar

    Returns:
        Lista de tokens
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


def preprocess_text(text: str, vocab: Dict[str, int], max_length: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocesa el texto completo para ser usado por el modelo.

    Args:
        text: Texto a preprocesar
        vocab: Diccionario del vocabulario
        max_length: Longitud maxima de la secuencia

    Returns:
        Tupla de (tensor de indices, tensor de longitudes)
    """
    tokens = tokenizer(text)
    indices = tokens_to_indices(tokens, vocab, max_length)

    # Calcular longitud real (sin padding)
    pad_idx = vocab.get('<pad>', 0)
    length = (indices[0] != pad_idx).sum().item()
    lengths = torch.tensor([length], dtype=torch.long)

    return indices, lengths


# CARGA Y PREDICCION DEL MODELO LSTM 2

# Cache global para el modelo LSTM 2
_lstm2_model = None
_lstm2_vocab = None
_lstm2_label_classes = None
_lstm2_device = None


def load_lstm2_model(force_reload: bool = False) -> Tuple[nn.Module, Dict[str, int], List[str], torch.device]:
    """
    Carga el modelo LSTM 2 desde el archivo guardado.
    Implementa caching para no recargar el modelo en cada prediccion.

    Args:
        force_reload: Si True, fuerza la recarga del modelo aunque este en cache

    Returns:
        Tupla con (modelo, vocabulario, clases, device)
    """
    global _lstm2_model, _lstm2_vocab, _lstm2_label_classes, _lstm2_device

    # Si ya esta cargado y no se fuerza recarga, devolver cache
    if not force_reload and _lstm2_model is not None:
        return _lstm2_model, _lstm2_vocab, _lstm2_label_classes, _lstm2_device

    # Ruta al modelo
    current_dir = Path(__file__).parent.parent.parent
    model_path = current_dir / "snapshots" / "lstm" / "lstm_attn_best.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo en: {model_path}")

    # Determinar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extraer configuracion
    vocab = checkpoint['vocab']
    label_classes = checkpoint['label_classes']

    # Parametros del modelo (basados en el notebook)
    vocab_size = len(vocab)
    embed_dim = 256
    hidden_size = 320
    num_layers = 2
    num_classes = len(label_classes)
    dropout = 0.35
    bidirectional = True
    padding_idx = vocab.get('<pad>', 0)
    attn_dim = 128

    # Crear modelo con la arquitectura
    model = LSTMWithAttention(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
        padding_idx=padding_idx,
        use_aux=False,
        aux_dim=0,
        attn_dim=attn_dim
    )

    # Cargar pesos
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()

    # Guardar en cache
    _lstm2_model = model
    _lstm2_vocab = vocab
    _lstm2_label_classes = label_classes
    _lstm2_device = device

    return model, vocab, label_classes, device


def predict_with_lstm2(text: str) -> Dict[str, any]:
    """
    Realiza una prediccion usando el modelo LSTM 2.

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
    model, vocab, label_classes, device = load_lstm2_model()

    # Preprocesar texto
    input_tensor, lengths = preprocess_text(text, vocab, max_length=200)
    input_tensor = input_tensor.to(device)
    lengths = lengths.to(device)

    # Hacer prediccion
    with torch.no_grad():
        outputs, attn_weights = model(input_tensor, lengths, aux=None)
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


def get_lstm2_info() -> Dict[str, any]:
    """
    Obtiene informacion sobre el modelo LSTM 2 sin cargarlo completamente.

    Returns:
        Diccionario con informacion del modelo
    """
    current_dir = Path(__file__).parent.parent.parent
    model_path = current_dir / "snapshots" / "lstm" / "lstm_attn_best.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo en: {model_path}")

    # Cargar solo metadata
    checkpoint = torch.load(model_path, map_location='cpu')

    return {
        'model_name': 'LSTM_2',
        'model_type': 'LSTM Bidireccional con Atencion',
        'classes': checkpoint['label_classes'],
        'vocab_size': len(checkpoint['vocab']),
        'embedding_dim': 256,
        'hidden_dim': 320,
        'num_layers': 2,
        'bidirectional': True,
        'attention': True
    }
