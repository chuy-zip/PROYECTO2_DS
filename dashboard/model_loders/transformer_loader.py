#Modulo para cargar y ejecutar predicciones con el modelo Transformer.


from typing import Dict
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from pathlib import Path


class EssayGroupModel2(nn.Module):
    """
    Model that processes full essays and predicts effectiveness for each discourse.

    How it works:
    1. Takes ONE full essay as input
    2. Passes it through the backbone (DeBERTa/RoBERTa)
    3. Pools embeddings between [START] and [END] tokens for each discourse
    4. Predicts effectiveness for each discourse independently
    """

    def __init__(
        self,
        model_name='microsoft/deberta-base',
        num_labels=3,  # Inadequate, Adequate, Effective
        dropout=0.1
    ):
        super().__init__()

        # Load pretrained backbone
        print(f"Loading pretrained model: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = self.backbone.config
        self.hidden_size = self.config.hidden_size

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Final classification layer
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        print(f"Model initialized with hidden_size={self.hidden_size}")

    def resize_token_embeddings(self, new_num_tokens):
        """
        Call this after adding special tokens to the tokenizer.
        This resizes the model's embedding layer to accommodate new tokens.
        """
        self.backbone.resize_token_embeddings(new_num_tokens)
        print(f"Resized token embeddings to {new_num_tokens}")

    def find_start_end_positions(self, input_ids, start_token_id, end_token_id):
        """
        Find all [START] and [END] token positions in the input.

        Args:
            input_ids: [seq_len] tensor of token IDs
            start_token_id: ID of [START] token
            end_token_id: ID of [END] token

        Returns:
            start_positions: List of start token positions
            end_positions: List of end token positions
        """
        start_positions = []
        end_positions = []

        for i, token_id in enumerate(input_ids):
            if token_id == start_token_id:
                start_positions.append(i)
            elif token_id == end_token_id:
                end_positions.append(i)

        return start_positions, end_positions

    def pool_discourse_embeddings(self, sequence_output, start_positions, end_positions):
        """
        Pool embeddings between [START] and [END] tokens for each discourse.

        Args:
            sequence_output: [seq_len, hidden_size] - output from backbone
            start_positions: List of start token positions
            end_positions: List of end token positions

        Returns:
            pooled_embeddings: [num_discourses, hidden_size]
        """
        num_discourses = len(start_positions)
        pooled_embeddings = []

        for i in range(num_discourses):
            start_idx = start_positions[i]
            end_idx = end_positions[i]

            # Extract tokens between START and END (exclusive of the markers)
            # discourse_embeddings shape: [num_tokens_in_discourse, hidden_size]
            discourse_embeddings = sequence_output[start_idx + 1:end_idx]

            if len(discourse_embeddings) == 0:
                # If empty (shouldn't happen), use the START token
                pooled = sequence_output[start_idx]
            else:
                # Mean pooling
                pooled = discourse_embeddings.mean(dim=0)

            pooled_embeddings.append(pooled)

        # Stack into tensor: [num_discourses, hidden_size]
        pooled_embeddings = torch.stack(pooled_embeddings, dim=0)

        return pooled_embeddings

    def forward(
        self,
        input_ids,
        attention_mask,
        start_token_id,
        end_token_id,
        labels=None,
        loss_fct=None
    ):
        """
        Forward pass.

        Args:
            input_ids: [batch_size=1, seq_len] - tokenized essay
            attention_mask: [batch_size=1, seq_len] - attention mask
            start_token_id: int - ID of [START] token
            end_token_id: int - ID of [END] token
            labels: [num_discourses] - effectiveness labels (optional, for training)
            loss_fct: función de pérdida opcional (ej. ponderada)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = sequence_output.squeeze(0)
        input_ids_1d = input_ids.squeeze(0)

        start_positions, end_positions = self.find_start_end_positions(
            input_ids_1d, start_token_id, end_token_id
        )

        pooled_embeddings = self.pool_discourse_embeddings(
            sequence_output, start_positions, end_positions
        )

        pooled_embeddings = self.dropout(pooled_embeddings)
        logits = self.classifier(pooled_embeddings)
        probabilities = torch.softmax(logits, dim=-1)

        # Usa el loss_fct externo si se pasa
        loss = None
        if labels is not None:
            if loss_fct is not None:
                loss = loss_fct(logits, labels)
            else:
                default_loss = nn.CrossEntropyLoss()
                loss = default_loss(logits, labels)

        return {
            'loss': loss,
            'logits': logits,
            'probabilities': probabilities,
            'num_discourses': len(start_positions)
        }


_transformer_model = None
_transformer_tokenizer = None
_transformer_device = None

def load_transformer(force_reload: bool = False):

    global _transformer_model, _transformer_tokenizer, _transformer_device

    if not force_reload and _transformer_model is not None:
        return _transformer_model, _transformer_device, _transformer_tokenizer

    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base', use_fast=False)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[START]', '[END]']})

    current_dir = Path(__file__).parent.parent.parent
    model_path = current_dir / "snapshots" / "transformers" / "essay_model2_full.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(model_path, map_location=device, weights_only=False)

    _transformer_model = model
    _transformer_tokenizer = tokenizer
    _transformer_device = device

    return model, device, tokenizer


def preprocess_text(text: str):
    """
    Preprocesa el texto completo para ser usado por el modelo.

    Args:
        text: Texto a preprocesar
        vocab: Diccionario del vocabulario
        max_length: Longitud maxima de la secuencia

    Returns:
        Tupla de (tensor de indices, tensor de longitudes)
    """
    discourse_parts = []
    discourse = [line for line in text.strip().split('\n') if line.strip() != '']
    for i in discourse:
        discourse_parts.append(f"[START] {i} [END]")

    return "".join(discourse_parts)

    
    


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
    preprocessed_text = preprocess_text(text)

    model, device, tokenizer = load_transformer()

    encoded = tokenizer(
        preprocessed_text,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding='max_length'
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_token_id=tokenizer.convert_tokens_to_ids('[START]'),
            end_token_id=tokenizer.convert_tokens_to_ids('[END]')
        )

    probs = outputs["probabilities"]
    preds = torch.argmax(probs, dim=-1)

    label_map = {0: "Inadequate", 1: "Adequate", 2: "Effective"}

    print("\nPredicciones por discurso:")
    results = []
    for i in range(preds.shape[0]):
        pred_label = label_map[preds[i].item()]
        confidence = probs[i, preds[i]].item()
        probs_dict = {label_map[j]: float(probs[i, j]) for j in range(probs.shape[1])}
        results.append({
            "discourse": i + 1,
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": probs_dict
        })

    return {
        'results': results
    }
    # raise NotImplementedError("El modelo Transformer aun no esta implementado")


def get_transformer_info() -> Dict[str, any]:
    """
    Obtiene informacion sobre el modelo Transformer.

    Returns:
        Diccionario con informacion del modelo

    Raises:
        NotImplementedError: Este modelo aun no esta implementado
    """
    raise {
        'a': "a",
    }
