# PROYECTO2_DS

Proyecto de Data Science - 2025

Este es un proyecto para calificar ensayos en efectivo, adecuado e inadecuado, utilizando modelos LSTM y Transformers.

# Necesary files
Los modelos que entrenamos se encuentran en la carpeta `snapshots`. Dentro de esta carpeta están otras 2 carpetas:
* `lstm`: Los modelos LSTM realizados, que en total fueron 3
* `transformers`: El modelo de transformer implementado

**Nota:** Es muy importante que todos los modelos se encuentren dentro de las carpetas para que la pestaña de predictor funcione dentro del dashboard. Hay 2 modelos que no están en el repositorio debido a que son muy pesados.

## Descarga de modelos desde Google Drive

Algunos modelos no están incluidos en este repositorio de GitHub debido a su tamaño. Estos hay que descargarlos manualmente desde Google Drive y colocarlos en sus respectivas carpetas:

### 1. Modelo LSTM Completo

**Archivo:** `lstm_model_complete.pth`

**Link de descarga:** [Carpeta de Drive con modelo LSTM](https://drive.google.com/drive/folders/1Ub-0wxDa-YRzaYbyoeQ81G7Xmm1lnSk4?usp=sharing)

**Ubicación destino:** `snapshots/lstm/lstm_model_complete.pth`

### 2. Modelo Transformer

**Archivo:** `essay_model2_full.pt`

**Link de descarga:** [Modelo Transformer en Drive](https://drive.google.com/file/d/1gzYaapiWFrosqQEeM-gSv4hTTCFsDpma/view?usp=sharing)

**Ubicación destino:** `snapshots/transformers/essay_model2_full.pt`

## Estructura de carpetas esperada

Una vez descargados los modelos, tu carpeta `snapshots` debe verse así:

```
snapshots/
├── lstm/
│   ├── lstm_model_complete.pth      # DESCARGAR DE DRIVE
│   ├── lstm_model_v1.pth             # (si existe en repo)
│   └── lstm_model_v2.pth             # (si existe en repo)
└── transformers/
    └── essay_model2_full.pt          # DESCARGAR DE DRIVE
```

**⚠️ IMPORTANTE:** Sin estos modelos, la funcionalidad de predicción en el dashboard no funcionará correctamente. 

# How to run

### Create virtual environment

```cmd
python -m venv .venv
```

### Activate it

### On Windows 

```cmd
.venv\Scripts\activate
```

### On macOS / Linux

```cmd
source .venv/bin/activate
```

### Install dependencies

Los requerimientos del proyecto con respecto a las dependencias se encuentran dentro del archivo requeriments.txt

```
pip install -r requirements.txt
```

Estas fueron las dependencias específicas que usamos:   

```
pip install streamlit

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install transformers

pip install matplotlib

pip install scikit-learn

pip install wordcloud

pip install plotly

```

### Run the app

Navegar hacia la carpeta del proyecto

```
cd dashboard
```

Correr la aplicación con streamlit

```
streamlit run dashboard/dashboard.py
```
