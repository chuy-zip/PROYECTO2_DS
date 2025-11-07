# pagina donde se veran las comparasiones de las visualizaciones que antes eran estaticas
# idealmente aqui es donde podemos usar los json que hice.
# en estos json principalmente guardé 
# las métricas principales, matrices de confusión y las métricas por clase

from metric_loader import load_lstm1_metrics, load_lstm2_metrics, load_transformer1_metrics 
import streamlit as st


def render():

    st.header("Comparacion de Modelos")

    st.markdown("""
    En esta pagina se podra comparar el rendimiento de los diferentes modelos
    entrenados (LSTM_1, LSTM_2, Transformer_1) mediante metricas y visualizaciones.
    """)

    # Aqui ira el contenido de comparacion de modelos
    st.info("Contenido en desarrollo...")
