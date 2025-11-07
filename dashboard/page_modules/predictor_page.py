# pagina donde se haran predicciones en base a la eleccion delo modelo del usuario

import streamlit as st


def render():

    st.header("Predictor de Efectividad de Discursos")

    st.markdown("""
    En esta pagina se podra ingresar un discurso y utilizar los modelos entrenados
    para predecir su nivel de efectividad.
    """)

    # Aqui ira el contenido del predictor
    st.info("Contenido en desarrollo...")
