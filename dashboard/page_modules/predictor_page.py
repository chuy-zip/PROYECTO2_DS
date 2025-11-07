# pagina donde se haran predicciones en base a la eleccion delo modelo del usuario

import streamlit as st
from io import StringIO

def render():

    st.header("Predictor de Efectividad de Discursos")

    st.markdown("""
    Ingrese un texto para clasificar su efectividad o bien suba un archivo de texto.
    """)

    option_map = {
        0: "Texto",
        1: "Archivo"
    }

    input_method = st.pills(
        "Selección de entrada de datos",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        selection_mode="single"
    )

    selection = None if input_method is None else option_map[input_method]


    input = ""
    
    if selection == "Texto":
        input = st.text_area(
            label="Escriba el texto a probar porfavor",
            placeholder="...",
            height=300
        )

        st.write(f"Longitud del texto: {len(input.split())} palabras")
    
    if selection == "Archivo":
        uploaded_file = st.file_uploader(
            label="Eliga un archivo de texto porfavor",
            type=["txt"]
        )
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

            input = stringio.read()

            st.text_area(
                label="Texto ingresado",
                value= input,
                height=300,
                disabled=True
            )
            st.write(f"Longitud del texto: {len(input.split())} palabras")

    
    option_map = {
        0: "Texto",
        1: "Archivo"
    }

    input_method = st.pills(
        "Selección de entrada de datos",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        selection_mode="single"
    )


