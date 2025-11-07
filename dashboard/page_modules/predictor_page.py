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

    st.markdown("---")

    if input and selection:    
        model_options = ["LSTM 1", "LSTM Bidireccional", "Transformer"]

        model_selection = st.pills(
            "Seleccione al menos 1 modelo",
            options=model_options,
            selection_mode="multi"
        )

        st.markdown(f"Modelos elegidos: {model_selection}.")

        if st.button("Evaluar efectividad", type="primary") and model_selection:

            st.markdown("---")
            st.subheader("Resultados de Prediccion")

            # Crear tabs dinamicas basadas en los modelos seleccionados
            tabs = st.tabs(model_selection)

            # Iterar sobre cada tab y modelo seleccionado
            for tab, model_name in zip(tabs, model_selection):
                with tab:
                    # Aqui va el contenido de cada modelo
                    render_model_prediction(model_name, input)

            # Seccion de comparacion de modelos
            st.markdown("---")
            st.subheader("Comparacion de Modelos")

            if len(model_selection) > 1:
                render_model_comparison(model_selection)
            else:
                st.info("Seleccione al menos 2 modelos para ver la comparacion.")


def render_model_prediction(model_name: str, input_text: str):
    """
    Renderiza la prediccion para un modelo especifico.

    Args:
        model_name: Nombre del modelo seleccionado
        input_text: Texto a predecir
    """
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Resultado de Clasificacion")

        # TODO: Aqui ira la logica de prediccion real
        # Por ahora, placeholder
        st.info(f"Cargando prediccion con {model_name}...")

        # Placeholder para grafico de probabilidades
        st.write("**Probabilidades por clase:**")
        st.write("- Adequate: XX%")
        st.write("- Effective: XX%")
        st.write("- Ineffective: XX%")

        # Placeholder para clasificacion final
        st.success("**Clasificacion: [Clase predicha]**")

    with col2:
        st.markdown("### Resumen de la Prediccion")

        # TODO: Aqui ira el resumen generado
        st.write(f"**Modelo utilizado:** {model_name}")
        st.write(f"**Longitud del texto:** {len(input_text.split())} palabras")
        st.write("**Confianza:** XX%")

        st.markdown("**Interpretacion:**")
        st.write("Placeholder para resumen o interpretacion de la prediccion...")


def render_model_comparison(selected_models: list):
    """
    Renderiza una tabla comparativa de las predicciones de los modelos.

    Args:
        selected_models: Lista de nombres de modelos seleccionados
    """
    st.markdown("Tabla comparativa de las predicciones realizadas por cada modelo:")

    # TODO: Aqui ira la tabla real con las predicciones
    # Por ahora, placeholder con estructura

    comparison_data = {
        "Modelo": selected_models,
        "Clasificacion": ["Placeholder"] * len(selected_models),
        "Confianza": ["XX%"] * len(selected_models),
        "Clase Primaria": ["Clase"] * len(selected_models),
        "Prob. Primaria": ["XX%"] * len(selected_models)
    }

    st.table(comparison_data)

    st.markdown("**Analisis:**")
    st.write("Placeholder para analisis comparativo entre modelos...")

