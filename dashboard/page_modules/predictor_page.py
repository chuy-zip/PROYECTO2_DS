# pagina donde se haran predicciones en base a la eleccion delo modelo del usuario

import streamlit as st
from io import StringIO

def render():

    st.header("Predictor de Efectividad de Discursos")

    st.markdown("""
    Utilice los modelos entrenados para clasificar la efectividad de un discurso.
    Ingrese el texto directamente o cargue un archivo.
    """)

    # Seccion 1: Seleccion de metodo de entrada
    st.subheader("1. Metodo de entrada")

    option_map = {
        0: "Texto",
        1: "Archivo"
    }

    input_method = st.pills(
        "Seleccione como desea ingresar el discurso:",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        selection_mode="single"
    )

    selection = None if input_method is None else option_map[input_method]

    st.markdown("")  # Espacio

    input = ""

    if selection == "Texto":
        input = st.text_area(
            label="Texto del discurso",
            placeholder="Escriba o pegue aqui el texto del discurso que desea analizar...",
            height=300,
            help="Ingrese el texto completo del discurso para su analisis"
        )

        if input:
            word_count = len(input.split())
            char_count = len(input)
            st.caption(f"Archivo: **{word_count}** palabras | **{char_count}** caracteres")

    if selection == "Archivo":
        uploaded_file = st.file_uploader(
            label="Archivo de texto",
            type=["txt"],
            help="Suba un archivo .txt con el contenido del discurso"
        )
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

            input = stringio.read()

            st.text_area(
                label="Contenido del archivo",
                value= input,
                height=300,
                disabled=True
            )

            word_count = len(input.split())
            char_count = len(input)
            st.caption(f"Archivo: **{uploaded_file.name}** | **{word_count}** palabras | **{char_count}** caracteres")

    st.markdown("---")

    # Seccion 2: Seleccion de modelos
    if input and selection:
        st.subheader("2. Seleccion de modelos")

        st.markdown("Elija uno o mas modelos para realizar la prediccion:")

        model_options = ["LSTM 1", "LSTM Bidireccional", "Transformer"]

        model_selection = st.pills(
            "Modelos disponibles:",
            options=model_options,
            selection_mode="multi",
            help="Puede seleccionar multiples modelos para comparar sus predicciones"
        )

        if model_selection:
            st.caption(f"**{len(model_selection)}** modelo(s) seleccionado(s): {', '.join(model_selection)}")

        st.markdown("")  # Espacio

        if st.button("Evaluar efectividad", type="primary", width='stretch') and model_selection:

            st.markdown("---")
            st.subheader("3. Resultados de Prediccion")

            # Crear tabs dinamicas basadas en los modelos seleccionados
            tabs = st.tabs(model_selection)

            # Iterar sobre cada tab y modelo seleccionado
            for tab, model_name in zip(tabs, model_selection):
                with tab:
                    # Aqui va el contenido de cada modelo
                    render_model_prediction(model_name, input)

            # Seccion de comparacion de modelos
            if len(model_selection) > 1:
                st.markdown("---")
                st.subheader("4. Comparacion entre Modelos")
                render_model_comparison(model_selection)


def render_model_prediction(model_name: str, input_text: str):
    """
    Renderiza la prediccion para un modelo especifico.

    Args:
        model_name: Nombre del modelo seleccionado
        input_text: Texto a predecir
    """
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("####Resultado de Clasificacion")

        # TODO: Aqui ira la logica de prediccion real
        # Por ahora, placeholder
        with st.spinner(f"Procesando con {model_name}..."):
            pass

        # Placeholder para grafico de probabilidades
        st.markdown("**Distribuccion de probabilidades:**")
        st.progress(0.65, text="Adequate: 65%")
        st.progress(0.25, text="Effective: 25%")
        st.progress(0.10, text="Ineffective: 10%")

        st.markdown("")
        # Placeholder para clasificacion final
        st.success("**Clasificacion:** Adequate")
        st.metric(label="Confianza", value="65%", delta="Alta")

    with col2:
        st.markdown("####Informacion de la Prediccion")

        # TODO: Aqui ira el resumen generado
        st.markdown(f"""
        **Modelo:** `{model_name}`

        **Estadisticas del texto:**
        - Palabras: {len(input_text.split())}
        - Caracteres: {len(input_text)}

        **Metricas de confianza:**
        - Nivel de certeza: Alta
        - Probabilidad maxima: 65%
        """)

        # with st.expander("ℹInterpretacion del resultado"):
        #     st.write("El modelo ha clasificado este discurso como **Adequate** con un nivel de confianza del 65%. "
        #             "Esto indica que el discurso cumple con los requisitos basicos de efectividad.")
        #     st.caption("Placeholder para interpretacion detallada...")


def render_model_comparison(selected_models: list):
    """
    Renderiza una tabla comparativa de las predicciones de los modelos.

    Args:
        selected_models: Lista de nombres de modelos seleccionados
    """
    st.markdown("Comparativa de predicciones entre los modelos seleccionados:")

    st.markdown("")

    # TODO: Aqui ira la tabla real con las predicciones
    # Por ahora, placeholder con estructura

    comparison_data = {
        "Modelo": selected_models,
        "Clasificacion": ["Adequate"] * len(selected_models),
        "Confianza": ["65%", "58%", "72%"][:len(selected_models)],
        "Segunda Clase": ["Effective", "Ineffective", "Effective"][:len(selected_models)],
        "Prob. Segunda": ["25%", "30%", "20%"][:len(selected_models)]
    }

    st.dataframe(
        comparison_data,
        width='stretch',
        hide_index=True
    )

    st.markdown("")

    # Metricas comparativas
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Consenso",
            value="100%",
            help="Porcentaje de modelos que coinciden en la clasificacion"
        )

    with col2:
        st.metric(
            label="Confianza promedio",
            value="65%",
            help="Promedio de confianza entre todos los modelos"
        )

    with col3:
        st.metric(
            label="Clasificacion final",
            value="Adequate",
            help="Clasificacion mas frecuente entre los modelos"
        )

    st.markdown("")

    # with st.expander("Analisis comparativo detallado"):
    #     st.markdown("""
    #     **Observaciones:**
    #     - Todos los modelos coinciden en clasificar el discurso como **Adequate**
    #     - El modelo Transformer muestra la mayor confianza (72%)
    #     - La segunda clase mas probable varia entre modelos

    #     **Recomendacion:**
    #     La alta coincidencia entre modelos sugiere una clasificacion confiable.
    #     """)
    #     st.caption("Placeholder para analisis comparativo detallado...")

