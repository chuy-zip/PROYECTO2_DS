# pagina donde se haran predicciones en base a la eleccion delo modelo del usuario

import streamlit as st
from io import StringIO
import sys
from pathlib import Path

# Agregar el directorio padre al path para importar model_loader
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_loader import predict_with_lstm1, predict_with_lstm2, predict_with_lstm3, predict_with_transformer

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

        model_options = ["LSTM 1", "LSTM Bidireccional", "LSTM Focal", "Transformers"]

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

            # Guardar resultados de todos los modelos para comparacion
            all_predictions = {}

            # Iterar sobre cada tab y modelo seleccionado
            for tab, model_name in zip(tabs, model_selection):
                with tab:
                    # Aqui va el contenido de cada modelo
                    result = render_model_prediction(model_name, input)
                    if result is not None:
                        all_predictions[model_name] = result

            # Seccion de comparacion de modelos
            if len(model_selection) > 1 and len(all_predictions) > 1:
                st.markdown("---")
                st.subheader("4. Comparacion entre Modelos")
                render_model_comparison(model_selection, all_predictions)


def render_model_prediction(model_name: str, input_text: str):
    """
    Renderiza la prediccion para un modelo especifico.

    Args:
        model_name: Nombre del modelo seleccionado
        input_text: Texto a predecir
    """
    col1, col2 = st.columns([1, 1], gap="large")

    # Hacer prediccion segun el modelo seleccionado
    prediction_result = None

    try:
        with st.spinner(f"Procesando con {model_name}..."):
            if model_name == "LSTM 1":
                prediction_result = predict_with_lstm1(input_text)
            elif model_name == "LSTM Bidireccional":
                prediction_result = predict_with_lstm2(input_text)
            elif model_name == "LSTM Focal":
                prediction_result = predict_with_lstm3(input_text)
            elif model_name == "Transformers":
                st.warning("El modelo de Transformers aun no esta implementado.")
                a = input_text.split("\n")
                st.info(a)
                pr = predict_with_transformer(input_text)
                st.info(pr)
                for line in a:
                    st.info(line)
                return
    except Exception as e:
        st.error(f"Error al realizar la prediccion: {str(e)}")
        return

    if prediction_result is None:
        return

    # Extraer resultados
    predicted_class = prediction_result['predicted_class']
    probabilities = prediction_result['probabilities']
    confidence = prediction_result['confidence']
    all_classes = prediction_result['all_classes']

    with col1:
        st.markdown("#### Resultado de Clasificacion")

        # Grafico de probabilidades
        st.markdown("**Distribuccion de probabilidades:**")

        # Ordenar clases para mostrarlas consistentemente
        for class_name in all_classes:
            prob = probabilities[class_name]
            st.progress(prob, text=f"{class_name}: {prob:.1%}")

        st.markdown("")

        # Clasificacion final con color segun la clase
        if predicted_class == "Effective":
            st.success(f"**Clasificacion:** {predicted_class}")
        elif predicted_class == "Adequate":
            st.info(f"**Clasificacion:** {predicted_class}")
        else:  # Ineffective
            st.warning(f"**Clasificacion:** {predicted_class}")

        # Metrica de confianza
        confidence_level = "Alta" if confidence > 0.7 else "Media" if confidence > 0.5 else "Baja"
        st.metric(label="Confianza", value=f"{confidence:.1%}", delta=confidence_level)

    with col2:
        st.markdown("#### Informacion de la Prediccion")

        # Informacion del texto y modelo
        st.markdown(f"""
        **Modelo:** `{model_name}`

        **Estadisticas del texto:**
        - Palabras: {len(input_text.split())}
        - Caracteres: {len(input_text)}

        **Metricas de confianza:**
        - Nivel de certeza: {confidence_level}
        - Probabilidad maxima: {confidence:.1%}

        **Segunda clase mas probable:**
        """)

        # Encontrar segunda clase mas probable
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_probs) > 1:
            second_class, second_prob = sorted_probs[1]
            st.write(f"- {second_class}: {second_prob:.1%}")

    return prediction_result


def render_model_comparison(selected_models: list, predictions: dict):
    """
    Renderiza una tabla comparativa de las predicciones de los modelos.

    Args:
        selected_models: Lista de nombres de modelos seleccionados
        predictions: Diccionario con los resultados de prediccion de cada modelo
    """
    st.markdown("Comparativa de predicciones entre los modelos seleccionados:")

    st.markdown("")

    # Construir datos de comparacion con resultados reales
    model_names = []
    classifications = []
    confidences = []
    second_classes = []
    second_probs = []

    for model_name in selected_models:
        if model_name in predictions:
            result = predictions[model_name]
            model_names.append(model_name)
            classifications.append(result['predicted_class'])
            confidences.append(f"{result['confidence']:.1%}")

            # Encontrar segunda clase mas probable
            sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
            if len(sorted_probs) > 1:
                second_class, second_prob = sorted_probs[1]
                second_classes.append(second_class)
                second_probs.append(f"{second_prob:.1%}")
            else:
                second_classes.append("-")
                second_probs.append("-")

    comparison_data = {
        "Modelo": model_names,
        "Clasificacion": classifications,
        "Confianza": confidences,
        "Segunda Clase": second_classes,
        "Prob. Segunda": second_probs
    }

    st.dataframe(
        comparison_data,
        width='stretch',
        hide_index=True
    )

    st.markdown("")

    # Calcular metricas comparativas
    # Consenso: porcentaje de modelos que coinciden en la clasificacion
    from collections import Counter
    class_counts = Counter(classifications)
    most_common_class, most_common_count = class_counts.most_common(1)[0]
    consensus = (most_common_count / len(classifications)) * 100

    # Confianza promedio
    avg_confidence = sum([predictions[m]['confidence'] for m in model_names]) / len(model_names)

    # Metricas comparativas
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Consenso",
            value=f"{consensus:.0f}%",
            help="Porcentaje de modelos que coinciden en la clasificacion"
        )

    with col2:
        st.metric(
            label="Confianza promedio",
            value=f"{avg_confidence:.1%}",
            help="Promedio de confianza entre todos los modelos"
        )

    with col3:
        st.metric(
            label="Clasificacion final",
            value=most_common_class,
            help="Clasificacion mas frecuente entre los modelos"
        )

    st.markdown("")

    # Analisis adicional
    with st.expander("Ver analisis comparativo detallado"):
        st.markdown("**Observaciones:**")

        # Verificar consenso
        if consensus == 100:
            st.write(f"- Todos los modelos coinciden en clasificar como **{most_common_class}**")
        else:
            st.write(f"- Los modelos no tienen consenso completo ({consensus:.0f}%)")
            st.write(f"- Clasificacion mas frecuente: **{most_common_class}**")

        # Modelo con mayor confianza
        max_conf_model = max(model_names, key=lambda m: predictions[m]['confidence'])
        max_conf = predictions[max_conf_model]['confidence']
        st.write(f"- El modelo **{max_conf_model}** muestra la mayor confianza ({max_conf:.1%})")

        # Variabilidad en segunda clase
        unique_second_classes = set([s for s in second_classes if s != "-"])
        if len(unique_second_classes) > 1:
            st.write(f"- La segunda clase mas probable varia entre modelos")
        elif len(unique_second_classes) == 1:
            st.write(f"- Todos coinciden en la segunda opcion: **{list(unique_second_classes)[0]}**")

        st.markdown("**Recomendacion:**")
        if consensus >= 80 and avg_confidence >= 0.6:
            st.success("La alta coincidencia y confianza entre modelos sugiere una clasificacion muy confiable.")
        elif consensus >= 60:
            st.info("Hay consenso moderado entre los modelos. Considere el contexto adicional.")
        else:
            st.warning("Los modelos no tienen consenso. Se recomienda precaucion con esta clasificacion.")
