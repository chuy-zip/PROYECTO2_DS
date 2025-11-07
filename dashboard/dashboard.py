
import streamlit as st
from page_modules import data_explore_page, model_comparison_page, predictor_page


# Configuración de la página
st.set_page_config(
    page_title="Speech Effectiveness Dashboard",
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():


    # Título principal
    st.title("Speech Effectiveness Dashboard")
    st.markdown("---")

    st.success("Pequeño disclaimer, esta pagina principal creo que va a ser importante porque " \
                "sino recuerdo mal en el doc aparte de las visualizaciones creo recordar que tambien pide algo de documentacion o algo parecido dentro de la app")

    # Sidebar para navegación
    st.sidebar.title("Navegación")
    st.sidebar.markdown("Selecciona una página:")

    # Opciones de páginas
    page = st.sidebar.radio(
        "Ir a:",
        [
            "Inicio",
            "Exploración de Datos",
            "Comparación de Modelos",
            "Predictor"
        ],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Información del Proyecto")
    st.sidebar.info(
        "Dashboard interactivo para analizar y comparar modelos de "
        "clasificación de efectividad de discursos."
    )

    # Renderizar página seleccionada
    if page == "Inicio":
        render_home_page()
    elif page == "Exploración de Datos":
        data_explore_page.render()
    elif page == "Comparación de Modelos":
        model_comparison_page.render()
    elif page == "Predictor":
        predictor_page.render()


def render_home_page():
    """Renderiza la página de inicio."""
    st.header("Bienvenido")

    st.markdown("""
    ### Acerca de este Dashboard

    Este dashboard permite:

    - **Explorar Datos**: Visualizar y analizar el dataset de discursos
    - **Comparar Modelos**: Analizar el rendimiento de diferentes modelos (LSTM_1, LSTM_2, Transformer_1)
    - **Hacer Predicciones**: Utilizar los modelos entrenados para clasificar nuevos discursos

    ### Modelos Disponibles

    1. **LSTM_1**: Modelo LSTM 
    2. **LSTM_2**: Modelo LSTM bidireccional
    3. **Transformer_1**: Modelo basado en arquitectura Transformer

    ### Clases de Clasificación

    - **Adequate**: Discurso adecuado
    - **Effective**: Discurso efectivo
    - **Ineffective**: Discurso inefectivo

    ---

    **Seleccione una página para ver los distintos dashboards**
    """)


if __name__ == "__main__":
    main()