
import streamlit as st
from page_modules import data_explore_page, model_comparison_page, predictor_page
from model_loader import EssayGroupModel2 # No borrar es importante para el transformer


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

    st.success("Pequeño disclaimer, esta pagina principal creo que va a ser importante porque " \
                "sino recuerdo mal en el doc aparte de las visualizaciones creo recordar que tambien pide algo de documentacion o algo parecido dentro de la app")

    st.header("Bienvenido")

    import streamlit as st

def render_home_page():
    """Renderiza la página de inicio."""
    st.header("Bienvenido")

    st.markdown("""
    ### Objetivo General
    Analizar exploratoriamente el dataset Feedback Prize – Effectiveness para identificar patrones y características en los argumentos que permitan comprender qué factores influyen en su clasificación como efectivos, adecuados o inefectivos.
    
    ### Objetivos
    - Obtener un listado de características descriptivas (longitud, cantidad, palabras más frecuentes) de los ensayos hechos por niños de los Estados Unidos entre 6to y 12vo grado.
    - Implementar varios modelos que predigan la efectividad de una sección de ensayo, con una precisión mayor a 75%.
    - Determinar qué tipo de algoritmos da mejores resultados con la misma cantidad de tiempo de entrenamiento.
    
    ### Cómo funciona
    """, unsafe_allow_html=True)

    st.markdown("""
    <table>
    <tr>
    <td>
    <ul>
    <li>Introducción</li>
    <li>Postura</li>
    <li>Argumento</li>
    <li>Contraargumento</li>
    <li>Refutación</li>
    <li>Evidencia</li>
    <li>Conclusión</li>
    </ul>
    </td>
    <td style="text-align:center; font-size:24px;">➡️</td>
    <td>
    <ul>
    <li>Inadecuado</li>
    <li>Adecuado</li>
    <li>Efectivo</li>
    </ul>
    </td>
    </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Acerca de este Dashboard

    Este dashboard permite:

    - **Explorar Datos**: Visualizar y analizar el dataset de discursos.
    - **Comparar Modelos**: Analizar el rendimiento de diferentes modelos (LSTM_1, LSTM_2, LSTM_3).
    - **Hacer Predicciones**: Utilizar los modelos entrenados para clasificar nuevos discursos.

    ### Modelos Disponibles
    1. **LSTM 1**: LSTM Bidireccional con Atención  
    2. **LSTM Bidireccional**: LSTM con Attention Pooling  
    3. **LSTM Focal**: LSTM entrenado con Focal Loss para clases desbalanceadas  

    ### Clases de Clasificación
    - **Adequate**: Discurso adecuado  
    - **Effective**: Discurso efectivo  
    - **Ineffective**: Discurso inefectivo  

    ---
    **Seleccione una página para ver los distintos dashboards.**
    """, unsafe_allow_html=True)



if __name__ == "__main__":
    main()