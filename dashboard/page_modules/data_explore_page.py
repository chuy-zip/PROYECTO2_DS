# data_explore_page_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import re
from collections import Counter, defaultdict
import networkx as nx
import nltk
from itertools import combinations
from io import StringIO, BytesIO
import os

# Configuración de página
st.set_page_config(page_title="Exploración de Datos", layout="wide")

# Descargas necesarias de NLTK (silenciosas)
# Descargar variantes necesarias para POS tagging y tokenizers.
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
# algunos entornos requieren esta variante nombrada
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
# universal tagset no es un paquete descargable; se usa como parámetro en pos_tag
# nltk.download("universal_tagset", quiet=True)  # no necesario

STOPWORDS = set(nltk.corpus.stopwords.words("spanish")) | set(nltk.corpus.stopwords.words("english"))

# Utilidades de texto
_word_split_re = re.compile(r"\w+", flags=re.UNICODE)

def tokenize(text):
    if not isinstance(text, str):
        return []
    return _word_split_re.findall(text.lower())

def preprocess_dataframe(df, text_col):
    # Normalizaciones y features básicos
    df = df.copy()
    df["__text_str"] = df[text_col].astype(str)
    df["text_length_chars"] = df["__text_str"].str.len()
    df["text_word_tokens"] = df["__text_str"].apply(lambda t: tokenize(t))
    df["text_word_count"] = df["text_word_tokens"].apply(len)
    df["text_unique_words"] = df["text_word_tokens"].apply(lambda tokens: len(set(tokens)))
    df["text_avg_word_len"] = df["text_word_tokens"].apply(lambda tokens: np.mean([len(w) for w in tokens]) if tokens else 0)
    # frecuencia por documento (Counter)
    df["token_counts"] = df["text_word_tokens"].apply(Counter)
    return df

@st.cache_data
def pos_tag_texts(list_of_texts):
    # Intentamos usar pos_tag universal
    tagged = []
    for t in list_of_texts:
        tokens = tokenize(t)
        if not tokens:
            tagged.append([])
            continue
        try:
            tags = nltk.pos_tag(tokens, tagset="universal")
        except LookupError:
            # fallback: intentar sin tagset
            try:
                tags = nltk.pos_tag(tokens)
            except Exception:
                tags = []
        tagged.append(tags)
    return tagged

@st.cache_data
def build_wordcloud(counter, max_words=200):
    wc = WordCloud(width=900, height=400, background_color="white", max_words=max_words)
    img = wc.generate_from_frequencies(dict(counter))
    return img

@st.cache_data
def top_n_vocab(counters, n=30):
    total = Counter()
    for c in counters:
        total.update(c)
    return total.most_common(n)

@st.cache_data
def build_cooccurrence_graph(counters, top_words=None, window_size=2, top_k_edges=50):
    # counters: list of Counter per doc
    total = Counter()
    for c in counters:
        total.update(c)
    if top_words is None:
        top_words = [w for w, _ in total.most_common(100)]
    # build co-occurrence counts via document-level cooccurrence (unordered pairs)
    pair_counts = Counter()
    for c in counters:
        present = [w for w in c.keys() if w in top_words]
        for (a, b) in combinations(sorted(set(present)), 2):
            pair_counts[(a, b)] += 1
    # build graph
    G = nx.Graph()
    for w in top_words:
        G.add_node(w, size=total[w])
    for (a, b), cnt in pair_counts.most_common(top_k_edges):
        G.add_edge(a, b, weight=cnt)
    return G

# helpers para n-gramas
def build_ngrams(counters, n=2, top_k=30):
    total = Counter()
    for c in counters:
        # expand each document into repeated tokens based on counts
        tokens = []
        for w, cnt in c.items():
            tokens.extend([w] * cnt)
        # create ngrams in the document (sequence-less approximation: use tokens order as-is)
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i:i+n])
            total[ng] += 1
    # return list of tuples with stringified ngrams and counts
    most = total.most_common(top_k)
    return [(" ".join(k), v) for k, v in most]

# --------------------------------------------------
# UI: carga de datos y configuración
# --------------------------------------------------

def render():
    st.title("Exploración de Datos")

    st.markdown(
        """
    En esta página se podrá visualizar y explorar el dataset de discursos utilizado para entrenar los modelos de clasificación.
    Selecciona o sube el archivo que contiene tus datos (CSV, Parquet). El dataframe debe tener una columna de texto y una columna con la etiqueta/clase.
    """
    )

    with st.expander("Cargar datos"):
        uploaded = st.file_uploader("Sube un CSV/Parquet (o deja vacío y selecciona ruta local)", type=["csv", "parquet", "parq"], accept_multiple_files=False)
        # Ruta por defecto basada en el notebook de análisis exploratorio (subir un nivel desde dashboard/)
        default_path = os.path.join("..", "data", "train_clean.csv")
        use_sample_path = st.text_input(
            "Ruta local (opcional)",
            value=default_path,
            help="Ruta por defecto basada en ExploratoryAnalysis.ipynb (relativa al folder 'dashboard/')"
        )

        # dejamos los inputs vacíos por defecto para detectar automáticamente
        text_col_input = st.text_input("Nombre columna texto (si se detecta automáticamente puedes dejar vacío)", value="")
        label_col_input = st.text_input("Nombre columna etiqueta/clase (si se detecta automáticamente puedes dejar vacío)", value="")

    df = None
    if uploaded:
        try:
            bytes_data = uploaded.read()
            if uploaded.type == "text/csv" or uploaded.name.endswith(".csv"):
                df = pd.read_csv(StringIO(bytes_data.decode("utf-8")), encoding="utf-8", low_memory=False)
            else:
                df = pd.read_parquet(BytesIO(bytes_data))
        except Exception as e:
            st.error(f"Error al leer archivo subido: {e}")
    elif use_sample_path:
        try:
            if use_sample_path.endswith(".csv"):
                df = pd.read_csv(use_sample_path, low_memory=False)
            else:
                df = pd.read_parquet(use_sample_path)
        except Exception as e:
            st.error(f"Error al leer ruta local: {e}")
    else:
        st.info("Sube un archivo o escribe la ruta local para cargar los datos. Si ya cargaste el notebook con datos, puedes arrastrar el CSV aquí.")

    # Si al leer aparece una columna de índice como 'Unnamed: 0', la removemos
    if df is not None:
        unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)

    if df is not None:
        st.success(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

        # Detección automática de columnas de texto y etiqueta (incluye tus nombres reales)
        text_col = text_col_input.strip() or next(
            (c for c in df.columns if str(c).lower() in (
                "text", "texto", "utterance", "message", "sentence", "text_clean", "textclean", "text_cleaned"
            )), None)

        label_col = label_col_input.strip() or next(
            (c for c in df.columns if str(c).lower() in (
                "label", "class", "clase", "target", "discourse_effectiveness", "discourse_type", "discourseeffectiveness"
            )), None)

        if text_col is None or label_col is None:
            st.warning("No se detectó automáticamente la columna de texto o etiqueta. Ajusta los nombres arriba.")
            st.write("Columnas disponibles:", list(df.columns))
        else:
            st.write(f"Usando columna de texto: **{text_col}**, columna de clase: **{label_col}**")

            # Preprocesamiento y creación de features
            dfp = preprocess_dataframe(df, text_col)
            # cache POS tags por lista de textos reducida
            dfp["pos_tags"] = pos_tag_texts(dfp["__text_str"].tolist())
            # Contadores globales por fila ya en 'token_counts'
            unique_classes = sorted(dfp[label_col].astype(str).unique().tolist())

            # filtros en sidebar
            st.sidebar.header("Filtros")
            selected_classes = st.sidebar.multiselect("Filtrar por clase", options=unique_classes, default=unique_classes)

            df_filtered = dfp[dfp[label_col].astype(str).isin(selected_classes)].copy()

            # --------------------------------------------------
            # 1) Gráfica horizontal: distribución por clases (centrada bajo el título)
            # --------------------------------------------------
            st.markdown("### Distribución por clases")
            class_counts = dfp[label_col].astype(str).value_counts().reindex(unique_classes).fillna(0)
            fig_bar = px.bar(
                x=class_counts.values,
                y=class_counts.index,
                orientation="h",
                labels={"x": "Cantidad", "y": "Clase"},
                text=class_counts.values
            )
            fig_bar.update_layout(height=350, margin=dict(l=40, r=20, t=20, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

            # --------------------------------------------------
            # 2) Boxplot por clases (izquierda) y 3) Correlation matrix (derecha)
            # --------------------------------------------------
            st.markdown("### Boxplot por clases y Matriz de correlación")
            # Permitimos elegir la variable numérica a usar en boxplot
            numeric_options = ["text_word_count", "text_length_chars", "text_unique_words", "text_avg_word_len"]
            col_box, col_corr = st.columns([1, 1])
            with col_box:
                var_box = st.selectbox("Variable numérica para boxplot", options=numeric_options, index=0)
                plt.figure(figsize=(6, 4))
                sns.boxplot(x=label_col, y=var_box, data=df_filtered, order=unique_classes)
                plt.xticks(rotation=45)
                plt.title(f"Boxplot de {var_box} por clase")
                st.pyplot(plt.gcf())
                plt.clf()

            with col_corr:
                # construimos matriz de features numéricas
                corr_df = df_filtered[numeric_options].copy()
                if corr_df.shape[0] < 2:
                    st.info("No hay suficientes filas después del filtro para calcular correlación.")
                else:
                    corr = corr_df.corr()
                    plt.figure(figsize=(6, 4))
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", square=True)
                    plt.title("Matriz de correlación (features textuales)")
                    st.pyplot(plt.gcf())
                    plt.clf()

            # --------------------------------------------------
            # 4) Nube de palabras (centrada) con filtro por clase
            # --------------------------------------------------
            st.markdown("### Nube de palabras")
            # selector de clase para wordcloud
            wc_col = st.selectbox("Clase para la nube de palabras", options=["Todas"] + unique_classes, index=0)
            if wc_col == "Todas":
                counters = df_filtered["token_counts"].tolist()
                global_counter = Counter()
                for c in counters:
                    global_counter.update(c)
            else:
                counters = df_filtered[df_filtered[label_col].astype(str) == wc_col]["token_counts"].tolist()
                global_counter = Counter()
                for c in counters:
                    global_counter.update(c)
            # eliminar stopwords y tokens numericos cortos
            for w in list(global_counter.keys()):
                if w in STOPWORDS or w.isdigit() or len(w) <= 1:
                    del global_counter[w]
            if not global_counter:
                st.info("No hay palabras suficientes para generar la nube.")
            else:
                wc_img = build_wordcloud(global_counter, max_words=200)
                fig = plt.figure(figsize=(12, 4))
                plt.imshow(wc_img, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(fig)
                plt.clf()

            # --------------------------------------------------
            # 5) Gráfico de frecuencias de POS (izquierda) y 6) Distribución por vocabulario (derecha)
            # --------------------------------------------------
            st.markdown("### POS frequency y Distribución por vocabulario")
            col_pos, col_vocab = st.columns([1, 1])
            with col_pos:
                st.subheader("Frecuencia POS")
                pos_class_filter = st.selectbox("Filtrar POS por clase (opcional)", options=["Todas"] + unique_classes, index=0, key="pos_filter")
                if pos_class_filter == "Todas":
                    pos_list = df_filtered["pos_tags"].tolist()
                else:
                    pos_list = df_filtered[df_filtered[label_col].astype(str) == pos_class_filter]["pos_tags"].tolist()
                pos_counter = Counter()
                for doc in pos_list:
                    for token, tag in doc:
                        pos_counter[tag] += 1
                if pos_counter:
                    pos_df = pd.DataFrame(pos_counter.most_common(), columns=["POS", "count"])
                    fig = px.bar(pos_df, x="POS", y="count", text="count")
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No hay tags POS detectados para el filtro seleccionado.")

            with col_vocab:
                st.subheader("Distribución por vocabulario (top N)")
                vocab_class_filter = st.selectbox("Clase para vocabulario", options=["Todas"] + unique_classes, index=0, key="vocab_filter")
                top_n = st.slider("Top N palabras", 10, 100, 30)
                if vocab_class_filter == "Todas":
                    counters_list = df_filtered["token_counts"].tolist()
                else:
                    counters_list = df_filtered[df_filtered[label_col].astype(str) == vocab_class_filter]["token_counts"].tolist()
                total_vocab = Counter()
                for c in counters_list:
                    total_vocab.update(c)
                for w in list(total_vocab.keys()):
                    if w in STOPWORDS or w.isdigit() or len(w) <= 1:
                        del total_vocab[w]
                if total_vocab:
                    top_words = total_vocab.most_common(top_n)
                    vocab_df = pd.DataFrame(top_words, columns=["word", "count"])
                    fig = px.bar(vocab_df, x="count", y="word", orientation="h", text="count")
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No hay vocabulario relevante después del filtro.")

            # --------------------------------------------------
            # Bigramas y Trigramas (antes de la vista previa)
            # --------------------------------------------------
            st.markdown("### Bigramas (Top)")
            bigram_top_n = st.slider("Top N bigramas", min_value=10, max_value=100, value=20, step=5, key="bigram_n")
            if wc_col == "Todas":
                counters_for_grams = df_filtered["token_counts"].tolist()
            else:
                counters_for_grams = df_filtered[df_filtered[label_col].astype(str) == wc_col]["token_counts"].tolist()
            bigrams = build_ngrams(counters_for_grams, n=2, top_k=bigram_top_n)
            if bigrams:
                big_df = pd.DataFrame(bigrams, columns=["bigram", "count"])
                fig = px.bar(big_df, x="count", y="bigram", orientation="h", text="count")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay bigramas suficientes para mostrar.")

            st.markdown("### Trigramas (Top)")
            trigram_top_n = st.slider("Top N trigramas", min_value=10, max_value=100, value=20, step=5, key="trigram_n")
            trigrams = build_ngrams(counters_for_grams, n=3, top_k=trigram_top_n)
            if trigrams:
                tri_df = pd.DataFrame(trigrams, columns=["trigram", "count"])
                fig = px.bar(tri_df, x="count", y="trigram", orientation="h", text="count")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay trigramas suficientes para mostrar.")

            # --------------------------------------------------
            # 7) Mapa de concurrencia de palabras (co-occurrence) centrado en la fila siguiente
            # --------------------------------------------------
            st.markdown("### Mapa de concurrencia de palabras (co-occurrence)")
            co_class_filter = st.selectbox("Filtrar co-ocurrencias por clase (opcional)", options=["Todas"] + unique_classes, index=0, key="co_filter")
            if co_class_filter == "Todas":
                counters_for_co = df_filtered["token_counts"].tolist()
            else:
                counters_for_co = df_filtered[df_filtered[label_col].astype(str) == co_class_filter]["token_counts"].tolist()
            # calcular top words y grafo
            most_common_n = st.slider("Top palabras a considerar (para el grafo)", min_value=20, max_value=200, value=80, step=10)
            total_counter = Counter()
            for c in counters_for_co:
                total_counter.update(c)
            top_words = [w for w, _ in total_counter.most_common(most_common_n)]
            if not top_words:
                st.info("No hay palabras para construir el grafo.")
            else:
                G = build_cooccurrence_graph(counters_for_co, top_words=top_words, top_k_edges=80)
                # dibujar grafo con networkx
                plt.figure(figsize=(10, 8))
                pos = nx.spring_layout(G, k=0.5, seed=42)
                sizes = [G.nodes[n]["size"] * 10 for n in G.nodes()]
                weights = [d["weight"] for (_, _, d) in G.edges(data=True)]
                nx.draw_networkx_nodes(G, pos, node_size=sizes, alpha=0.8)
                nx.draw_networkx_edges(G, pos, width=[max(1, w / 3) for w in weights], alpha=0.6)
                nx.draw_networkx_labels(G, pos, font_size=8)
                plt.axis("off")
                st.pyplot(plt.gcf())
                plt.clf()

            # --------------------------------------------------
            # Muestra tabla de datos filtrada (opcional para inspección)
            # --------------------------------------------------
            st.markdown("### Vista previa de datos (filtrados)")
            st.dataframe(df_filtered[[text_col, label_col, "text_word_count", "text_unique_words", "text_avg_word_len"]].reset_index(drop=True))

    else:
        st.info("Aún no hay datos cargados. Sube un CSV/Parquet con tus discursos y clases para comenzar.")

if __name__ == "__main__":
    # Permite ejecutar el módulo directamente para pruebas rápidas
    render()
