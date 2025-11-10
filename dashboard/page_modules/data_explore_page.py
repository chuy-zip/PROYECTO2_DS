import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import re
from collections import Counter
import nltk
from io import StringIO, BytesIO
import os

# Configuración de página
st.set_page_config(page_title="Exploración de Datos", layout="wide")

# Descargas necesarias de NLTK (silenciosas)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

STOPWORDS = set(nltk.corpus.stopwords.words("spanish")) | set(nltk.corpus.stopwords.words("english"))

# Utilidades de texto
_word_split_re = re.compile(r"\w+", flags=re.UNICODE)

def tokenize(text):
    if not isinstance(text, str):
        return []
    return _word_split_re.findall(text.lower())

def preprocess_dataframe(df, text_col):
    df = df.copy()
    df["__text_str"] = df[text_col].astype(str)
    df["text_length_chars"] = df["__text_str"].str.len()
    df["text_word_tokens"] = df["__text_str"].apply(lambda t: tokenize(t))
    df["text_word_count"] = df["text_word_tokens"].apply(len)
    df["text_unique_words"] = df["text_word_tokens"].apply(lambda tokens: len(set(tokens)))
    df["text_avg_word_len"] = df["text_word_tokens"].apply(
        lambda tokens: np.mean([len(w) for w in tokens]) if tokens else 0
    )
    df["token_counts"] = df["text_word_tokens"].apply(Counter)
    return df

@st.cache_data
def pos_tag_texts(list_of_texts):
    tagged = []
    for t in list_of_texts:
        tokens = tokenize(t)
        if not tokens:
            tagged.append([])
            continue
        try:
            tags = nltk.pos_tag(tokens, tagset="universal")
        except LookupError:
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
def build_ngrams(counters, n=2, top_k=30):
    total = Counter()
    for c in counters:
        tokens = []
        for w, cnt in c.items():
            tokens.extend([w] * cnt)
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i:i + n])
            total[ng] += 1
    most = total.most_common(top_k)
    return [(" ".join(k), v) for k, v in most]

# --------------------------------------------------
# UI PRINCIPAL (sin cargador manual)
# --------------------------------------------------

def render():
    st.title("Exploración de Datos")

    st.markdown(
        """
        En esta página se podrá visualizar y explorar el dataset de discursos utilizado para entrenar los modelos de clasificación.
        Los datos se cargan automáticamente desde la ruta local configurada.
        """
    )

    # Carga directa del dataset sin mostrar uploader
    default_path = os.path.join("..", "data", "train_clean.csv")
    df = None

    try:
        if default_path.endswith(".csv"):
            df = pd.read_csv(default_path, low_memory=False)
        else:
            df = pd.read_parquet(default_path)
    except Exception as e:
        st.error(f"Error al leer los datos desde {default_path}: {e}")

    if df is not None:
        unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)

        st.success(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

        # Forzamos la detección de columnas
        text_col = next(
            (c for c in df.columns if str(c).lower() in (
                "text", "texto", "utterance", "message", "sentence", "text_clean", "textclean", "text_cleaned"
            )), None)

        # Forzar que la clase sea discourse_effectiveness
        if "discourse_effectiveness" in df.columns:
            label_col = "discourse_effectiveness"
        else:
            label_col = next(
                (c for c in df.columns if str(c).lower() in (
                    "label", "class", "clase", "target"
                )), None)

        if text_col is None or label_col is None:
            st.warning("No se detectó automáticamente la columna de texto o la columna 'discourse_effectiveness'.")
            st.write("Columnas disponibles:", list(df.columns))
            return

        st.write(f"Usando columna de texto: **{text_col}**, columna de clase: **{label_col}**")

        dfp = preprocess_dataframe(df, text_col)
        dfp["pos_tags"] = pos_tag_texts(dfp["__text_str"].tolist())
        unique_classes = sorted(dfp[label_col].astype(str).unique().tolist())

        st.sidebar.header("Filtros")
        selected_classes = st.sidebar.multiselect("Filtrar por clase (discourse_effectiveness)", options=unique_classes, default=unique_classes)
        df_filtered = dfp[dfp[label_col].astype(str).isin(selected_classes)].copy()

        # Distribución por clases
        st.markdown("### Distribución por clases (discourse_effectiveness)")
        class_counts = dfp[label_col].astype(str).value_counts().reindex(unique_classes).fillna(0)
        fig_bar = px.bar(
            x=class_counts.values,
            y=class_counts.index,
            orientation="h",
            labels={"x": "Cantidad", "y": "Clase (discourse_effectiveness)"},
            text=class_counts.values,
            color_discrete_sequence=["#D4AA7D"]
        )
        fig_bar.update_layout(height=350, margin=dict(l=40, r=20, t=20, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Boxplot + Matriz de correlación
        st.markdown("### Boxplot por clases y Matriz de correlación")
        numeric_options = ["text_word_count", "text_length_chars", "text_unique_words", "text_avg_word_len"]
        col_box, col_corr = st.columns([1, 1])

        with col_box:
            var_box = st.selectbox("Variable numérica para boxplot", options=numeric_options, index=0)
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=label_col, y=var_box, data=df_filtered, order=unique_classes, palette=["#D4AA7D"])
            plt.xticks(rotation=45)
            plt.title(f"Boxplot de {var_box} por clase (discourse_effectiveness)")
            st.pyplot(plt.gcf())
            plt.clf()

        with col_corr:
            corr_df = df_filtered[numeric_options].copy()
            if corr_df.shape[0] < 2:
                st.info("No hay suficientes filas después del filtro para calcular correlación.")
            else:
                corr = corr_df.corr()
                plt.figure(figsize=(6, 4))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
                plt.title("Matriz de correlación (features textuales)")
                st.pyplot(plt.gcf())
                plt.clf()

        # Nube de palabras
        st.markdown("### Nube de palabras por clase")
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

        # POS frequency y Distribución por vocabulario
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
                fig = px.bar(pos_df, x="POS", y="count", text="count", color_discrete_sequence=["#90A9B7"])
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
                fig = px.bar(vocab_df, x="count", y="word", orientation="h", text="count", color_discrete_sequence=["#D2D8B3"])
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay vocabulario relevante después del filtro.")

        # Bigramas y Trigramas
        st.markdown("### Bigramas (Top)")
        bigram_top_n = st.slider("Top N bigramas", min_value=10, max_value=100, value=20, step=5, key="bigram_n")
        if wc_col == "Todas":
            counters_for_grams = df_filtered["token_counts"].tolist()
        else:
            counters_for_grams = df_filtered[df_filtered[label_col].astype(str) == wc_col]["token_counts"].tolist()
        bigrams = build_ngrams(counters_for_grams, n=2, top_k=bigram_top_n)
        if bigrams:
            big_df = pd.DataFrame(bigrams, columns=["bigram", "count"])
            fig = px.bar(big_df, x="count", y="bigram", orientation="h", text="count", color_discrete_sequence=["#D4AA7D"])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay bigramas suficientes para mostrar.")

        st.markdown("### Trigramas (Top)")
        trigram_top_n = st.slider("Top N trigramas", min_value=10, max_value=100, value=20, step=5, key="trigram_n")
        trigrams = build_ngrams(counters_for_grams, n=3, top_k=trigram_top_n)
        if trigrams:
            tri_df = pd.DataFrame(trigrams, columns=["trigram", "count"])
            fig = px.bar(tri_df, x="count", y="trigram", orientation="h", text="count", color_discrete_sequence=["#EFD09E"])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay trigramas suficientes para mostrar.")

        # Vista previa
        st.markdown("### Vista previa de datos (filtrados)")
        st.dataframe(dfp[[text_col, label_col, "text_word_count", "text_unique_words", "text_avg_word_len"]].reset_index(drop=True))

    else:
        st.error("No se pudieron cargar los datos automáticamente.")

if __name__ == "__main__":
    render()