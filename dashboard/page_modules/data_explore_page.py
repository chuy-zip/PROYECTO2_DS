# pagina para explorar los datos usados para entrenar el modelo
# de todas creo que esta es la que tiene mayor potencial para ser dinamica

import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

# Densidad léxica osea la cantidad de palabras únicas / total palabras
def lexical_density(text):
    words = str(text).split()
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words)

# longitud promedio de las palabras
def avg_word_length(text):
    words = str(text).split()
    if len(words) == 0:
        return 0
    return sum(len(word) for word in words) / len(words)

def get_top_ngrams(corpus, ngram_range=(2, 2), n=10):
    corpus = corpus.dropna().astype(str)
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(words_freq[:n], columns=['ngram', 'count'])

def render():

    current_dir = Path(__file__).parent.parent.parent
    model_path = current_dir / "data" / "train_clean.csv"

    df = pd.read_csv(model_path)
    df = df.drop('Unnamed: 0', axis=1)
    df_type = df["discourse_type"].value_counts().reset_index()
    df_type.columns = ["discourse_type", "count"]

    st.markdown("## Exploración de Datos")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Cantidad de discursos", value=f"{len(df):,}")
    with col2:
        st.metric(label="Cantidad de columnas", value=f"{len(df.columns)}")

    st.markdown("Estos datos corresponden al conjunto de entrenamiento utilizado para los modelos de clasificación.")

    st.markdown("---")
    st.markdown("## Graficos y Estadísticas")

    # === Gráfico 1: tipos de discurso ===
    df_type = df["discourse_type"].value_counts().reset_index()
    df_type.columns = ["discourse_type", "count"]  # Renombrar columnas

    fig1 = px.bar(
        df_type,
        x="discourse_type",
        y="count",
        color="discourse_type",
        title="Frecuencia de los tipos de discurso",
        labels={"discourse_type": "Tipo de discurso", "count": "Frecuencia"}
    )
    st.plotly_chart(fig1, use_container_width=True)

    # === Gráfico 2: efectividad ===
    df_eff = df["discourse_effectiveness"].value_counts().reset_index()
    df_eff.columns = ["effectiveness", "count"]

    fig2 = px.bar(
        df_eff,
        x="effectiveness",
        y="count",
        color="effectiveness",
        title="Frecuencia de los tipos de efectividad en discursos",
        labels={"effectiveness": "Efectividad", "count": "Frecuencia"}
    )
    st.plotly_chart(fig2, use_container_width=True)

    
    df['text_char_count'] = df['text_clean'].str.len()

    df['text_word_count'] = df['text_clean'].str.split().str.len()

    df['avg_word_length'] = df['text_clean'].apply(avg_word_length)

    df['lexical_density'] = df['text_clean'].apply(lexical_density)

    st.header("📈 Estadísticas por nivel de efectividad")
    # === Densidad léxica ===
    lexical_density_stats = df.groupby('discourse_effectiveness')['lexical_density'].mean().reset_index()
    fig1 = px.bar(
        lexical_density_stats,
        x='discourse_effectiveness',
        y='lexical_density',
        color='discourse_effectiveness',
        title='Densidad léxica promedio por efectividad',
        labels={'lexical_density': 'Densidad léxica promedio', 'discourse_effectiveness': 'Efectividad'}
    )
    st.plotly_chart(fig1, use_container_width=True)

    # === Longitud promedio de palabras ===
    avg_word_length_stats = df.groupby('discourse_effectiveness')['avg_word_length'].mean().reset_index()
    fig2 = px.bar(
        avg_word_length_stats,
        x='discourse_effectiveness',
        y='avg_word_length',
        color='discourse_effectiveness',
        title='Tamaño promedio de palabra por efectividad',
        labels={'avg_word_length': 'Longitud promedio (caracteres)', 'discourse_effectiveness': 'Efectividad'}
    )
    st.plotly_chart(fig2, use_container_width=True)

    # === Cantidad de palabras ===
    text_word_count_stats = df.groupby('discourse_effectiveness')['text_word_count'].agg(['mean', 'std']).reset_index()
    fig3 = px.box(
        df,
        x='discourse_effectiveness',
        y='text_word_count',
        color='discourse_effectiveness',
        title='Distribución de cantidad de palabras por efectividad',
        labels={'text_word_count': 'Cantidad de palabras', 'discourse_effectiveness': 'Efectividad'}
    )
    st.plotly_chart(fig3, use_container_width=True)

    # === Largo del texto ===
    text_char_count_stats = df.groupby('discourse_effectiveness')['text_char_count'].agg(['mean', 'std']).reset_index()
    fig4 = px.box(
        df,
        x='discourse_effectiveness',
        y='text_char_count',
        color='discourse_effectiveness',
        title='Distribución del largo del texto (caracteres) por efectividad',
        labels={'text_char_count': 'Largo del texto (caracteres)', 'discourse_effectiveness': 'Efectividad'}
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.header("🧩 N-gramas más comunes por efectividad")

    for eff in df['discourse_effectiveness'].unique():
        subset = df[df['discourse_effectiveness'] == eff]

        st.subheader(f"**{eff}**")

        # === BIGRAMAS ===
        top_bi = get_top_ngrams(subset['text_clean'], ngram_range=(2,2), n=10)
        fig_bi = px.bar(
            top_bi,
            x='count',
            y='ngram',
            orientation='h',
            title=f"Top 10 Bigramas - {eff}",
            labels={'count': 'Frecuencia', 'ngram': 'Bigramas'},
            color='count',
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig_bi, use_container_width=True)

        # === TRIGRAMAS ===
        top_tri = get_top_ngrams(subset['text_clean'], ngram_range=(3,3), n=10)
        fig_tri = px.bar(
            top_tri,
            x='count',
            y='ngram',
            orientation='h',
            title=f"Top 10 Trigramas - {eff}",
            labels={'count': 'Frecuencia', 'ngram': 'Trigramas'},
            color='count',
            color_continuous_scale='greens'
        )
        st.plotly_chart(fig_tri, use_container_width=True)

        st.markdown("---")

    st.header("☁️ Nubes de Palabras por Efectividad")
    for eff in df['discourse_effectiveness'].unique():
        subset = df[df['discourse_effectiveness'] == eff]
        text = " ".join(subset['text_clean'].astype(str))

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            stopwords="english"
        ).generate(text)

        st.subheader(f"Nube de Palabras - {eff}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    st.header("🧮 Palabras más relevantes por Efectividad (TF-IDF)")

    tfidf = TfidfVectorizer(stop_words="english", max_features=20)

    for eff in df["discourse_effectiveness"].unique():
        subset = df[df["discourse_effectiveness"] == eff]["text_clean"].dropna().astype(str)
        if len(subset) == 0:
            continue

        tfidf_matrix = tfidf.fit_transform(subset)
        scores = zip(tfidf.get_feature_names_out(), tfidf.idf_)

        # Convertir a DataFrame ordenado
        tfidf_df = pd.DataFrame(sorted(scores, key=lambda x: x[1], reverse=True), columns=["word", "score"]).head(10)

        st.subheader(f"{eff}")
        fig = px.bar(
            tfidf_df,
            x="score",
            y="word",
            orientation="h",
            title=f"Palabras más relevantes (TF-IDF alto) en {eff}",
            labels={"score": "Peso TF-IDF", "word": "Palabra"},
            color="score",
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig, use_container_width=True)