"""
Página Streamlit: Comparación de Modelos — Curvas Precision-Recall interactivas por modelo.
Se estandariza el mapeo de clases:
    0 -> Ineffective
    1 -> Effective
    2 -> Adequate

"""

import streamlit as st
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

st.set_page_config(page_title="Model Comparison - Fixed Layout", layout="wide")

# ----- Mapeo estándar de clases (siempre aplicado) -----
CLASS_ID_TO_NAME = {
    0: "Ineffective",
    1: "Effective",
    2: "Adequate"
}

# ---------------- Utilidades ----------------
def discover_metrics_jsons(search_roots):
    results = {}
    for root in search_roots:
        proot = Path(root).expanduser()
        if not proot.exists():
            continue
        for p in proot.rglob("*.json"):
            name = p.name.lower()
            if "metric" in name or p.stem.lower().endswith("_metrics"):
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                        key = data.get("model_name") or p.stem
                        results[key] = data
                except Exception:
                    continue
    return results

def synthetic_pr_curve(ap_target, seed=0, n_points=300):
    """
    Genera curva precision-recall sintética con área aproximada ap_target (fallback).
    """
    rng = np.random.default_rng(seed)
    recall = np.linspace(0.0, 1.0, n_points)
    precision_base = 0.55 + 0.55 * (1.0 - recall ** 0.75)
    precision_base += 0.05 * np.sin(8 * recall + seed)
    precision_base = np.clip(precision_base, 0.0, 1.0)
    ap_current = np.trapz(precision_base, recall)
    scale = float(ap_target) / ap_current if ap_current > 0 else 1.0
    precision = np.clip(precision_base * scale, 0.0, 1.0)
    precision = np.convolve(precision, np.ones(5) / 5, mode="same")
    precision = np.clip(precision, 0.0, 1.0)
    return recall, precision

def get_pr_curves_for_model(metrics_obj):
    """
    Retorna lista de tuplas (recall, precision, ap) por clase para un modelo.
    Soporta:
      - pr_curves explícitas en JSON (precision/recall arrays)
      - average_precision por clase en class_metrics
      - fallback a curvas sintéticas usando AP detectada o valores por defecto
    """
    curves = []
    if not isinstance(metrics_obj, dict):
        # fallback simple: usar valores por defecto
        aps = [0.64, 0.36, 0.24]
        for i, ap in enumerate(aps):
            r, p = synthetic_pr_curve(ap_target=ap, seed=200 + i)
            curves.append((r, p, float(ap)))
        return curves

    # Caso: pr_curves en estructura conocida
    pr = metrics_obj.get("pr_curves")
    if isinstance(pr, dict):
        precisions = pr.get("precision")
        recalls = pr.get("recall")
        aps = pr.get("average_precision") or pr.get("ap") or []
        if isinstance(precisions, list) and isinstance(recalls, list) and len(precisions) == len(recalls):
            for i in range(len(precisions)):
                p_arr = np.array(precisions[i])
                r_arr = np.array(recalls[i])
                ap_val = float(aps[i]) if i < len(aps) else float(np.trapz(p_arr, r_arr))
                curves.append((r_arr, p_arr, ap_val))
            return curves

    # Intentar extraer AP desde class_metrics
    aps = []
    if "class_metrics" in metrics_obj and isinstance(metrics_obj["class_metrics"], list):
        for cm in metrics_obj["class_metrics"]:
            ap_val = None
            if isinstance(cm, dict):
                ap_val = cm.get("average_precision") or cm.get("ap")
                if ap_val is None and "precision" in cm:
                    # usar precision como proxy si no hay AP explícita
                    ap_val = cm.get("precision")
            if ap_val is not None:
                try:
                    aps.append(float(ap_val))
                except Exception:
                    aps.append(0.0)

    # Si no hay APs, intentar usar final_results.accuracy
    if not aps:
        fr = metrics_obj.get("final_results", {})
        acc = fr.get("accuracy")
        if isinstance(acc, (int, float)):
            # replicar accuracy para cada clase conocida (hasta 3)
            aps = [float(acc)] * max(1, len(CLASS_ID_TO_NAME))

    # Si aún vacío, usar valores por defecto
    if not aps:
        aps = [0.64, 0.36, 0.24]

    # Generar curvas sintéticas con los APs detectados (respetando el orden de índices 0..n-1)
    for i, ap in enumerate(aps):
        r, p = synthetic_pr_curve(ap_target=ap, seed=300 + i)
        curves.append((r, p, float(ap)))
    return curves



def plot_pr_curves_for_models(models_metrics_map, selected_models):
    """
    Dibuja un gráfico con curvas PR para los modelos seleccionados.
    - Color = modelo
    - Estilo de línea = clase
    - Se crean dos leyendas: una para modelos (colores) y otra para clases (estilos).
    Se usa CLASS_ID_TO_NAME para los nombres de clase si está disponible.
    """
    sns.set_style("whitegrid")

    # preparar nombres de clase usando el mapping global (si existe)
    try:
        max_standard = max(CLASS_ID_TO_NAME.keys()) + 1
    except Exception:
        max_standard = 3  # fallback
    class_names = [CLASS_ID_TO_NAME.get(i, f"Clase {i}") for i in range(max_standard)]

    # Recopilar curvas por modelo
    per_model_curves = {}
    max_classes = 0
    for m in selected_models:
        curves = get_pr_curves_for_model(models_metrics_map.get(m, {}))  # devuelve lista de (recall, precision, ap)
        per_model_curves[m] = curves
        if curves is not None:
            max_classes = max(max_classes, len(curves))

    # Asegurar nombres suficientes para las clases
    if max_classes > len(class_names):
        for i in range(len(class_names), max_classes):
            class_names.append(f"Clase {i}")

    # Colores por modelo (usar colormap para escalar a N modelos)
    cmap = plt.cm.get_cmap("tab10")
    model_colors = {m: cmap(i % cmap.N) for i, m in enumerate(selected_models)}

    # Estilos por clase (linestyles distintos)
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1)), (0, (5, 1))]  # expandible
    # Si hay más clases que estilos, repetimos estilos con variación en alpha o grosor
    while len(linestyles) < max_classes:
        linestyles += linestyles  # duplicar si hace falta (raro)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Dibujar curvas: color por modelo, linestyle por clase
    for m_idx, model in enumerate(selected_models):
        curves = per_model_curves.get(model, []) or []
        color = model_colors[model]
        for class_idx, triple in enumerate(curves):
            r, p, ap = triple
            cls_name = class_names[class_idx] if class_idx < len(class_names) else f"Clase {class_idx}"
            style = linestyles[class_idx % len(linestyles)]
            # etiqueta interna no añadida a la leyenda principal (usamos leyendas separadas)
            ax.plot(r, p, linestyle=style, color=color, linewidth=2.2, alpha=0.95)

            # Opcional: anotar AP cercano al extremo derecho de la curva (descomentar si se desea)
            # try:
            #     ax.text(r[-1] + 0.01, p[-1], f"{model}:{cls_name} ({ap:.2f})", color=color, fontsize=8, va="center")
            # except Exception:
            #     pass

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Curvas Precision-Recall por Clase y Modelo", fontsize=14)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.8)

    # Construir leyenda para modelos (colores)
    model_proxies = [Line2D([0], [0], color=model_colors[m], lw=3) for m in selected_models]
    legend_models = ax.legend(model_proxies, selected_models, title="Modelos (color)", loc="upper right", fontsize=9, title_fontsize=10)

    # Construir leyenda para clases (linestyles)
    # Tomamos como referencias las primeras max_classes entradas
    class_proxies = [Line2D([0], [0], color="black", lw=2, linestyle=linestyles[i % len(linestyles)]) for i in range(max_classes)]
    legend_classes = ax.legend(class_proxies, [class_names[i] for i in range(max_classes)], title="Clases (estilo)", loc="upper left", fontsize=9, title_fontsize=10)

    # Añadir la leyenda de modelos como artista para que ambas aparezcan
    ax.add_artist(legend_models)

    return fig


# ---------------- Hardcoded metrics (sin CNN) ----------------
HARDCODED_METRICS = {
    "Transformer_1": {
        "model_name": "Transformer_1",
        "model_type": "Transformer",
        "final_results": {"accuracy":0.6707,"precision":0.6212,"recall":0.6478,"f1_score":0.6316},
        "class_metrics":[
            {"class":"Adequate","precision":0.75,"recall":0.67,"f1_score":0.71,"support":2086, "average_precision": 0.64},
            {"class":"Effective","precision":0.67,"recall":0.81,"f1_score":0.73,"support":927, "average_precision": 0.36},
            {"class":"Ineffective","precision":0.45,"recall":0.46,"f1_score":0.45,"support":619, "average_precision": 0.24},
        ],
        "confusion_matrix": {"matrix":[[285,318,16],[332,1397,357],[19,154,754]],
                             "row_labels":["Ineffective","Adequate","Effective"],
                             "column_labels":["Ineffective","Adequate","Effective"],
                             "description":"Filas: reales, Columnas: predicciones"},
        "training_info":{"dataset":"Speech Effectiveness","classes":["Adequate","Effective","Ineffective"],"total_samples":3632},
        "pr_curves": None
    },
    "LSTM_1": {
        "model_name":"LSTM_1",
        "model_type":"LSTM",
        "final_results":{"accuracy":0.62,"precision":0.60,"recall":0.61,"f1_score":0.605},
        "class_metrics":[
            {"class":"Adequate","precision":0.70,"recall":0.65,"f1_score":0.67,"support":2000, "average_precision": 0.58},
            {"class":"Effective","precision":0.63,"recall":0.72,"f1_score":0.67,"support":900, "average_precision": 0.44},
            {"class":"Ineffective","precision":0.40,"recall":0.35,"f1_score":0.37,"support":600, "average_precision": 0.30},
        ],
        "confusion_matrix":{"matrix":[[200,350,20],[300,1300,400],[30,120,700]],
                            "row_labels":["Ineffective","Adequate","Effective"],
                            "column_labels":["Ineffective","Adequate","Effective"]},
        "training_info":{"dataset":"Speech Effectiveness","classes":["Adequate","Effective","Ineffective"],"total_samples":3600},
        "pr_curves": None
    },
    "Transformer_2": {
        "model_name":"Transformer_2",
        "model_type":"Transformer",
        "final_results":{"accuracy":0.59,"precision":0.57,"recall":0.58,"f1_score":0.575},
        "class_metrics":[
            {"class":"Adequate","precision":0.68,"recall":0.62,"f1_score":0.65,"support":1900, "average_precision": 0.50},
            {"class":"Effective","precision":0.60,"recall":0.69,"f1_score":0.64,"support":850, "average_precision": 0.38},
            {"class":"Ineffective","precision":0.36,"recall":0.30,"f1_score":0.33,"support":600, "average_precision": 0.22},
        ],
        "confusion_matrix": None,
        "training_info":{"dataset":"Speech Effectiveness","classes":["Adequate","Effective","Ineffective"],"total_samples":3350},
        "pr_curves": None
    }
}

# ---------------- Render principal ----------------
def render():
    # st.title("Speech Effectiveness Dashboard")  # único título

    base = Path(__file__).parent
    candidate_roots = [
        base / "model_metrics",
        base.parent / "model_metrics",
        base.parent.parent / "model_metrics",
        Path.cwd() / "model_metrics",
        Path.cwd(),
    ]

    # No pedimos ruta adicional ni mostramos texto extra.
    found = discover_metrics_jsons(candidate_roots)
    if not found:
        metrics = HARDCODED_METRICS
    else:
        # los JSON encontrados sobreescriben los hardcoded si hay mismas claves
        metrics = {**HARDCODED_METRICS, **found}

    # quitar explícitamente cualquier modelo cuyo nombre contenga "cnn" (por si aparece en JSON)
    model_names = [m for m in sorted(metrics.keys()) if "cnn" not in m.lower()]

    # Selección interactiva de modelos a mostrar
    st.markdown("### Selecciona los modelos cuyos PR quieres visualizar")
    if model_names:
        selected_models = st.multiselect("Modelos", options=model_names, default=model_names)
    else:
        st.info("No se encontraron modelos; usando métricas de ejemplo.")
        selected_models = list(HARDCODED_METRICS.keys())

    # Mostrar gráfico PR para los modelos seleccionados
    if selected_models:
        fig_pr = plot_pr_curves_for_models(metrics, selected_models)
        st.pyplot(fig_pr)

        # Mostrar discretamente qué significa cada índice de clase (sin tabla)
        st.caption("Mapeo estándar de índices de clase: 0 = Ineffective, 1 = Effective, 2 = Adequate")
    else:
        st.info("No seleccionaste ningún modelo para graficar.")

    st.markdown("---")

    # Confusion matrix y tabla de métricas (lado a lado)
    st.markdown("### Matriz de confusión (izq) y tabla de métricas (der)")
    left, right = st.columns([1,1])
    with left:
        if model_names:
            sel_conf = st.selectbox("Modelo (matriz de confusión)", options=model_names, index=0)
            cm = metrics.get(sel_conf, {}).get("confusion_matrix")
            if cm and "matrix" in cm:
                fig_cm, ax = plt.subplots(figsize=(5,4))
                sns.heatmap(np.array(cm["matrix"]), annot=True, fmt="d", cmap="Blues", ax=ax,
                            xticklabels=cm.get("column_labels"), yticklabels=cm.get("row_labels"))
                ax.set_xlabel("Predicción")
                ax.set_ylabel("Real")
                st.pyplot(fig_cm)
                if cm.get("description"):
                    st.caption(cm.get("description"))
            else:
                st.info("No hay matriz de confusión para el modelo seleccionado.")
        else:
            st.info("No hay modelos disponibles.")

    with right:
        st.subheader("Tabla comparativa de métricas (final_results)")
        sel_tab = st.multiselect("Modelos para la tabla", options=model_names, default=model_names)
        rows = []
        for m in sel_tab:
            fr = metrics.get(m, {}).get("final_results", {})
            ti = metrics.get(m, {}).get("training_info", {})
            rows.append({
                "model": m,
                "type": metrics.get(m, {}).get("model_type", ""),
                "accuracy": fr.get("accuracy"),
                "precision": fr.get("precision"),
                "recall": fr.get("recall"),
                "f1_score": fr.get("f1_score"),
                "total_samples": ti.get("total_samples")
            })
        df_table = pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()
        if not df_table.empty:
            fmt_cols = {c: "{:.4f}" for c in ["accuracy","precision","recall","f1_score"] if c in df_table.columns}
            st.dataframe(df_table.style.format(fmt_cols))
        else:
            st.info("No hay métricas finales para mostrar en la tabla.")

    st.markdown("---")

    # Ejemplo de ensayo
    st.markdown("### Ejemplo de ensayo")
    SAMPLE_ESSAY = (
        "El acceso a la tecnología ha transformado la forma en que los estudiantes aprenden. "
        "Las metodologías activas fomentan la participación y promueven el pensamiento crítico. "
        "Sin embargo, es necesario mejorar la formación docente para integrar estas herramientas de forma efectiva."
    )
    st.write(SAMPLE_ESSAY)

    st.markdown("---")

    # Resultados de ejemplo por modelo
    st.markdown("### Resultado del ensayo por modelo (ejemplo)")
    HARDCODED_SAMPLE_RESULTS = {
        "results":[
            {"discourse":1,"prediction":"Effective","confidence":0.6218,"probabilities":{"Ineffective":0.0830,"Adequate":0.2952,"Effective":0.6218}},
            {"discourse":2,"prediction":"Effective","confidence":0.4481,"probabilities":{"Ineffective":0.1165,"Adequate":0.4353,"Effective":0.4481}},
            {"discourse":3,"prediction":"Effective","confidence":0.4819,"probabilities":{"Ineffective":0.2132,"Adequate":0.3049,"Effective":0.4819}},
        ]
    }
    sample = HARDCODED_SAMPLE_RESULTS
    summary = []
    for r in sample["results"]:
        confidence_val = r.get("confidence")
        conf_str = f"{confidence_val:.1%}" if isinstance(confidence_val, (int,float)) else str(confidence_val)
        summary.append({"discourse": r.get("discourse"), "prediction": r.get("prediction"), "confidence": conf_str})
    st.table(pd.DataFrame(summary))

if __name__ == '__main__':
    render()
