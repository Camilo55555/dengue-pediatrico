import io

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Predicción de Dengue Pediátrico",
    page_icon="🦠",
    layout="wide",
)

try:
    with open("style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

st.title("🩺 Predicción de Complicaciones por Dengue Pediátrico")
st.markdown(
    "Herramienta de apoyo clínico basada en Machine Learning que compara "
    "**Regresión Logística** y **Red Neuronal (MLP)** para estimar el riesgo de dengue complicado."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Definición de variables
# ---------------------------------------------------------------------------
# Nota sobre la codificación del CSV original (verificada con los datos):
#   - Síntomas (cefalea, mialgia, dolor abdominal, rash, vómitos, dificultad
#     respiratoria, sangrado, organomegalia): 1 = Sí, 2 = No
#   - BP AT ADMISSION: 1 = normal, 2 = alterada
#   - AGE: 1 = ≤ 6 años, 2 = > 6 años
#   - Grupo: 1 = dengue no complicado, 2 = dengue complicado
# Internamente todo se convierte a 0/1 (1 = presente / alterado / complicado).

BASE_FEATURES = [
    "AGE",
    "DURATION OF FEVER",
    "HEADACHE",
    "MYALGIA",
    "ABDOMINAL PAIN",
    "RASH",
    "VOMITING",
    "BREATHLESSNESS",
    "BLEEDING",
    "BP AT ADMISSION",
    "ORGANOMEGALY",
    "HB",
    "PCV",
    "LOWEST WBC",
    "HIGHEST WBC",
    "PLATELET COUNT",
    "ALT",
    "CREATININE",
]

SINTOMA_COLS = [
    "HEADACHE",
    "MYALGIA",
    "ABDOMINAL PAIN",
    "RASH",
    "VOMITING",
    "BREATHLESSNESS",
    "BLEEDING",
    "ORGANOMEGALY",
]

COL_GRUPO = "GROUPS: DENGUE FEVER OR COMPLICATED DENGUE"

ETIQUETAS_CLASE = {
    0: "Dengue No Complicado",
    1: "Dengue Complicado",
}


def _a_numerico(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _valores(s: pd.Series) -> set:
    return set(s.dropna().unique())


def limpiar_base_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Limpia comas decimales y convierte la codificación del dataset a 0/1.

    Acepta tanto la codificación original (1/2) como datos ya en 0/1.
    """
    df_clean = pd.DataFrame(index=df_raw.index)
    for col in BASE_FEATURES:
        if col in df_raw.columns:
            df_clean[col] = _a_numerico(df_raw[col])
        else:
            df_clean[col] = np.nan

    # Síntomas: original 1 = Sí, 2 = No  ->  1 = presente
    for col in SINTOMA_COLS:
        s = df_clean[col]
        if _valores(s).issubset({0, 1}):
            continue
        df_clean[col] = np.where(s.isna(), np.nan, (s == 1).astype(float))

    # Presión arterial: original 1 = normal, 2 = alterada  ->  1 = alterada
    s = df_clean["BP AT ADMISSION"]
    if not _valores(s).issubset({0, 1}):
        df_clean["BP AT ADMISSION"] = np.where(s.isna(), np.nan, (s == 2).astype(float))

    # Edad: original 1 = ≤ 6 años, 2 = > 6 años; o edad en años  ->  1 = > 6 años
    s = df_clean["AGE"]
    vals = _valores(s)
    if vals.issubset({0, 1}):
        pass
    elif vals.issubset({1, 2}):
        df_clean["AGE"] = np.where(s.isna(), np.nan, (s == 2).astype(float))
    else:
        df_clean["AGE"] = np.where(s.isna(), np.nan, (s > 6).astype(float))

    return df_clean


def construir_variables_predictoras(
    df_base: pd.DataFrame, medianas_base: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Imputa nulos con la mediana y agrega variables derivadas."""
    df = df_base.copy()
    if medianas_base is None:
        medianas_base = df.median()
    df = df.fillna(medianas_base)

    df["n_sintomas"] = df[SINTOMA_COLS].sum(axis=1)
    df["rango_leuco"] = (df["HIGHEST WBC"] - df["LOWEST WBC"]).clip(lower=0)
    df["log_alt"] = np.log1p(df["ALT"].clip(lower=0))

    df["plaquetas_bajas"] = (df["PLATELET COUNT"] <= 75000).astype(int)
    df["pcv_alto"] = (df["PCV"] >= 42).astype(int)
    df["leuco_alto"] = (df["HIGHEST WBC"] >= 7000).astype(int)
    df["alt_alta"] = (df["ALT"] >= 70).astype(int)

    return df, medianas_base


def traducir_grupo_real(serie: pd.Series) -> pd.Series:
    """Convierte la etiqueta original (1 = no complicado, 2 = complicado) a 0/1."""
    s_num = pd.to_numeric(serie, errors="coerce")
    if _valores(s_num).issubset({0, 1}):
        return s_num.fillna(0).astype(int)
    return (s_num == 2).astype(int)


# Umbral común para ambos modelos (más bajo que 0.5 para no dejar escapar casos graves)
UMBRAL = 0.35

NOMBRES_VARIABLES = {
    "AGE": "edad > 6 años",
    "DURATION OF FEVER": "duración de fiebre",
    "HEADACHE": "cefalea",
    "MYALGIA": "mialgia",
    "ABDOMINAL PAIN": "dolor abdominal",
    "RASH": "erupción",
    "VOMITING": "vómitos",
    "BREATHLESSNESS": "dificultad respiratoria",
    "BLEEDING": "sangrado",
    "BP AT ADMISSION": "hipotensión",
    "ORGANOMEGALY": "organomegalia",
    "HB": "hemoglobina",
    "PCV": "hematocrito (PCV)",
    "LOWEST WBC": "leucocitos mínimos",
    "HIGHEST WBC": "leucocitos máximos",
    "PLATELET COUNT": "plaquetas (por mm³)",
    "ALT": "ALT (U/L)",
    "CREATININE": "creatinina",
    "n_sintomas": "n.º de síntomas",
    "rango_leuco": "rango de leucocitos",
    "log_alt": "ALT (log)",
    "plaquetas_bajas": "plaquetas ≤ 75.000",
    "pcv_alto": "PCV ≥ 42",
    "leuco_alto": "leucocitos ≥ 7.000",
    "alt_alta": "ALT ≥ 70",
}


def predecir(modelo, X):
    """Devuelve (clase, probabilidad) usando el umbral común de ambos modelos."""
    prob = modelo.predict_proba(X)[:, 1]
    return (prob >= UMBRAL).astype(int), prob


@st.cache_resource
def entrenar_modelos():
    df = pd.read_csv(
        "MASTER_CHART_F1000Research.csv",
        sep=None,
        engine="python",
        on_bad_lines="skip",
    )
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)

    y = traducir_grupo_real(df[COL_GRUPO])

    df_base = limpiar_base_df(df)
    X, medianas_base = construir_variables_predictoras(df_base)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelos = {
        "Regresión Logística": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=3000, random_state=42),
        ),
        "Red Neuronal (MLP)": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(8,),
                alpha=3.0,
                learning_rate_init=0.004,
                max_iter=3000,
                random_state=42,
            ),
        ),
    }

    for modelo in modelos.values():
        modelo.fit(X_train, y_train)

    n_complicados = int(y.sum())
    pct_complicados = (n_complicados / len(y) * 100) if len(y) > 0 else 0.0
    binarias = [c for c in X.columns if set(X[c].unique()).issubset({0, 1})]

    return (
        binarias,
        modelos,
        medianas_base,
        list(X.columns),
        X_test,
        y_test,
        len(df),
        len(X_train),
        len(X_test),
        n_complicados,
        pct_complicados,
    )


(
    binarias,
    modelos,
    medianas_base,
    feature_cols_finales,
    X_test,
    y_test,
    n_total,
    n_train,
    n_test,
    n_complicados,
    pct_complicados,
) = entrenar_modelos()

with st.expander("🧹 Documentación de Limpieza y Preprocesamiento de Datos"):
    st.markdown(
        f"""
- **Fuente de datos:** `MASTER_CHART_F1000Research.csv` ({n_total} registros tras eliminar duplicados; {n_complicados} complicados, {pct_complicados:.1f}%; {n_train} entrenamiento / {n_test} prueba, división 80/20 estratificada).
- **Codificación original del CSV:** síntomas `1 = Sí`, `2 = No`; presión arterial `1 = normal`, `2 = alterada`; grupo `1 = no complicado`, `2 = complicado`. Todo se convierte internamente a `0/1` (1 = presente/alterado/complicado).
- **Limpieza:** nombres de columnas sin espacios sobrantes y comas decimales (`,`) convertidas a puntos (`.`).
- **Exclusión de fuga de información:** no se usan ferritina ni tratamientos (líquidos IV, antibióticos, antipiréticos, esteroides), porque se registran después del diagnóstico. El género tampoco se usa (no aporta señal clínica y su codificación no está documentada).
- **Imputación de nulos:** mediana.
- **Variables derivadas:** `n_sintomas`, `rango_leuco`, `log_alt` e indicadores (`plaquetas ≤ 75000`, `PCV ≥ 42`, `leucocitos máx. ≥ 7000`, `ALT ≥ 70`).
- **Modelos:** **Regresión Logística** (con regularización moderada) y **Red Neuronal (MLP)**, ambos con `StandardScaler`.
        """
    )

pestana_individual, pestana_rendimiento, pestana_lote = st.tabs(
    ["👤 Predicción Individual", "📈 Rendimiento y Factores de Riesgo", "📁 Predicción por Lote"]
)

SI_NO = {0: "No", 1: "Sí"}

# ---------------------------------------------------------------------------
# Pestaña individual
# ---------------------------------------------------------------------------
with pestana_individual:
    st.subheader("📋 Parámetros Clínicos del Paciente")
    st.caption(
        "Los valores por defecto representan un paciente sin síntomas ni alteraciones de laboratorio."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 📌 Datos Generales")
        age_anios = st.slider("Edad del paciente (años)", 1, 18, 4)
        age_codificado = 1 if age_anios > 6 else 0
        fever_duration = st.slider("Duración de fiebre (días)", 1, 10, 3)

        st.markdown("##### 🤒 Síntomas (Parte 1)")
        headache = st.selectbox("Dolor de cabeza (Cefalea)", [0, 1], format_func=SI_NO.get)
        myalgia = st.selectbox("Mialgia (Dolor muscular)", [0, 1], format_func=SI_NO.get)
        abdominal_pain = st.selectbox("Dolor abdominal", [0, 1], format_func=SI_NO.get)
        rash = st.selectbox("Erupción (Rash)", [0, 1], format_func=SI_NO.get)

    with col2:
        st.markdown("##### 🤒 Síntomas (Parte 2)")
        vomiting = st.selectbox("Vómitos", [0, 1], format_func=SI_NO.get)
        breathlessness = st.selectbox("Dificultad respiratoria", [0, 1], format_func=SI_NO.get)
        bleeding = st.selectbox("Sangrado (Hemorragias)", [0, 1], format_func=SI_NO.get)
        organomegaly = st.selectbox("Organomegalia", [0, 1], format_func=SI_NO.get)
        bp_admission = st.selectbox(
            "Presión arterial al ingreso",
            [0, 1],
            format_func=lambda x: "Normal" if x == 0 else "Alterada / Hipotensión",
        )

    with col3:
        st.markdown("##### 🧪 Exámenes de Laboratorio")
        hb = st.number_input("Hemoglobina (g/dL)", 5.0, 20.0, 13.0)
        pcv = st.number_input("Hematocrito - PCV (%)", 20.0, 60.0, 38.0)
        lowest_wbc = st.number_input("Leucocitos mínimos (WBC)", 1000, 30000, 4500)
        highest_wbc = st.number_input("Leucocitos máximos (WBC)", 1000, 30000, 6000)
        platelet = st.number_input("Plaquetas (conteo)", 5000, 500000, 180000, step=5000)
        alt = st.number_input("Transaminasa ALT (U/L)", 5.0, 500.0, 25.0)
        creatinine = st.number_input("Creatinina (mg/dL)", 0.2, 5.0, 0.5)

    datos_paciente = {
        "AGE": age_codificado,
        "DURATION OF FEVER": fever_duration,
        "HEADACHE": headache,
        "MYALGIA": myalgia,
        "ABDOMINAL PAIN": abdominal_pain,
        "RASH": rash,
        "VOMITING": vomiting,
        "BREATHLESSNESS": breathlessness,
        "BLEEDING": bleeding,
        "BP AT ADMISSION": bp_admission,
        "ORGANOMEGALY": organomegaly,
        "HB": hb,
        "PCV": pcv,
        "LOWEST WBC": lowest_wbc,
        "HIGHEST WBC": highest_wbc,
        "PLATELET COUNT": platelet,
        "ALT": alt,
        "CREATININE": creatinine,
    }
    # Ya está en 0/1: se construyen directamente las variables derivadas.
    df_usuario, _ = construir_variables_predictoras(pd.DataFrame([datos_paciente]), medianas_base)
    df_usuario = df_usuario[feature_cols_finales]

    st.markdown("---")
    if st.button("⚡ Ejecutar Predicción y Comparar Modelos", type="primary", use_container_width=True):
        st.markdown("### 🔬 Resultado por Modelo")
        cols_modelos = st.columns(len(modelos))
        resumen_ind = []
        alerta_complicado = False
        probs_complicado = []

        for idx, (nombre_mod, mod) in enumerate(modelos.items()):
            pred_arr, prob_arr = predecir(mod, df_usuario)
            pred_i = int(pred_arr[0])
            prob_i = float(prob_arr[0])
            prob_pct = round(prob_i * 100, 1)
            etiqueta_i = ETIQUETAS_CLASE[pred_i]
            probs_complicado.append(prob_pct)
            if pred_i == 1:
                alerta_complicado = True

            with cols_modelos[idx]:
                st.metric(
                    label=nombre_mod,
                    value=f"{prob_pct:.1f}%",
                    delta=etiqueta_i,
                    delta_color="inverse" if pred_i == 1 else "normal",
                )

            resumen_ind.append(
                {
                    "Modelo": nombre_mod,
                    "Probabilidad de complicación (%)": prob_pct,
                    "Clasificación": etiqueta_i,
                }
            )

        st.dataframe(pd.DataFrame(resumen_ind), use_container_width=True, hide_index=True)

        max_prob = max(probs_complicado)
        if alerta_complicado:
            st.error(
                f"⚠️ **ALERTA CLÍNICA — probabilidad máxima: {max_prob:.1f}%**\n\n"
                "*Al menos un modelo clasifica el caso como Dengue Complicado. Se recomienda "
                "evaluación médica prioritaria y vigilancia estrecha de signos de alarma.*"
            )
        else:
            st.success(
                f"✅ **RIESGO BAJO — probabilidad máxima: {max_prob:.1f}%**\n\n"
                "*Ningún modelo clasifica el caso como complicado. Mantener seguimiento clínico "
                "y educación sobre signos de alarma.*"
            )
        st.caption(
            "Herramienta de apoyo: no reemplaza el criterio médico ni los signos de alarma clínicos."
        )

# ---------------------------------------------------------------------------
# Pestaña por lote
# ---------------------------------------------------------------------------
with pestana_lote:
    st.subheader("📁 Predicción por Lote y Métricas")
    st.markdown(
        "Evalúa un lote de pacientes con ambos modelos, revisa las métricas y descarga el reporte."
    )

    modelo_seleccionado = st.selectbox(
        "Modelo principal para matriz de confusión y reporte:",
        list(modelos.keys()),
    )

    tipo_eval = st.radio(
        "Fuente de datos del lote:",
        ["Usar dataset de prueba del sistema", "Subir archivo CSV personalizado"],
    )

    df_lote = None
    if tipo_eval == "Subir archivo CSV personalizado":
        st.caption(
            "El CSV debe tener las mismas columnas que el dataset original "
            "(síntomas 1 = Sí / 2 = No; presión 1 = normal / 2 = alterada). "
            "También se aceptan valores 0/1."
        )
        archivo_subido = st.file_uploader("Sube tu archivo CSV", type=["csv"])
        if archivo_subido is not None:
            df_lote = pd.read_csv(archivo_subido, sep=None, engine="python", on_bad_lines="skip")
            df_lote.columns = [c.strip() for c in df_lote.columns]
    else:
        df_lote = X_test.copy()
        df_lote["Etiqueta Real"] = y_test.map(ETIQUETAS_CLASE).values
        df_lote["_y_real_bin"] = y_test.values

    if df_lote is not None:
        st.markdown("### 📊 Vista previa del lote")
        st.dataframe(df_lote.head(10), use_container_width=True)

        if st.button("🚀 Ejecutar Predicción por Lote y Calcular Métricas", type="primary"):
            faltantes_base = [c for c in BASE_FEATURES if c not in df_lote.columns]
            ya_preparado = all(c in df_lote.columns for c in feature_cols_finales)

            if faltantes_base and not ya_preparado:
                st.error(f"⚠️ Faltan columnas requeridas por el modelo: {faltantes_base}")
            else:
                if ya_preparado:
                    X_lote = df_lote[feature_cols_finales].fillna(0)
                else:
                    X_lote, _ = construir_variables_predictoras(
                        limpiar_base_df(df_lote), medianas_base
                    )
                    X_lote = X_lote[feature_cols_finales]

                df_resultados = df_lote.copy()
                for nombre_mod, mod in modelos.items():
                    preds, probs = predecir(mod, X_lote)
                    df_resultados[f"{nombre_mod} (Prob %)"] = np.round(probs * 100, 1)
                    df_resultados[f"{nombre_mod} (Predicción)"] = pd.Series(
                        preds, index=df_lote.index
                    ).map(ETIQUETAS_CLASE)

                pred_prin, _ = predecir(modelos[modelo_seleccionado], X_lote)

                st.markdown("### 📋 Resultados por Lote")
                cols_mostrar = [
                    c
                    for c in [
                        "AGE",
                        "PLATELET COUNT",
                        "n_sintomas",
                        "Etiqueta Real",
                        "Regresión Logística (Predicción)",
                        "Regresión Logística (Prob %)",
                        "Red Neuronal (MLP) (Predicción)",
                        "Red Neuronal (MLP) (Prob %)",
                    ]
                    if c in df_resultados.columns
                ]
                st.dataframe(df_resultados[cols_mostrar].head(20), use_container_width=True)

                csv_buffer = io.StringIO()
                df_export = df_resultados.drop(columns=["_y_real_bin"], errors="ignore")
                df_export.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="⬇️ Descargar Reporte (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="reporte_prediccion_dengue_pediatrico.csv",
                    mime="text/csv",
                )

                y_real_eval = None
                if "_y_real_bin" in df_lote.columns:
                    y_real_eval = df_lote["_y_real_bin"].values
                elif COL_GRUPO in df_lote.columns:
                    y_real_eval = traducir_grupo_real(df_lote[COL_GRUPO]).values

                if y_real_eval is not None:
                    st.markdown("---")
                    st.subheader("📊 Comparativa de Modelos en el Lote")

                    filas_comp = []
                    for nombre_mod, mod in modelos.items():
                        p_m, pr_m = predecir(mod, X_lote)
                        auc_m = (
                            roc_auc_score(y_real_eval, pr_m)
                            if len(np.unique(y_real_eval)) > 1
                            else np.nan
                        )
                        filas_comp.append(
                            {
                                "Modelo": nombre_mod,
                                "Exactitud": round(accuracy_score(y_real_eval, p_m), 2),
                                "Precisión (Complicado)": round(
                                    precision_score(y_real_eval, p_m, zero_division=0), 2
                                ),
                                "Recall (Complicado)": round(
                                    recall_score(y_real_eval, p_m, zero_division=0), 2
                                ),
                                "F1 (Complicado)": round(
                                    f1_score(y_real_eval, p_m, zero_division=0), 2
                                ),
                                "ROC-AUC": round(auc_m, 2) if pd.notna(auc_m) else "N/A",
                            }
                        )
                    st.dataframe(pd.DataFrame(filas_comp), use_container_width=True, hide_index=True)

                    st.markdown(f"### 📈 Detalle: **{modelo_seleccionado}**")
                    acc_sel = accuracy_score(y_real_eval, pred_prin)
                    rec_sel = recall_score(y_real_eval, pred_prin, zero_division=0)
                    cm_sel = confusion_matrix(y_real_eval, pred_prin, labels=[0, 1])

                    mcol1, mcol2 = st.columns(2)
                    with mcol1:
                        st.metric("Exactitud Global", f"{acc_sel * 100:.2f}%")
                        st.metric("Recall Dengue Complicado", f"{rec_sel * 100:.2f}%")
                        st.markdown("**Matriz de Confusión:**")
                        cm_df = pd.DataFrame(
                            cm_sel,
                            columns=["Pred: No Complicado", "Pred: Complicado"],
                            index=["Real: No Complicado", "Real: Complicado"],
                        )
                        st.dataframe(cm_df, use_container_width=True)

                    with mcol2:
                        st.markdown("**Reporte de Clasificación:**")
                        reporte_dict = classification_report(
                            y_real_eval,
                            pred_prin,
                            labels=[0, 1],
                            target_names=["Dengue No Complicado", "Dengue Complicado"],
                            output_dict=True,
                            zero_division=0,
                        )
                        st.dataframe(
                            pd.DataFrame(reporte_dict).transpose().round(2),
                            use_container_width=True,
                        )

                    st.markdown("### 🧠 Interpretación Clínica y Desbalance de Clases")
                    st.info(
                        "El conjunto de datos tiene **desbalance de clases**: predominan los casos "
                        "no complicados. Una exactitud alta puede ocultar casos graves no detectados "
                        "(falsos negativos). En apoyo clínico pediátrico, revisa el **recall** de "
                        "Dengue Complicado, la matriz de confusión y las probabilidades junto con "
                        "los signos de alarma."
                    )


# ---------------------------------------------------------------------------
# Pestaña de rendimiento y factores de riesgo
# ---------------------------------------------------------------------------
def _metricas_modelo(modelo) -> dict:
    pred, prob = predecir(modelo, X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "Exactitud": accuracy_score(y_test, pred),
        "Precisión": precision_score(y_test, pred, zero_division=0),
        "Sensibilidad": recall_score(y_test, pred, zero_division=0),
        "Especificidad": tn / (tn + fp) if (tn + fp) else 0.0,
        "F1": f1_score(y_test, pred, zero_division=0),
        "AUC": roc_auc_score(y_test, prob),
        "_cm": (int(tn), int(fp), int(fn), int(tp)),
    }


def _html_matriz(titulo: str, cm: tuple) -> str:
    tn, fp, fn, tp = cm
    celdas = [
        (tn, "Verdaderos negativos", "ok"),
        (fp, "Falsos positivos", "err"),
        (fn, "Falsos negativos", "err"),
        (tp, "Verdaderos positivos", "ok"),
    ]
    html = "".join(
        f'<div class="celda celda-{tipo}"><div class="celda-num">{n}</div>'
        f'<div class="celda-txt">{txt}</div></div>'
        for n, txt, tipo in celdas
    )
    return f'<div><div class="matriz-titulo">{titulo}</div><div class="matriz-grid">{html}</div></div>'


def _html_rendimiento() -> str:
    nombres = list(modelos.keys())
    m_lr = _metricas_modelo(modelos[nombres[0]])
    m_nn = _metricas_modelo(modelos[nombres[1]])

    filas = ""
    for k in ["Exactitud", "Precisión", "Sensibilidad", "Especificidad", "F1", "AUC"]:
        a, b = m_lr[k] * 100, m_nn[k] * 100
        filas += (
            f'<div class="metrica"><div class="metrica-nombre">{k}</div>'
            f'<div class="fila-barra"><div class="barra-fondo"><div class="barra barra-lr" style="width:{a:.1f}%"></div></div>'
            f'<span class="valor">{a:.1f} %</span></div>'
            f'<div class="fila-barra"><div class="barra-fondo"><div class="barra barra-nn" style="width:{b:.1f}%"></div></div>'
            f'<span class="valor">{b:.1f} %</span></div></div>'
        )

    n_comp_test = int(y_test.sum())
    return (
        '<div class="tarjeta">'
        '<h2 class="tarjeta-titulo">¿Qué tan bien funcionan?</h2>'
        '<div class="leyenda"><span><i class="punto punto-lr"></i>Regresión logística</span>'
        '<span><i class="punto punto-nn"></i>Red neuronal</span></div>'
        f"{filas}"
        '<div class="matrices">'
        f'{_html_matriz("Regresión logística", m_lr["_cm"])}'
        f'{_html_matriz("Red neuronal", m_nn["_cm"])}'
        "</div>"
        f'<p class="nota">Evaluación con {n_test} pacientes de prueba (20 %, partición estratificada) y '
        f"{n_train} de entrenamiento, con umbral de decisión {UMBRAL:.2f} para ambos modelos. "
        f"La muestra de prueba es pequeña ({n_comp_test} casos complicados), así que las diferencias "
        "entre modelos deben leerse con cautela.</p>"
        "</div>"
    )


def _html_factores(top: int = 6) -> str:
    pipe = modelos["Regresión Logística"]
    scaler = pipe.named_steps["standardscaler"]
    clf = pipe.named_steps["logisticregression"]
    coef_std = clf.coef_[0]

    filas = []
    for i, col in enumerate(feature_cols_finales):
        # Binaria: razón de momios por presencia. Continua: por 1 desviación estándar.
        log_or = coef_std[i] / scaler.scale_[i] if col in binarias else coef_std[i]
        filas.append((col, float(np.exp(log_or)), abs(float(coef_std[i]))))

    filas.sort(key=lambda t: t[2], reverse=True)
    filas = filas[:top]
    maximo = max(f[2] for f in filas) or 1.0

    html = ""
    for col, odds, imp in filas:
        clase = "barra-riesgo" if odds >= 1 else "barra-protector"
        html += (
            f'<div class="factor"><div class="factor-cab"><span class="metrica-nombre">'
            f'{NOMBRES_VARIABLES.get(col, col)}</span><span class="valor">{odds:.2f}</span></div>'
            f'<div class="barra-fondo"><div class="barra {clase}" style="width:{imp / maximo * 100:.1f}%"></div></div></div>'
        )

    return (
        '<div class="tarjeta">'
        '<h2 class="tarjeta-titulo">Lo que más pesa en el riesgo</h2>'
        f"{html}"
        '<p class="nota">Razón de momios (regresión logística) por aumento de 1 desviación estándar, '
        "o por presencia si la variable es sí/no. Valores &gt; 1 aumentan el riesgo de complicación "
        "(barra magenta); valores &lt; 1 lo reducen (barra cian).</p>"
        "</div>"
    )


with pestana_rendimiento:
    st.markdown(_html_rendimiento(), unsafe_allow_html=True)
    st.markdown(_html_factores(), unsafe_allow_html=True)