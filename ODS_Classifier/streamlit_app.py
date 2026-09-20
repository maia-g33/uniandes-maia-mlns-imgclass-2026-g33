#  We ensure proper path handling in Python
import Definitions
import os.path as osp

import altair as alt
import streamlit as st

from src.ModelController import ModelController

### Setup and configuration

st.set_page_config(page_title="Clasificador de textos ODS", page_icon="🌍", layout="centered")

RUTA_IMAGENES = osp.join(Definitions.ROOT_DIR, "resources", "images")
LOGO_UNIANDES = osp.join(RUTA_IMAGENES, "logo_uniandes.png")
RUEDA_ODS = osp.join(RUTA_IMAGENES, "ods_rueda.png")

EJEMPLOS = {
    "— Escribe tu propio texto —": "",
    "Agua potable en zonas rurales": (
        "Es urgente ampliar la cobertura de acueducto y alcantarillado en las zonas "
        "rurales del país, donde miles de familias todavía no tienen acceso a agua "
        "potable y las aguas residuales se vierten sin tratamiento a los ríos. Se "
        "necesita una mejor gestión de las cuencas y de los recursos hídricos."
    ),
    "Brecha salarial de género": (
        "Las mujeres siguen recibiendo salarios menores que los hombres por realizar "
        "el mismo trabajo y están claramente subrepresentadas en los cargos directivos "
        "de las empresas, además de asumir la mayor parte del trabajo de cuidado no "
        "remunerado en el hogar."
    ),
    "Energías renovables": (
        "La instalación de paneles solares y parques eólicos permitiría reducir la "
        "dependencia de los combustibles fósiles, bajar las emisiones del sector "
        "eléctrico y llevar electricidad limpia y asequible a comunidades aisladas "
        "que hoy dependen de plantas diésel."
    ),
    "Deserción escolar": (
        "Muchos niños y niñas abandonan la escuela primaria por la falta de docentes "
        "capacitados y de material educativo adecuado. Mejorar la calidad del "
        "aprendizaje y garantizar la permanencia de los estudiantes en el sistema "
        "escolar es una prioridad."
    ),
}


### My vars

@st.cache_resource
def get_controller():
    """Carga el modelo una sola vez y lo reutiliza entre interacciones."""
    return ModelController()


ctrl = get_controller()
dp = ctrl.d_processing

### My UI starting here

col_rueda, col_titulo = st.columns([1, 5], vertical_alignment="center")
col_rueda.image(RUEDA_ODS, width=110)
col_titulo.title("Clasificador de textos según los ODS")

# Franja con los 17 colores oficiales de los ODS.
st.markdown(
    '<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;margin:4px 0 18px">'
    + "".join(f'<div style="flex:1;background:{c}"></div>' for c in dp.COLORES_ODS.values())
    + "</div>",
    unsafe_allow_html=True,
)

st.write(
    "Esta aplicación identifica con cuál de los Objetivos de Desarrollo Sostenible "
    "de la Agenda 2030 se relaciona un texto. Usa el mismo pipeline construido en el "
    "notebook del microproyecto: bolsa de palabras con pesado TF-IDF, reducción de "
    "dimensionalidad con SVD truncado y una máquina de vectores de soporte lineal."
)

with st.sidebar:
    st.image(LOGO_UNIANDES, width=85)
    st.header("Sobre el modelo")
    st.markdown(
        """
        **Datos:** 9.656 textos en español del *OSDG Community Dataset*.

        **Pipeline:** TF-IDF (unigramas y bigramas) → TruncatedSVD (300 componentes)
        → LinearSVC (C = 0,5).

        **Desempeño en el conjunto de prueba:**
        - Exactitud: 88,2%
        - F1 macro: 0,855

        **Nota:** el conjunto de datos no contiene textos del ODS 17, por lo que el
        modelo solo puede predecir los objetivos del 1 al 16.
        """
    )
    st.divider()
    st.markdown(
        """
        **Diseñadores:**
        - Giovanny Andres Jurado Torres
        - Nelson Fabian Ibañez Piedrahita
        """
    )

ejemplo = st.selectbox("Puedes partir de un ejemplo o escribir tu propio texto:", list(EJEMPLOS))
texto = st.text_area(
    "Texto a clasificar",
    value=EJEMPLOS[ejemplo],
    height=200,
    placeholder="Escribe o pega aquí el texto que quieres clasificar...",
)

if st.button("Clasificar texto", type="primary"):
    es_valido, mensaje = dp.validate(texto)
    if not es_valido:
        st.warning(mensaje)
    else:
        if mensaje:
            st.info(mensaje)

        ods, ranking = ctrl.predict(texto.strip())
        # Diferencia de puntaje con el segundo objetivo: es lo que de verdad indica
        # si el modelo tiene un ganador claro o si el texto quedo entre varios ODS.
        ventaja = ranking.loc[0, "Puntaje"] - ranking.loc[1, "Puntaje"]

        # Letra oscura sobre los colores claros (p. ej. el amarillo del ODS 7) para que se lea.
        color = dp.get_color(ods)
        r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
        color_letra = "#1F2A37" if 0.299 * r + 0.587 * g + 0.114 * b > 160 else "white"

        st.markdown(
            f"""
            <div style="background-color:{color};padding:22px;border-radius:10px;
                        color:{color_letra};text-align:center;margin-top:10px">
                <div style="font-size:19px;opacity:0.9">ODS {ods}</div>
                <div style="font-size:30px;font-weight:700">{dp.get_cat_name(ods)}</div>
                <div style="font-size:15px;opacity:0.9;margin-top:6px">
                    Afinidad relativa: {ranking.loc[0, 'Afinidad']:.1f}%
                    &nbsp;·&nbsp; ventaja de {ventaja:.2f} sobre el ODS {ranking.loc[1, 'ODS']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Umbral medido sobre los 1.932 textos de prueba: con una ventaja menor a 0,5
        # el modelo acierta el 59% de las veces, frente al 96% cuando es mayor.
        if ventaja < 0.5:
            st.info(
                f"La ventaja sobre el ODS {ranking.loc[1, 'ODS']} es de solo {ventaja:.2f}, "
                "así que el texto quedó cerca de la frontera entre varios objetivos. "
                "En estos casos el modelo acierta cerca del 59% de las veces, frente al 96% "
                "cuando la ventaja es amplia: conviene revisar los primeros de la lista y no "
                "quedarse solo con el primero."
            )

        st.subheader("Los 5 objetivos con mayor afinidad")
        top5 = ranking.head(5).copy()
        top5["Etiqueta"] = [f"ODS {o} - {n}" for o, n in zip(top5["ODS"], top5["Objetivo"])]
        top5["Color"] = [dp.get_color(o) for o in top5["ODS"]]
        grafico = alt.Chart(top5).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Afinidad:Q", title="Afinidad relativa (%)"),
            y=alt.Y("Etiqueta:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color("Color:N", scale=None),
            tooltip=["Etiqueta", alt.Tooltip("Afinidad:Q", format=".1f")],
        ).properties(height=260)
        st.altair_chart(grafico, use_container_width=True)

        st.dataframe(
            top5[["ODS", "Objetivo", "Puntaje", "Afinidad"]].style.format(
                {"Puntaje": "{:+.3f}", "Afinidad": "{:.1f}%"}
            ),
            hide_index=True,
        )

        with st.expander("¿Cómo se leen el puntaje y la afinidad?"):
            st.markdown(
                """
                El modelo entrena **una frontera por cada ODS**, y para un texto nuevo
                calcula un **puntaje** por objetivo, proporcional a la distancia del
                texto a esa frontera. Un puntaje positivo significa que el texto cae
                del lado del objetivo; uno negativo, que no. Si los 16 puntajes salen
                negativos, ninguna frontera reclama el texto y el modelo entrega el
                objetivo *menos descartado*.

                La **afinidad** reparte esos puntajes en porcentajes que suman 100,
                solo para poderlos comparar de un vistazo. **No es una probabilidad
                de acierto:** el modelo acierta el 88,2% de las veces, pero incluso en
                un texto muy claro la afinidad del ganador rara vez pasa del 50%,
                porque el porcentaje se reparte entre 16 objetivos. En el otro extremo,
                un valor cercano al 6,25% (1 entre 16) equivale a "sin información".

                Lo que de verdad indica seguridad es la **ventaja**: la diferencia de
                puntaje entre el primer objetivo y el segundo. En los textos de prueba
                la ventaja mediana es de 1,23; cuando supera 0,5 el modelo acierta el
                96% de las veces, y cuando no la alcanza el acierto cae al 59%. Una
                ventaja pequeña
                significa que el texto está entre dos o más objetivos, algo normal
                cuando comparten vocabulario, como los ODS 1, 8, 9 y 10, que hablan
                todos de ingresos, empleo y crecimiento.
                """
            )

        with st.expander("Ver el texto después del preprocesamiento"):
            st.caption(
                "Así es como el modelo ve el texto: en minúsculas, sin tildes, sin "
                "palabras vacías y con cada palabra reducida a su raíz."
            )
            st.code(dp.transform(texto), wrap_lines=True)

### Pie de pagina

st.divider()
st.markdown(
    '<div style="text-align:center;font-size:0.85rem;opacity:0.75">'
    "Machine Learning no Supervisado · 2026<br>Universidad de los Andes"
    "</div>",
    unsafe_allow_html=True,
)
