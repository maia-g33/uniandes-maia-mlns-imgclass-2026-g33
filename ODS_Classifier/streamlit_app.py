#  We ensure proper path handling in Python
import Definitions
import streamlit as st

from src.ModelController import ModelController

### Setup and configuration

st.set_page_config(page_title="Clasificador de textos ODS", page_icon="🌍", layout="centered")

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

st.title("🌍 Clasificador de textos según los ODS")
st.write(
    "Esta aplicación identifica con cuál de los Objetivos de Desarrollo Sostenible "
    "de la Agenda 2030 se relaciona un texto. Usa el mismo pipeline construido en el "
    "notebook del microproyecto: bolsa de palabras con pesado TF-IDF, reducción de "
    "dimensionalidad con SVD truncado y una máquina de vectores de soporte lineal."
)

with st.sidebar:
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
        - Nelson Fabian Ibanez Piedrahita
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

        st.markdown(
            f"""
            <div style="background-color:{dp.get_color(ods)};padding:22px;border-radius:10px;
                        color:white;text-align:center;margin-top:10px">
                <div style="font-size:19px;opacity:0.9">ODS {ods}</div>
                <div style="font-size:30px;font-weight:700">{dp.get_cat_name(ods)}</div>
                <div style="font-size:15px;opacity:0.9;margin-top:6px">
                    Confianza relativa: {ranking.loc[0, 'Confianza']:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Los 5 objetivos más probables")
        top5 = ranking.head(5).copy()
        top5["Etiqueta"] = [f"ODS {o} - {n}" for o, n in zip(top5["ODS"], top5["Objetivo"])]
        st.bar_chart(top5.set_index("Etiqueta")["Confianza"], horizontal=True, height=260)

        st.dataframe(
            top5[["ODS", "Objetivo", "Puntaje", "Confianza"]].style.format(
                {"Puntaje": "{:+.3f}", "Confianza": "{:.1f}%"}
            ),
            hide_index=True,
        )

        with st.expander("Ver el texto después del preprocesamiento"):
            st.caption(
                "Así es como el modelo ve el texto: en minúsculas, sin tildes, sin "
                "palabras vacías y con cada palabra reducida a su raíz."
            )
            st.code(dp.transform(texto), wrap_lines=True)
