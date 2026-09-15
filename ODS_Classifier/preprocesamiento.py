"""
Funcion de preprocesamiento de textos del Microproyecto 2 (ODS).

La funcion `limpiar_texto` se deja en este modulo aparte, y no dentro del
notebook, porque el TfidfVectorizer la recibe en el parametro `preprocessor`.
Cuando guardamos el pipeline con joblib, la funcion no se serializa: lo que se
guarda es una referencia al modulo donde fue definida ("preprocesamiento").

Por eso este archivo debe quedar en la misma carpeta que `streamlit_app.py` y
no dentro de `src/`: si cambiara de ubicacion, la referencia pasaria a ser
"src.preprocesamiento" y joblib no podria cargar el modelo.
"""

import unicodedata

import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# En Streamlit Community Cloud el servidor no trae los datos de nltk. Si la
# lista de palabras vacias no esta disponible, la descargamos una sola vez.
try:
    stopwords.words("spanish")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Objetos que se crean una sola vez al importar el modulo, para no volver a
# construirlos en cada texto (son miles de llamadas).
tokenizador = RegexpTokenizer(r"[a-z]+")
stemmer = SnowballStemmer("spanish")

# Palabras vacias del espanol de nltk. Les quitamos las tildes porque en
# `limpiar_texto` normalizamos el texto antes de comparar.
STOPWORDS_ES = {
    unicodedata.normalize("NFKD", palabra).encode("ascii", "ignore").decode("ascii")
    for palabra in stopwords.words("spanish")
}

# Palabras muy frecuentes en este corpus que no ayudan a distinguir un ODS de
# otro (aparecen en textos de casi todos los objetivos). Las agregamos despues
# de revisar la nube de palabras en el notebook.
STOPWORDS_EXTRA = {
    "ods", "objetivo", "objetivos", "desarrollo", "sostenible", "sostenibles",
    "agenda", "meta", "metas", "pais", "paises", "nacional", "internacional",
    "mundial", "mundo", "naciones", "unidas", "onu", "asi", "ademas", "tambien",
    "ser", "puede", "pueden", "debe", "deben", "mas", "menos", "ejemplo",
    "traves", "parte", "partes", "general", "gran", "cada", "aunque", "segun",
    "decir", "hacer", "tener", "sido", "tanto", "muchos", "muchas", "ello",
}

PALABRAS_VACIAS = STOPWORDS_ES | STOPWORDS_EXTRA


def quitar_tildes(texto):
    """Reemplaza las vocales acentuadas y la dieresis por su letra simple."""
    normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in normalizado if not unicodedata.combining(c))


def limpiar_texto(texto):
    """Aplica todo el preprocesamiento a un texto y devuelve una cadena.

    Pasos: minusculas -> quitar tildes -> tokenizar dejando solo letras ->
    quitar palabras vacias y tokens muy cortos -> stemming en espanol.
    """
    # Al quitar las tildes la "ñ" tambien queda convertida en "n", por eso el
    # tokenizador solo necesita buscar letras de la "a" a la "z".
    texto = quitar_tildes(str(texto).lower())
    tokens = tokenizador.tokenize(texto)
    tokens = [t for t in tokens if t not in PALABRAS_VACIAS and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)
