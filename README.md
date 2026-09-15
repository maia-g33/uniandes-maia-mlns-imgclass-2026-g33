# Clasificador de textos según los ODS

Microproyecto 2 del curso Machine Learning No Supervisado (MAIA, Universidad de los Andes), grupo 33.

Aplicación de Streamlit que recibe un texto libre en español y predice con cuál de los Objetivos de Desarrollo Sostenible se relaciona.

**Aplicación desplegada:** https://uniandes-maia-mlns-imgclass-2026-g33.streamlit.app

## Modelo

El pipeline se construyó en [notebook/Microproyecto2_ODS.ipynb](notebook/Microproyecto2_ODS.ipynb):

1. Preprocesamiento con nltk: minúsculas, eliminación de tildes, stopwords en español y stemming Snowball.
2. TF-IDF con unigramas y bigramas.
3. TruncatedSVD con 300 componentes (análisis semántico latente).
4. LinearSVC con C = 0,5, elegido por GridSearchCV.

| Métrica en el conjunto de prueba | Valor |
|---|---|
| Exactitud | 88,2 % |
| F1 macro | 0,855 |

El conjunto de datos no contiene textos del ODS 17, así que el modelo predice los objetivos del 1 al 16.

## Estructura

```
ODS_Classifier/
  Definitions.py              ruta raíz del proyecto
  preprocesamiento.py         función limpiar_texto usada por el pipeline
  streamlit_app.py            interfaz de usuario
  requirements.txt
  resources/
    batch/streamlit.bat       ejecución local en Windows
    models/model.joblib       pipeline entrenado
  src/
    DataPreprocessing.py      nombres de los ODS y validación del texto
    ModelController.py        carga del modelo y predicción
notebook/
  Microproyecto2_ODS.ipynb
  Microproyecto2_ODS.html
```

## Diferencias con la plantilla del curso

- **Un solo archivo de modelo.** La plantilla guarda `scaler`, `pca` y `model` por separado. Aquí el `Pipeline` de scikit-learn ya encapsula la vectorización, la reducción de dimensionalidad y el clasificador, así que basta con `model.joblib`.
- **`preprocesamiento.py` junto a `streamlit_app.py`.** El pipeline guardado referencia la función `preprocesamiento.limpiar_texto`. Si el archivo se mueve a `src/`, joblib no puede cargar el modelo.
- **Python 3.11.** El modelo se entrenó con scikit-learn 1.9.0, que requiere Python 3.11 o superior. Las versiones de `requirements.txt` están fijadas para que coincidan con las del entrenamiento.

## Despliegue en Streamlit Community Cloud

| Campo | Valor |
|---|---|
| Repository | `maia-g33/uniandes-maia-mlns-imgclass-2026-g33` |
| Branch | `main` |
| Main file path | `ODS_Classifier/streamlit_app.py` |
| App URL | `uniandes-maia-mlns-imgclass-2026-g33` |
| Advanced settings → Python version | `3.11` |

## Ejecución local

Desde la raíz del repositorio:

```
pip install -r ODS_Classifier/requirements.txt
streamlit run ODS_Classifier/streamlit_app.py
```

En Windows también puede usarse `ODS_Classifier/resources/batch/streamlit.bat`, con la variable de entorno `PYTHONPATH` apuntando a la carpeta de Python.
