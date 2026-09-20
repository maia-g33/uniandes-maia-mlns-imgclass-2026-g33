import Definitions

import os.path as osp

import joblib
import numpy as np
import pandas as pd

from src.DataPreprocessing import DataPreprocessing


class ModelController:
    """Carga el pipeline entrenado y genera las predicciones."""

    def __init__(self):
        # Pipeline ya encapsula TF-IDF -> TruncatedSVD -> LinearSVC
        self.model_path = osp.join(Definitions.ROOT_DIR, "resources", "models", "model.joblib")
        self.model = joblib.load(self.model_path)
        # En el entrenamiento el pipeline usaba una carpeta temporal como cache
        # Para predecir no se necesita, asi que se desactiva
        self.model.set_params(memory=None)
        self.d_processing = DataPreprocessing()

    def get_categories(self):
        return self.d_processing.get_categories()

    def predict(self, texto):
        """Devuelve el ODS predicho y un DataFrame con todos los ODS ordenados.

        LinearSVC entrena una frontera por ODS (uno contra el resto) y
        `decision_function()` devuelve un puntaje por cada una, proporcional a la
        distancia del texto a esa frontera: positivo si cae del lado del objetivo

        Como esos puntajes no tienen una escala fija, para mostrarlos se pasan con
        softmax a una columna "Afinidad" de 0 a 100. No es una probabilidad: no
        esta calibrada, asi que una afinidad del 47% no significa que el modelo
        acierte 47 de cada 100 veces (su exactitud real es del 88%, y el reparto
        entre 16 objetivos hace que el ganador rara vez pase del 50%). Solo sirve
        para comparar objetivos entre si; lo informativo es cuanto se destaca el
        primero sobre el segundo
        """
        puntajes = self.model.decision_function([texto])[0]
        exp = np.exp(puntajes - puntajes.max())
        ranking = pd.DataFrame({
            "ODS": self.model.classes_.astype(int),
            "Objetivo": [self.d_processing.get_cat_name(c) for c in self.model.classes_],
            "Puntaje": puntajes,
            "Afinidad": 100 * exp / exp.sum(),
        }).sort_values("Puntaje", ascending=False).reset_index(drop=True)
        return int(ranking.loc[0, "ODS"]), ranking
