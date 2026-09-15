import Definitions

import os.path as osp

import joblib
import numpy as np
import pandas as pd

from src.DataPreprocessing import DataPreprocessing


class ModelController:
    """Carga el pipeline entrenado y genera las predicciones."""

    def __init__(self):
        # A diferencia de la plantilla (scaler + pca + model), aqui hay un solo
        # archivo: el Pipeline ya encapsula TF-IDF -> TruncatedSVD -> LinearSVC.
        self.model_path = osp.join(Definitions.ROOT_DIR, "resources", "models", "model.joblib")
        self.model = joblib.load(self.model_path)
        # En el entrenamiento el pipeline usaba una carpeta temporal como cache.
        # Para predecir no se necesita, asi que la desactivamos.
        self.model.set_params(memory=None)
        self.d_processing = DataPreprocessing()

    def get_categories(self):
        return self.d_processing.get_categories()

    def predict(self, texto):
        """Devuelve el ODS predicho y un DataFrame con todos los ODS ordenados.

        LinearSVC no entrega probabilidades, sino un puntaje por clase con
        `decision_function()`. Para mostrarlo de forma comprensible lo pasamos a
        una escala de 0 a 100 con softmax: no son probabilidades reales, sino
        una medida relativa de confianza.
        """
        puntajes = self.model.decision_function([texto])[0]
        exp = np.exp(puntajes - puntajes.max())
        ranking = pd.DataFrame({
            "ODS": self.model.classes_.astype(int),
            "Objetivo": [self.d_processing.get_cat_name(c) for c in self.model.classes_],
            "Puntaje": puntajes,
            "Confianza": 100 * exp / exp.sum(),
        }).sort_values("Puntaje", ascending=False).reset_index(drop=True)
        return int(ranking.loc[0, "ODS"]), ranking
