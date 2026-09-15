from preprocesamiento import limpiar_texto


class DataPreprocessing:
    """Informacion de las clases (ODS) y preparacion del texto de entrada."""

    NOMBRES_ODS = {
        1: "Fin de la pobreza",
        2: "Hambre cero",
        3: "Salud y bienestar",
        4: "Educación de calidad",
        5: "Igualdad de género",
        6: "Agua limpia y saneamiento",
        7: "Energía asequible y no contaminante",
        8: "Trabajo decente y crecimiento económico",
        9: "Industria, innovación e infraestructura",
        10: "Reducción de las desigualdades",
        11: "Ciudades y comunidades sostenibles",
        12: "Producción y consumo responsables",
        13: "Acción por el clima",
        14: "Vida submarina",
        15: "Vida de ecosistemas terrestres",
        16: "Paz, justicia e instituciones sólidas",
        17: "Alianzas para lograr los objetivos",
    }

    # Colores oficiales de cada ODS, para que el resultado se reconozca a simple vista.
    COLORES_ODS = {
        1: "#E5243B", 2: "#DDA63A", 3: "#4C9F38", 4: "#C5192D", 5: "#FF3A21",
        6: "#26BDE2", 7: "#FCC30B", 8: "#A21942", 9: "#FD6925", 10: "#DD1367",
        11: "#FD9D24", 12: "#BF8B2E", 13: "#3F7E44", 14: "#0A97D9", 15: "#56C02B",
        16: "#00689D", 17: "#19486A",
    }

    # El modelo se entreno con parrafos de unas 100 palabras.
    MINIMO_PALABRAS = 15

    def transform(self, texto):
        """Devuelve el texto tal como lo ve el modelo despues de la limpieza."""
        return limpiar_texto(texto)

    def get_categories(self):
        return list(self.NOMBRES_ODS.values())

    def get_cat_name(self, ods):
        return self.NOMBRES_ODS.get(int(ods), "")

    def get_color(self, ods):
        return self.COLORES_ODS.get(int(ods), "#555555")

    def validate(self, texto):
        """Devuelve (es_valido, mensaje). El mensaje es None si no hay nada que avisar."""
        texto = texto.strip()
        if not texto:
            return False, "Escribe un texto antes de clasificar."
        n_palabras = len(texto.split())
        if n_palabras < self.MINIMO_PALABRAS:
            return True, (
                f"El texto tiene solo {n_palabras} palabras. El modelo fue entrenado con "
                "párrafos de unas 100 palabras, así que con textos muy cortos la "
                "predicción puede ser poco confiable."
            )
        return True, None
