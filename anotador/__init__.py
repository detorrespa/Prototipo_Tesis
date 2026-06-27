"""Paquete del prototipo de anotación clínica con LLM (TDAH / BRIEF-2).

Separa la lógica de dominio (instrumento, evaluación) del acceso a datos
(SQL vía SQLAlchemy) y de la capa de transporte hacia el modelo
(backends: directo / langchain / instructor).

El cuaderno `prototipo_anotador_tdah.ipynb` queda como capa de demo y
visualización; toda la lógica reutilizable vive aquí.
"""

from anotador.config import Config
from anotador.modelos import AnotacionClinica, Mensaje, construir_modelo_anotacion

__all__ = [
    "Config",
    "Mensaje",
    "AnotacionClinica",
    "construir_modelo_anotacion",
]
