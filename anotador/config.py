"""Configuración central: conexión a Ollama, a la base de datos y rutas.

Todo configurable por variables de entorno para no atar el prototipo a una
máquina concreta (mercurio usa el puerto 11002, no el 11434 por defecto).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Raíz del proyecto (carpeta que contiene este paquete)
RAIZ = Path(__file__).resolve().parent.parent

# Ollama
OLLAMA_PORT = os.environ.get("OLLAMA_PORT","11002")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST_IP", "127.0.0.1")
OLLAMA_BASE = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
# Endpoint compatible con OpenAI que expone Ollama (lo usan langchain e instructor)
OLLAMA_OPENAI_BASE = f"{OLLAMA_BASE}/v1"

MODELO_POR_DEFECTO = os.environ.get("OLLAMA_MODEL", "gemma4:26b")

# Base de datos
RUTA_BD = Path(os.environ.get("ANOTADOR_DB", RAIZ / "datos" / "anotador.db"))
URL_BD = os.environ.get("ANOTADOR_DB_URL", f"sqlite:///{RUTA_BD}")

# Rutas de recursos
RUTA_ESQUEMA_SQL = RAIZ / "datos" / "esquema.sql"
RUTA_INSTRUMENTO_DEFECTO = RAIZ / "instrumentos" / "brief2.json"


@dataclass(frozen=True)
class Config:
    """Una combinación concreta de parámetros para una ejecución del pipeline.

    Cada campo es una variable del experimento. La hago inmutable para poder
    usarla como clave al agrupar resultados.
    """

    backend: str = "directo"            # directo | langchain | instructor
    modelo: str = MODELO_POR_DEFECTO
    temperature: float = 0.1
    top_p: float | None = None
    seed: int | None = None
    num_predict: int = 2048
    estrategia_prompt: str = "zero_shot"  # zero_shot | few_shot | cot

    def como_dict(self) -> dict:
        return {
            "backend": self.backend,
            "modelo": self.modelo,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "num_predict": self.num_predict,
            "estrategia_prompt": self.estrategia_prompt,
        }
