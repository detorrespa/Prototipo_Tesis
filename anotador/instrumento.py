"""Carga de instrumentos clínicos (BRIEF-2 y compatibles).

El instrumento es un dato de entrada: cambiar de cuestionario = cambiar el
fichero JSON, sin tocar el código de anotación.
"""

from __future__ import annotations

import json
from pathlib import Path

from anotador.config import RUTA_INSTRUMENTO_DEFECTO


def cargar_instrumento(ruta: str | Path | None = None) -> dict:
    """Carga un instrumento clínico desde un fichero JSON."""
    ruta = Path(ruta) if ruta else RUTA_INSTRUMENTO_DEFECTO
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)
