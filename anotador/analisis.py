"""Análisis de la variabilidad y la fiabilidad de las anotaciones.

Carga la tabla `anotacion` y calcula métricas de fiabilidad tomadas de la
psicometría, para responder a la pregunta de fondo: ¿es estable lo que anota el
modelo aunque sea probabilístico, y qué parámetros lo estabilizan?

Métricas principales:
- Consistencia entre las N repeticiones de una misma (entrada × configuración):
  Jaccard para ítems y escalas, acuerdo modal para el nivel de alerta.
- Alpha de Krippendorff: fiabilidad entre réplicas del nivel de alerta (ordinal),
  tratando cada repetición como un codificador.
- Resúmenes por parámetro: cómo cambian consistencia, calidad, fallo de formato
  y latencia al mover cada variable.
"""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
from sqlalchemy import select

from anotador.db import Anotacion, Session

# Columnas que identifican una configuración del experimento
COLS_CONFIG = [
    "backend", "modelo", "temperature", "top_p", "seed",
    "estrategia_prompt", "agregacion", "n_muestras",
]

# Orden ordinal de los niveles de alerta (para Krippendorff y comparaciones)
ORDEN_NIVEL = {"bajo": 0, "moderado": 1, "alto": 2}


def cargar_df() -> pd.DataFrame:
    """Carga la tabla `anotacion` en un DataFrame, parseando los campos JSON."""
    with Session() as s:
        filas = s.execute(select(Anotacion)).scalars().all()
        registros = []
        for a in filas:
            registros.append(
                {
                    "id_anotacion": a.id_anotacion,
                    "id_entrada": a.id_entrada,
                    "instrumento": a.instrumento,
                    "backend": a.backend,
                    "modelo": a.modelo,
                    "temperature": a.temperature,
                    "top_p": a.top_p,
                    "seed": a.seed,
                    "estrategia_prompt": a.estrategia_prompt,
                    "repeticion": a.repeticion,
                    "agregacion": a.agregacion,
                    "n_muestras": a.n_muestras,
                    "formato_ok": bool(a.formato_ok),
                    "items_detectados": json.loads(a.items_detectados or "[]"),
                    "escalas_afectadas": json.loads(a.escalas_afectadas or "[]"),
                    "nivel_alerta": a.nivel_alerta,
                    "n_metricas_ok": a.n_metricas_ok,
                    "latencia_s": a.latencia_s,
                }
            )
    df = pd.DataFrame(registros)
    # top_p/seed pueden ser None: los normalizamos para poder agrupar
    if not df.empty:
        df["top_p"] = df["top_p"].fillna(-1.0)
        df["seed"] = df["seed"].fillna(-1).astype(int)
    return df


def _jaccard(a: set, b: set) -> float:
    """Índice de Jaccard. Dos conjuntos vacíos se consideran idénticos (1.0)."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _jaccard_medio(conjuntos: list[set]) -> float:
    """Jaccard medio sobre todos los pares de una lista de conjuntos."""
    if len(conjuntos) < 2:
        return 1.0
    pares = list(combinations(conjuntos, 2))
    return float(np.mean([_jaccard(x, y) for x, y in pares]))


def _acuerdo_modal(valores: list) -> float:
    """Fracción de valores iguales al valor más frecuente (moda)."""
    valores = [v for v in valores if v is not None]
    if not valores:
        return np.nan
    serie = pd.Series(valores)
    return serie.value_counts().iloc[0] / len(serie)


def consistencia_por_grupo(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-consistencia entre repeticiones de cada (entrada × configuración).

    Devuelve una fila por grupo con:
    - jaccard_items / jaccard_escalas: estabilidad de la selección.
    - acuerdo_nivel: acuerdo modal del nivel de alerta.
    - n_rep, tasa_formato_ok, media_metricas, media_latencia.
    """
    if df.empty:
        return df
    filas = []
    claves = ["id_entrada", *COLS_CONFIG]
    for clave, g in df.groupby(claves, dropna=False):
        items = [set(x) for x in g["items_detectados"]]
        escalas = [set(x) for x in g["escalas_afectadas"]]
        fila = dict(zip(claves, clave))
        fila.update(
            {
                "n_rep": len(g),
                "jaccard_items": _jaccard_medio(items),
                "jaccard_escalas": _jaccard_medio(escalas),
                "acuerdo_nivel": _acuerdo_modal(list(g["nivel_alerta"])),
                "tasa_formato_ok": g["formato_ok"].mean(),
                "media_metricas": g["n_metricas_ok"].mean(),
                "media_latencia": g["latencia_s"].mean(),
            }
        )
        filas.append(fila)
    return pd.DataFrame(filas)


def krippendorff_nivel(df: pd.DataFrame) -> pd.DataFrame:
    """Krippendorff's alpha (ordinal) del nivel de alerta, por configuración.

    Trata cada repetición como un codificador y cada entrada como una unidad.
    Requiere ≥2 repeticiones y ≥2 entradas para ser informativo.
    """
    try:
        import krippendorff
    except ImportError as e:  # pragma: no cover
        raise ImportError("Instala 'krippendorff' para esta métrica") from e

    if df.empty:
        return df

    resultados = []
    for clave, g in df.groupby(COLS_CONFIG, dropna=False):
        # Matriz codificadores (repeticiones) × unidades (entradas)
        pivot = (
            g.assign(nivel_cod=g["nivel_alerta"].map(ORDEN_NIVEL))
            .pivot_table(
                index="repeticion",
                columns="id_entrada",
                values="nivel_cod",
                aggfunc="first",
            )
        )
        datos = pivot.to_numpy(dtype=float)
        alpha = np.nan
        if datos.shape[0] >= 2 and datos.shape[1] >= 2:
            try:
                alpha = krippendorff.alpha(
                    reliability_data=datos, level_of_measurement="ordinal"
                )
            except Exception:
                alpha = np.nan
        resultados.append({**dict(zip(COLS_CONFIG, clave)), "alpha_nivel": alpha})
    return pd.DataFrame(resultados)


def resumen_por_parametro(df: pd.DataFrame, parametro: str) -> pd.DataFrame:
    """Agrupa consistencia, calidad, fallo de formato y latencia por un parámetro.

    `parametro` puede ser cualquier columna de configuración (por ejemplo
    'temperature', 'modelo' o 'backend').
    """
    cons = consistencia_por_grupo(df)
    if cons.empty:
        return cons
    agg = (
        cons.groupby(parametro)
        .agg(
            jaccard_items=("jaccard_items", "mean"),
            jaccard_escalas=("jaccard_escalas", "mean"),
            acuerdo_nivel=("acuerdo_nivel", "mean"),
            media_metricas=("media_metricas", "mean"),
            n_grupos=("n_rep", "size"),
        )
        .reset_index()
    )
    # Tasa de fallo de formato y latencia desde el df bruto
    extra = (
        df.groupby(parametro)
        .agg(
            tasa_fallo_formato=("formato_ok", lambda x: 1 - x.mean()),
            media_latencia=("latencia_s", "mean"),
        )
        .reset_index()
    )
    return agg.merge(extra, on=parametro)
