"""Voto mayoritario (self-consistency) sobre varias anotaciones.

Es la forma de reducir la variabilidad: en lugar de fiarme de una sola ejecución
del modelo, lanzo N veces el mismo caso y construyo una anotación de consenso
juntando las N salidas.

Reglas para juntarlas:
- `items_detectados`: un ítem entra si lo eligen al menos `umbral` de las
  ejecuciones válidas (umbral=0.5 es mayoría simple).
- `escalas_afectadas`: salen de los ítems de consenso a través del instrumento,
  así no hay incoherencias ítem-escala.
- `nivel_alerta`: voto mayoritario; los empates se resuelven hacia el nivel más
  alto, que es lo prudente en clínica.
- `nota_clinica` / `justificacion`: se cogen de la ejecución más representativa,
  la que más coincide con los ítems de consenso.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

# Severidad para desempatar el nivel de alerta (mayor = más severo)
_SEVERIDAD = {"bajo": 0, "moderado": 1, "alto": 2}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _jaccard_medio(conjuntos: list[set]) -> float:
    if len(conjuntos) < 2:
        return 1.0
    pares = list(combinations(conjuntos, 2))
    return sum(_jaccard(x, y) for x, y in pares) / len(pares)


def votar(
    anotaciones: list[dict | None],
    instrumento: dict,
    umbral: float = 0.5,
) -> tuple[dict, dict]:
    """Agrega N anotaciones en una de consenso por voto mayoritario.

    Devuelve `(consenso, diagnostico)`. `diagnostico` incluye la frecuencia de
    cada ítem, el acuerdo del nivel y la estabilidad del lote (Jaccard medio).
    """
    validas = [a for a in anotaciones if a]
    item_escala = {it["id"]: it["escala"] for it in instrumento["items"]}
    niveles = list(instrumento["niveles_alerta"])

    if not validas:
        consenso = {
            "items_detectados": [],
            "escalas_afectadas": [],
            "nivel_alerta": niveles[-1],
            "nota_clinica": "",
            "justificacion": "Sin anotaciones válidas para el consenso.",
        }
        return consenso, {"n_muestras": len(anotaciones), "n_validas": 0}

    n = len(validas)

    # 1) Ítems por frecuencia ≥ umbral
    conteo_items: Counter = Counter()
    for a in validas:
        for i in set(a.get("items_detectados", []) or []):
            conteo_items[i] += 1
    frecuencia = {i: c / n for i, c in conteo_items.items()}
    items_consenso = sorted(i for i, f in frecuencia.items() if f >= umbral)

    # 2) Escalas derivadas de los ítems de consenso (coherencia garantizada)
    escalas_consenso = sorted(
        {item_escala[i] for i in items_consenso if i in item_escala}
    )

    # 3) Nivel de alerta por mayoría; empate -> más severo
    conteo_nivel = Counter(a.get("nivel_alerta") for a in validas if a.get("nivel_alerta"))
    if conteo_nivel:
        max_votos = max(conteo_nivel.values())
        candidatos = [niv for niv, c in conteo_nivel.items() if c == max_votos]
        nivel_consenso = max(candidatos, key=lambda x: _SEVERIDAD.get(x, 0))
        acuerdo_nivel = max_votos / n
    else:
        nivel_consenso = niveles[-1]
        acuerdo_nivel = 0.0

    # 4) Nota/justificación de la ejecución más representativa
    objetivo = set(items_consenso)
    representativa = max(
        validas,
        key=lambda a: (
            _jaccard(set(a.get("items_detectados", []) or []), objetivo),
            len((a.get("nota_clinica") or "")),
        ),
    )

    consenso = {
        "items_detectados": items_consenso,
        "escalas_afectadas": escalas_consenso,
        "nivel_alerta": nivel_consenso,
        "nota_clinica": representativa.get("nota_clinica", ""),
        "justificacion": representativa.get("justificacion", ""),
    }

    diagnostico = {
        "n_muestras": len(anotaciones),
        "n_validas": n,
        "umbral": umbral,
        "frecuencia_items": {int(i): round(f, 3) for i, f in sorted(frecuencia.items())},
        "acuerdo_nivel": round(acuerdo_nivel, 3),
        "jaccard_medio_items": round(
            _jaccard_medio([set(a.get("items_detectados", []) or []) for a in validas]), 3
        ),
    }
    return consenso, diagnostico
