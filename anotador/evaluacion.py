"""Evaluación de la anotación sin ground truth (viene del cuaderno, fase 6).

Mira propiedades de estructura y coherencia interna que no necesitan que un
clínico haya anotado nada: que los ítems y escalas existan, que el ítem case con
su escala, que el nivel de alerta cuadre con los ítems y la longitud de la nota.
"""

from __future__ import annotations

from typing import Any


def evaluar_anotacion(anotacion: dict, instrumento: dict) -> dict[str, dict[str, Any]]:
    """Evalúa la anotación con métricas que no requieren ground truth."""
    resultados: dict[str, dict[str, Any]] = {}
    reglas = instrumento["reglas_coherencia"]

    item_escala = {it["id"]: it["escala"] for it in instrumento["items"]}
    escalas_validas = set(instrumento["escalas"].keys())

    items = anotacion.get("items_detectados", []) or []
    escalas = anotacion.get("escalas_afectadas", []) or []
    nivel = anotacion.get("nivel_alerta", "")
    nota = anotacion.get("nota_clinica", "") or ""

    # 1. Validez de ítems
    items_invalidos = [i for i in items if i not in item_escala]
    resultados["items_validos"] = {
        "pasa": len(items_invalidos) == 0,
        "detalle": (
            f"{len(items)} ítems, {len(items_invalidos)} inválidos: "
            f"{items_invalidos or 'ninguno'}"
        ),
    }

    # 2. Validez de escalas
    escalas_invalidas = [e for e in escalas if e not in escalas_validas]
    resultados["escalas_validas"] = {
        "pasa": len(escalas_invalidas) == 0,
        "detalle": f"{escalas_invalidas or 'todas válidas'}",
    }

    # 3. Coherencia ítem-escala
    escalas_de_items = {item_escala[i] for i in items if i in item_escala}
    huerfanas = set(escalas) - escalas_de_items
    resultados["coherencia_item_escala"] = {
        "pasa": len(huerfanas) == 0,
        "detalle": f"escalas sin ítem que las respalde: {huerfanas or 'ninguna'}",
    }

    # 4. Coherencia alerta-ítems
    min_alto = reglas["alerta_alto_min_items"]
    min_mod = reglas["alerta_moderado_min_items"]
    if nivel == "alto":
        pasa = len(items) >= min_alto
    elif nivel == "moderado":
        pasa = len(items) >= min_mod
    else:
        pasa = True
    resultados["coherencia_alerta"] = {
        "pasa": pasa,
        "detalle": f"nivel '{nivel}' con {len(items)} ítems",
    }

    # 5. Longitud de nota clínica
    n_palabras = len(nota.split())
    min_p = reglas["nota_clinica_min_palabras"]
    max_p = reglas["nota_clinica_max_palabras"]
    resultados["longitud_nota"] = {
        "pasa": min_p <= n_palabras <= max_p,
        "detalle": f"{n_palabras} palabras (rango {min_p}-{max_p})",
    }

    return resultados


def contar_ok(evaluacion: dict[str, dict[str, Any]]) -> int:
    """Número de métricas superadas."""
    return sum(1 for r in evaluacion.values() if r["pasa"])
