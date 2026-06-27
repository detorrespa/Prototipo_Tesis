"""Construcción de prompts a partir del instrumento y del mensaje.

Portado del cuaderno. Separa la *plantilla* de los *datos*:
el instrumento genera el prompt de sistema; el mensaje genera el de usuario.
La estrategia (zero-shot / few-shot / cot) modula el prompt de sistema.
"""

from __future__ import annotations

from anotador.modelos import Mensaje

_INSTRUCCION_COT = (
    "\nAntes de responder, razona internamente paso a paso qué ítems encajan, "
    "pero la salida final debe ser ÚNICAMENTE el objeto JSON pedido."
)


def construir_prompt_sistema(instrumento: dict, estrategia: str = "zero_shot") -> str:
    """Genera el prompt de sistema a partir de la definición del instrumento."""
    catalogo = "\n".join(
        f"  {it['id']}: [{it['escala']}] {it['texto']}" for it in instrumento["items"]
    )
    escalas_desc = "\n".join(
        f"  - {e}: {d}" for e, d in instrumento["escalas"].items()
    )
    niveles = " | ".join(instrumento["niveles_alerta"])
    n0 = instrumento["niveles_alerta"]

    extra = _INSTRUCCION_COT if estrategia == "cot" else ""

    return f"""Eres un {instrumento['rol_anotador']}.

Tu tarea es analizar el texto libre de observación de un padre/madre sobre su hijo/a
y producir una anotación clínica estructurada en formato JSON, basada en el instrumento
{instrumento['nombre']}.

## CATÁLOGO DE ÍTEMS ({len(instrumento['items'])} ítems)
{catalogo}

## ESCALAS
{escalas_desc}

## NIVELES DE ALERTA
{niveles}

## INSTRUCCIONES DE SALIDA
Responde ÚNICAMENTE con un objeto JSON válido, sin texto antes ni después, sin markdown.
Estructura requerida:
{{
  "items_detectados": [lista de números de ítem observables en el texto],
  "escalas_afectadas": [lista de escalas correspondientes],
  "nivel_alerta": "{n0[0]}|{n0[1]}|{n0[2]}",
  "nota_clinica": "resumen clínico de 1-3 frases para el médico",
  "justificacion": "explicación del razonamiento (para auditoría)"
}}{extra}"""


def construir_prompt_usuario(mensaje: Mensaje | dict) -> str:
    """Genera el prompt de usuario con el contexto del paciente y su observación."""
    m = mensaje  # admite Mensaje (acceso por clave) o dict
    return f"""## CONTEXTO DEL PACIENTE
- Edad: {m['edad']} años
- Sexo: {m['sexo']}
- Informante: {m['rol_informante']}

## TEXTO DEL PADRE/MADRE
\"\"\"{m['entrada']}\"\"\"

Analiza el texto y genera el JSON de anotación clínica."""
