"""Las cuatro formas de pedirle al modelo una salida estructurada.

Cada backend es una opción más del experimento. Se eligen con Config(backend=...):

- directo: POST a /api/generate y parseo el JSON a mano (el del notebook original).
- langchain: ChatOllama + with_structured_output(json_schema).
- tool_calling: /api/chat con `tools` (function calling).
- instructor: cliente compatible con OpenAI + Pydantic con reintentos.

Todos devuelven (anotacion: dict | None, metodo: str, respuesta_cruda: str).
"""

from __future__ import annotations

import json
import re
from typing import Protocol

import requests

from anotador.config import OLLAMA_BASE, OLLAMA_OPENAI_BASE, Config


# Saca el JSON de la respuesta del modelo, usado sobre todo por el backend directo
def parsear_json(texto_crudo: str) -> dict | None:
    """Intenta sacar el primer JSON válido del texto que devuelve el modelo."""
    if not texto_crudo:
        return None
    try:
        return json.loads(texto_crudo.strip())
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", texto_crudo, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    limpio = re.sub(r"```(?:json)?", "", texto_crudo).strip()
    try:
        return json.loads(limpio)
    except json.JSONDecodeError:
        return None


def _opciones_ollama(config: Config) -> dict:
    opts: dict = {"temperature": config.temperature, "num_predict": config.num_predict}
    if config.top_p is not None:
        opts["top_p"] = config.top_p
    if config.seed is not None:
        opts["seed"] = config.seed
    return opts


class Backend(Protocol):
    nombre: str

    def anotar(
        self, system_prompt: str, user_prompt: str, modelo_salida: type | None
    ) -> tuple[dict | None, str, str]: ...


# 1) Directo: requests a /api/generate
class BackendDirecto:
    nombre = "directo"

    def __init__(self, config: Config):
        self.config = config
        self.url = f"{OLLAMA_BASE}/api/generate"

    def anotar(self, system_prompt, user_prompt, modelo_salida=None):
        payload = {
            "model": self.config.modelo,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": _opciones_ollama(self.config),
        }
        r = requests.post(self.url, json=payload, timeout=180)
        r.raise_for_status()
        cruda = r.json()["response"]
        return parsear_json(cruda), "parse_json", cruda


# 2) LangChain: ChatOllama + salida estructurada
class BackendLangchain:
    """Salida estructurada con LangChain.

    `metodo` decide el mecanismo:
    - 'json_schema': el modelo queda obligado a seguir el esquema JSON (Ollama
      con format=<schema>). Funciona aunque el modelo no domine tool calling.
    - 'function_calling': function calling vía la API `tools` de Ollama.
    """

    nombre = "langchain"
    metodo = "json_schema"

    def __init__(self, config: Config):
        self.config = config
        from langchain_ollama import ChatOllama

        kwargs = dict(
            model=config.modelo,
            base_url=OLLAMA_BASE,
            temperature=config.temperature,
            num_predict=config.num_predict,
        )
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.seed is not None:
            kwargs["seed"] = config.seed
        self.llm = ChatOllama(**kwargs)

    def anotar(self, system_prompt, user_prompt, modelo_salida=None):
        from langchain_core.messages import HumanMessage, SystemMessage

        mensajes = [SystemMessage(system_prompt), HumanMessage(user_prompt)]
        if modelo_salida is not None:
            try:
                estructurado = self.llm.with_structured_output(
                    modelo_salida, method=self.metodo
                )
                obj = estructurado.invoke(mensajes)
                if obj is not None:
                    datos = obj if isinstance(obj, dict) else obj.model_dump()
                    return datos, f"structured_output:{self.metodo}", json.dumps(
                        datos, ensure_ascii=False
                    )
            except Exception:
                pass  # caemos a texto libre + parsing
        cruda = self.llm.invoke(mensajes).content
        return parsear_json(cruda), "parse_json", cruda


# 2b) Tool calling llamando directamente a /api/chat
class BackendToolCalling:
    """Function calling directo contra el endpoint `/api/chat`.

    Defino la salida como una función con su esquema y el modelo devuelve los
    argumentos ya estructurados en `message.tool_calls`. Funciona en modelos con
    soporte de tools (Qwen, Llama 3.1+, Hermes, Command-R y Gemma vía Ollama). Si
    el modelo no llama a la función, vuelvo a parsear el texto.

    Uso la API de Ollama directamente y no LangChain porque
    `with_structured_output(method='function_calling')` de langchain-ollama
    devuelve None con estos modelos.
    """

    nombre = "tool_calling"

    def __init__(self, config: Config):
        self.config = config
        self.url = f"{OLLAMA_BASE}/api/chat"

    def anotar(self, system_prompt, user_prompt, modelo_salida=None):
        parametros = (
            modelo_salida.model_json_schema()
            if modelo_salida is not None
            else {"type": "object"}
        )
        tool = {
            "type": "function",
            "function": {
                "name": "registrar_anotacion_clinica",
                "description": "Registra la anotación clínica estructurada de la observación.",
                "parameters": parametros,
            },
        }
        # El prompt compartido pide "responder en JSON"; con tools adjuntas eso
        # hace que el modelo escriba el JSON en el texto en vez de llamar a la
        # función. Añadimos una directiva explícita para forzar la tool call.
        directiva_tool = (
            "\n\nIMPORTANTE: para entregar el resultado DEBES invocar la función "
            "`registrar_anotacion_clinica` con los argumentos correspondientes. "
            "No respondas con texto ni con JSON en el cuerpo del mensaje."
        )
        payload = {
            "model": self.config.modelo,
            "messages": [
                {"role": "system", "content": system_prompt + directiva_tool},
                {"role": "user", "content": user_prompt},
            ],
            "tools": [tool],
            "stream": False,
            # Desactivamos el "thinking" de los modelos de razonamiento (p. ej.
            # Qwen3): si no, el presupuesto de num_predict se consume razonando y
            # el modelo nunca llega a emitir la tool call. La tarea de registrar
            # argumentos no requiere cadena de pensamiento.
            "think": False,
            "options": _opciones_ollama(self.config),
        }
        try:
            r = requests.post(self.url, json=payload, timeout=180)
            r.raise_for_status()
        except requests.HTTPError:
            # Algunos modelos no aceptan el campo `think`: reintentar sin él.
            payload.pop("think", None)
            r = requests.post(self.url, json=payload, timeout=180)
            r.raise_for_status()
        msg = r.json().get("message", {})
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            args = tool_calls[0]["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            return args, "tool_calling", json.dumps(args, ensure_ascii=False)
        # El modelo no emitió tool call: fallback al texto
        contenido = msg.get("content", "")
        return parsear_json(contenido), "fallback_parse", contenido


# 3) Instructor: Pydantic con reintentos automáticos
class BackendInstructor:
    nombre = "instructor"

    def __init__(self, config: Config):
        self.config = config
        import instructor
        from openai import OpenAI

        oai = OpenAI(base_url=OLLAMA_OPENAI_BASE, api_key="ollama")
        self.client = instructor.from_openai(oai, mode=instructor.Mode.JSON)

    def anotar(self, system_prompt, user_prompt, modelo_salida=None):
        if modelo_salida is None:
            from anotador.modelos import AnotacionClinica

            modelo_salida = AnotacionClinica
        kwargs = dict(temperature=self.config.temperature)
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        try:
            obj = self.client.chat.completions.create(
                model=self.config.modelo,
                response_model=modelo_salida,
                max_retries=2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **kwargs,
            )
            datos = obj.model_dump()
            return datos, "instructor_retry", json.dumps(datos, ensure_ascii=False)
        except Exception as e:
            return None, f"error:{type(e).__name__}", str(e)


_BACKENDS = {
    "directo": BackendDirecto,
    "langchain": BackendLangchain,
    "tool_calling": BackendToolCalling,
    "instructor": BackendInstructor,
}

# Backends que dependen de que el modelo sepa hacer tool calling.
BACKENDS_TOOL_CALLING = {"tool_calling"}

# Familias de modelos (por prefijo) que sé que hacen tool calling bien en Ollama.
# Lo comprobé con Qwen3 y Gemma. Esta lista solo se usa si activo el filtro
# opcional (Rejilla.permitir_tool_sin_soporte=False); por defecto está apagado.
MODELOS_TOOL_CALLING = (
    "qwen",          # Qwen 2.5 / Qwen 3 / QwQ
    "gemma",         # Gemma 2 / 3 / 4 (verificado vía Ollama)
    "command-r",     # Cohere Command-R / R+
    "hermes",        # NousResearch Hermes 3
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "mistral-nemo",
    "mistral-large",
    "firefunction",
)


def soporta_tool_calling(modelo: str) -> bool:
    """¿El modelo está en la lista de soporte fiable de tool calling?"""
    m = modelo.lower()
    return any(m.startswith(p) or p in m for p in MODELOS_TOOL_CALLING)


def combinacion_valida(backend: str, modelo: str) -> bool:
    """False si es un backend de tool calling sobre un modelo sin soporte."""
    if backend in BACKENDS_TOOL_CALLING:
        return soporta_tool_calling(modelo)
    return True


def crear_backend(config: Config) -> Backend:
    """Instancia el backend indicado por `config.backend`."""
    try:
        return _BACKENDS[config.backend](config)
    except KeyError:
        raise ValueError(
            f"backend desconocido: '{config.backend}'. "
            f"Opciones: {list(_BACKENDS)}"
        )
