"""El pipeline de anotación, de principio a fin.

Junta las piezas: carga el mensaje de la base, arma los prompts, llama al
backend elegido, evalúa la salida y (si se quiere) la guarda en la tabla
`anotacion`. Lo usan tanto el cuaderno como el motor de simulaciones.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from anotador.backends import crear_backend
from anotador.config import Config
from anotador.consenso import votar
from anotador.evaluacion import contar_ok, evaluar_anotacion
from anotador.modelos import Mensaje, construir_modelo_anotacion
from anotador.prompts import construir_prompt_sistema, construir_prompt_usuario
from anotador.repositorio import cargar_entrada, guardar_anotacion


@dataclass
class Resultado:
    anotacion: dict | None
    evaluacion: dict
    n_metricas_ok: int
    formato_ok: bool
    latencia_s: float
    metodo: str
    respuesta_cruda: str
    id_anotacion: int | None = None
    diagnostico: dict = field(default_factory=dict)


def anotar(
    entrada: int | Mensaje,
    instrumento: dict,
    config: Config | None = None,
    repeticion: int = 0,
    persistir: bool = True,
) -> Resultado:
    """Anota una entrada con la configuración dada.

    `entrada` puede ser un id (se carga de la BD) o un `Mensaje` ya construido.
    """
    config = config or Config()
    mensaje = cargar_entrada(entrada) if isinstance(entrada, int) else entrada

    system = construir_prompt_sistema(instrumento, config.estrategia_prompt)
    user = construir_prompt_usuario(mensaje)
    modelo_salida = construir_modelo_anotacion(instrumento)
    backend = crear_backend(config)

    t0 = time.time()
    anotacion, metodo, cruda = backend.anotar(system, user, modelo_salida)
    latencia = time.time() - t0

    formato_ok = anotacion is not None
    evaluacion = evaluar_anotacion(anotacion, instrumento) if formato_ok else {}
    n_ok = contar_ok(evaluacion) if formato_ok else 0

    id_anot = None
    if persistir and mensaje.id_entrada is not None:
        id_anot = guardar_anotacion(
            id_entrada=mensaje.id_entrada,
            instrumento_nombre=instrumento["nombre"],
            config=config,
            repeticion=repeticion,
            formato_ok=formato_ok,
            anotacion=anotacion,
            metricas=evaluacion or None,
            n_metricas_ok=n_ok,
            latencia_s=latencia,
            respuesta_cruda=cruda,
        )

    return Resultado(
        anotacion=anotacion,
        evaluacion=evaluacion,
        n_metricas_ok=n_ok,
        formato_ok=formato_ok,
        latencia_s=latencia,
        metodo=metodo,
        respuesta_cruda=cruda,
        id_anotacion=id_anot,
    )


def anotar_consenso(
    entrada: int | Mensaje,
    instrumento: dict,
    config: Config | None = None,
    n_muestras: int = 5,
    umbral: float = 0.5,
    repeticion: int = 0,
    persistir: bool = True,
) -> Resultado:
    """Anota por voto mayoritario (self-consistency).

    Ejecuta el backend `n_muestras` veces sobre la misma entrada y junta las
    salidas en una anotación de consenso. Reduce la variabilidad a cambio de
    gastar `n_muestras` veces más cómputo. Se guarda como una sola fila con
    `agregacion='consenso_voto'`.
    """
    config = config or Config()
    mensaje = cargar_entrada(entrada) if isinstance(entrada, int) else entrada

    system = construir_prompt_sistema(instrumento, config.estrategia_prompt)
    user = construir_prompt_usuario(mensaje)
    modelo_salida = construir_modelo_anotacion(instrumento)
    backend = crear_backend(config)

    t0 = time.time()
    muestras = [backend.anotar(system, user, modelo_salida)[0] for _ in range(n_muestras)]
    latencia = time.time() - t0

    consenso, diagnostico = votar(muestras, instrumento, umbral=umbral)
    formato_ok = diagnostico.get("n_validas", 0) > 0
    evaluacion = evaluar_anotacion(consenso, instrumento) if formato_ok else {}
    n_ok = contar_ok(evaluacion) if formato_ok else 0

    id_anot = None
    if persistir and mensaje.id_entrada is not None:
        id_anot = guardar_anotacion(
            id_entrada=mensaje.id_entrada,
            instrumento_nombre=instrumento["nombre"],
            config=config,
            repeticion=repeticion,
            formato_ok=formato_ok,
            anotacion=consenso,
            metricas=evaluacion or None,
            n_metricas_ok=n_ok,
            latencia_s=latencia,
            respuesta_cruda=json.dumps(diagnostico, ensure_ascii=False),
            agregacion="consenso_voto",
            n_muestras=n_muestras,
        )

    return Resultado(
        anotacion=consenso,
        evaluacion=evaluacion,
        n_metricas_ok=n_ok,
        formato_ok=formato_ok,
        latencia_s=latencia,
        metodo="consenso_voto",
        respuesta_cruda=json.dumps(diagnostico, ensure_ascii=False),
        id_anotacion=id_anot,
        diagnostico=diagnostico,
    )
