"""Acceso a datos: convierte filas de la BD en objetos del dominio.

El pipeline no sabe nada de SQL: pide un `Mensaje` por id de entrada y el
repositorio se encarga del JOIN paciente + cuidador + entrada y de calcular
la edad en el momento de la observación.
"""

from __future__ import annotations

import datetime as dt
import json

from sqlalchemy import select

from anotador.config import Config
from anotador.db import Anotacion, Entrada, ReferenciaSintetica, Session
from anotador.modelos import Mensaje


def _edad(fecha_nac: dt.date, fecha_obs: dt.date) -> int:
    """Edad en años cumplidos en la fecha de la observación."""
    return (
        fecha_obs.year
        - fecha_nac.year
        - ((fecha_obs.month, fecha_obs.day) < (fecha_nac.month, fecha_nac.day))
    )


def cargar_entrada(id_entrada: int) -> Mensaje:
    """Reconstruye un `Mensaje` a partir de las tres tablas de entrada."""
    with Session() as s:
        e = s.get(Entrada, id_entrada)
        if e is None:
            raise ValueError(f"No existe la entrada {id_entrada}")
        return Mensaje(
            id_paciente=e.id_paciente,
            id_familiar=e.id_cuidador,
            rol_informante=e.cuidador.rol,
            fecha=e.fecha.isoformat(),
            edad=_edad(e.paciente.fecha_nacimiento, e.fecha),
            sexo=e.paciente.sexo,
            entrada=e.texto,
            id_entrada=e.id_entrada,
        )


def listar_entradas() -> list[Mensaje]:
    """Todas las entradas como mensajes (útil para barridos de simulación)."""
    with Session() as s:
        ids = s.execute(select(Entrada.id_entrada).order_by(Entrada.id_entrada)).scalars().all()
    return [cargar_entrada(i) for i in ids]


def semanas_disponibles() -> list[int]:
    """Semanas de seguimiento presentes en el dataset (vía referencia_sintetica).

    El dataset sintético es longitudinal: cada paciente aporta una entrada por
    semana. La semana vive en `referencia_sintetica`, ligada 1:1 a `entrada`.
    """
    with Session() as s:
        semanas = s.execute(
            select(ReferenciaSintetica.semana)
            .where(ReferenciaSintetica.semana.is_not(None))
            .distinct()
            .order_by(ReferenciaSintetica.semana)
        ).scalars().all()
    return list(semanas)


def ids_entradas_semana(semana: int) -> list[int]:
    """Ids de las entradas de una semana concreta (todos los pacientes).

    Es la unidad natural del experimento: la anotación del reporte semanal.
    Cada semana aporta tantas entradas como pacientes tenga el dataset.
    """
    with Session() as s:
        ids = s.execute(
            select(ReferenciaSintetica.id_entrada)
            .where(ReferenciaSintetica.semana == semana)
            .order_by(ReferenciaSintetica.id_entrada)
        ).scalars().all()
    return list(ids)


def guardar_anotacion(
    id_entrada: int,
    instrumento_nombre: str,
    config: Config,
    repeticion: int,
    formato_ok: bool,
    anotacion: dict | None,
    metricas: dict | None,
    n_metricas_ok: int | None,
    latencia_s: float | None,
    respuesta_cruda: str | None = None,
    agregacion: str = "individual",
    n_muestras: int = 1,
) -> int:
    """Persiste el resultado de una ejecución del pipeline. Devuelve su id."""
    anot = anotacion or {}
    fila = Anotacion(
        id_entrada=id_entrada,
        creada_en=dt.datetime.now(),
        instrumento=instrumento_nombre,
        backend=config.backend,
        modelo=config.modelo,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        estrategia_prompt=config.estrategia_prompt,
        repeticion=repeticion,
        agregacion=agregacion,
        n_muestras=n_muestras,
        formato_ok=int(formato_ok),
        items_detectados=json.dumps(anot.get("items_detectados", [])),
        escalas_afectadas=json.dumps(anot.get("escalas_afectadas", [])),
        nivel_alerta=anot.get("nivel_alerta"),
        nota_clinica=anot.get("nota_clinica"),
        justificacion=anot.get("justificacion"),
        metricas=json.dumps(metricas) if metricas is not None else None,
        n_metricas_ok=n_metricas_ok,
        latencia_s=latencia_s,
        respuesta_cruda=respuesta_cruda,
    )
    with Session() as s:
        s.add(fila)
        s.commit()
        return fila.id_anotacion
