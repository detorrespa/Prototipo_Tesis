"""Inicializa la base de datos y la puebla con datos de ejemplo.

- Crea las tablas (a partir de los modelos ORM).
- Migra el antiguo `data/mensaje_ejemplo.json` a las tablas normalizadas.
- Añade un par de pacientes/entradas más para poder experimentar.

Uso:
    python -m datos.seed            # crea y puebla (idempotente: reinicia datos)
    python -m datos.seed --reset    # borra el fichero .db antes de poblar
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

# Permite ejecutar el script directamente (añade la raíz al path)
RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from anotador.config import RUTA_BD  # noqa: E402
from anotador.db import (  # noqa: E402
    Cuidador,
    Entrada,
    Paciente,
    Session,
    crear_tablas,
)


def _fecha_nac_desde_edad(edad: int, fecha_obs: dt.date) -> dt.date:
    """Aproxima una fecha de nacimiento dado que el JSON antiguo solo tenía edad."""
    return fecha_obs.replace(year=fecha_obs.year - edad)


# Datos de ejemplo (incluye la migración del mensaje_ejemplo.json original)
PACIENTES = [
    # id_paciente, edad_en_primera_obs, sexo, fecha_obs, id_cuidador, rol, texto
    {
        "id_paciente": "PAC_T01",
        "sexo": "masculino",
        "edad": 8,
        "cuidador": ("FAM_T01_01", "madre"),
        "entradas": [
            (
                "2024-01-08",
                "Primera semana con la pastilla. Marcos ha estado más tranquilo en "
                "casa, pero en el colegio la maestra dice que sigue levantándose de la "
                "silla. Le costó dormirse el lunes y el martes. Los deberes los hizo en "
                "40 minutos, antes tardaba casi dos horas.",
            ),
        ],
    },
    {
        "id_paciente": "PAC_T02",
        "sexo": "femenino",
        "edad": 11,
        "cuidador": ("FAM_T02_01", "madre"),
        "entradas": [
            (
                "2024-01-08",
                "Sofía ha tenido una semana difícil. Lloró mucho el jueves porque pensó "
                "que una amiga estaba enfadada con ella. En clase la profesora dice que "
                "se distrae y se pierde en los detalles. El apetito sigue muy bajo con la "
                "medicación.",
            ),
        ],
    },
    {
        "id_paciente": "PAC_T03",
        "sexo": "masculino",
        "edad": 9,
        "cuidador": ("FAM_T03_01", "padre"),
        "entradas": [
            (
                "2024-02-12",
                "Esta semana Hugo ha discutido varias veces con su hermano. Pierde el "
                "estuche casi a diario y olvida los deberes en el colegio aunque los haya "
                "hecho. Cuando se enfada grita y tarda mucho en calmarse.",
            ),
        ],
    },
]


def poblar() -> None:
    crear_tablas()
    with Session() as s:
        for p in PACIENTES:
            if s.get(Paciente, p["id_paciente"]) is not None:
                continue  # ya existe, no duplicar
            primera_fecha = dt.date.fromisoformat(p["entradas"][0][0])
            s.add(
                Paciente(
                    id_paciente=p["id_paciente"],
                    fecha_nacimiento=_fecha_nac_desde_edad(p["edad"], primera_fecha),
                    sexo=p["sexo"],
                )
            )
            id_cuidador, rol = p["cuidador"]
            s.add(
                Cuidador(
                    id_cuidador=id_cuidador,
                    id_paciente=p["id_paciente"],
                    rol=rol,
                )
            )
            for fecha, texto in p["entradas"]:
                s.add(
                    Entrada(
                        id_paciente=p["id_paciente"],
                        id_cuidador=id_cuidador,
                        fecha=dt.date.fromisoformat(fecha),
                        texto=texto,
                    )
                )
        s.commit()

        n_pac = s.query(Paciente).count()
        n_cui = s.query(Cuidador).count()
        n_ent = s.query(Entrada).count()
    print(f"Base de datos lista en: {RUTA_BD}")
    print(f"  Pacientes: {n_pac}  |  Cuidadores: {n_cui}  |  Entradas: {n_ent}")


if __name__ == "__main__":
    if "--reset" in sys.argv and RUTA_BD.exists():
        RUTA_BD.unlink()
        print(f"Borrada BD previa: {RUTA_BD}")
    poblar()
