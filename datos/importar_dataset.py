"""Importa el dataset SINTÉTICO de prototipo (TDAH) a las tablas SQL.

El dataset (en `data/`, p. ej. `referencia_sintetica.json`) contiene 30
pacientes sintéticos con seguimiento semanal (24 semanas → 720 observaciones), cada una
con una anotación de referencia. IMPORTANTE: son datos sintéticos sin validez
diagnóstica; esa anotación NO es un ground truth clínico y se guarda solo para
una fase futura. Conviven dos formatos de paciente:

- Formato A (3): identificador `patient_id` y perfil anidado en `profile`.
- Formato B (27): claves planas (`pid`, `edad`, `informante`, ...).

Este script normaliza ambos y puebla: paciente, cuidador, entrada y
referencia_sintetica.

Uso:
    python -m datos.importar_dataset            # importa (idempotente)
    python -m datos.importar_dataset --reset    # borra la BD y reimporta
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from anotador.config import RUTA_BD  # noqa: E402
from anotador.db import (  # noqa: E402
    Cuidador,
    Entrada,
    Paciente,
    ReferenciaSintetica,
    Session,
    crear_tablas,
)

# Nombres posibles del dataset (admite renombrados). Se usa el primero que exista.
_CANDIDATOS_DATASET = [
    "referencia_sintetica.json",
    "dataset_sintetico_prototipo.json",
    "tdah_golden_dataset_v02_completo.json",
]


def _localizar_dataset() -> Path:
    carpeta = RAIZ / "data"
    for nombre in _CANDIDATOS_DATASET:
        ruta = carpeta / nombre
        if ruta.exists():
            return ruta
    # Último recurso: cualquier .json con 'sintetic' o 'dataset' en el nombre
    for ruta in carpeta.glob("*.json"):
        if any(k in ruta.name.lower() for k in ("sintetic", "dataset", "tdah")):
            return ruta
    return carpeta / _CANDIDATOS_DATASET[0]


RUTA_DATASET = _localizar_dataset()


def _normalizar_paciente(p: dict) -> dict:
    """Devuelve un dict homogéneo independientemente del formato (A o B)."""
    if "profile" in p:  # Formato A
        prof = p["profile"]
        inf = prof.get("informante_principal", {})
        return {
            "id_paciente": p["patient_id"],
            "edad": prof["edad"],
            "sexo": prof["sexo"],
            "rol_informante": inf.get("relacion", "otro"),
            "seguimiento": p["seguimiento_semanal"],
        }
    # Formato B (claves planas)
    inf = p.get("informante", {})
    return {
        "id_paciente": p["pid"],
        "edad": p["edad"],
        "sexo": p["sexo"],
        "rol_informante": inf.get("relacion", "otro"),
        "seguimiento": p["seguimiento_semanal"],
    }


def _fecha_nac(edad: int, fecha_ref: dt.date) -> dt.date:
    """Aproxima la fecha de nacimiento a partir de la edad y una fecha de referencia."""
    try:
        return fecha_ref.replace(year=fecha_ref.year - edad)
    except ValueError:  # 29 de febrero
        return fecha_ref.replace(year=fecha_ref.year - edad, day=28)


def importar() -> None:
    if not RUTA_DATASET.exists():
        raise FileNotFoundError(f"No se encuentra el dataset: {RUTA_DATASET}")

    with open(RUTA_DATASET, encoding="utf-8") as f:
        data = json.load(f)

    crear_tablas()
    n_pac = n_cui = n_ent = n_ref = 0

    with Session() as s:
        for bruto in data["pacientes"]:
            p = _normalizar_paciente(bruto)
            if s.get(Paciente, p["id_paciente"]) is not None:
                continue  # ya importado

            seguimiento = [
                w for w in p["seguimiento"] if (w.get("entrada_padre") or "").strip()
            ]
            if not seguimiento:
                continue
            primera_fecha = dt.date.fromisoformat(seguimiento[0]["fecha"])

            s.add(
                Paciente(
                    id_paciente=p["id_paciente"],
                    fecha_nacimiento=_fecha_nac(p["edad"], primera_fecha),
                    sexo=p["sexo"],
                )
            )
            id_cuidador = f"{p['id_paciente']}_C01"
            s.add(
                Cuidador(
                    id_cuidador=id_cuidador,
                    id_paciente=p["id_paciente"],
                    rol=p["rol_informante"],
                )
            )
            n_pac += 1
            n_cui += 1

            for w in seguimiento:
                entrada = Entrada(
                    id_paciente=p["id_paciente"],
                    id_cuidador=id_cuidador,
                    fecha=dt.date.fromisoformat(w["fecha"]),
                    texto=w["entrada_padre"].strip(),
                )
                s.add(entrada)
                s.flush()  # asigna id_entrada
                n_ent += 1

                s.add(
                    ReferenciaSintetica(
                        id_entrada=entrada.id_entrada,
                        semana=w.get("semana"),
                        fase=w.get("fase"),
                        senal_adherencia=w.get("senal_adherencia"),
                        items_detectados=json.dumps(w.get("items_brief2_detectados", [])),
                        escalas_afectadas=json.dumps(w.get("escalas_afectadas", [])),
                        nivel_alerta=w.get("nivel_alerta"),
                        nota_clinica=w.get("nota_clinica"),
                    )
                )
                n_ref += 1
        s.commit()

        tot_pac = s.query(Paciente).count()
        tot_ent = s.query(Entrada).count()
        tot_ref = s.query(ReferenciaSintetica).count()

    print(f"Importación desde: {RUTA_DATASET.name}")
    print(f"  Nuevos → pacientes: {n_pac}  cuidadores: {n_cui}  entradas: {n_ent}  referencia_sintetica: {n_ref}")
    print(f"  Totales en BD → pacientes: {tot_pac}  entradas: {tot_ent}  referencia_sintetica: {tot_ref}")
    print(f"  BD: {RUTA_BD}")


if __name__ == "__main__":
    if "--reset" in sys.argv and RUTA_BD.exists():
        RUTA_BD.unlink()
        print(f"Borrada BD previa: {RUTA_BD}")
    importar()
