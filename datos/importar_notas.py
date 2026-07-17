"""Importa el dataset de notas de diario a la base SQLite.

Lee data/notas_diario.json y data/pacientes_contexto.json y puebla las tablas
paciente, cuidador, entrada y referencia_sintetica. Los ref_items de cada nota
(ground truth sintético) van a referencia_sintetica; el anotador NO los lee.

Autocontenido: solo usa la librería estándar (sqlite3, json).

Uso:
    python datos/importar_notas.py            # importa (borra antes las notas
                                              # previas de estos 5 pacientes)
    python datos/importar_notas.py --bd otra.db
"""

import argparse
import json
import sqlite3
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
OFFSET_ID = 10000  # ids de entrada que no chocan con el dataset semanal (1-720)


def importar(ruta_bd: str) -> None:
    notas = json.load(open(RAIZ / "data" / "notas_diario.json", encoding="utf-8"))["notas"]
    pacientes = json.load(open(RAIZ / "data" / "pacientes_contexto.json", encoding="utf-8"))["pacientes"]

    con = sqlite3.connect(ruta_bd)
    con.executescript(open(RAIZ / "datos" / "esquema.sql", encoding="utf-8").read())

    pids = [p["patient_id"] for p in pacientes]
    marcas = ",".join("?" * len(pids))

    # Limpiar importaciones previas de estos pacientes (orden por claves foráneas)
    con.execute(
        f"DELETE FROM referencia_sintetica WHERE id_entrada IN "
        f"(SELECT id_entrada FROM entrada WHERE id_paciente IN ({marcas}))", pids)
    con.execute(f"DELETE FROM entrada WHERE id_paciente IN ({marcas})", pids)
    con.execute(f"DELETE FROM cuidador WHERE id_paciente IN ({marcas})", pids)
    con.execute(f"DELETE FROM paciente WHERE id_paciente IN ({marcas})", pids)

    # Pacientes y cuidadores
    id_cuidador = {}  # (patient_id, relacion) -> id
    for p in pacientes:
        con.execute("INSERT INTO paciente VALUES (?,?,?)",
                    (p["patient_id"], p["fecha_nacimiento"], p["sexo"]))
        for c in p["cuidadores"]:
            cid = f"{p['patient_id']}-{c['relacion']}"
            id_cuidador[(p["patient_id"], c["relacion"])] = cid
            con.execute("INSERT INTO cuidador VALUES (?,?,?)",
                        (cid, p["patient_id"], c["relacion"]))

    # Notas → entrada (+ ground truth → referencia_sintetica)
    for i, n in enumerate(notas):
        eid = OFFSET_ID + i
        cid = id_cuidador[(n["patient_id"], n["cuidador"]["relacion"])]
        con.execute("INSERT INTO entrada VALUES (?,?,?,?,?)",
                    (eid, n["patient_id"], cid, n["fecha"], n["texto"]))
        con.execute(
            "INSERT INTO referencia_sintetica (id_entrada, semana, fase, "
            "senal_adherencia, items_detectados) VALUES (?,?,?,?,?)",
            (eid, n["ref"]["semana"], n["ref"]["fase"], n["ref"]["adherencia"],
             json.dumps(n["ref_items"])))

    con.commit()

    n_pac = con.execute(f"SELECT COUNT(*) FROM paciente WHERE id_paciente IN ({marcas})", pids).fetchone()[0]
    n_ent = con.execute(f"SELECT COUNT(*) FROM entrada WHERE id_paciente IN ({marcas})", pids).fetchone()[0]
    print(f"Importado en {ruta_bd}:")
    print(f"  {n_pac} pacientes, {len(id_cuidador)} cuidadores, {n_ent} notas "
          f"(ids de entrada desde {OFFSET_ID})")
    for pid in pids:
        filas = con.execute(
            "SELECT COUNT(*), MIN(fecha), MAX(fecha) FROM entrada WHERE id_paciente = ?",
            (pid,)).fetchone()
        print(f"    {pid}: {filas[0]} notas ({filas[1][:10]} → {filas[2][:10]})")
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bd", default=str(RAIZ / "datos" / "anotador.db"),
                    help="ruta de la base SQLite (por defecto datos/anotador.db)")
    args = ap.parse_args()
    importar(args.bd)
