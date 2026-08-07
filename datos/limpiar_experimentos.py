"""Borra filas de la tabla `experimento` (deja pacientes/notas intactos).

Uso:
    python3 datos/limpiar_experimentos.py              # borra TODO experimento
    python3 datos/limpiar_experimentos.py --flujo A    # solo códigos flujoA-*
    python3 datos/limpiar_experimentos.py --codigo '...'
    python3 datos/limpiar_experimentos.py --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

RUTA_BD = Path(__file__).resolve().parent / "anotador.db"


def main() -> None:
    p = argparse.ArgumentParser(description="Limpia resultados de experimento.")
    p.add_argument("--flujo", help="Solo códigos que empiezan por flujoX-")
    p.add_argument("--codigo", help="Solo un código exacto")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    con = sqlite3.connect(RUTA_BD)
    if args.codigo:
        where, params = "codigo = ?", [args.codigo]
    elif args.flujo:
        where, params = "codigo LIKE ?", [f"flujo{args.flujo}-%"]
    else:
        where, params = "1=1", []

    n = con.execute(f"SELECT COUNT(*) FROM experimento WHERE {where}", params).fetchone()[0]
    print(f"Filas a borrar: {n}")
    if args.dry_run or n == 0:
        con.close()
        return
    con.execute(f"DELETE FROM experimento WHERE {where}", params)
    con.commit()
    print("OK")
    con.close()


if __name__ == "__main__":
    main()
