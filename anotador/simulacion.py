"""El motor del experimento factorial.

Recorre todas las combinaciones de parámetros (backend × modelo × temperatura ×
top_p × seed × estrategia) y, para cada una, repite N veces sobre cada entrada.
Cada ejecución se anota y se guarda en la tabla `anotacion`, que luego uso para
analizar la variabilidad.

La idea es la de un experimento de toda la vida: variables independientes (los
parámetros) por réplicas, y de ahí salen las dependientes (ítems, escalas,
nivel, métricas, latencia).
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

from anotador.backends import combinacion_valida
from anotador.config import Config, MODELO_POR_DEFECTO
from anotador.pipeline import Resultado, anotar, anotar_consenso
from anotador.repositorio import listar_entradas


@dataclass
class Rejilla:
    """Define el espacio del experimento. Cada lista es un eje del factorial.

    Con `entradas=None` usa todas las entradas de la base.
    `repeticiones` es el número de réplicas por (entrada × configuración), y es
    lo que permite medir la variabilidad (no quitarla).

    `agregaciones` sirve para comparar la anotación individual con el consenso
    por voto mayoritario: cada repetición 'consenso_voto' lanza `n_muestras`
    ejecuciones y las junta en una sola.
    """

    entradas: list[int] | None = None
    backends: list[str] = field(default_factory=lambda: ["directo"])
    modelos: list[str] = field(default_factory=lambda: [MODELO_POR_DEFECTO])
    temperaturas: list[float] = field(default_factory=lambda: [0.1])
    top_ps: list[float | None] = field(default_factory=lambda: [None])
    seeds: list[int | None] = field(default_factory=lambda: [None])
    estrategias: list[str] = field(default_factory=lambda: ["zero_shot"])
    repeticiones: int = 10
    agregaciones: list[str] = field(default_factory=lambda: ["individual"])
    n_muestras: int = 5
    umbral_voto: float = 0.5
    # El tool calling funciona en todos los modelos que probé (Gemma, Qwen,
    # Llama 3.1+, Hermes, Command-R), así que por defecto no filtro nada y comparo
    # los 4 backends en todos. Lo pongo en False solo si meto un modelo que no
    # soporte `tools`.
    permitir_tool_sin_soporte: bool = True

    def configuraciones(self) -> list[Config]:
        """Configuraciones válidas (sin contar repeticiones).

        Por defecto incluye todas las combinaciones. Si
        `permitir_tool_sin_soporte=False`, omite los backends de tool calling
        sobre modelos marcados como sin soporte en `backends.MODELOS_TOOL_CALLING`.
        """
        combos = itertools.product(
            self.backends,
            self.modelos,
            self.temperaturas,
            self.top_ps,
            self.seeds,
            self.estrategias,
        )
        configs = []
        for (b, m, t, tp, s, est) in combos:
            if not self.permitir_tool_sin_soporte and not combinacion_valida(b, m):
                continue
            configs.append(
                Config(backend=b, modelo=m, temperature=t, top_p=tp, seed=s,
                       estrategia_prompt=est)
            )
        return configs

    def _coste_por_repeticion(self) -> int:
        """Llamadas al modelo por repetición, sumando todas las agregaciones."""
        coste = 0
        for ag in self.agregaciones:
            coste += self.n_muestras if ag == "consenso_voto" else 1
        return coste

    def total_ejecuciones(self, n_entradas: int) -> int:
        """Total de unidades de resultado (filas en `anotacion`)."""
        return (
            len(self.configuraciones())
            * self.repeticiones
            * len(self.agregaciones)
            * n_entradas
        )

    def total_llamadas_modelo(self, n_entradas: int) -> int:
        """Total de llamadas reales al modelo (el consenso multiplica por n_muestras)."""
        return (
            len(self.configuraciones())
            * self.repeticiones
            * self._coste_por_repeticion()
            * n_entradas
        )


def ejecutar(
    rejilla: Rejilla,
    instrumento: dict,
    persistir: bool = True,
    verbose: bool = True,
) -> list[Resultado]:
    """Ejecuta toda la rejilla. Devuelve la lista de resultados (también persiste).

    Aviso: el número de llamadas al modelo crece muy rápido
    (configs × repeticiones × entradas). Usa `total_ejecuciones()` antes.
    """
    ids = rejilla.entradas
    if ids is None:
        ids = [m.id_entrada for m in listar_entradas()]

    configs = rejilla.configuraciones()
    total = rejilla.total_ejecuciones(len(ids))
    total_llamadas = rejilla.total_llamadas_modelo(len(ids))
    if verbose:
        n_pedidas = (
            len(rejilla.backends) * len(rejilla.modelos) * len(rejilla.temperaturas)
            * len(rejilla.top_ps) * len(rejilla.seeds) * len(rejilla.estrategias)
        )
        omitidas = n_pedidas - len(configs)
        if omitidas:
            print(
                f"  (omitidas {omitidas} combinaciones: tool calling sobre modelos "
                f"marcados como sin soporte)\n"
            )
        print(
            f"Simulación: {len(configs)} configuraciones × "
            f"{rejilla.repeticiones} repeticiones × {len(rejilla.agregaciones)} "
            f"agregaciones × {len(ids)} entradas = {total} resultados\n"
            f"  Llamadas reales al modelo: {total_llamadas} "
            f"(consenso usa n_muestras={rejilla.n_muestras})\n"
        )

    resultados: list[Resultado] = []
    hecho = 0
    t_ini = time.time()
    for id_entrada in ids:
        for cfg in configs:
            for rep in range(rejilla.repeticiones):
                for ag in rejilla.agregaciones:
                    if ag == "consenso_voto":
                        r = anotar_consenso(
                            id_entrada, instrumento, cfg,
                            n_muestras=rejilla.n_muestras,
                            umbral=rejilla.umbral_voto,
                            repeticion=rep, persistir=persistir,
                        )
                    else:
                        r = anotar(id_entrada, instrumento, cfg, repeticion=rep, persistir=persistir)
                    resultados.append(r)
                    hecho += 1
                    if verbose:
                        transcurrido = time.time() - t_ini
                        eta = transcurrido / hecho * (total - hecho)
                        print(
                            f"  [{hecho:>4}/{total}] entrada={id_entrada} "
                            f"{cfg.backend}/{cfg.modelo} T={cfg.temperature} "
                            f"{ag} rep={rep} → ok={r.formato_ok} {r.n_metricas_ok}/5 "
                            f"{r.latencia_s:.1f}s | ETA {eta/60:.1f} min",
                            flush=True,
                        )
    if verbose:
        print(f"\nHecho en {(time.time() - t_ini)/60:.1f} min.")
    return resultados
