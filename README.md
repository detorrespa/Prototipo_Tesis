# Anotación clínica con modelos de lenguaje (TDAH / BRIEF-2)

Este es el prototipo de la Tesis doctoral en el TDAH con LLMs.

La idea central es tomar lo que escriben los padres o las madres sobre su hijo (texto libre) y convertirlo en una anotación clínica
ordenada: qué ítems del instrumento aparecen, qué escalas se ven afectadas, un nivel de alerta y una nota. Para hacer esa conversión uso modelos de lenguaje en local con Ollama.

Usaremos los LLms para hacer la anotación de los comentarios segun un instruemnto clínoc, (tipo Brief 2). Como estos modelos son
probabilísticos, el mismo texto puede dar resultados distintos cada vez que se ejecuta. Así que el objetivo es medir hasta qué punto la anotación es estable y ver qué cosas la estabilizan (el modelo, la temperatura, la forma de pedirlo, agregar varias respuestas, etc.). Para eso he creado un experimento factorial, guardando todas las ejecuciones para luego analizarlas.

Aviso importante: los datos de pacientes son **sintéticos** no tienen validez clínica. En esta fase no comparo contra un "ground truth"; eso lo dejo para más adelante.

## Estructura del proyecto

```
├── anotador/                 # El paquete con toda la lógica
│   ├── config.py             # Configuración: Ollama, base de datos, clase Config
│   ├── modelos.py            # Modelos Pydantic (Mensaje, AnotacionClinica)
│   ├── instrumento.py        # Carga el instrumento desde brief2.json
│   ├── prompts.py            # Arma los prompts (zero_shot / few_shot / cot)
│   ├── backends.py           # Las 4 formas de hablar con el modelo
│   ├── evaluacion.py         # 5 métricas de calidad de la salida
│   ├── consenso.py           # Voto mayoritario (self-consistency)
│   ├── pipeline.py           # Junta todo: anotar() y anotar_consenso()
│   ├── repositorio.py        # Leer y guardar en la base de datos
│   ├── db.py                 # Tablas con SQLAlchemy
│   ├── simulacion.py         # La rejilla del experimento y ejecutar()
│   └── analisis.py           # Métricas de fiabilidad (Jaccard, Krippendorff)
├── datos/
│   ├── esquema.sql           # Las tablas en SQL
│   ├── importar_dataset.py   # Mete el dataset sintético en la base
│   ├── seed.py               # Datos mínimos de prueba
│   └── anotador.db           # La base SQLite (se genera sola; no la subo)
├── instrumentos/
│   └── brief2.json           # El instrumento BRIEF-2
├── data/                     # Dataset sintético de partida (JSON)
├── prototipo_anotador_tdah.ipynb   # El cuaderno original, paso a paso
├── 01_backend_directo.ipynb        # Experimento del backend directo
├── 02_backend_langchain.ipynb      # Experimento del backend langchain
├── 03_backend_tool_calling.ipynb   # Experimento del backend tool_calling
├── 04_backend_instructor.ipynb     # Experimento del backend instructor
├── 05_comparacion_backends.ipynb   # Comparación (solo lee la BD)
├── requirements.txt
└── README.md
```

## Instalación

Hace falta Python 3.12 y Ollama corriendo en local.

```bash
# Entorno virtual
python3.12 -m venv .venv
source .venv/bin/activate

# Dependencias
pip install -r requirements.txt

# Modelos en Ollama
ollama pull gemma4:e4b
ollama pull qwen3:8b
```

En la máquina donde trabajo Ollama escucha en el puerto 11002 (no el 11434 de
por defecto). Se cambia con la variable de entorno `OLLAMA_PORT`.

## Empezar a usarlo

```bash
# Crear las tablas e importar el dataset sintético
python -m datos.importar_dataset --reset

# Comprobar que Ollama responde
curl http://localhost:11002/api/tags

# Anotar una entrada desde Python
python -c "
from anotador.config import Config
from anotador.instrumento import cargar_instrumento
from anotador.pipeline import anotar
instr = cargar_instrumento()
r = anotar(1, instr, Config(backend='langchain', modelo='gemma4:e4b'), persistir=False)
print(r.metodo, r.formato_ok, r.anotacion)
"
```

Para los experimentos completos, los cuadernos `01` a `05` (ver siguiente sección).

## Los cuadernos del experimento

Un cuaderno por backend, todos con la misma estructura, y un quinto que compara.
La unidad del experimento es **la anotación del reporte semanal**: el dataset es
longitudinal (30 pacientes × 24 semanas) y la simulación va semana a semana, no
sobre el conjunto de entradas como si fueran intercambiables.

| Cuaderno | Qué hace |
|----------|----------|
| `01_backend_directo.ipynb` | Simula y mide el backend `directo` semana a semana. |
| `02_backend_langchain.ipynb` | Lo mismo con `langchain` (json_schema). |
| `03_backend_tool_calling.ipynb` | Lo mismo con `tool_calling`. |
| `04_backend_instructor.ipynb` | Lo mismo con `instructor` (Pydantic + reintentos). |
| `05_comparacion_backends.ipynb` | No llama al modelo: lee la tabla `anotacion` y cruza los cuatro. |

Cada cuaderno de backend sigue los mismos pasos: (1) configuración en una sola
celda, (2) una anotación de ejemplo para ver la salida, (3) simulación semana a
semana, (4) análisis de **solo esa simulación** (`cargar_df(backend=...,
desde=inicio)`), (5) fiabilidad por semana (Jaccard, acuerdo de nivel, alpha de
Krippendorff) y (6) lectura del resultado. Los cuatro pueden ejecutarse en
cualquier orden y son independientes entre sí.

La versión anterior del experimento (un solo cuaderno `simulaciones.ipynb` con
la rejilla factorial completa) está en la rama `version-1`.

## Las tablas

Hay cinco tablas (el SQL está en `datos/esquema.sql`):

| Tabla | Para qué |
|-------|----------|
| `paciente` | El menor evaluado (fecha de nacimiento, sexo). |
| `cuidador` | Quien aporta la observación (madre, padre...). |
| `entrada` | Una observación en texto libre. Es lo que se anota. |
| `anotacion` | El resultado de anotar una entrada. Una fila por ejecución. |
| `referencia_sintetica` | Referencia sintética para una fase posterior. |

La tabla `anotacion` es la importante para el experimento: ahí se guarda la configuración usada, lo que devolvió el modelo y las métricas. Una misma entrada  genera muchas filas (repeticiones × configuraciones), y eso es lo que permite medir la variabilidad.

## Los backends

Hay cuatro maneras distintas de pedirle al modelo una salida estructurada. Se eligen con `Config(backend=...)` y se comparan entre sí en el experimento:

| Backend | Cómo funciona |
|---------|---------------|
| `directo` | `POST /api/generate` y luego parseo el JSON a mano. Es el del cuaderno original y el más frágil. |
| `langchain` | `ChatOllama` con `with_structured_output(method="json_schema")`. El modelo queda obligado a seguir el esquema JSON. |
| `tool_calling` | `POST /api/chat` con `tools` (function calling). Suele ser el más rápido. Uso `think: False` para que los modelos de razonamiento como Qwen3 no se queden pensando. |
| `instructor` | Cliente compatible con OpenAI + Pydantic con reintentos. Valida y vuelve a intentar si falla. |

Sobre tool calling: probé que tanto Gemma como Qwen3 devuelven `tool_calls`  a través de Ollama. Por eso el filtro por modelo está quitado por defecto (`Rejilla.permitir_tool_sin_soporte=True`); solo hay que volver a ponerlo cuando no soporte `tools`.

Aprendizaje: al principio parecía que el tool calling no funcionaba, pero el problema era que `with_structured_output(method="function_calling")` de langchain-ollama devuelve `None`. Llamando directamente a la API de Ollama sí funciona en los dos modelos.

## Los experimentos

Todas las simulaciones estan en `anotador/simulacion.py`. Recorre todas las combinaciones de parámetros y, para cada una, repite N veces sobre cada entrada.

```python
from anotador.instrumento import cargar_instrumento
from anotador.simulacion import Rejilla, ejecutar

instr = cargar_instrumento()
rej = Rejilla(
    entradas=[1, 2, 3],                       # None = todas las de la base
    backends=["directo", "langchain", "tool_calling", "instructor"],
    modelos=["gemma4:e4b", "qwen3:8b"],
    temperaturas=[0.0, 0.7],
    repeticiones=3,                           # las réplicas dan la varianza
)
print("Llamadas al modelo:", rej.total_llamadas_modelo(len(rej.entradas)))
resultados = ejecutar(rej, instr, verbose=True)
```

Importante: el número de llamadas se dispara rápido (configuraciones × repeticiones × entradas, y el consenso lo multiplica por
`n_muestras`). Conviene mirar `total_llamadas_modelo()` antes de lanzar algo grande.

## Dónde quedan los resultados

Todo se guarda en una sola base SQLite, no genero ficheros JSON sueltos:

```
datos/anotador.db  →  tabla `anotacion`
```

Los JSON que devuelve el modelo van dentro de columnas de esa tabla
(`items_detectados`, `escalas_afectadas`, `metricas`...).

Para mirar los resultados:

```python
from anotador.analisis import (
    cargar_df, consistencia_por_grupo, resumen_por_parametro, krippendorff_nivel,
    resumen_por_semana, krippendorff_por_semana,
)
df = cargar_df()                       # todas las filas, con semana y paciente
df = cargar_df(backend="langchain", desde=inicio)  # solo una simulación
consistencia_por_grupo(df)             # consistencia por (entrada × config)
resumen_por_parametro(df, "backend")   # efecto de un parámetro
resumen_por_semana(df)                 # fiabilidad semana a semana
krippendorff_nivel(df)                 # fiabilidad entre réplicas del nivel
krippendorff_por_semana(df)            # el alpha, semana a semana
```

También se puede abrir `datos/anotador.db` con DB Browser for SQLite o la extensión SQLite de VS Code, o consultarlo desde la terminal con `sqlite3`.

## Las métricas de fiabilidad

Están en `anotador/analisis.py`. Me basé en ideas de psicometría y todas funcionan sin ground truth:

- Consistencia entre repeticiones: `jaccard_items` y `jaccard_escalas` miden si   el modelo elige lo mismo cada vez; `acuerdo_nivel` mira si coincide el nivel   de alerta.
- Alpha de Krippendorff (ordinal) sobre el nivel de alerta: trata cada   repetición como un codificador y cada entrada como una unidad.
- Calidad de la salida (`evaluacion.py`, 5 métricas): ítems válidos, escalas   válidas, coherencia ítem-escala, coherencia alerta-ítems y longitud de la nota.
- Coste: latencia y cuántas veces falla el formato.

Para reducir la variabilidad uso el voto mayoritario (`consenso.py`,`pipeline.anotar_consenso`): agrega varias respuestas en una con más cómputo.

## Cambiar cosas

| Qué | Dónde |
|-----|-------|
| Los parámetros del experimento | La celda de configuración de cada cuaderno `0X_backend_*.ipynb`. |
| Una ejecución suelta | `Config(backend=..., modelo=..., temperature=..., seed=...)`. |
| Modelo por defecto | Variable de entorno `OLLAMA_MODEL`. |
| Puerto / host de Ollama | `OLLAMA_PORT` (11002 por defecto), `OLLAMA_HOST_IP`. |
| Ruta de la base | `ANOTADOR_DB` o `ANOTADOR_DB_URL` (para PostgreSQL). |
| Instrumento | `cargar_instrumento("ruta/al/instrumento.json")`. |
| Reiniciar la base | `python -m datos.importar_dataset --reset`. |

## Límites de esta fase

- Los datos son sintéticos, no sirven para decisiones reales.
- Todavía no hay comparación contra ground truth.
- Uso SQLite para desarrollar, pero el ORM se puede llevar a PostgreSQL.

