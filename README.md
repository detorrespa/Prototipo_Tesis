# Anotación clínica con modelos de lenguaje (TDAH / BRIEF-2)

Prototipo de la tesis doctoral sobre TDAH con LLMs.

La idea central: tomar lo que escriben los padres o las madres sobre su hijo
(texto libre) y convertirlo en una anotación clínica ordenada según el
instrumento BRIEF-2: qué ítems aparecen, qué escalas se ven afectadas, un nivel
de alerta y una nota para el médico. La conversión la hace un modelo de
lenguaje en local con Ollama.

Como los modelos son probabilísticos, el mismo texto puede dar anotaciones
distintas en ejecuciones distintas. Cada cuaderno lanza un **experimento**
(anotar las entradas de una semana varias veces) y guarda los resultados en la
base de datos con un código, para poder comparar configuraciones entre sí.

Aviso importante: los datos de pacientes son **sintéticos**, sin validez
clínica. En esta fase no se compara contra un "ground truth".

## Estructura del proyecto

```
├── 01_backend_directo.ipynb        # POST /api/generate + parseo manual (línea base)
├── 02_backend_langchain.ipynb      # salida estructurada con json_schema
├── 03_backend_tool_calling.ipynb   # function calling vía /api/chat
├── 04_backend_instructor.ipynb     # Pydantic con reintentos automáticos
├── 05_comparacion_experimentos.ipynb  # compara experimentos (no llama al modelo)
├── instrumentos/
│   └── brief2.json                 # el instrumento BRIEF-2 (63 ítems, 9 escalas)
├── datos/
│   ├── esquema.sql                 # referencia de las tablas
│   └── anotador.db                 # la base SQLite (NO está en git, ver abajo)
├── data/                           # dataset sintético de partida (JSON)
├── requirements.txt
└── README.md
```

Los cuadernos son **independientes y autocontenidos**: no hay paquete de
Python que importar. Cada uno lleva dentro todo lo que usa (lectura de la BD,
prompts, backend, guardado), y se lee de arriba abajo.

## Cómo funciona cada cuaderno (01–04)

Los cuatro siguen los mismos pasos; solo cambia la forma de pedirle la salida
al modelo (el backend):

1. **Parámetros** — una sola celda con todo lo configurable:

```python
SEMANA       = 1            # semana de seguimiento (el dataset llega a la 24)
PACIENTES    = None         # None = todos; o lista: ["P001", "P003"]
REPETICIONES = 3            # veces que se anota cada entrada
TEMPERATURA  = 0.7
MODELO       = "gemma4:e4b"
```

2. **Datos** — lee de la BD las entradas de esa semana (texto del padre +
   contexto del paciente).
3. **Prompts** — se construyen desde `brief2.json`, visibles en el cuaderno.
4. **Backend** — la función `anotar()`, la única parte distinta entre cuadernos.
5. **Una anotación de ejemplo** — para ver la salida antes de lanzar nada.
6. **El experimento** — entradas × repeticiones, cada resultado a la tabla
   `experimento` con un código (p. ej. `directo-s1-t0.7-20260712`).
7. **Resultados** — formato válido, acuerdo del nivel de alerta entre
   repeticiones y latencia.

El cuaderno **05** no llama al modelo: lee la tabla `experimento` y compara los
experimentos guardados por sus códigos (estabilidad frente a coste).

## La tabla `experimento`

Todo resultado queda en `datos/anotador.db`, tabla `experimento` (el DDL está
en `datos/esquema.sql`). Las columnas clave:

| Columna | Qué guarda |
|---------|------------|
| `codigo` | Identificador del experimento, para comparar entre sí. |
| `backend`, `modelo`, `temperatura`, `semana` | La configuración usada. |
| `id_paciente`, `id_entrada`, `repeticion` | Qué se anotó y en qué réplica. |
| `formato_ok` | Si la salida fue un JSON válido. |
| `items_detectados`, `escalas_afectadas`, `nivel_alerta`, `nota_clinica` | La anotación. |
| `latencia_s` | Coste en segundos. |

También se puede consultar con DB Browser for SQLite, la extensión SQLite de
VS Code o `sqlite3` en la terminal.

## Instalación

Hace falta Python 3.12 y Ollama corriendo (en Mercurio escucha en el puerto
11002; se ajusta en la celda de parámetros de cada cuaderno).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Modelos en Ollama
ollama pull gemma4:e4b

# Comprobar que Ollama responde
curl http://localhost:11002/api/tags
```

**La base de datos** (`datos/anotador.db`) no está en git: contiene las tablas
de pacientes/entradas del dataset sintético y los resultados acumulados. Se
comparte por Synology Drive; hay que copiarla en `datos/` antes de ejecutar
los cuadernos.

## Los backends

| Cuaderno | Backend | Cómo pide la salida estructurada |
|----------|---------|----------------------------------|
| 01 | `directo` | `POST /api/generate` y extraer el JSON del texto. Frágil a propósito: es la línea base. |
| 02 | `langchain` | `ChatOllama` + `with_structured_output(method="json_schema")`: el esquema se impone al decodificar. |
| 03 | `tool_calling` | `POST /api/chat` con `tools`: el modelo invoca una función con los argumentos ya estructurados. |
| 04 | `instructor` | Cliente OpenAI-compatible + Pydantic: valida la salida y reintenta si falla. |

## Dónde está lo demás

La versión anterior del prototipo —paquete `anotador/` completo, el
experimento factorial con `Rejilla`, el consenso por voto mayoritario, las
métricas psicométricas (Jaccard, alpha de Krippendorff) y los scripts de
generación de la base de datos— está íntegra en la rama **`version-1`**.

## Límites de esta fase

- Los datos son sintéticos, no sirven para decisiones reales.
- Todavía no hay comparación contra ground truth.
- SQLite para desarrollar; el esquema es portable a PostgreSQL.
