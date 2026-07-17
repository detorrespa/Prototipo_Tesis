# Anotación clínica con modelos de lenguaje (TDAH / BRIEF-2)

Prototipo de la tesis doctoral sobre TDAH con LLMs.

El objetivo es investigar cómo un LLM local puede ayudar al médico en el
seguimiento del TDAH: los cuidadores escriben **notas espontáneas** sobre su
hijo (texto libre, cuando ellos quieran) y el sistema convierte cada nota en
una **anotación clínica estructurada** según el instrumento BRIEF-2 en la que
**cada ítem detectado va acompañado de su evidencia**: la cita textual de la
nota que lo sustenta y una justificación. La conversión la hace un modelo de
lenguaje en local con Ollama.

La evidencia por ítem cumple dos funciones:

1. **Auditabilidad clínica** — el médico puede verificar cada ítem contra el
   fragmento exacto de la nota, sin releerla entera.
2. **Verificación automática** — como la evidencia debe ser una cita literal,
   el sistema comprueba si el fragmento existe realmente en la nota
   (`evidencia_ok`). Una evidencia inexistente es una alucinación detectada
   sin revisión humana.

Como los modelos son probabilísticos, cada nota se anota varias veces
(**repeticiones**) y los resultados se guardan con un código de experimento
para medir la estabilidad y comparar configuraciones.

Los datos de pacientes son **sintéticos**, sin validez clínica. En esta fase
no se compara contra un ground truth.

## Estructura del proyecto

```
├── anotador.ipynb                  # EL cuaderno: nota → anotación con evidencia
├── comparacion_experimentos.ipynb  # compara experimentos (no llama al modelo)
├── instrumentos/
│   └── brief2.json                 # el instrumento BRIEF-2 (63 ítems, 9 escalas)
├── datos/
│   ├── esquema.sql                 # referencia de las tablas
│   ├── importar_notas.py           # carga el dataset de notas en la BD
│   └── anotador.db                 # la base SQLite (NO está en git, ver abajo)
├── data/
│   ├── notas_diario.json           # dataset de notas de diario (5 pacientes, 92 notas)
│   ├── pacientes_contexto.json     # contexto de los 5 pacientes
│   └── referencia_sintetica.json   # dataset semanal original (30 pacientes × 24 semanas)
├── requirements.txt
└── README.md
```

Los cuadernos son **autocontenidos**: no hay paquete de Python que importar.
Todo lo que usan (lectura de la BD, prompts, backend, verificación, guardado)
está dentro, y se leen de arriba abajo.

## El cuaderno `anotador.ipynb`

1. **Parámetros** — una sola celda con todo lo configurable:

```python
NOTAS        = None         # None = todas las notas; o lista de id_entrada
PACIENTES    = None         # None = todos; o lista: ["PAC001", "PAC003"]
MUESTRA      = 10           # nº máximo de notas a anotar
REPETICIONES = 3            # veces que se anota cada nota
TEMPERATURA  = 0.7
MODELO       = "gemma4:e4b"
```

2. **Datos** — las notas de los cuidadores, con el contexto del paciente. La
   unidad es la nota: cadencia irregular (varias una semana, ninguna otra).
3. **Esquema de salida** — el contrato Pydantic: cada ítem es un objeto
   `{id, evidencia, justificacion}`.
4. **Prompts** — construidos desde `brief2.json`; exigen cita literal y
   prohíben incluir ítems sin evidencia citable.
5. **Backend** — LangChain con `with_structured_output(method="json_schema")`:
   el esquema se impone al decodificar.
6. **Verificación de evidencia** — ¿la cita existe literalmente en la nota?
   (normalizando mayúsculas, tildes y espacios).
7. **Una anotación de ejemplo** — con cada evidencia marcada `OK`/`??`.
8. **El experimento** — notas × repeticiones → tabla `experimento`.
9. **Resultados** — formato válido, acuerdo del nivel entre repeticiones,
   evidencia verificada media y latencia.

Cada fila guarda también `respuesta_cruda` (salida exacta del modelo) para
auditoría, `items_detalle` (los ítems con evidencia y justificación) y
`evidencia_ok` (fracción de evidencias verificadas).

Para relanzar un experimento: borrar filas con el mismo `codigo` o cambiar el
código antes de ejecutar de nuevo.

## El cuaderno `comparacion_experimentos.ipynb`

No llama al modelo. Lee la tabla `experimento` y compara los experimentos
guardados (distintas temperaturas, modelos o versiones del prompt) por su
código: estabilidad de la anotación, formato válido y latencia, con detalle
por paciente de los casos inestables.

## La tabla `experimento`

Todo resultado queda en `datos/anotador.db`, tabla `experimento` (el DDL está
en `datos/esquema.sql`). Las columnas clave:

| Columna | Qué guarda |
|---------|------------|
| `codigo` | Identificador del experimento, para comparar entre sí. |
| `backend`, `modelo`, `temperatura` | La configuración usada. |
| `id_paciente`, `id_entrada`, `repeticion` | Qué nota se anotó y en qué réplica. |
| `formato_ok` | Si la salida fue válida. |
| `items_detectados`, `escalas_afectadas`, `nivel_alerta` | La anotación estructurada. |
| `items_detalle` | Los ítems con su evidencia y justificación (JSON). |
| `evidencia_ok` | Fracción de ítems cuya evidencia existe literalmente en la nota. |
| `nota_clinica` | Resumen para el médico. |
| `latencia_s` | Coste en segundos. |
| `respuesta_cruda` | Salida bruta del modelo (auditoría). |

También se puede consultar con DB Browser for SQLite, la extensión SQLite de
VS Code o `sqlite3` en la terminal.

## Instalación

Hace falta Python 3.12 y Ollama corriendo (en Mercurio escucha en el puerto
11002; se ajusta en la celda de parámetros del cuaderno).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Modelo en Ollama
ollama pull gemma4:e4b

# Comprobar que Ollama responde
curl http://localhost:11002/api/tags
```

**La base de datos** (`datos/anotador.db`) no está en git: contiene las tablas
de pacientes/notas del dataset sintético y los resultados acumulados. Se
comparte por Synology Drive; hay que copiarla en `datos/` antes de ejecutar
los cuadernos. Las notas de diario se cargan (o recargan) con:

```bash
python datos/importar_notas.py
```

## El dataset de notas de diario

`data/notas_diario.json` contiene **92 notas espontáneas de cuidadores** sobre
5 pacientes sintéticos, reescritas como entradas de diario a partir del
dataset semanal original (`referencia_sintetica.json`, 30 pacientes × 24
semanas). Es el formato que pide el flujo de trabajo: los cuidadores escriben
cuando quieren, con **cadencia irregular**.

| Paciente | Edad | Escenario | Cuidador(es) | Cadencia |
|----------|------|-----------|--------------|----------|
| PAC001 | 7 | Arco clásico: pre-medicación → MTF → estabilización | Madre | Metódica, ~3 notas/semana |
| PAC005 | 8 | Dos hogares, adherencia irregular interparental | Madre y padre | Madre regular, padre escaso |
| PAC013 | 10 | Sin medicación → decisión → inicio MTF | Padre | Poco metódico, **semanas vacías** |
| PAC017 | 11 | Tics por MTF → transición a atomoxetina | Madre | Ráfagas en crisis, calma después |
| PAC022 | 14 | Resistencia adolescente → psicoeducación → adherencia | Madre | Irregular |

Cada nota lleva `ref_items`: los ítems BRIEF-2 que el texto expresa (ground
truth sintético, para validación futura). **El anotador no los recibe** — el
importador los guarda en la tabla `referencia_sintetica`, separada de lo que
lee el cuaderno. Un 26 % de las notas no expresa ningún ítem a propósito
(insomnio, gestiones médicas, días buenos): son los casos negativos con los
que comprobar que el anotador no inventa.

## Por qué el backend es LangChain (json_schema)

En una fase previa se compararon empíricamente cuatro mecanismos de salida
estructurada (parseo directo, json_schema, tool calling e instructor) sobre
los mismos datos y modelos. La salida estructurada por esquema elimina los
fallos de formato por construcción sin la latencia de los reintentos. Esa
comparación —cuadernos ejecutados con sus resultados— está íntegra en la rama
**`fase1-backends`**; la justificación de la elección no forma parte del
objeto de la tesis.

## Las ramas del repositorio

| Rama | Contenido |
|------|-----------|
| `main` | El anotador con evidencia por ítem (fase actual). |
| `fase1-backends` | La comparación de los 4 backends, con los experimentos ejecutados. |
| `version-1` | La versión inicial: paquete `anotador/`, rejilla factorial, consenso y métricas psicométricas. |

## Límites de esta fase

- Los datos son sintéticos, no sirven para decisiones reales.
- Todavía no hay comparación contra ground truth; los pares (ítem, evidencia)
  están pensados para la validación futura con clínicos.
- Las notas espontáneas tienen sesgo de selección de eventos: la ausencia de
  notas no implica estabilidad clínica.
- SQLite para desarrollar; el esquema es portable a PostgreSQL.
