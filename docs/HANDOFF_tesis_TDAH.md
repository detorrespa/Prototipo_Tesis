# Tesis TDAH + LLMs — Documento de traspaso (contexto para nuevo chat)

> Estado a fecha de este documento. Sirve para arrancar una sesión nueva con
> todo el contexto: qué es el proyecto, qué ha indicado el director, qué hay
> construido, las líneas de investigación y la planificación.

---

## 1 · Qué es el proyecto

Prototipo de tesis doctoral: usar un **LLM local** (Ollama, modelo Gemma) para
convertir las **notas espontáneas que escriben los cuidadores** sobre un niño
con TDAH (texto libre) en una **anotación clínica estructurada** según el
instrumento **BRIEF-2**, de forma **auditable** (cada ítem detectado va con la
cita textual de la nota que lo justifica).

Objetivo último del sistema (visión, no fase actual): que el médico pueda pedir
informes de evolución por paciente y recibir alertas de adherencia/eficacia.

Los datos de pacientes son **100 % sintéticos**, sin validez clínica.

### Infraestructura
- Servidor **Mercurio** (universidad), acceso por túnel SSH.
- **Ollama** escuchando en el puerto **11002** (no el 11434 por defecto).
- Modelos: `gemma4:e4b` (pequeño, para iterar) y `gemma4:26b` (~30B, en Mercurio).
- BD **SQLite** `datos/anotador.db` (~82 MB): NO está en git, se comparte por
  **Synology Drive** (vault de Obsidian compartido con el director).

### Repositorio
`https://github.com/detorrespa/Prototipo_Tesis` (público)

| Rama | Contenido |
|------|-----------|
| `main` | **Fase actual**: flujos A/B/C + dataset de notas de diario + comparador. |
| `version-1` | Prototipo inicial completo: paquete `anotador/`, rejilla factorial, consenso por voto, métricas psicométricas (Jaccard, alpha de Krippendorff). |
| `fase1-backends` *(local; puede subirse si se quiere)* | La comparación de los 4 backends con experimentos ejecutados. También está en el historial de `main`. |

---

## 2 · Lo que ha indicado el director (acumulado en las reuniones)

Estas son las directrices que marcan la dirección. **Son la brújula del proyecto.**

1. **La tesis NO va de justificar backends.** Comparar mecanismos de salida
   estructurada (directo / langchain / tool calling / instructor) ya está
   investigado en la literatura. Frase textual: *"no estás evaluando los
   modelos, estás evaluando el arnés que le pones al modelo"*. Los 4 backends
   sirvieron para aprender la tecnología; se descartan como objeto de estudio.

2. **Usar un solo backend: LangChain** (con salida estructurada / llamada a
   herramientas). Las librerías de abstracción (LangChain, y en su momento
   instructor) están bien porque permiten prototipar rápido; nunca se hará una
   llamada a la API a bajo nivel. Los modelos actuales ya se entrenan para
   generar datos estructurados vía tool calling.

3. **La contribución es la auditabilidad clínica.** Por cada ítem que el LLM
   detecte, debe dar la **cita textual** de la nota en la que se basa +
   justificación. Eso da **trazabilidad** y es el puente hacia la validación
   con médicos (juzgar pares ítem–evidencia es rápido).

4. **Adelgazar el esquema de salida.** El LLM debe producir **solo la lista de
   ítems** (con evidencia y justificación por ítem). En concreto:
   - `escalas_afectadas` → **fuera**: es redundante, se deriva del ID del ítem.
   - `nivel_alerta` → **fuera del LLM**: se **calcula** de forma determinista
     con las `reglas_coherencia` del instrumento (alerta alta ≥ 4 ítems,
     moderada ≥ 2 ítems).
   - `nota_clinica` → **para un segundo agente** más adelante (informe/resumen),
     no para esta fase.

5. **La unidad es la nota (entrada de diario), no la semana.** Los cuidadores
   escriben **cuando quieren**: unos días 2-3 notas, semanas sin ninguna, padres
   metódicos y padres que no. El dataset se ha rehecho para reflejar esto.

6. **Definir el flujo de trabajo del agente por bloques individualizados**
   (más fácil de diseñar, depurar y entender), y separar bien la parte de
   *prompting*. Elogió el enfoque de un notebook con funciones mínimas legibles.

7. **Lo evaluable son las DIFERENCIAS DE FLUJO, no el backend.** Hay que
   proponer variantes de flujo de trabajo y ver cuáles funcionan bien. Sugirió
   explícitamente:
   - **Variante A**: una entrada + el catálogo completo de ítems → lista de ítems.
   - **Variante B**: la entrada + los ítems **uno a uno** → sí/no por ítem, con
     trazabilidad total (incluidos los negativos).
   - También experimentar con **cuánta información meter en el contexto**: varias
     entradas con un solo ítem, varias entradas con todos los ítems, etc.
   - *"Qué información hay que meter en el contexto y cómo estructurar el
     tratamiento de los datos es lo que va a definir nuestro flujo de trabajo."*

8. **Un solo paciente por ejecución** (sin listas de pacientes). Usar el Gemma
   grande (la infraestructura lo aguanta; los contextos son pequeños).

9. **Arquitectura de dos bloques**: (1) el anotador clasifica entradas → datos
   estructurados; (2) un **segundo agente** genera el informe/resumen a partir
   de esos datos ya estructurados. El **RAG pertenece a este segundo bloque**,
   más adelante — no al anotador (meterle historial al anotador contaminaría la
   medida de fiabilidad).

10. **Validación con médicos (futura)**: presentar pares (ítem, evidencia) y que
    un médico juzgue qué ítems se cumplen; comparar con lo que detecta el modelo;
    otro médico revisor verifica las justificaciones. **Todavía NO se hace
    validación oficial** — se está construyendo el flujo preliminar.

11. **Calendario**: el director está fuera gran parte del verano; revisión la
    **primera semana de septiembre** con lo avanzado.

---

## 3 · Lo que tenemos construido (en `main`)

### Dataset de notas de diario
- `data/notas_diario.json`: **92 notas** de **5 pacientes**, reescritas como
  entradas de diario a partir del dataset semanal original
  (`referencia_sintetica.json`, 30 pacientes × 24 semanas).
- 5 pacientes elegidos por franja de edad y escenario:
  | Paciente | Edad | Escenario | Cadencia |
  |---|---|---|---|
  | PAC001 | 7 | Arco clásico pre-med → MTF → estabilización | Madre metódica ~3/sem |
  | PAC005 | 8 | Dos hogares, adherencia irregular interparental | Madre + padre (2 informantes) |
  | PAC013 | 10 | Solo conductual → decisión → inicio MTF | Padre, **semanas vacías** |
  | PAC017 | 11 | Tics por MTF → cambio a atomoxetina | Ráfagas en crisis |
  | PAC022 | 14 | Resistencia adolescente → psicoeducación | Madre, irregular |
- Cada nota lleva `ref_items`: los ítems BRIEF-2 que ESE texto expresa (ground
  truth sintético a nivel de nota, para validación futura). **El anotador no los
  recibe** — se guardan en la tabla `referencia_sintetica`.
- **26 % de notas sin ítems** a propósito (insomnio, gestiones médicas, días
  buenos): casos negativos para comprobar que el modelo no inventa.
- `data/pacientes_contexto.json`: perfil mínimo de los 5 pacientes.
- `datos/importar_notas.py`: importador autocontenido a SQLite (idempotente;
  ids de entrada desde 10000 para no chocar con el dataset semanal).

### Los tres cuadernos de flujo (la ablación)
Misma tarea (nota → ítems con evidencia), variando cómo se presenta el instrumento:

| Cuaderno | Llamadas/nota | Contexto | Nota |
|----------|---------------|----------|------|
| `flujo_A_catalogo.ipynb` | 1 | 63 ítems a la vez | Barato; riesgo de falsos negativos |
| `flujo_B_item_a_item.ipynb` | 63 | 1 ítem (sí/no + cita) | Trazabilidad total (negativos incluidos); 63× coste |
| `flujo_C_por_escala.ipynb` | 9 | 4-9 ítems de una escala | Equilibrio; descarta ids fuera de escala |

Comunes a los tres:
- El LLM solo produce ítems `{id, evidencia, justificacion}`.
- Escalas y nivel de alerta **derivados** con `reglas_coherencia` (determinista).
- `PACIENTE` único en la celda de parámetros.
- **Verificación automática de evidencia** (`evidencia_ok`): fracción de citas
  que aparecen literalmente en la nota → detecta alucinaciones sin médico.
- Backend: LangChain `with_structured_output(method="json_schema")`.

### Comparador
- `comparacion_experimentos.ipynb`: no llama al modelo; lee la tabla
  `experimento` y compara flujos por su código de experimento.

### La tabla `experimento` (en `anotador.db`)
Cada fila = anotación completa de una nota en una repetición. Columnas clave:
`codigo`, `backend`, `modelo`, `temperatura`, `id_paciente`, `id_entrada`,
`repeticion`, `formato_ok`, `items_detectados`, `escalas_afectadas` (derivadas),
`nivel_alerta` (derivado), `items_detalle` (JSON con evidencia y justificación
por ítem), `evidencia_ok`, `latencia_s`, `respuesta_cruda` (traza de todas las
llamadas del flujo, negativos incluidos).

### El instrumento
`instrumentos/brief2.json`: BRIEF-2 Familia, **63 ítems**, **9 escalas**
(inhibición, flexibilidad, memoria de trabajo, supervisión de conducta,
supervisión de tarea, control emocional, iniciativa, planificación,
organización de materiales), 3 niveles ordinales (bajo < moderado < alto),
`reglas_coherencia` (alto ≥ 4 ítems, moderado ≥ 2).

### Métricas de evaluación (todas sin ground truth clínico)
- `formato_ok`: fracción de salidas válidas.
- `jaccard_items`: estabilidad de la selección de ítems entre repeticiones.
- `acuerdo_nivel`: acuerdo modal del nivel de alerta (derivado) entre repeticiones.
- `evidencia_ok`: fracción de citas que existen literalmente en la nota.
- `latencia_s`: coste por nota, comparable entre flujos.

---

## 4 · Líneas de investigación

Marco psicométrico (secuencia COSMIN): **fiabilidad → validez → sensibilidad al
cambio**. La fiabilidad es condición necesaria de la validez.

| Fase | Pregunta | Diseño | Estado |
|------|----------|--------|--------|
| **0. Flujo de trabajo** | ¿Qué forma de estructurar el contexto anota mejor? | Ablación A/B/C sobre las mismas notas, N repeticiones | **En curso** |
| **1. Fiabilidad** | ¿El anotador es estable y reproducible? | Repeticiones por nota; Jaccard, acuerdo de nivel, evidencia_ok | Instrumentado |
| **2. Validez de criterio** | ¿Anota "bien"? | Pares (ítem, evidencia) juzgados por clínicos; comparar con el modelo | Diseño futuro |
| **3. Sensibilidad al cambio** | ¿Detecta la evolución del niño (24 semanas)? | Segundo agente + RAG sobre datos ya estructurados | Diseño futuro |

Preguntas de investigación abiertas (para afinar con el director):
- ¿Cuánta estructura de contexto necesita el modelo para anotar de forma
  estable? (A vs B vs C: estabilidad frente a coste).
- ¿Mejora la fidelidad de la evidencia (menos alucinación) con contexto más
  acotado (B/C) frente al catálogo completo (A)?
- Efecto de meter varias entradas en el contexto vs una sola.
- Más adelante: ¿el segundo agente + RAG detecta la trayectoria del paciente
  y las alertas de adherencia?

---

## 5 · Planificación

### Ahora / verano (construcción del flujo preliminar) — HECHO en su mayoría
- [x] Reorientar a un solo backend (langchain) y esquema con evidencia por ítem.
- [x] Rehacer el dataset como notas de diario con cadencia irregular (92 notas).
- [x] Implementar los 3 flujos A/B/C con esquema adelgazado y derivación
      determinista de escalas/nivel.
- [ ] **Ejecutar la ablación en Mercurio con Gemma real** (pendiente del usuario):
      `python datos/importar_notas.py`, luego flujo A → C → B (barato a caro),
      mismo `PACIENTE` y parámetros, y comparar en `comparacion_experimentos.ipynb`.
      Coste flujo B: 4 notas × 3 rep = 756 llamadas (~1-2 h con gemma4:e4b).

### Reunión de septiembre — llevar
- Resultados de la ablación A/B/C: qué flujo da la anotación más estable, con
  qué fidelidad de evidencia y a qué coste.
- La métrica `evidencia_ok` como aportación cuantitativa nueva (mide alucinación
  sin necesidad de clínicos).

### Después (fases 2-3)
- Diseñar la validación con médicos sobre pares (ítem, evidencia).
- Diseñar el segundo agente (informe/resumen) y, en su capa, el RAG por paciente.
- Escalar el dataset (extender los 5 pacientes a más semanas, o añadir pacientes).

---

## 6 · Cómo trabajar el repo (recordatorio operativo)

Esta sesión de Claude tiene credenciales del proxy solo para el repo `AIE9`, no
para `Prototipo_Tesis`. El flujo de traspaso usado ha sido: subir a una rama
`transfer/prototipo-main` en AIE9, y el usuario reenviarla a Prototipo_Tesis
desde su máquina:
```bash
git clone -b transfer/prototipo-main https://github.com/detorrespa/AIE9 tmp_transfer
cd tmp_transfer
git remote add proto https://github.com/detorrespa/Prototipo_Tesis
git push proto HEAD:main
cd ..; rm -rf tmp_transfer   # (Windows: Remove-Item -Recurse -Force tmp_transfer)
```
(En PowerShell, git escribe por stderr y aparece en rojo aunque el comando vaya
bien; verificar con `git ls-remote proto main`.)

Commits con la identidad del usuario (Alberto de Torres), nunca con "Claude"
como autor.

### Ejecución en Mercurio
```bash
git pull origin main
python datos/importar_notas.py          # carga las 92 notas en anotador.db
# abrir flujo_A_catalogo.ipynb, ajustar PACIENTE y ejecutar de arriba abajo
# repetir con flujo_C y flujo_B; comparar en comparacion_experimentos.ipynb
```

---

## 7 · Documentos de apoyo ya generados (en Obsidian / conversaciones previas)
- **Defensa metodológica** (fiabilidad → validez → sensibilidad; alpha de
  Krippendorff explicado; umbrales; referencias COSMIN, Krippendorff, Hallgren…).
- **Nota técnica sobre `instructor`** y el protocolo OpenAI (por si se retoma).
- **Sección de tesis "El prototipo desarrollado"** con citas formales.
- Nota: parte de esos textos hablan de "semana" y de los 4 backends —
  **desactualizados** respecto a la dirección actual (nota, un solo backend,
  ablación de flujos). Reutilizar el marco de fiabilidad, actualizar lo demás.

---

## 8 · Advertencias / decisiones abiertas
- Confirmar con el director el **punto exacto de la investigación** (el propio
  usuario expresó dudas): la hipótesis de trabajo es *"¿puede un LLM local
  convertir notas espontáneas de cuidadores en anotaciones BRIEF-2 auditables
  ítem a ítem, y qué flujo de trabajo lo hace de forma más estable y fiel?"*.
- Las notas espontáneas tienen **sesgo de selección de eventos** (se escribe
  cuando algo va mal); la ausencia de notas ≠ estabilidad clínica. Documentar
  como limitación.
- Verificar las citas bibliográficas antes de la entrega formal.
- La tabla `experimento` acumula todas las ejecuciones: filtrar por `codigo` al
  analizar, no mezclar experimentos.
