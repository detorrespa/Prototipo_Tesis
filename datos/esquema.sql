-- Esquema de la base de datos del prototipo de anotación clínica (TDAH / BRIEF-2).
--
-- Tres tablas de entrada (paciente, cuidador, entrada) y una de resultados
-- (anotacion), que es la que uso para las simulaciones.
--
-- Está escrito para SQLite, pero el ORM lo lleva a PostgreSQL sin cambios. En
-- SQLite hay que activar las claves foráneas con PRAGMA foreign_keys = ON.

PRAGMA foreign_keys = ON;

-- paciente: un menor evaluado. No guardo la edad (cambia con el tiempo) sino la
-- fecha de nacimiento; la edad la calculo en cada entrada.
CREATE TABLE IF NOT EXISTS paciente (
    id_paciente       TEXT PRIMARY KEY,
    fecha_nacimiento  DATE NOT NULL,
    sexo              TEXT NOT NULL CHECK (sexo IN ('masculino', 'femenino', 'otro'))
);

-- cuidador: quien aporta la observación (madre, padre, tutor...). Cada cuidador
-- está asociado a un paciente.
CREATE TABLE IF NOT EXISTS cuidador (
    id_cuidador  TEXT PRIMARY KEY,
    id_paciente  TEXT NOT NULL REFERENCES paciente(id_paciente),
    rol          TEXT NOT NULL  -- madre, padre, tutor, abuela_materna...
);

-- entrada: una observación en texto libre, con fecha, ligada a un paciente y al
-- cuidador que la aporta. Es lo que se anota.
CREATE TABLE IF NOT EXISTS entrada (
    id_entrada   INTEGER PRIMARY KEY,
    id_paciente  TEXT NOT NULL REFERENCES paciente(id_paciente),
    id_cuidador  TEXT NOT NULL REFERENCES cuidador(id_cuidador),
    fecha        DATE NOT NULL,
    texto        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entrada_paciente ON entrada(id_paciente);
CREATE INDEX IF NOT EXISTS idx_cuidador_paciente ON cuidador(id_paciente);

-- anotacion: resultado de anotar una entrada con un LLM. Cada fila es una
-- ejecución del pipeline, con su configuración completa y sus métricas. Una
-- misma entrada genera muchas anotaciones (repeticiones × configuraciones) para
-- poder estudiar la variabilidad.
CREATE TABLE IF NOT EXISTS anotacion (
    id_anotacion       INTEGER PRIMARY KEY,
    id_entrada         INTEGER NOT NULL REFERENCES entrada(id_entrada),
    creada_en          TIMESTAMP NOT NULL,

    -- Parámetros del experimento (variables independientes)
    instrumento        TEXT NOT NULL,
    backend            TEXT NOT NULL,   -- directo | langchain | instructor
    modelo             TEXT NOT NULL,   -- gemma4:26b, gemma4:12b, ...
    temperature        REAL NOT NULL,
    top_p              REAL,
    seed               INTEGER,
    estrategia_prompt  TEXT NOT NULL,   -- zero_shot | few_shot | cot
    repeticion         INTEGER NOT NULL DEFAULT 0,
    agregacion         TEXT NOT NULL DEFAULT 'individual',  -- individual | consenso_voto
    n_muestras         INTEGER NOT NULL DEFAULT 1,          -- nº de ejecuciones agregadas

    -- Salida del modelo (variables dependientes)
    formato_ok         INTEGER NOT NULL,  -- 0/1: se pudo parsear/validar
    items_detectados   TEXT,              -- JSON: [int]
    escalas_afectadas  TEXT,              -- JSON: [str]
    nivel_alerta       TEXT,
    nota_clinica       TEXT,
    justificacion      TEXT,

    -- Métricas y coste
    metricas           TEXT,              -- JSON con el dict de evaluar_anotacion
    n_metricas_ok      INTEGER,
    latencia_s         REAL,
    respuesta_cruda    TEXT               -- texto bruto del modelo (auditoría)
);

CREATE INDEX IF NOT EXISTS idx_anotacion_entrada ON anotacion(id_entrada);
CREATE INDEX IF NOT EXISTS idx_anotacion_config  ON anotacion(backend, modelo, temperature);

-- referencia_sintetica: anotación de referencia del dataset sintético, ligada a
-- una entrada (1:1). No es un ground truth clínico validado, son datos
-- sintéticos sin validez diagnóstica. La guardo para una fase posterior; por
-- ahora no se usa para evaluar la validez del sistema.
CREATE TABLE IF NOT EXISTS referencia_sintetica (
    id_entrada         INTEGER PRIMARY KEY REFERENCES entrada(id_entrada),
    semana             INTEGER,
    fase               TEXT,
    senal_adherencia   TEXT,
    items_detectados   TEXT,   -- JSON: [int]
    escalas_afectadas  TEXT,   -- JSON: [str]
    nivel_alerta       TEXT,
    nota_clinica       TEXT
);

