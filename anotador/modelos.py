"""Modelos Pydantic del dominio.

- `Mensaje`: la observación parental que entra al pipeline (reconstruida desde
  las tablas paciente + cuidador + entrada).
- `AnotacionClinica`: la salida estructurada del LLM.
- `construir_modelo_anotacion`: fábrica que genera un modelo de salida
  *específico del instrumento* (rango de ítems y escalas válidas como
  restricciones), para que `instructor`/structured output validen y reintenten
  automáticamente.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field, create_model, field_validator


class Mensaje(BaseModel):
    """Observación parental lista para anotar.

    """

    id_paciente: str
    id_familiar: str          # = id_cuidador
    rol_informante: str       # rol del cuidador (madre, padre, ...)
    fecha: str                # fecha de la entrada (ISO)
    edad: int                 # calculada: fecha_entrada - fecha_nacimiento
    sexo: str
    entrada: str              # texto libre de la observación
    id_entrada: int | None = None

    def __getitem__(self, clave: str) -> Any:  # compat. con mensaje['...']
        return getattr(self, clave)

    def get(self, clave: str, defecto: Any = None) -> Any:
        return getattr(self, clave, defecto)


class AnotacionClinica(BaseModel):
    """Salida estructurada genérica (sin restricciones del instrumento).

    Útil como modelo base. Para validación estricta usa
    `construir_modelo_anotacion(instrumento)`.
    """

    items_detectados: list[int] = Field(default_factory=list)
    escalas_afectadas: list[str] = Field(default_factory=list)
    nivel_alerta: str = "bajo"
    nota_clinica: str = ""
    justificacion: str = ""


def construir_modelo_anotacion(instrumento: dict) -> type[BaseModel]:
    """Crea un modelo Pydantic de salida ajustado al instrumento dado.

    - `items_detectados`: enteros dentro del rango real de ítems.
    - `escalas_afectadas`: solo escalas existentes en el instrumento.
    - `nivel_alerta`: solo niveles definidos por el instrumento.

    Estas restricciones permiten que `instructor` reintente automáticamente
    cuando el modelo se sale del esquema, sin lógica de parsing manual.
    """
    ids = [it["id"] for it in instrumento["items"]]
    id_min, id_max = min(ids), max(ids)
    escalas_validas = set(instrumento["escalas"].keys())
    niveles = list(instrumento["niveles_alerta"])

    ItemValido = Annotated[int, Field(ge=id_min, le=id_max)]

    def _valida_escalas(cls, v: list[str]) -> list[str]:
        invalidas = [e for e in v if e not in escalas_validas]
        if invalidas:
            raise ValueError(f"escalas no válidas para el instrumento: {invalidas}")
        return v

    def _valida_nivel(cls, v: str) -> str:
        if v not in niveles:
            raise ValueError(f"nivel_alerta debe ser uno de {niveles}, recibido '{v}'")
        return v

    modelo = create_model(
        "AnotacionInstrumento",
        items_detectados=(list[ItemValido], Field(default_factory=list)),
        escalas_afectadas=(list[str], Field(default_factory=list)),
        nivel_alerta=(str, Field(default=niveles[-1])),
        nota_clinica=(str, Field(default="")),
        justificacion=(str, Field(default="")),
        __validators__={
            "_valida_escalas": field_validator("escalas_afectadas")(_valida_escalas),
            "_valida_nivel": field_validator("nivel_alerta")(_valida_nivel),
        },
    )
    return modelo
