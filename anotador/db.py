"""Capa de base de datos con SQLAlchemy 2.0.

Define el motor, la sesión y los modelos ORM que mapean las tablas del
esquema (paciente, cuidador, entrada, anotacion). El mismo código funciona
sobre SQLite (desarrollo) y PostgreSQL (producción) cambiando solo `URL_BD`.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from anotador.config import URL_BD


class Base(DeclarativeBase):
    pass


class Paciente(Base):
    __tablename__ = "paciente"

    id_paciente: Mapped[str] = mapped_column(String, primary_key=True)
    fecha_nacimiento: Mapped[dt.date] = mapped_column(Date, nullable=False)
    sexo: Mapped[str] = mapped_column(String, nullable=False)

    cuidadores: Mapped[list["Cuidador"]] = relationship(back_populates="paciente")
    entradas: Mapped[list["Entrada"]] = relationship(back_populates="paciente")


class Cuidador(Base):
    __tablename__ = "cuidador"

    id_cuidador: Mapped[str] = mapped_column(String, primary_key=True)
    id_paciente: Mapped[str] = mapped_column(
        ForeignKey("paciente.id_paciente"), nullable=False
    )
    rol: Mapped[str] = mapped_column(String, nullable=False)

    paciente: Mapped["Paciente"] = relationship(back_populates="cuidadores")
    entradas: Mapped[list["Entrada"]] = relationship(back_populates="cuidador")


class Entrada(Base):
    __tablename__ = "entrada"

    id_entrada: Mapped[int] = mapped_column(Integer, primary_key=True)
    id_paciente: Mapped[str] = mapped_column(
        ForeignKey("paciente.id_paciente"), nullable=False
    )
    id_cuidador: Mapped[str] = mapped_column(
        ForeignKey("cuidador.id_cuidador"), nullable=False
    )
    fecha: Mapped[dt.date] = mapped_column(Date, nullable=False)
    texto: Mapped[str] = mapped_column(Text, nullable=False)

    paciente: Mapped["Paciente"] = relationship(back_populates="entradas")
    cuidador: Mapped["Cuidador"] = relationship(back_populates="entradas")
    anotaciones: Mapped[list["Anotacion"]] = relationship(back_populates="entrada")


class Anotacion(Base):
    """Resultado de una ejecución del pipeline sobre una entrada.

    Una entrada tiene N anotaciones (repeticiones × configuraciones); son los
    datos con los que después estudio la variabilidad.
    """

    __tablename__ = "anotacion"

    id_anotacion: Mapped[int] = mapped_column(Integer, primary_key=True)
    id_entrada: Mapped[int] = mapped_column(
        ForeignKey("entrada.id_entrada"), nullable=False
    )
    creada_en: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.now, nullable=False
    )

    # Palancas (variables independientes)
    instrumento: Mapped[str] = mapped_column(String, nullable=False)
    backend: Mapped[str] = mapped_column(String, nullable=False)
    modelo: Mapped[str] = mapped_column(String, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    top_p: Mapped[float | None] = mapped_column(Float)
    seed: Mapped[int | None] = mapped_column(Integer)
    estrategia_prompt: Mapped[str] = mapped_column(String, nullable=False)
    repeticion: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    agregacion: Mapped[str] = mapped_column(String, default="individual", nullable=False)
    n_muestras: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # Salida (variables dependientes)
    formato_ok: Mapped[int] = mapped_column(Integer, nullable=False)
    items_detectados: Mapped[str | None] = mapped_column(Text)   # JSON
    escalas_afectadas: Mapped[str | None] = mapped_column(Text)  # JSON
    nivel_alerta: Mapped[str | None] = mapped_column(String)
    nota_clinica: Mapped[str | None] = mapped_column(Text)
    justificacion: Mapped[str | None] = mapped_column(Text)

    # Métricas y coste
    metricas: Mapped[str | None] = mapped_column(Text)  # JSON
    n_metricas_ok: Mapped[int | None] = mapped_column(Integer)
    latencia_s: Mapped[float | None] = mapped_column(Float)
    respuesta_cruda: Mapped[str | None] = mapped_column(Text)

    entrada: Mapped["Entrada"] = relationship(back_populates="anotaciones")


class ReferenciaSintetica(Base):
    """Anotación de referencia del dataset SINTÉTICO, 1:1 con `entrada`.

    NO es un ground truth clínico validado: son datos sintéticos sin validez
    diagnóstica. Se almacena para una fase futura (cuando se disponga de
    anotaciones reales de clínicos); por ahora NO debe usarse para evaluar la
    validez del sistema.
    """

    __tablename__ = "referencia_sintetica"

    id_entrada: Mapped[int] = mapped_column(
        ForeignKey("entrada.id_entrada"), primary_key=True
    )
    semana: Mapped[int | None] = mapped_column(Integer)
    fase: Mapped[str | None] = mapped_column(String)
    senal_adherencia: Mapped[str | None] = mapped_column(String)
    items_detectados: Mapped[str | None] = mapped_column(Text)   # JSON
    escalas_afectadas: Mapped[str | None] = mapped_column(Text)  # JSON
    nivel_alerta: Mapped[str | None] = mapped_column(String)
    nota_clinica: Mapped[str | None] = mapped_column(Text)

    entrada: Mapped["Entrada"] = relationship()


_engine: Engine = create_engine(URL_BD, future=True)


@event.listens_for(_engine, "connect")
def _activar_fk(dbapi_conn, _record):
    """Activa claves foráneas en SQLite (desactivadas por defecto)."""
    if URL_BD.startswith("sqlite"):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


Session = sessionmaker(bind=_engine, future=True)


def obtener_engine() -> Engine:
    return _engine


def crear_tablas() -> None:
    """Crea todas las tablas si no existen (idempotente)."""
    Base.metadata.create_all(_engine)
