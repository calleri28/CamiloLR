"""Lectura de los archivos fuente (todos se leen como texto para poder validar su calidad)."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pandas as pd

from .utilidades import LOG

ARCHIVOS = {
    # nombre lógico: (archivo, separador)
    "sap": ("SAP.csv", ","),
    "tms": ("TMS.csv", ";"),
    "capacidad_plantas": ("capacidad_plantas.csv", ";"),
    "homologacion_centros": ("homologacion_centros.csv", ";"),
    "info_cedi": ("info_cedi.csv", ";"),
    "maestro_clientes": ("maestro_clientes.csv", ";"),
    "maestro_nodos": ("maestro_nodos.csv", ";"),
    "maestro_referencias": ("maestro_referencias.csv", ";"),
    "parametros_plantas": ("parametros_plantas.csv", ";"),
}

COLUMNAS_MINIMAS = {
    "sap": ["Viaje", "Cliente", "Referencia", "Unidad Negocio", "Centro productor", "CEDI",
            "CorredorTransporte", "Poblacion", "Periodo", "Mercado", "Tipo", "Cantidad",
            "IMPORTE SAP", "Volumen"],
    "tms": ["Viaje", "Población Origen", "Población Destino", "Fecha salida real",
            "Fecha llegada real", "Peso", "Volumen", "Volumen viaje"],
    "maestro_nodos": ["ID_Nodo", "Ciudad", "Latitud", "Longitud", "Estado", "Modalidad"],
    "maestro_referencias": ["Cod_Ref", "Familia", "Planta_Produccion_Antes", "Situacion_Post_Cambio",
                            "Planta_Produccion_Nueva", "Cajas_x_Estiba", "Volumen_Caja_m3"],
    "maestro_clientes": ["Nombre_Cliente", "Ciudad", "Departamento", "Latitud", "Longitud"],
    "info_cedi": ["ID_Nodo", "Posiciones_Estiba"],
    "capacidad_plantas": ["ID_Nodo", "Familia", "Capacidad_M3_Mes"],
    "parametros_plantas": ["ID_Nodo", "Costo_Fijo_COP_Mes"],
}


def _buscar_archivo(carpeta: Path, nombre: str) -> Path:
    directo = carpeta / nombre
    if directo.exists():
        return directo
    # tolera subcarpetas (p. ej. datos/Datos/SAP.csv al descomprimir el .rar) y mayúsculas
    for p in carpeta.rglob("*"):
        if p.is_file() and p.name.lower() == nombre.lower():
            return p
    raise FileNotFoundError(f"No se encontró {nombre} dentro de {carpeta.resolve()}")


def _leer_csv(ruta: Path, sep: str) -> pd.DataFrame:
    df = pd.read_csv(ruta, sep=sep, encoding="utf-8-sig", dtype=str, keep_default_na=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def cargar_todo(carpeta_datos: Path, carpeta_cache: Path | None = None) -> dict[str, pd.DataFrame]:
    carpeta_datos = Path(carpeta_datos)
    datos: dict[str, pd.DataFrame] = {}
    for clave, (archivo, sep) in ARCHIVOS.items():
        ruta = _buscar_archivo(carpeta_datos, archivo)
        firma = f"{ruta.resolve()}|{ruta.stat().st_size}|{ruta.stat().st_mtime_ns}"
        cache = None
        if carpeta_cache is not None and clave in ("sap", "tms"):
            carpeta_cache.mkdir(parents=True, exist_ok=True)
            cache = carpeta_cache / f"{clave}_{hashlib.md5(firma.encode()).hexdigest()[:10]}.pkl"
        if cache is not None and cache.exists():
            with open(cache, "rb") as fh:
                df = pickle.load(fh)
            LOG.info("Leído %-22s desde caché (%s filas)", archivo, f"{len(df):,}")
        else:
            df = _leer_csv(ruta, sep)
            LOG.info("Leído %-22s %s filas x %s columnas", archivo, f"{len(df):,}", df.shape[1])
            if cache is not None:
                with open(cache, "wb") as fh:
                    pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
        faltantes = [c for c in COLUMNAS_MINIMAS.get(clave, []) if c not in df.columns]
        if faltantes:
            raise ValueError(f"{archivo}: faltan columnas obligatorias {faltantes}")
        datos[clave] = df
    return datos
