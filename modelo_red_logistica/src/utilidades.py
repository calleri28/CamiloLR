"""Funciones auxiliares: conversión de números en formato colombiano, textos, distancias y log."""
from __future__ import annotations

import logging
import math
import unicodedata

import numpy as np
import pandas as pd

LOG = logging.getLogger("modelo_red")


def configurar_log(ruta_log=None) -> logging.Logger:
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    consola = logging.StreamHandler()
    consola.setFormatter(fmt)
    LOG.addHandler(consola)
    if ruta_log:
        archivo = logging.FileHandler(ruta_log, mode="w", encoding="utf-8")
        archivo.setFormatter(fmt)
        LOG.addHandler(archivo)
    return LOG


def a_numero(serie: pd.Series) -> pd.Series:
    """Convierte textos tipo '1.234,56' / ' 24.300 ' / '0,029' a float.

    En todos los archivos la coma es el separador decimal y el punto el de miles.
    """
    s = serie.astype("string").str.strip()
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def normalizar_texto(valor) -> str | None:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)) or valor is pd.NA:
        return None
    txt = unicodedata.normalize("NFKD", str(valor)).encode("ascii", "ignore").decode("ascii")
    txt = " ".join(txt.upper().split())
    return txt or None


def normalizar_serie_texto(serie: pd.Series) -> pd.Series:
    unicos = serie.dropna().unique()
    mapa = {u: normalizar_texto(u) for u in unicos}
    return serie.map(mapa)


def normalizar_viaje(serie: pd.Series) -> pd.Series:
    """El viaje en SAP trae ceros a la izquierda ('0984938') y en TMS no ('984938')."""
    s = serie.astype("string").str.strip().str.lstrip("0")
    return s.where(s.str.len() > 0)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def pct(a, b) -> float:
    return float(a) / float(b) if b else 0.0
