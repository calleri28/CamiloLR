"""Coordenadas y ubicación de los clientes (para el mapa por cliente)."""
from __future__ import annotations

import pandas as pd

from .preparacion import Preparado


def clientes_geo(prep: Preparado, d: dict) -> pd.DataFrame:
    """Clientes con ciudad, departamento, zona y coordenadas (cliente -> ciudad -> departamento)."""
    from .utilidades import a_numero, normalizar_serie_texto
    cl = d["maestro_clientes"].copy()
    cl["ciudad"] = normalizar_serie_texto(cl["Ciudad"])
    cl["lat"] = a_numero(cl["Latitud"])
    cl["lon"] = a_numero(cl["Longitud"])
    cc = prep.ciudades.set_index("ciudad")
    cl["lat"] = cl["lat"].fillna(cl.ciudad.map(cc.lat))
    cl["lon"] = cl["lon"].fillna(cl.ciudad.map(cc.lon))
    cl["zona"] = cl.ciudad.map(cc.zona)
    cl["depto"] = cl["Departamento"].str.strip().str.title()
    return cl.rename(columns={"Nombre_Cliente": "cliente"})[["cliente", "ciudad", "depto", "zona", "lat", "lon"]]
