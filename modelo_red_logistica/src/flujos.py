"""Flujos detallados por escenario para el mapa: planta -> CEDI -> cliente, traslados XI y mercado N/E.

El modelo optimiza a nivel ciudad y producto (planta de origen). El detalle por cliente y mercado se obtiene
repartiendo cada flujo ciudad-producto según la participación histórica de cada cliente y mercado en esa ciudad
y producto (el costo no cambia porque la tarifa es la misma para todos los clientes de la ciudad).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .preparacion import Preparado


def _participaciones(prep: Preparado, cfg: dict, clave: str) -> pd.DataFrame:
    sap = prep.sap
    b = sap[sap.en_base & ~sap.es_xi & sap.ciudad.notna()]
    if cfg["produccion"].get("excluir_descontinuadas", True):
        b = b[~b.descontinuada]
    k = {"asis": "planta_asis", "tobe": "planta_tobe", "flexible": "Unidad Negocio"}[clave]
    g = b.groupby(["ciudad", k, "Cliente", "Mercado"]).vol.sum().reset_index().rename(columns={k: "producto"})
    g["share"] = g.vol / g.groupby(["ciudad", "producto"]).vol.transform("sum")
    return g[["ciudad", "producto", "Cliente", "Mercado", "share"]]


def flujos_cliente(prep: Preparado, cfg: dict, res: dict) -> dict:
    """Devuelve por escenario: F = [planta, cedi, cliente, mercado, m3, costo] y X = [origen, destino, mercado, m3, costo]."""
    cache = {}
    salida = {}
    sap = prep.sap
    M = prep.meses_base
    for sid, r in res.items():
        if sid == "ASIS_REAL":
            b = sap[sap.en_base & ~sap.es_xi & sap.ciudad.notna()]
            f = (b.groupby(["planta_asis", "CEDI", "Cliente", "Mercado", "ciudad"]).vol.sum() / M).reset_index()
            f = f.rename(columns={"planta_asis": "planta", "CEDI": "cedi", "vol": "m3"})
            tar = r["flujos_distribucion"].set_index(["almacen", "ciudad"]).tarifa_cop_m3
            f["costo"] = f.m3 * [tar.get((c, z), np.nan) for c, z in zip(f.cedi, f.ciudad)]
            xi = sap[sap.en_base & sap.es_xi]
            xi = xi.assign(costo=np.where(xi.flota_propia, xi.imp_imputado, xi.imp),
                           destino=xi.nodo_destino.astype(object).where(xi.nodo_destino.notna(), "EXTERNO"))
            x = (xi.groupby(["nodo_origen", "destino", "Mercado"])[["vol", "costo"]].sum() / M).reset_index()
            x = x.rename(columns={"nodo_origen": "origen", "vol": "m3"})
            x = x[x.origen != x.destino]
            salida[sid] = {"F": f[["planta", "cedi", "Cliente", "Mercado", "m3", "costo"]], "X": x[["origen", "destino", "Mercado", "m3", "costo"]]}
            continue
        modo = r.get("modo_produccion", "tobe")
        if modo not in cache:
            cache[modo] = _participaciones(prep, cfg, modo)
        part = cache[modo]
        fy = r["flujos_distribucion"][["almacen", "ciudad", "producto", "m3_mes", "tarifa_cop_m3"]]
        fx = r["flujos_traslado"]
        # planta que abastece cada (almacén, producto): directa en modo fijo, proporcional a x en modo flexible
        px = fx.groupby(["almacen", "producto", "planta"]).m3_mes.sum().reset_index()
        px["sp"] = px.m3_mes / px.groupby(["almacen", "producto"]).m3_mes.transform("sum")
        f = fy.merge(px[["almacen", "producto", "planta", "sp"]], on=["almacen", "producto"], how="left")
        f["planta"] = f.planta.fillna(f.producto)
        f["sp"] = f.sp.fillna(1.0)
        f = f.merge(part, on=["ciudad", "producto"], how="left")
        f["Cliente"] = f.Cliente.fillna("(ciudad " + f.ciudad.astype(str) + ")")
        f["Mercado"] = f.Mercado.fillna("N")
        f["share"] = f.share.fillna(1.0)
        f["m3"] = f.m3_mes * f.sp * f.share
        f["costo"] = f.m3 * f.tarifa_cop_m3
        f = f.groupby(["planta", "almacen", "Cliente", "Mercado"], as_index=False)[["m3", "costo"]].sum() \
            .rename(columns={"almacen": "cedi"})
        # mercado de los traslados: según la mezcla N/E de lo que despacha cada almacén por producto
        mez = f.assign(producto=None).merge(px[["almacen", "planta"]].drop_duplicates(), left_on=["cedi", "planta"],
                                            right_on=["almacen", "planta"], how="inner")
        mez = mez.groupby(["planta", "cedi", "Mercado"]).m3.sum().reset_index()
        mez["sm"] = mez.m3 / mez.groupby(["planta", "cedi"]).m3.transform("sum")
        t = fx[fx.planta != fx.almacen].groupby(["planta", "almacen"], as_index=False)[["m3_mes", "costo_cop_mes"]].sum()
        x = t.merge(mez[["planta", "cedi", "Mercado", "sm"]], left_on=["planta", "almacen"], right_on=["planta", "cedi"], how="left")
        x["Mercado"] = x.Mercado.fillna("N")
        x["sm"] = x.sm.fillna(1.0)
        x["m3"] = x.m3_mes * x.sm
        x["costo"] = x.costo_cop_mes * x.sm
        x = x.rename(columns={"planta": "origen", "almacen": "destino"})
        salida[sid] = {"F": f[["planta", "cedi", "Cliente", "Mercado", "m3", "costo"]],
                       "X": x[["origen", "destino", "Mercado", "m3", "costo"]]}
    return salida


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
