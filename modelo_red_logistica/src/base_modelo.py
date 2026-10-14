"""Base de datos del modelo multiperiodo diario.

Produce un único diccionario JSON-serializable (`base`) que consumen por igual el modelo en Python
(`modelo_diario.py`) y el motor del navegador (`recursos/motor.js`). Así ambos resuelven exactamente
el mismo problema con los mismos datos.

Origen de cada bloque (H = histórico observado, E = estimado/derivado, S = supuesto parametrizable):
  demanda diaria por grupo ................ H (SAP, líneas no XI, ventana histórica)
  asignación histórica CEDI por grupo ..... H (SAP, participación de cada CEDI en la ciudad y planta)
  flujos observados (despachos y XI) ...... H (SAP; XI de flota propia con flete imputado = E)
  tarifas de traslado y distribución ...... H (rutas observadas) / E (regresión por distancia)
  tiempos de tránsito ..................... H (TMS) / E (km / 45 km/h + 3 h)
  capacidades ............................. H (posiciones de info_cedi) x S (m3 por posición, utilización)
  costos fijos ............................ H (parametros_plantas) / S (bodegas candidatas)
  costos de manejo y almacenaje ........... S (parametros.yaml)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .modelo_optimizacion import Tarifario
from .preparacion import Preparado
from .utilidades import LOG


def _horas_lane(prep: Preparado, tar: Tarifario):
    tl = prep.tiempos_lane
    hx = {(o, d): float(h) for o, d, h in zip(tl[tl.es_xi].origen, tl[tl.es_xi].destino, tl[tl.es_xi].transito_h_mediana)}
    hd = {(o, d): float(h) for o, d, h in zip(tl[~tl.es_xi].origen, tl[~tl.es_xi].destino, tl[~tl.es_xi].transito_h_mediana)}
    return hx, hd


def construir_base(prep: Preparado, cfg: dict, tar: Tarifario) -> dict:
    hz = cfg["horizonte"]
    mcfg = cfg["modelo"]
    sap = prep.sap
    n_dias = int(hz["dias_ventana"])
    fechas = pd.date_range(hz["inicio_ventana"], periods=n_dias, freq="D")
    f0, f1 = fechas[0], fechas[-1]
    m3p = prep.m3_por_posicion
    cerrada = cfg["produccion"]["planta_cerrada"]
    opcionales = set(cfg["almacenamiento"].get("nodos_opcionales_tobe") or [])

    # ------------------------------------------------------------------ nodos
    nd = prep.nodos.copy()
    cmod = mcfg["costos_nodo"]
    plantas_cap = prep.plantas
    nodos = []
    for nid in nd.index:
        r = nd.loc[nid]
        cand = bool(r["candidata"])
        es_be = cand or nid in opcionales
        mod = cmod.get(r["Modalidad"], cmod["Propio"])
        if cand:
            bc = next(b for b in cfg["bodegas_candidatas"] if b["id"] == nid)
            man_s = man_e = float(bc["manejo_cop_m3"]) / 2
            alm = float(bc["posicion_cop_mes"]) / 30 / m3p
        elif es_be:
            man_s, man_e = float(mod["manejo_salida"]), float(mod["manejo_entrada"])
            alm = float(r["posicion_cop_mes"] or 0) / 30 / m3p
        else:
            man_s, man_e, alm = float(mod["manejo_salida"]), float(mod["manejo_entrada"]), float(mod["almac_dia"])
        tipo = ("candidata" if cand else "bodega_externa" if es_be else
                "planta" if "Planta" in str(r["Tipo_Nodo"]) else "3pl" if "3PL" in str(r["Modalidad"]) else "cedi")
        cap_p = plantas_cap["capacidad_m3_mes"].get(nid) if nid in plantas_cap.index else None
        nodos.append({
            "id": nid, "nombre": str(r["nombre"]), "tipo": tipo, "es_planta": tipo == "planta",
            "ciudad": str(r["Ciudad"]), "lat": float(r["lat"]), "lon": float(r["lon"]),
            "posiciones": float(r["posiciones_total"] or 0), "costo_fijo_mes": float(r["costo_fijo"] or 0),
            "manejo_salida": man_s, "manejo_entrada": man_e, "almac_dia": alm,
            "bodega_externa": es_be, "cap_produccion_mes": float(cap_p) if cap_p is not None and pd.notna(cap_p) else None,
            "familias_planta": (plantas_cap.at[nid, "familias"] if nid in plantas_cap.index and
                                isinstance(plantas_cap.at[nid, "familias"], list) else []),
        })
    ids = [n["id"] for n in nodos]
    idx_n = {n: i for i, n in enumerate(ids)}

    # ------------------------------------------------------------------ demanda diaria por grupo
    b = sap[(sap.fecha >= f0) & (sap.fecha <= f1) & ~sap.es_xi & sap.ciudad.notna()]
    if cfg["produccion"].get("excluir_descontinuadas", True):
        b = b[~b.descontinuada]
    b = b[b.planta_asis.isin(ids) & b.planta_tobe.isin(ids)]
    clave = ["ciudad", "Unidad Negocio", "planta_asis", "planta_tobe", "Mercado"]
    gdf = b.groupby(clave).vol.sum().reset_index()
    gdf = gdf[gdf.vol > 1e-6].reset_index(drop=True)
    ciudades = sorted(gdf.ciudad.unique())
    idx_z = {c: i for i, c in enumerate(ciudades)}
    familias = sorted(gdf["Unidad Negocio"].unique())
    idx_f = {f: i for i, f in enumerate(familias)}
    gkey = {tuple(r[c] for c in clave): i for i, r in gdf.iterrows()}
    grupos = [[idx_z[r.ciudad], idx_f[r["Unidad Negocio"]], idx_n[r.planta_asis], idx_n[r.planta_tobe],
               1 if r.Mercado == "E" else 0] for _, r in gdf.iterrows()]
    b = b.assign(dia=(b.fecha - f0).dt.days)
    dd = b.groupby(["dia"] + clave).vol.sum().reset_index()
    demanda = [[int(r.dia), gkey[tuple(r[c] for c in clave)], round(float(r.vol), 4)] for _, r in dd.iterrows() if r.vol > 1e-6]

    # ------------------------------------------------------------------ zonas (ciudades de demanda)
    cc = prep.ciudades.set_index("ciudad")
    zonas = [{"id": z, "zona": str(cc.zona.get(z)) if z in cc.index else None, "depto": str(cc.depto.get(z)) if z in cc.index else None,
              "lat": float(cc.lat.get(z)) if z in cc.index and pd.notna(cc.lat.get(z)) else None,
              "lon": float(cc.lon.get(z)) if z in cc.index and pd.notna(cc.lon.get(z)) else None} for z in ciudades]

    # ------------------------------------------------------------------ asignación histórica (año base) por (ciudad, planta AS-IS)
    yb = sap[sap.en_base & ~sap.es_xi & sap.ciudad.notna() & sap.CEDI.isin(ids)]
    hist = yb.groupby(["ciudad", "planta_asis", "CEDI"]).vol.sum()
    hist_c = yb.groupby(["ciudad", "CEDI"]).vol.sum()
    asign_hist = []
    for _, r in gdf.iterrows():
        try:
            s_ = hist.loc[(r.ciudad, r.planta_asis)]
        except KeyError:
            s_ = hist_c.loc[r.ciudad] if r.ciudad in hist_c.index.get_level_values(0) else pd.Series(dtype=float)
        s_ = s_[s_ > 0]
        asign_hist.append([[idx_n[w], round(float(v / s_.sum()), 6)] for w, v in s_.items()] if len(s_) else [])

    # ------------------------------------------------------------------ distribución: tarifa y tiempo nodo -> ciudad
    hx, hd = _horas_lane(prep, tar)
    c_out, h_out, obs_out = [], [], []
    for n in nodos:
        w = n["id"]
        como = nd.at[w, "tarifas_salida_como"] if isinstance(nd.at[w, "tarifas_salida_como"], str) else w
        fila_c, fila_h, fila_o = [], [], []
        for z in ciudades:
            fila_c.append(round(tar.c_out(w, z), 2))
            fila_o.append(0 if str(tar.fuente.get(("out", w, z), "")).startswith("estimada") else 1)
            h = hd.get((como, z))
            if h is None:
                km = tar.km_nc(w, z)
                h = km / 45 + 3 if not np.isnan(km) else 72.0
            fila_h.append(round(float(h), 2))
        c_out.append(fila_c)
        h_out.append(fila_h)
        obs_out.append(fila_o)

    # ------------------------------------------------------------------ traslados: tarifa terceros / flota propia, km, tiempos
    m3v_def = float(prep.tarifas_xi.m3_viaje_propia.median()) if prep.tarifas_xi.m3_viaje_propia.notna().any() else 50.0
    lanes = {}
    for u in ids:
        for v in ids:
            if u == v:
                continue
            como = nd.at[v, "tarifa_entrada_como"] if isinstance(nd.at[v, "tarifa_entrada_como"], str) else None
            key = (u, como) if como and (u, como) in tar.lane_xi else (u, v)
            L = tar.lane_xi.get(key) if key in tar.obs_xi else None
            c = tar.c_in(u, v)
            km = tar.km_nn(u, v)
            h = hx.get(key) or hx.get((u, v))
            if h is None:
                h = (km / 45 + 3) if not np.isnan(km) else 48.0
            lanes[f"{u}|{v}"] = {"tt": round(float(L["t_terc"]) if L and L["t_terc"] is not None else float(c), 2),
                                 "tp": round(float(L["t_prop"]), 2) if L and L["t_prop"] is not None else None,
                                 "s": round(float(L["s"]), 4) if L else 0.0, "km": round(float(km), 1) if not np.isnan(km) else 0.0,
                                 "m3v": round(float(L["m3v"]), 2) if L and L["m3v"] else None,
                                 "horas": round(float(h), 1), "obs": bool(L)}
    xa = prep.xi_asis.dropna(subset=["destino"])
    arcos_hist = sorted({f"{o}|{d_}" for o, d_, v in zip(xa.origen, xa.destino, xa.vol_mes) if o != d_ and v > 1 and o in idx_n and d_ in idx_n})

    # ------------------------------------------------------------------ flujos observados en la ventana (escenario Actual)
    dis = b[b.CEDI.isin(ids)].groupby(["dia", "CEDI"] + clave).vol.sum().reset_index()
    obs_desp = [[int(r.dia), idx_n[r.CEDI], gkey[tuple(r[c] for c in clave)], round(float(r.vol), 4)]
                for _, r in dis.iterrows() if r.vol > 1e-6]
    xi = sap[(sap.fecha >= f0) & (sap.fecha <= f1) & sap.es_xi]
    if cfg["produccion"].get("excluir_descontinuadas", True):
        xi = xi[~xi.descontinuada]
    xi = xi.assign(dia=(xi.fecha - f0).dt.days, costo=np.where(xi.flota_propia, xi.imp_imputado, xi.imp),
                   destino=xi.nodo_destino.astype(object).where(xi.nodo_destino.notna(), "EXTERNO"))
    xi = xi[xi.nodo_origen.isin(ids) & xi["Unidad Negocio"].isin(familias) & xi.planta_asis.isin(ids)]
    gx = xi.groupby(["dia", "nodo_origen", "destino", "Unidad Negocio", "planta_asis", "flota_propia"])[["vol", "costo"]].sum().reset_index()
    obs_xi = [[int(r.dia), idx_n[r.nodo_origen], idx_n.get(r.destino, -1), idx_f[r["Unidad Negocio"]], idx_n[r.planta_asis],
               round(float(r.vol), 4), round(float(r.costo), 0), 1 if r.flota_propia else 0] for _, r in gx.iterrows()]

    # ------------------------------------------------------------------ clientes y referencias (para reportes)
    cl = sap[sap.en_base & ~sap.es_xi & sap.ciudad.notna() & ~sap.descontinuada]
    gc = cl.groupby(["ciudad", "Unidad Negocio", "planta_asis", "Mercado", "Cliente"]).vol.sum().reset_index()
    gc["share"] = gc.vol / gc.groupby(["ciudad", "Unidad Negocio", "planta_asis", "Mercado"]).vol.transform("sum")
    clientes = sorted(gc.Cliente.unique())
    idx_c = {c: i for i, c in enumerate(clientes)}
    cli_shares = [[] for _ in grupos]
    gk2 = {(r[0], r[1], r[2], r[4]): i for i, r in enumerate(grupos)}
    for r in gc.itertuples():
        key = (idx_z.get(r.ciudad), idx_f.get(r[2]), idx_n.get(r.planta_asis), 1 if r.Mercado == "E" else 0)
        gi = gk2.get(key)
        if gi is not None:
            cli_shares[gi].append([idx_c[r.Cliente], round(float(r.share), 5)])
    refs = {}
    ref_df = cl.groupby(["Unidad Negocio", "planta_tobe", "Referencia"]).vol.sum().reset_index()
    for (f, pl), g in ref_df.groupby(["Unidad Negocio", "planta_tobe"]):
        g = g.sort_values("vol", ascending=False)
        tot = g.vol.sum()
        refs[f"{f}|{pl}"] = [[r.Referencia, round(float(r.vol / tot), 4)] for r in g.head(8).itertuples()]
    LOG.info("Base diaria: %d días, %d grupos de demanda, %d registros diarios, %d ciudades, %d clientes, %d nodos",
             n_dias, len(grupos), len(demanda), len(ciudades), len(clientes), len(nodos))

    imp = prep.info["imputacion_xi"]
    return {
        "meta": {"fechas": [str(x.date()) for x in fechas], "dia_semana": [int(x.dayofweek) for x in fechas],
                 "m3_por_posicion": m3p, "planta_cerrada": cerrada, "m3_viaje_def": m3v_def,
                 "imputacion": {"metodo": imp["metodo"], "costo_km": imp["costo_km_cop"], "fijo_viaje": imp["costo_fijo_viaje_cop"]}},
        "nodos": nodos, "familias": familias, "zonas": zonas, "grupos": grupos, "demanda": demanda,
        "asign_hist": asign_hist, "c_out": c_out, "h_out": h_out, "obs_out": obs_out, "lanes": lanes, "arcos_hist": arcos_hist,
        "obs_desp": obs_desp, "obs_xi": obs_xi, "clientes": clientes, "cli_shares": cli_shares, "refs": refs,
    }


def parametros_defecto(cfg: dict, base: dict) -> dict:
    """Parámetros del modelo (estructura idéntica en Python y JavaScript)."""
    m = cfg["modelo"]
    hz = cfg["horizonte"]
    nodos = {}
    for n in base["nodos"]:
        modo = "fijo"
        if n["bodega_externa"]:
            modo = "opcional"
        nodos[n["id"]] = {"modo": modo, "posiciones": n["posiciones"], "manejo_salida": n["manejo_salida"],
                          "manejo_entrada": n["manejo_entrada"], "almac_dia": n["almac_dia"], "costo_fijo_mes": n["costo_fijo_mes"]}
    imp = base["meta"]["imputacion"]
    return {
        "horizonte_dias": int(hz["horizonte_dias"]), "desplazamiento": int(hz.get("desplazamiento", 0)),
        "proyeccion": m.get("proyeccion", "replica"), "crecimiento": float(m.get("crecimiento", 0)),
        "crec_N": float(m.get("crec_N", 0)), "crec_E": float(m.get("crec_E", 0)), "crec_zona": {}, "crec_familia": {},
        "planta_cerrada": base["meta"]["planta_cerrada"], "usar_reparto": False, "reparto": {},
        "asignacion": "libre", "utilizacion_maxima": float(m["utilizacion_maxima"]),
        "produccion": dict(m["produccion"]), "inventario": dict(m["inventario"]),
        "traslados": m.get("traslados", "plantas_y_observados"),
        "servicio": dict(m["servicio"]), "holguras": dict(m["holguras"]),
        "costo_capital_dia": float(m.get("costo_capital_dia", 0)),
        "transporte": {"f_terc": float(m["transporte"]["f_terc"]), "f_prop": float(m["transporte"]["f_prop"]),
                       "f_dist": float(m["transporte"]["f_dist"]), "metodo": imp["metodo"] if imp["metodo"] == "km" else "similares",
                       "costo_km": imp["costo_km"], "fijo_viaje": imp["fijo_viaje"], "s_propia": None, "f_zona": {}},
        "nodos": nodos,
    }


def parametros_escenario(cfg: dict, base: dict, esc: dict) -> dict:
    import copy
    p = copy.deepcopy(parametros_defecto(cfg, base))
    for k, v in esc.items():
        if k in ("id", "nombre", "bodegas", "nodos"):
            continue
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            p[k].update(v)
        else:
            p[k] = v
    for nid, modo in (esc.get("bodegas") or {}).items():
        if nid in p["nodos"]:
            p["nodos"][nid]["modo"] = modo
    for nid, aj in (esc.get("nodos") or {}).items():
        p["nodos"].setdefault(nid, {}).update(aj)
    return p


def _redondeo(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x
