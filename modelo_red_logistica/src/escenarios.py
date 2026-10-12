"""Corre los escenarios definidos en parametros.yaml, la sensibilidad y arma el modelo agregado del simulador."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .modelo_optimizacion import (ModeloLineal, Tarifario, construir_instancia, evaluar_asis_real,
                                  evaluar_red_actual, parametros_escenario, resolver)
from .preparacion import Preparado
from .utilidades import LOG

COMPARACION = ["E1_TRASLADO_CEDI", "E2_BODEGA_EXTERNA", "E3_OPTIMO"]


def _fijo_bodegas(res: dict, cfg: dict) -> float:
    opc = set(cfg["almacenamiento"].get("nodos_opcionales_tobe") or []) | {b["id"] for b in cfg.get("bodegas_candidatas") or []}
    rn = res["nodos"]
    return float(rn.loc[rn.almacen.isin(opc) & rn.abierto, "costo_fijo_cop"].sum())


def correr_escenarios(prep: Preparado, cfg: dict) -> tuple[dict, Tarifario]:
    tar = Tarifario(prep, cfg)
    res = {"ASIS_REAL": evaluar_asis_real(prep, tar, cfg)}
    LOG.info("Optimizando %d escenarios...", len(cfg["escenarios"]))
    for esc in cfg["escenarios"]:
        if esc.get("tipo") == "red_actual":
            r = evaluar_red_actual(prep, tar, cfg, esc)
            LOG.info("  %-18s evaluación  z=%s MCOP", esc["id"], f"{r['costos']['Total'] / 1e6:,.0f}")
            r["fijo_bodegas_externas"] = r["costos"]["Fijo bodegas externas"]
            r["descripcion"] = esc
            res[esc["id"]] = r
            continue
        prm = parametros_escenario(cfg, esc)
        inst = construir_instancia(prep, cfg, esc, prm)
        r = resolver(inst, tar, prep, cfg)
        r["fijo_bodegas_externas"] = _fijo_bodegas(r, cfg)
        r["descripcion"] = esc
        res[esc["id"]] = r
    return res, tar


def sensibilidad(prep: Preparado, cfg: dict, tar: Tarifario) -> pd.DataFrame:
    s = cfg.get("sensibilidad") or {}
    if not s.get("activa"):
        return pd.DataFrame()
    escs = {e["id"]: e for e in cfg["escenarios"]}
    ids = [i for i in COMPARACION if i in escs]
    filas = []
    LOG.info("Sensibilidad: %d corridas", len(ids) * (len(s.get("dias_inventario", [])) + len(s.get("factor_flete", []))))
    for var, valores in (("dias_inventario", s.get("dias_inventario", [])), ("factor_flete", s.get("factor_flete", []))):
        for v in valores:
            for i in ids:
                ov = {"dias_inventario": float(v)} if var == "dias_inventario" else \
                    {"factor_flete_traslado": float(v), "factor_flete_distribucion": float(v)}
                prm = parametros_escenario(cfg, escs[i], ov)
                inst = construir_instancia(prep, cfg, escs[i], prm)
                r = resolver(inst, tar, prep, cfg)
                filas.append({"variable": var, "valor": float(v), "escenario": i, "nombre": escs[i]["nombre"],
                              "costo_total_cop": r["costos"]["Total"],
                              "costo_variable_cop": r["costos"]["Total variable + fijo opcional"],
                              "fijo_bodegas_externas": _fijo_bodegas(r, cfg),
                              "deficit_m3": r["kpis"]["deficit_m3"],
                              "bodegas_abiertas": ", ".join(r["kpis"]["bodegas_abiertas"]) or "-"})
    return pd.DataFrame(filas)


def punto_equilibrio(res: dict) -> dict:
    """Costo fijo máximo que podría pagar la bodega externa para empatar con el traslado a CEDI."""
    if not all(k in res for k in ("E1_TRASLADO_CEDI", "E2_BODEGA_EXTERNA")):
        return {}
    b, a = res["E1_TRASLADO_CEDI"], res["E2_BODEGA_EXTERNA"]
    var_a = a["costos"]["Total"] - a["fijo_bodegas_externas"]
    return {"costo_total_traslado": b["costos"]["Total"], "costo_bodega_sin_fijo": var_a,
            "fijo_bodega_actual": a["fijo_bodegas_externas"],
            "fijo_maximo_bodega_para_empatar": b["costos"]["Total"] - var_a}


# ---------------------------------------------------------------------------
# Modelo agregado por zona comercial (simulador HTML / Excel)
# ---------------------------------------------------------------------------
PRESETS = {
    "ACTUAL": {"cerrar": False, "co99": "abierta", "candidatas": False, "limitar": False},
    "B": {"cerrar": True, "co99": "cerrada", "candidatas": False, "limitar": False},
    "A": {"cerrar": True, "co99": "abierta", "candidatas": True, "limitar": True},
    "OPT": {"cerrar": True, "co99": "decide", "candidatas": True, "limitar": False},
}


def datos_simulador(prep: Preparado, cfg: dict, tar: Tarifario, res: dict) -> dict:
    cerrada = cfg["produccion"]["planta_cerrada"]
    opcionales = set(cfg["almacenamiento"].get("nodos_opcionales_tobe") or [])
    esc = {"id": "SIM", "produccion": "tobe", "cerrar_planta": False, "bodegas_candidatas": True}
    prm = parametros_escenario(cfg, esc)
    inst = construir_instancia(prep, cfg, esc, prm)
    W = inst.W
    nd = inst.nodos
    sap = prep.sap
    b = sap[sap.en_base & ~sap.es_xi & sap.ciudad.notna()]
    if cfg["produccion"].get("excluir_descontinuadas", True):
        b = b[~b.descontinuada]
    crec = 1.0 + float(cfg["produccion"].get("crecimiento_demanda", 0.0))
    g = (b.groupby(["zona", "Mercado", "planta_asis", "planta_tobe"]).vol.sum() / prep.meses_base * crec).reset_index()
    R = g.groupby("zona").vol.sum().sort_values(ascending=False).index.tolist()
    K = sorted(set(g.planta_asis) | set(g.planta_tobe))
    demanda = [[r.zona, r.Mercado, r.planta_asis, r.planta_tobe, float(r.vol)] for r in g.itertuples() if r.vol > 0]
    mov = g[g.planta_asis == cerrada].groupby("planta_tobe").vol.sum()
    reparto = (mov / mov.sum()).to_dict() if mov.sum() > 0 else {}
    # tarifas de distribución por zona: promedio ponderado por demanda de cada ciudad
    dz = b.groupby(["zona", "ciudad"]).vol.sum().reset_index()
    c_out = {}
    for w in W:
        c_out[w] = {}
        for r in R:
            sub = dz[dz.zona == r]
            c = np.array([tar.c_out(w, z) for z in sub.ciudad])
            c_out[w][r] = float(np.average(c, weights=sub.vol)) if sub.vol.sum() > 0 else float(c.mean())
    lanes = {}
    for k in K:
        lanes[k] = {}
        for w in W:
            if k == w:
                continue
            como = nd.at[w, "tarifa_entrada_como"] if w in nd.index else None
            key = (k, como) if isinstance(como, str) and (k, como) in tar.lane_xi else (k, w)
            L = tar.lane_xi.get(key) if key in tar.obs_xi else None
            c = tar.c_in(k, w)
            km = tar.km_nn(k, w)
            lanes[k][w] = {"tt": float(L["t_terc"]) if L and L["t_terc"] is not None else float(c),
                           "tp": float(L["t_prop"]) if L and L["t_prop"] is not None else None,
                           "s": float(L["s"]) if L else 0.0, "km": float(km) if not np.isnan(km) else 0.0,
                           "m3v": float(L["m3v"]) if L and L["m3v"] else None, "obs": bool(L)}
    m3v_def = float(prep.tarifas_xi.m3_viaje_propia.median()) if prep.tarifas_xi.m3_viaje_propia.notna().any() else 50.0
    xa = prep.xi_asis.dropna(subset=["destino"])
    obs = {(o, d_): v for o, d_, v in zip(xa.origen, xa.destino, xa.vol_mes)}
    nodos = []
    for w in W:
        tipo = "candidata" if nd.at[w, "candidata"] else ("bodega_externa_actual" if w in opcionales else
                                                          ("3pl" if "3PL" in str(nd.at[w, "Modalidad"]) else "propio"))
        nodos.append({"id": w, "nombre": str(nd.at[w, "nombre"]), "tipo": tipo, "posiciones": float(nd.at[w, "posiciones_total"]),
                      "manejo": float(nd.at[w, "manejo_cop_m3"]), "posicion_cop": float(nd.at[w, "posicion_cop_mes"]),
                      "fijo": float(nd.at[w, "costo_fijo"]), "lat": float(nd.at[w, "lat"]), "lon": float(nd.at[w, "lon"]),
                      "es_planta": w in K})
    cc = prep.ciudades.set_index("ciudad")
    zg = dz.assign(lat=dz.ciudad.map(cc.lat), lon=dz.ciudad.map(cc.lon)).dropna(subset=["lat"])
    zonas_geo = {r: [float(np.average(s_.lat, weights=s_.vol)), float(np.average(s_.lon, weights=s_.vol))]
                 for r, s_ in zg.groupby("zona") if s_.vol.sum() > 0}
    imp = prep.info["imputacion_xi"]
    cap = prep.plantas.capacidad_m3_mes.dropna()
    return {
        "plantas": K, "planta_cerrada": cerrada, "regiones": R, "zonas_geo": zonas_geo, "almacenes": nodos,
        "demanda": demanda, "lanes": lanes, "c_out": c_out, "m3_por_posicion": prep.m3_por_posicion, "m3_viaje_def": m3v_def,
        "lim_xi_asis": {k: {w: float(obs.get((k, w), 0.0)) for w in W if w != k} for k in K},
        "cap_planta": {k: float(v) for k, v in cap.items()},
        "defaults": {"dias_inventario": prm["dias_inventario"], "utilizacion_maxima": prm["utilizacion_maxima"],
                     "penalidad": prm["penalidad"], "factor_flete_traslado": 1.0, "factor_flota_propia": 1.0,
                     "factor_flete_distribucion": 1.0, "metodo": imp["metodo"] if imp["metodo"] == "km" else "similares",
                     "costo_km": imp["costo_km_cop"], "costo_fijo_viaje": imp["costo_fijo_viaje_cop"], "s_propia": None,
                     "crecimiento": 0.0, "crec_N": 0.0, "crec_E": 0.0, "crec_zona": {}, "factor_zona": {},
                     "reparto": reparto, "usar_reparto": False, "ajustes": {},
                     "pers": {"cerrar": True, "co99": "decide", "candidatas": True, "limitar": False}},
        "referencia": {k: {"nombre": r["nombre"], "total": r["costos"]["Total"]} for k, r in res.items()},
    }


def resolver_reducido(sim: dict, prm: dict, modo: str) -> dict:
    """Mismo modelo que simulador.js (prueba de paridad Python <-> navegador)."""
    pol = prm["pers"] if modo == "PERS" else PRESETS[modo]
    R, m3p, cerr = sim["regiones"], sim["m3_por_posicion"], sim["planta_cerrada"]
    doi, util, fdi = prm["dias_inventario"], prm["utilizacion_maxima"], prm["factor_flete_distribucion"]
    reparto = prm["reparto"]
    tot = sum(v for v in reparto.values() if v)
    D: dict = {}
    for zona, merc, pa, pt, m3 in sim["demanda"]:
        gg = (1 + prm.get("crecimiento", 0)) * (1 + (prm.get("crec_E", 0) if merc == "E" else prm.get("crec_N", 0))) * \
             (1 + prm.get("crec_zona", {}).get(zona, 0))
        v = m3 * gg
        if not pol["cerrar"]:
            partes = {pa: v}
        elif pa == cerr:
            partes = {p: v * f / tot for p, f in reparto.items() if f and p != cerr} if (prm.get("usar_reparto") and tot > 0) else {pt: v}
        else:
            partes = {(pa if pt == cerr else pt): v}
        for k, q in partes.items():
            D.setdefault(zona, {}).setdefault(k, 0.0)
            D[zona][k] += q
    K = sorted({k for r in D for k in D[r]})
    W = []
    for n in sim["almacenes"]:
        n = dict(n, **prm.get("ajustes", {}).get(n["id"], {}))
        if n.get("activa") is False or (n["id"] == cerr and pol["cerrar"]):
            continue
        if n["tipo"] == "candidata" and not pol["candidatas"]:
            continue
        if n["tipo"] == "bodega_externa_actual" and pol["co99"] == "cerrada":
            continue
        W.append(n)

    def cin(k, w):
        if k == w:
            return 0.0
        L = sim["lanes"][k][w]
        s_ = L["s"] if prm.get("s_propia") is None else prm["s_propia"]
        tt = L["tt"] * prm["factor_flete_traslado"]
        if prm["metodo"] == "km":
            tp = (prm["costo_fijo_viaje"] + prm["costo_km"] * L["km"]) / (L["m3v"] or sim["m3_viaje_def"]) if L["km"] > 0 else 0.0
        else:
            tp = L["tt"] if L["tp"] is None else L["tp"]
        return (1 - s_) * tt + s_ * tp * prm["factor_flota_propia"]

    m = ModeloLineal()
    x, y, a, dfc = {}, {}, {}, {}
    for n in W:
        w = n["id"]
        var = n["manejo"] + n["posicion_cop"] * doi / 30 / m3p
        for k in K:
            x[(k, w)] = m.var(cin(k, w) / 1e6)
            for r in R:
                if D.get(r, {}).get(k, 0) > 1e-9:
                    fz = prm.get("factor_zona", {}).get(r, 1)
                    y[(w, r, k)] = m.var((sim["c_out"][w][r] * fdi * fz + var) / 1e6)
        if n["tipo"] == "candidata" or (n["tipo"] == "bodega_externa_actual" and pol["co99"] == "decide"):
            a[w] = m.var(n["fijo"] / 1e6, 0, 1, entera=True)
        else:
            dfc[w] = m.var(prm["penalidad"] / 1e6)
    for r in R:
        for k in K:
            dv = D.get(r, {}).get(k, 0)
            if dv > 1e-9:
                m.restr([y[(n["id"], r, k)] for n in W], [1.0] * len(W), dv, dv)
    for n in W:
        w = n["id"]
        for k in K:
            iy = [y[(w, r, k)] for r in R if (w, r, k) in y]
            m.restr([x[(k, w)]] + iy, [1.0] + [-1.0] * len(iy), 0, 0)
        iy = [i for (ww, _, _), i in y.items() if ww == w]
        cap = n["posiciones"] * m3p * util
        if w in a:
            m.restr(iy + [a[w]], [doi / 30] * len(iy) + [-cap], hi=0)
        else:
            m.restr(iy + [dfc[w]], [doi / 30] * len(iy) + [-1.0], hi=cap)
        if pol["limitar"]:
            for k in K:
                if w != k and n["tipo"] in ("propio", "3pl"):
                    m.restr([x[(k, w)]], [1.0], hi=sim["lim_xi_asis"].get(k, {}).get(w, 0.0))
    estado, fobj = m.resolver(60, 1e-6)
    fijo_const = sum(n["fijo"] for n in W if n["id"] not in a) / 1e6
    return {"estado": estado, "total_mcop": fobj + fijo_const, "abiertas": [w for w in a if m.v(a[w]) > 0.5]}
