"""Corre los escenarios definidos en parametros.yaml, la sensibilidad y arma el modelo agregado del simulador."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .modelo_optimizacion import (ModeloLineal, Tarifario, construir_instancia, evaluar_asis_real,
                                  parametros_escenario, resolver)
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
def datos_simulador(prep: Preparado, cfg: dict, tar: Tarifario, res: dict) -> dict:
    esc = {"id": "SIM", "produccion": "tobe", "cerrar_planta": True, "bodegas_candidatas": True}
    prm = parametros_escenario(cfg, esc)
    inst = construir_instancia(prep, cfg, esc, prm)
    opcionales = set(cfg["almacenamiento"].get("nodos_opcionales_tobe") or [])
    W = inst.W
    K = sorted({k for (_, k) in inst.D})
    dz = pd.Series(inst.D).rename_axis(["ciudad", "k"]).reset_index(name="m3")
    dz["zona"] = dz.ciudad.map(inst.zona_ciudad)
    R = dz.groupby("zona").m3.sum().sort_values(ascending=False).index.tolist()
    D = {r: {k: float(dz[(dz.zona == r) & (dz.k == k)].m3.sum()) for k in K} for r in R}
    peso = dz.groupby(["zona", "ciudad"]).m3.sum().reset_index()
    c_out = {}
    for w in W:
        c_out[w] = {}
        for r in R:
            sub = peso[peso.zona == r]
            c = np.array([tar.c_out(w, z) for z in sub.ciudad])
            c_out[w][r] = float(np.average(c, weights=sub.m3)) if sub.m3.sum() > 0 else float(c.mean())
    c_in = {k: {w: tar.c_in(k, w) for w in W} for k in K}
    xa = prep.xi_asis.dropna(subset=["destino"])
    obs = {(o, d): v for o, d, v in zip(xa.origen, xa.destino, xa.vol_mes)}
    nd = inst.nodos
    nodos = []
    for w in W:
        tipo = "candidata" if nd.at[w, "candidata"] else ("bodega_externa_actual" if w in opcionales else
                                                          ("3pl" if "3PL" in str(nd.at[w, "Modalidad"]) else "propio"))
        nodos.append({"id": w, "nombre": str(nd.at[w, "nombre"]), "tipo": tipo,
                      "posiciones": float(nd.at[w, "posiciones_total"]), "manejo": float(nd.at[w, "manejo_cop_m3"]),
                      "posicion_cop": float(nd.at[w, "posicion_cop_mes"]), "fijo": float(nd.at[w, "costo_fijo"]),
                      "lat": float(nd.at[w, "lat"]), "lon": float(nd.at[w, "lon"])})
    lim = {k: {w: float(obs.get((k, w), 0.0)) for w in W if w != k} for k in K}
    e0 = res.get("E0_ASIS_OPT")
    return {
        "plantas": K, "regiones": R, "almacenes": nodos, "demanda": D, "c_in": c_in, "c_out": c_out,
        "lim_xi_asis": lim, "m3_por_posicion": prep.m3_por_posicion,
        "defaults": {"dias_inventario": prm["dias_inventario"], "utilizacion_maxima": prm["utilizacion_maxima"],
                     "penalidad": prm["penalidad"], "factor_flete_traslado": 1.0, "factor_flete_distribucion": 1.0,
                     "crecimiento": 0.0},
        "referencia": {"asis_real_total": res["ASIS_REAL"]["costos"]["Total"],
                       "e0_total": e0["costos"]["Total"] if e0 else None,
                       "fijo_cerrada": float(prep.nodos.at[cfg["produccion"]["planta_cerrada"], "costo_fijo"])
                       if cfg["produccion"]["planta_cerrada"] in prep.nodos.index else 0.0},
    }


def resolver_reducido(sim: dict, prm: dict, modo: str, candidatas_activas: dict | None = None) -> dict:
    """Mismo modelo que el simulador JavaScript (para validar que ambos dan lo mismo)."""
    K, R = sim["plantas"], sim["regiones"]
    doi, util = prm["dias_inventario"], prm["utilizacion_maxima"]
    fxi, fdi, crec = prm["factor_flete_traslado"], prm["factor_flete_distribucion"], prm.get("crecimiento", 0.0)
    m3p = sim["m3_por_posicion"]
    W = []
    for n in sim["almacenes"]:
        if n["tipo"] == "candidata" and (modo == "B" or not (candidatas_activas or {}).get(n["id"], True)):
            continue
        if n["tipo"] == "bodega_externa_actual" and modo == "B":
            continue
        W.append(n)
    m = ModeloLineal()
    x, y, a, dfc = {}, {}, {}, {}
    for n in W:
        w = n["id"]
        var = n["manejo"] + n["posicion_cop"] * doi / 30 / m3p
        for k in K:
            x[(k, w)] = m.var(sim["c_in"][k][w] * fxi / 1e6)
            for r in R:
                if sim["demanda"][r][k] > 0:
                    y[(w, r, k)] = m.var((sim["c_out"][w][r] * fdi + var) / 1e6)
        binario = n["tipo"] == "candidata" or (n["tipo"] == "bodega_externa_actual" and modo == "OPT")
        if binario:
            a[w] = m.var(n["fijo"] / 1e6, 0, 1, entera=True)
        else:
            dfc[w] = m.var(prm["penalidad"] / 1e6)
    for r in R:
        for k in K:
            dv = sim["demanda"][r][k] * (1 + crec)
            if dv > 0:
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
    if modo == "A":
        for k in K:
            for n in W:
                if n["id"] != k and n["tipo"] in ("propio", "3pl"):
                    m.restr([x[(k, n["id"])]], [1.0], hi=sim["lim_xi_asis"][k][n["id"]])
    estado, fobj = m.resolver(60, 1e-6)
    fijo_const = sum(n["fijo"] for n in W if n["id"] not in a) / 1e6
    return {"estado": estado, "total_mcop": fobj + fijo_const,
            "abiertas": [w for w in a if m.v(a[w]) > 0.5],
            "x": {f"{k}->{w}": m.v(i) for (k, w), i in x.items() if m.v(i) > 1e-6}}
