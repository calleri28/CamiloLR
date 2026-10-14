"""Corre el escenario Actual (reconstruido), los escenarios del modelo diario y las combinaciones de bodegas externas."""
from __future__ import annotations

import copy
import itertools

from . import modelo_diario as md
from .base_modelo import parametros_escenario
from .utilidades import LOG


def correr(base: dict, cfg: dict) -> dict:
    sc = cfg.get("solver", {})
    tiempo, gap = sc.get("tiempo_maximo_seg", 300), sc.get("gap_relativo", 1e-3)
    escs = cfg["escenarios_diarios"]
    res, prms = {}, {}
    ref = next((e for e in escs if e["id"].startswith("S3")), escs[0])
    p_act = parametros_escenario(cfg, base, ref)
    a = md.reconstruir_actual(base, p_act)
    a["id"], a["nombre"] = "S0_ACTUAL", "Actual (operación observada)"
    res["S0_ACTUAL"], prms["S0_ACTUAL"] = a, p_act
    LOG.info("  %-20s %-26s %s MCOP/mes eq.", "S0_ACTUAL", a["estado"], f"{a['costos']['Total mensual equivalente'] / 1e6:,.0f}")
    for e in escs:
        prm = parametros_escenario(cfg, base, e)
        r = md.resolver(base, prm, tiempo, gap)
        r["id"], r["nombre"] = e["id"], e["nombre"]
        res[e["id"]], prms[e["id"]] = r, prm
        LOG.info("  %-20s %-10s %s MCOP/mes eq. (%d vars, %.1fs) bodegas: %s", e["id"], r["estado"],
                 f"{r['costos']['Total mensual equivalente'] / 1e6:,.0f}", r["variables"], r["tiempo_s"],
                 ", ".join(r["kpis"]["bodegas_abiertas"]) or "ninguna")
    return {"resultados": res, "parametros": prms}


def combinaciones_bodegas(base: dict, cfg: dict) -> list:
    cb = cfg.get("combinaciones_bodegas") or {}
    if not cb.get("activa"):
        return []
    sc = cfg.get("solver", {})
    escs = {e["id"]: e for e in cfg["escenarios_diarios"]}
    ref = escs.get("S3_OPTIMIZADO", cfg["escenarios_diarios"][0])
    ids = [b for b in cb.get("bodegas", []) if any(n["id"] == b for n in base["nodos"])]
    filas = []
    LOG.info("Combinaciones de bodegas externas: %d", 2 ** len(ids) + 1)
    combos = list(itertools.product([False, True], repeat=len(ids)))
    for combo in combos + [None]:
        e = copy.deepcopy(ref)
        if combo is None:
            e["bodegas"] = {b: "opcional" for b in ids}
            nombre = "El optimizador decide"
        else:
            e["bodegas"] = {b: ("forzado" if on else "cerrado") for b, on in zip(ids, combo)}
            abiertas = [b for b, on in zip(ids, combo) if on]
            nombre = "Ninguna bodega externa" if not abiertas else "Solo " + " + ".join(abiertas)
        prm = parametros_escenario(cfg, base, e)
        r = md.resolver(base, prm, sc.get("tiempo_maximo_seg", 300), sc.get("gap_relativo", 1e-3))
        c = r["costos"]
        uso = {b: max(r["nodos_dia"][b]["inv_fin"]) if b in r["nodos_dia"] else 0.0 for b in ids}
        filas.append({"combinacion": nombre, "bodegas_abiertas": ", ".join(r["kpis"]["bodegas_abiertas"]) or "-",
                      "estado": r["estado"], "total_mes": c["Total mensual equivalente"],
                      "logistico_mes": c["Costo logístico mensual equivalente"],
                      "transporte_mes": c["Total transporte"] * 30 / r["H"], "almacenamiento_mes": c["Total almacenamiento"] * 30 / r["H"],
                      "fijo_be_mes": c["Fijo bodegas externas"] * 30 / r["H"], "penalizaciones_mes": c["Total penalizaciones"] * 30 / r["H"], "fijo_nodos_mes": c["Total fijo nodos"] * 30 / r["H"],
                      "exceso_max_m3": r["kpis"]["exceso_max_m3"], "fill_rate": r["kpis"]["fill_rate"],
                      **{f"inv_max_{b}": uso[b] for b in ids}})
    mejor = min(filas[:-1], key=lambda f: f["total_mes"])
    for f in filas:
        f["vs_mejor_mes"] = f["total_mes"] - mejor["total_mes"]
    return filas
