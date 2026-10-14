"""Genera el reporte HTML autocontenido (funciona sin internet: Plotly, mapa base, solver HiGHS WebAssembly y motor del modelo van embebidos)."""
from __future__ import annotations

import base64
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import metodologia
from .preparacion import Preparado

RECURSOS = Path(__file__).resolve().parent / "recursos"
ORDEN = ["S0_ACTUAL", "S1_ACTUAL_OPT", "S2_SIN_PLANTA", "S3_OPTIMIZADO", "S4_BE_OBLIGATORIA", "S5_SIN_BE", "S6_DEMANDA_MAS20", "S7_PARAMETRIZADO"]


def _limpiar(o):
    if isinstance(o, dict):
        return {str(k): _limpiar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_limpiar(v) for v in o]
    if isinstance(o, np.ndarray):
        return _limpiar(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else (round(f, 4) if abs(f) < 1e12 else f)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if o is pd.NA or o is pd.NaT:
        return None
    if isinstance(o, (pd.Timestamp, dt.date)):
        return str(o)
    return o


def _registros(df: pd.DataFrame) -> list:
    d2 = df.copy()
    num = d2.select_dtypes(include=[np.number]).columns
    d2[num] = d2[num].astype(float).where(np.isfinite(d2[num].astype(float)))
    d2 = d2.astype(object).where(d2.notna(), None)
    return _limpiar(d2.to_dict("records"))


def _clientes_geo(base: dict, clientes: pd.DataFrame | None) -> list:
    """Lista alineada con base['clientes']: ciudad y coordenadas de cada cliente."""
    cg = clientes.drop_duplicates("cliente").set_index("cliente") if clientes is not None else None
    out = []
    for c in base["clientes"]:
        if cg is not None and c in cg.index:
            r = cg.loc[c]
            out.append({"ciudad": r.ciudad, "lat": r.lat, "lon": r.lon})
        else:
            out.append({"ciudad": None, "lat": None, "lon": None})
    return out


def _viajes_xi(prep: Preparado) -> list:
    v = prep.viajes_xi[prep.viajes_xi.en_base & prep.viajes_xi.propia]
    g = (v.groupby(["origen", "destino"], as_index=False)
         .agg(viajes=("viaje_k", "size"), m3=("vol", "sum"), km=("km", "first"), imp_similar=("imp_similar", "sum"), imp_km=("imp_km", "sum"),
              fuente=("fuente_similar", lambda s_: s_.mode().iat[0] if len(s_.dropna()) else "")))
    mb = prep.meses_base
    g = g.assign(m3=g.m3 / mb, imp_similar=g.imp_similar / mb, imp_km=g.imp_km / mb, viajes=g.viajes / mb).sort_values("m3", ascending=False)
    return _registros(g)


def generar_html(ruta: Path, prep: Preparado, cfg: dict, base: dict, corrida: dict, combos: list, pruebas: list,
                 calidad: pd.DataFrame, resumen: dict, clientes: pd.DataFrame | None = None):
    from plotly.offline import get_plotlyjs

    info = prep.info
    R, P = corrida["resultados"], corrida["parametros"]
    orden = [k for k in ORDEN if k in R] + [k for k in R if k not in ORDEN]
    esc = {k: {"nombre": R[k]["nombre"], "prm": P[k], "res": R[k]} for k in orden}
    for c in combos:
        c.setdefault("fijo_nodos_mes", c["total_mes"] - c["transporte_mes"] - c["almacenamiento_mes"] - c["penalizaciones_mes"])
    S3 = R.get("S3_OPTIMIZADO") or R[orden[-1]]
    hz = cfg.get("horizonte", {})
    D = {
        "meta": {"titulo": f"Red logística · cierre de planta {base['meta']['planta_cerrada']}",
                 "subtitulo": f"Modelo diario de {S3['H']} días ({S3['fechas'][0]} a {S3['fechas'][-1]}, ventana desde {hz.get('inicio_ventana')}) · "
                              f"datos SAP {info['periodo_datos'][0]} a {info['periodo_datos'][1]} · costos en COP por mes equivalente · "
                              f"generado {dt.datetime.now():%Y-%m-%d %H:%M}",
                 "match_tms": info["viajes_match_normalizado"] / max(info["viajes_sap"], 1), "pct_flete_valido": float(prep.sap.flete_valido.mean()),
                 "cerrada": base["meta"]["planta_cerrada"]},
        "base": base,
        "orden": orden,
        "escenarios": esc,
        "combos": combos,
        "pruebas": pruebas,
        "resumen": resumen,
        "calidad": _registros(calidad),
        "viajes_xi": _viajes_xi(prep),
        "clientes_geo": _clientes_geo(base, clientes),
        "metodologia_html": metodologia.html(base, cfg, pruebas, corrida),
    }
    datos = json.dumps(_limpiar(D), ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    html = (RECURSOS / "plantilla.html").read_text(encoding="utf-8")
    rep = {
        "/*__DATA__*/": datos,
        "/*__WASM__*/": base64.b64encode((RECURSOS / "highs.wasm").read_bytes()).decode("ascii"),
        "/*__TOPOJSON__*/": "window.PlotlyGeoAssets={topojson:{world_50m:" + (RECURSOS / "world_50m.json").read_text(encoding="utf-8") + "}};",
        "/*__PLOTLY__*/": get_plotlyjs().replace("</script", "<\\/script"),
        "/*__HIGHS__*/": (RECURSOS / "highs.js").read_text(encoding="utf-8").replace("</script", "<\\/script"),
        "/*__MOTOR__*/": (RECURSOS / "motor.js").read_text(encoding="utf-8"),
    }
    for k, v in rep.items():   # un solo reemplazo por marcador (los contenidos embebidos podrían contener otros marcadores)
        i = html.index(k)
        html = html[:i] + v + html[i + len(k):]
    Path(ruta).write_text(html, encoding="utf-8")
