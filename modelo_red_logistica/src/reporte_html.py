"""Genera el reporte HTML autocontenido (funciona sin internet: Plotly y el solver van embebidos)."""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

def _sin_inf(df: pd.DataFrame) -> pd.DataFrame:
    d2 = df.copy()
    num = d2.select_dtypes(include=[np.number]).columns
    d2[num] = d2[num].astype(float).where(np.isfinite(d2[num].astype(float)))
    return d2


from .preparacion import Preparado

RECURSOS = Path(__file__).resolve().parent / "recursos"

CORTOS = {"ASIS_REAL": "Actual real", "E0_ASIS_OPT": "Actual optimizado (con CO10)",
          "E5_CIERRE_RED_ACTUAL": "Cierre CO10 · red actual", "E1_TRASLADO_CEDI": "B · Traslado a CEDI",
          "E2_BODEGA_EXTERNA": "A · Bodega externa", "E3_OPTIMO": "Óptimo TO-BE", "E4_OPTIMO_FLEX": "Óptimo + reasignar producción"}


def _geo(prep: Preparado, cfg: dict, flujos: dict, clientes: pd.DataFrame) -> dict:
    nod = prep.nodos
    opc = set(cfg["almacenamiento"].get("nodos_opcionales_tobe") or [])
    cerrada = cfg["produccion"]["planta_cerrada"]
    nodos, idx_n = [], {}
    for i in nod.index:
        tipo = ("candidata" if nod.at[i, "candidata"] else "bodega_externa" if i in opc else
                "planta" if "Planta" in str(nod.at[i, "Tipo_Nodo"]) else "3pl" if "3PL" in str(nod.at[i, "Modalidad"]) else "cedi")
        idx_n[i] = len(nodos)
        nodos.append({"id": i, "nombre": str(nod.at[i, "nombre"]), "tipo": tipo, "cerrada": i == cerrada,
                      "lat": float(nod.at[i, "lat"]), "lon": float(nod.at[i, "lon"])})
    idx_n["EXTERNO"] = len(nodos)
    nodos.append({"id": "EXTERNO", "nombre": "Destino externo (sin nodo)", "tipo": "externo", "cerrada": False, "lat": None, "lon": None})
    cl = clientes.drop_duplicates("cliente").reset_index(drop=True)
    idx_c = {c: i for i, c in enumerate(cl.cliente)}
    lista_c = _registros(cl)
    cc = prep.ciudades.set_index("ciudad")
    payload = {}
    for sid, fl in flujos.items():
        F = fl["F"]
        F = F[F.m3 > 0.01]
        filas = []
        for r in F.itertuples():
            if r.Cliente not in idx_c:      # cliente agregado "(ciudad X)"
                ciudad = str(r.Cliente)[8:-1]
                idx_c[r.Cliente] = len(lista_c)
                lista_c.append({"cliente": r.Cliente, "ciudad": ciudad, "depto": None, "zona": cc.zona.get(ciudad),
                                "lat": _limpiar(cc.lat.get(ciudad)), "lon": _limpiar(cc.lon.get(ciudad))})
            filas.append([idx_n.get(r.planta, idx_n["EXTERNO"]), idx_n.get(r.cedi, idx_n["EXTERNO"]), idx_c[r.Cliente],
                          1 if r.Mercado == "E" else 0, round(float(r.m3), 3), round(float(r.costo), 0) if pd.notna(r.costo) else 0])
        X = fl["X"]
        X = X[X.m3 > 0.01]
        xs = [[idx_n.get(r.origen, idx_n["EXTERNO"]), idx_n.get(r.destino, idx_n["EXTERNO"]), 1 if r.Mercado == "E" else 0,
               round(float(r.m3), 3), round(float(r.costo), 0)] for r in X.itertuples()]
        payload[sid] = {"F": filas, "X": xs}
    return {"nodos": nodos, "clientes": lista_c, "flujos": payload}


def _limpiar(o):
    if isinstance(o, dict):
        return {str(k): _limpiar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_limpiar(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if o is pd.NA or o is pd.NaT:
        return None
    if isinstance(o, (pd.Timestamp, dt.date)):
        return str(o)
    return o


def _registros(df: pd.DataFrame) -> list:
    d2 = _sin_inf(df)
    d2 = d2.astype(object).where(d2.notna(), None)
    return _limpiar(d2.to_dict("records"))


def _sankey(r: dict) -> dict:
    fx, fy = r["flujos_traslado"], r["flujos_distribucion"]
    if "planta" not in fx.columns or "zona" not in fy.columns:
        return {"nodos": [], "s": [], "t": [], "v": []}
    a = fx.groupby(["planta", "almacen"]).m3_mes.sum().reset_index()
    b = fy.groupby(["almacen", "zona"]).m3_mes.sum().reset_index()
    nodos, idx = [], {}

    def nid(tipo, k, label):
        key = (tipo, k)
        if key not in idx:
            idx[key] = len(nodos)
            nodos.append({"tipo": tipo, "label": label})
        return idx[key]

    s, t, v = [], [], []
    for row in a.itertuples():
        s.append(nid("planta", row.planta, f"Planta {row.planta}"))
        t.append(nid("almacen", row.almacen, f"Almacén {row.almacen}"))
        v.append(float(row.m3_mes))
    for row in b.itertuples():
        s.append(nid("almacen", row.almacen, f"Almacén {row.almacen}"))
        t.append(nid("zona", row.zona, str(row.zona)))
        v.append(float(row.m3_mes))
    return {"nodos": nodos, "s": s, "t": t, "v": v}


def generar_html(ruta: Path, prep: Preparado, cfg: dict, res: dict, calidad: pd.DataFrame, sens: pd.DataFrame,
                 sim: dict, equilibrio: dict, resumen: dict, flujos: dict | None = None, clientes: pd.DataFrame | None = None):
    from plotly.offline import get_plotlyjs

    cerrada = cfg["produccion"]["planta_cerrada"]
    nod = prep.nodos
    fc = None
    if cerrada in nod.index and isinstance(nod.at[cerrada, "Fecha_Cierre"], str):
        fc = pd.to_datetime(nod.at[cerrada, "Fecha_Cierre"], dayfirst=True, errors="coerce")
    meses = prep.series["meses_datos"]
    ultimo = str(prep.sap.fecha.max().to_period("M"))
    if prep.sap.fecha.max().day < 25:
        meses = [m for m in meses if m != ultimo]
    vpm = prep.series["vol_planta_mes"].reindex(meses).fillna(0)
    vpm = vpm.loc[:, vpm.sum() > vpm.values.sum() * 0.002]

    escenarios = []
    for k, r in res.items():
        e = {"id": k, "nombre": r["nombre"], "corto": CORTOS.get(k, k), "costos": r["costos"], "kpis": r["kpis"], "params": r["params"]}
        fxr = r["flujos_traslado"]
        e["nodos"] = _registros(r["nodos"])
        e["traslados"] = _registros(fxr.groupby(["planta", "almacen"], as_index=False)
                                    .agg(m3_mes=("m3_mes", "sum"), costo_terceros_cop=("costo_terceros_cop", "sum"),
                                         costo_flota_propia_cop=("costo_flota_propia_cop", "sum"), costo_cop_mes=("costo_cop_mes", "sum"),
                                         pct_flota_propia=("pct_flota_propia", "mean"), viajes_mes=("viajes_mes", "sum"),
                                         transito_h=("transito_h", "first"), fuente_tarifa=("fuente_tarifa", "first"))
                                    .assign(tarifa_cop_m3=lambda d_: d_.costo_cop_mes / d_.m3_mes))
        dist = (r["flujos_distribucion"].groupby(["almacen", "ciudad", "zona"], as_index=False, dropna=False)
                .agg(m3_mes=("m3_mes", "sum"), tarifa_cop_m3=("tarifa_cop_m3", "first"), costo_cop_mes=("costo_cop_mes", "sum"),
                     fuente_tarifa=("fuente_tarifa", "first"))
                .sort_values("m3_mes", ascending=False))
        e["distribucion"] = _registros(dist.head(1500))
        e["sankey"] = _sankey(r)
        escenarios.append(e)
    orden = ["ASIS_REAL", "E0_ASIS_OPT", "E5_CIERRE_RED_ACTUAL", "E1_TRASLADO_CEDI", "E2_BODEGA_EXTERNA", "E3_OPTIMO", "E4_OPTIMO_FLEX"]
    escenarios.sort(key=lambda e: orden.index(e["id"]) if e["id"] in orden else 99)

    td = prep.tarifas_dist[prep.tarifas_dist.confiable & prep.tarifas_dist.km.notna()]
    tx = prep.tarifas_xi[prep.tarifas_xi.confiable & prep.tarifas_xi.km.notna()]
    pts = lambda df: _registros(df[["origen", "destino", "km", "tarifa", "vol"]])
    pl = prep.plantas.reset_index().rename(columns={"index": "planta"})
    pl["familias"] = pl["familias"].apply(lambda l: ", ".join(l) if isinstance(l, list) else "")
    ndf = nod.reset_index().rename(columns={"index": "ID_Nodo"})
    cols_n = ["ID_Nodo", "nombre", "Tipo_Nodo", "Modalidad", "Estado", "posiciones_total", "cap_inv_m3", "despacho_asis_m3_mes",
              "traslado_salida_asis_m3_mes", "costo_fijo"]
    base = prep.sap
    pct_valido = float(base.flete_valido.mean())
    info = prep.info
    escenarios_def = [{"id": e["id"], "nombre": e["nombre"], "produccion": e.get("produccion"),
                       "tipo": "Evaluación (sin optimizar)" if e.get("tipo") == "red_actual" else "Optimización",
                       "cerrar_planta": "Sí" if e.get("cerrar_planta") else "No",
                       "bodegas_candidatas": "Sí" if e.get("bodegas_candidatas") else "No",
                       "limitar_traslados_asis": "Sí" if e.get("limitar_traslados_asis") else "No",
                       "bodega_actual": "Cerrada" if e.get("cerrar_opcionales") else ("Abierta (fija)" if e.get("opcionales_fijos_abiertos") else "El modelo decide")}
                      for e in cfg["escenarios"]]
    lim = cfg["limpieza"]
    metodologia = [
        f"SAP es la fuente principal ({info['sap_filas_original']:,} líneas). Periodo base {prep.meses_lista[0]} a {prep.meses_lista[-1]} "
        f"({prep.meses_base} meses completos antes del cierre). Demanda = volumen no XI despachado por ciudad del cliente, promedio mensual.",
        f"Tipo {', '.join(lim['tipos_traslado'])} = traslado entre nodos: el campo CEDI es el origen y el Cliente representa el nodo destino "
        f"(homologado por ciudad o en parametros.yaml). Si TMS muestra otro origen (ej. CO99 sale desde Cajicá), se corrige con TMS.",
        f"Fletes con IMPORTE SAP <= {lim['umbral_importe_invalido']} COP se omiten de las tarifas (numerador y denominador). "
        f"También se excluyen líneas con COP/m3 fuera de los percentiles {lim['percentil_outlier_tarifa']}.",
        f"Tarifa de una ruta = suma importe / suma m3 de líneas válidas; se usa si la ruta tiene >= {lim['volumen_minimo_lane_m3']} m3 en el periodo. "
        f"Si no, se estima con una regresión lineal por distancia (km línea recta x {lim['factor_ruta_km']}) ajustada por el factor de cada CEDI.",
        f"Traslados XI de flota propia (viajes con todas sus líneas en 0/1 peso, {info['imputacion_xi']['viajes_propia_base']:,} viajes "
        f"en el periodo base): se imputa el flete por viajes similares de terceros (misma ruta, si no banda de distancia) o por "
        f"{info['imputacion_xi']['costo_fijo_viaje_cop']:,.0f} COP fijo + {info['imputacion_xi']['costo_km_cop']:,.0f} COP/km (ajustado con "
        f"viajes de terceros). Método usado: {info['imputacion_xi']['metodo']}. Viajes dentro del mismo nodo no se imputan.",
        "Costos desagregados en transporte (XI terceros, XI flota propia, distribución), almacenamiento (manejo, posiciones, "
        "fijo de bodegas externas) y fijos de plantas/CEDI (parametros_plantas).",
        "Escenario 'Cierre sin re-optimizar': cada ciudad sigue atendida por los mismos CEDI de hoy para cada producto; solo cambia la "
        "planta que produce. El inventario que no cabe se desborda a la bodega externa actual (traslado + manejo).",
        "Mapa: el modelo optimiza por ciudad; el detalle por cliente y mercado (N/E) reparte cada flujo según la participación "
        "histórica de cada cliente y mercado en esa ciudad y producto.",
        "TMS complementa por número de viaje (ceros a la izquierda normalizados): origen real del traslado, tiempos de tránsito, m3 por viaje "
        "(número de viajes de traslado) y población faltante en SAP.",
        "Planta TO-BE de cada referencia: maestro de referencias (Trasladada -> planta nueva homologada a nodo SAP; Sin cambio -> planta "
        "actual SAP; Descontinuada -> sale de la demanda).",
        f"Capacidad de almacén = posiciones de estiba x {prep.m3_por_posicion:.2f} m3/posición (maestro: cajas x estiba x m3 caja) x "
        f"utilización máxima. Inventario = despacho mensual x días de inventario / 30.",
        f"Días de inventario {'calibrados' if str(cfg['almacenamiento']['dias_inventario']).lower() == 'auto' else 'fijados'}: "
        f"{cfg['almacenamiento']['dias_inventario_efectivo']:.1f}. DOI implícito AS-IS en {cfg['almacenamiento'].get('nodo_calibracion_doi')}: "
        f"{info['doi_implicito']:.1f} días (supone CEDI lleno porque hoy desborda a la bodega externa).",
        "Costos fijos de nodos (parametros_plantas) se suman como constantes; la bodega externa actual y las candidatas se abren o cierran "
        "con variables binarias (pagan su costo fijo solo si se usan).",
        "Solver: HiGHS (scipy.optimize.milp) en Python; javascript-lp-solver en el simulador del navegador (modelo agregado por zona).",
    ]
    parametros = [{"p": "Días de inventario", "v": f"{cfg['almacenamiento']['dias_inventario_efectivo']:.1f}"},
                  {"p": "Utilización máxima posiciones", "v": f"{cfg['almacenamiento']['utilizacion_maxima']:.0%}"},
                  {"p": "m3 por posición", "v": f"{prep.m3_por_posicion:.2f}"},
                  {"p": "Penalización por m3 sin capacidad", "v": f"{cfg['almacenamiento']['penalidad_deficit_cop_m3']:,.0f} COP"},
                  {"p": "Regresión distribución", "v": f"{prep.regresion['distribucion']['a']:,.0f} + {prep.regresion['distribucion']['b']:,.1f}·km (R²={prep.regresion['distribucion']['r2']:.2f})"},
                  {"p": "Regresión traslado XI", "v": f"{prep.regresion['traslado']['a']:,.0f} + {prep.regresion['traslado']['b']:,.1f}·km (R²={prep.regresion['traslado']['r2']:.2f})"}]
    for modalidad, c in cfg["almacenamiento"]["costos_por_modalidad"].items():
        parametros.append({"p": f"Costos {modalidad}", "v": f"manejo {c['manejo_cop_m3']:,} COP/m3 · posición {c['posicion_cop_mes']:,} COP/mes"})
    for bc in cfg.get("bodegas_candidatas") or []:
        parametros.append({"p": f"Candidata {bc['id']}", "v": f"{bc['nombre']}: {bc['posiciones']:,} pos · fijo {bc['costo_fijo_cop_mes']:,} · "
                                                                 f"posición {bc['posicion_cop_mes']:,} · manejo {bc['manejo_cop_m3']:,}"})
    if equilibrio:
        parametros.append({"p": "Costo fijo máximo de bodega externa para empatar (modelo completo)",
                           "v": f"{equilibrio['fijo_maximo_bodega_para_empatar']:,.0f} COP/mes"})

    D = {
        "meta": {"titulo": f"Red logística TO-BE · cierre de planta {cerrada}",
                 "subtitulo": f"Periodo base {prep.meses_lista[0]} a {prep.meses_lista[-1]} · datos SAP {info['periodo_datos'][0]} a "
                              f"{info['periodo_datos'][1]} · costos en COP por mes · generado {dt.datetime.now():%Y-%m-%d %H:%M}",
                 "match_tms": info["viajes_match_normalizado"] / max(info["viajes_sap"], 1), "pct_flete_valido": pct_valido,
                 "umbral": lim["umbral_importe_invalido"], "cerrada": cerrada,
                 "mes_cierre": str(fc.to_period("M")) if fc is not None and not pd.isna(fc) else None,
                 "nombre_nodo": {i: str(nod.at[i, "Ciudad"]) for i in nod.index}, "util": cfg["almacenamiento"]["utilizacion_maxima"],
                 "meses": prep.meses_base},
        "resumen": {"recomendacion_lista": resumen["recomendacion_lista"], "hallazgos": resumen["hallazgos"]},
        "escenarios": escenarios,
        "calidad": _registros(calidad),
        "series": {"planta_mes": {"mes": list(vpm.index), "series": {str(c): [float(x) for x in vpm[c].values] for c in vpm.columns}},
                   "flete_tipo": _registros(prep.series["flete_valido_tipo"])},
        "tarifas": {"dist": pts(td), "xi": pts(tx), "reg": prep.regresion},
        "plantas": _registros(pl),
        "asis_nodos": _registros(res["ASIS_REAL"]["nodos"]),
        "nodos": _registros(ndf[[c for c in cols_n if c in ndf.columns]]),
        "sensibilidad": _registros(sens) if len(sens) else [],
        "sim": _limpiar(sim),
        "equilibrio": _limpiar(equilibrio),
        "escenarios_def": escenarios_def,
        "metodologia": metodologia,
        "parametros": parametros,
        "imputacion": _limpiar(info["imputacion_xi"]),
        "viajes_xi": _registros(prep.viajes_xi[prep.viajes_xi.en_base & prep.viajes_xi.propia]
                                .groupby(["origen", "destino"], as_index=False)
                                .agg(viajes=("viaje_k", "size"), m3=("vol", "sum"), km=("km", "first"),
                                     imp_similar=("imp_similar", "sum"), imp_km=("imp_km", "sum"),
                                     fuente=("fuente_similar", lambda s_: s_.mode().iat[0] if len(s_.dropna()) else ""))
                                .assign(m3=lambda d_: d_.m3 / prep.meses_base, imp_similar=lambda d_: d_.imp_similar / prep.meses_base,
                                        imp_km=lambda d_: d_.imp_km / prep.meses_base, viajes=lambda d_: d_.viajes / prep.meses_base)
                                .sort_values("m3", ascending=False)),
        "tarifas_xi": _registros(prep.tarifas_xi[["origen", "destino", "vol", "tarifa", "tarifa_tercero", "tarifa_propia",
                                                  "pct_flota_propia", "km", "confiable"]].sort_values("vol", ascending=False)),
        "geo": _geo(prep, cfg, flujos, clientes) if flujos else None,
    }
    datos = json.dumps(_limpiar(D), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    html = (RECURSOS / "plantilla.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/", datos)
    html = html.replace("/*__PLOTLY__*/", get_plotlyjs().replace("</script", "<\\/script"))
    html = html.replace("/*__LPSOLVER__*/", (RECURSOS / "jsLPSolver.js").read_text(encoding="utf-8"))
    html = html.replace("/*__SIMULADOR__*/", (RECURSOS / "simulador.js").read_text(encoding="utf-8"))
    topo = (RECURSOS / "world_50m.json").read_text(encoding="utf-8")
    html = html.replace("/*__TOPOJSON__*/", "window.PlotlyGeoAssets={topojson:{world_50m:" + topo + "}};")
    Path(ruta).write_text(html, encoding="utf-8")
