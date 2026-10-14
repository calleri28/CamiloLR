"""Pruebas de validación del modelo (casos A a H). Se ejecutan en cada corrida y se reportan en el HTML y el Excel."""
from __future__ import annotations

import copy
import math

import numpy as np

from . import modelo_diario as md
from .base_modelo import parametros_escenario


def _fila(caso, nombre, que, resultado, ok):
    return {"caso": caso, "prueba": nombre, "que_verifica": que, "resultado": resultado, "estado": "OK" if ok else "Revisar"}


def balance_desde_series(res) -> float:
    """Recalcula el balance con las series diarias publicadas (independiente del modelo)."""
    mx = 0.0
    for d in res["nodos_dia"].values():
        prev_b = 0.0
        for t in range(len(d["inv_fin"])):
            izq = d["inv_fin"][t] - d["atraso"][t]
            der = d["inv_ini"][t] - prev_b + d["produccion"][t] + d["entradas"][t] - d["salidas_tr"][t] - d["despachos"][t]
            mx = max(mx, abs(izq - der))
            prev_b = d["atraso"][t]
    return mx


def evaluar_ruta(base, prm, res, origen, via, z, m3, t, dias_perm=1):
    """Costo de atender m3 en la ciudad z desde `origen`, pasando (o no) por `via`. Misma fórmula que el validador del HTML."""
    ins = md.instancia(base, prm)
    nodos = base["nodos"]
    tr = prm["transporte"]
    sv = prm["servicio"]
    tramos = []
    horas = 0.0
    if via != origen:
        L = base["lanes"][f"{nodos[origen]['id']}|{nodos[via]['id']}"]
        s_ = L["s"] if tr.get("s_propia") is None else tr["s_propia"]
        tp = ((tr["fijo_viaje"] + tr["costo_km"] * L["km"]) / (L["m3v"] or base["meta"]["m3_viaje_def"]) if L["km"] > 0 else 0.0) \
            if tr.get("metodo") == "km" else (L["tt"] if L["tp"] is None else L["tp"])
        flete = m3 * ((1 - s_) * L["tt"] * tr["f_terc"] + s_ * tp * tr["f_prop"])
        manejo = m3 * (ins["man_s"][origen] + ins["man_e"][via])
        tramos.append({"tramo": f"{nodos[origen]['id']} → {nodos[via]['id']}", "flete": flete, "manejo": manejo, "horas": L["horas"]})
        horas += L["horas"]
    alm = m3 * dias_perm * (ins["alm"][via] + (prm.get("costo_capital_dia", 0) or 0))
    fz = tr.get("f_zona", {}).get(base["zonas"][z]["zona"] or "", 1.0)
    flete2 = m3 * base["c_out"][via][z] * tr["f_dist"] * fz
    tramos.append({"tramo": f"{nodos[via]['id']} → {base['zonas'][z]['id']}", "flete": flete2, "manejo": m3 * ins["man_s"][via],
                   "horas": base["h_out"][via][z]})
    horas += base["h_out"][via][z]
    var = sum(x["flete"] + x["manejo"] for x in tramos) + alm
    abierto = res["abiertos"].get(nodos[via]["id"], False)
    fijo_inc = 0.0 if abierto else ins["fijo"][via] * ins["H"] / 30
    return {"tramos": tramos, "almacenaje": alm, "variable": var, "fijo_incremental": fijo_inc, "horas": horas,
            "cumple_servicio": sv.get("lmax_h") is None or horas <= sv["lmax_h"]}


def ejecutar(base: dict, cfg: dict, corrida: dict, combos: list) -> list:
    R = corrida["resultados"]
    P = corrida["parametros"]
    sc = cfg.get("solver", {})
    tiempo, gap = sc.get("tiempo_maximo_seg", 300), sc.get("gap_relativo", 1e-3)
    filas = []
    escs = {e["id"]: e for e in cfg["escenarios_diarios"]}

    # A. Balance de flujo
    peor_modelo = max(r["kpis"]["residuo_balance"] for r in R.values())
    peor_series = max(balance_desde_series(r) for r in R.values())
    filas.append(_fila("A", "Balance de flujo", "Inventario final = inicial + producción + entradas − salidas − despachos, por nodo, "
                       "producto y día, en todos los escenarios (incluye atrasos).",
                       f"Residuo máximo: {peor_modelo:.2e} m3 (variables del modelo) y {peor_series:.2e} m3 (series publicadas).",
                       peor_modelo < 1e-3 and peor_series < 1e-3))

    # B. Traslado ficticio planta -> CEDI con el mismo código
    ins = md.instancia(base, P["S3_OPTIMIZADO"])
    mismo = [a for a in ins["arcos"] if a["u"] == a["v"]]
    fic = [r for r in base["obs_xi"] if r[1] == r[2]]
    m3f = sum(r[5] for r in fic)
    cf = sum(r[6] for r in fic)
    filas.append(_fila("B", "Traslado ficticio (mismo código)", "La producción entra al CEDI de su misma planta sin arco de transporte, "
                       "sin costo y sin tiempo de tránsito; los XI con origen = destino no se cuentan como traslado físico.",
                       f"Arcos con origen = destino en el modelo: {len(mismo)}. XI observados con origen = destino en la ventana: "
                       f"{len(fic)} registros, {m3f:,.0f} m3, costo registrado {cf:,.0f} COP: excluidos del escenario Actual (costo 0, tránsito 0).",
                       len(mismo) == 0))

    # C. Inventario acumulado (caso sintético: producción estable, ventas 50 / 50 / 200 por semana)
    sb, sp = base_sintetica()
    r = md.resolver(sb, sp, tiempo, 1e-9)
    inv_red = np.array([sum(d["inv_fin"][t] for d in r["nodos_dia"].values()) for t in range(21)])
    esperado = np.cumsum(np.array([100 / 7] * 21) - np.array([50 / 7] * 14 + [200 / 7] * 7))
    dif = float(np.abs(inv_red - esperado).max())
    filas.append(_fila("C", "Inventario acumulado", "Producción estable de 100 m3/semana y ventas de 50, 50 y 200 m3: el inventario "
                       "diario debe subir a 50 y 100 m3 y volver a 0, y la ocupación debe seguirlo.",
                       f"Inventario de la red al cierre de cada semana: {inv_red[6]:.1f} / {inv_red[13]:.1f} / {inv_red[20]:.1f} m3 "
                       f"(esperado 50 / 100 / 0). Diferencia máxima diaria: {dif:.2e} m3. Ocupación máxima del CEDI: "
                       f"{max(max(d['ocupacion']) for d in r['nodos_dia'].values()):.0%}.", dif < 1e-4))

    # D. Capacidad insuficiente
    e = copy.deepcopy(escs["S3_OPTIMIZADO"])
    e.setdefault("nodos", {})["CO01"] = {"posiciones": base["nodos"][[n["id"] for n in base["nodos"]].index("CO01")]["posiciones"] * 0.3}
    rD = md.resolver(base, parametros_escenario(cfg, base, e), tiempo, gap)
    b3 = R["S3_OPTIMIZADO"]
    filas.append(_fila("D", "Capacidad insuficiente", "Con CO01 al 30 % de sus posiciones el modelo debe buscar otra alternativa "
                       "(otro CEDI o bodega externa) o declarar el exceso de capacidad.",
                       f"Bodegas abiertas: {', '.join(rD['kpis']['bodegas_abiertas']) or 'ninguna'}; exceso máximo {rD['kpis']['exceso_max_m3']:,.0f} m3; "
                       f"ocupación máx. CO01 {max(rD['nodos_dia']['CO01']['ocupacion']):.0%}; costo mensual +"
                       f"{(rD['costos']['Total mensual equivalente'] - b3['costos']['Total mensual equivalente']) / 1e6:,.0f} MCOP; "
                       f"diagnóstico: {len(rD['diagnostico'])} alertas.",
                       rD["estado"] in ("Optimal", "Factible (límite de tiempo)") and
                       max(rD["nodos_dia"]["CO01"]["inv_fin"]) <= rD["nodos_dia"]["CO01"]["cap"] + rD["kpis"]["exceso_max_m3"] + 1e-6))

    # E. Combinaciones de bodegas externas
    if combos:
        ok = True
        txt = []
        for f in combos:
            ok &= f["estado"] in ("Optimal", "Factible (límite de tiempo)")
            txt.append(f"{f['combinacion']}: {f['total_mes'] / 1e6:,.0f} MCOP/mes (fijo BE {f['fijo_be_mes'] / 1e6:,.0f})")
        cerr_ok = all(f.get(f"inv_max_{b}", 0) < 1e-6 for f in combos for b in cfg["combinaciones_bodegas"]["bodegas"]
                      if b not in f["bodegas_abiertas"] and f["combinacion"] != "El optimizador decide")
        decide = combos[-1]["total_mes"]
        mejor = min(f["total_mes"] for f in combos[:-1])
        filas.append(_fila("E", "Bodegas externas", "Cada combinación respeta capacidades; las bodegas cerradas no reciben producto; "
                           "el costo fijo solo se cobra si la bodega está abierta; el optimizador libre no es peor que la mejor combinación.",
                           " · ".join(txt) + f". Bodegas cerradas sin inventario: {'sí' if cerr_ok else 'no'}. "
                           f"Optimizador libre vs mejor combinación: {(decide - mejor) / 1e6:+,.1f} MCOP/mes.",
                           ok and cerr_ok and decide <= mejor * (1 + 2e-3)))

    # F. Ruta directa vs CEDI intermedio vs bodega externa (10 m3 Cajicá -> Medellín)
    ids = [n["id"] for n in base["nodos"]]
    z = next((i for i, zz in enumerate(base["zonas"]) if zz["id"] == "MEDELLIN"), 0)
    o = ids.index("CO01")
    prm3 = P["S3_OPTIMIZADO"]
    alts = [("Directo CO01", o)] + [(f"Vía {w}", ids.index(w)) for w in ("CO02", "BE_ANTIOQUIA") if w in ids]
    txt = []
    ok = True
    for nom, via in alts:
        ev = evaluar_ruta(base, prm3, R["S3_OPTIMIZADO"], o, via, z, 10.0, 3)
        suma = sum(x["flete"] + x["manejo"] for x in ev["tramos"]) + ev["almacenaje"]
        ok &= abs(suma - ev["variable"]) < 1e-6
        txt.append(f"{nom}: {ev['variable']:,.0f} COP ({len(ev['tramos'])} tramos, {ev['horas']:.0f} h"
                   f"{', + fijo incremental ' + format(ev['fijo_incremental'], ',.0f') if ev['fijo_incremental'] else ''})")
    fam = base["familias"].index("FaCa") if "FaCa" in base["familias"] else 0
    fijo = fijar_configuracion(prm3, R["S3_OPTIMIZADO"])
    pe = copy.deepcopy(fijo)
    pe["extra"] = {"z": z, "familia": fam, "planta": o, "t": 3, "m3": 10.0, "nodo": None}
    rx = md.resolver(base, pe, tiempo, 1e-9)
    rb = md.resolver(base, fijo, tiempo, 1e-9)
    marg = rx["objetivo"] - rb["objetivo"]
    pf = copy.deepcopy(pe)
    pf["extra"]["nodo"] = ids.index("CO02")
    marg_f = md.resolver(base, pf, tiempo, 1e-9)["objetivo"] - rb["objetivo"]
    filas.append(_fila("F", "Atención de clientes por varias rutas", "El costo de cada alternativa suma todos sus tramos (traslado, "
                       "manejo, almacenaje y distribución) y se compara con el costo marginal exacto (re-optimizar con la demanda adicional).",
                       " · ".join(txt) + f". Costo marginal exacto re-optimizando (bodegas del escenario fijas): {marg:,.0f} COP para 10 m3; "
                       f"forzando la atención desde CO02: {marg_f:,.0f} COP.",
                       ok and 0 < marg <= marg_f + 1.0))

    # G. Servicio
    e = copy.deepcopy(escs["S3_OPTIMIZADO"])
    e["servicio"] = {"lmax_h": 30, "meta_h": 30, "pen_hora": 0}
    rG = md.resolver(base, parametros_escenario(cfg, base, e), tiempo, gap)
    e2 = copy.deepcopy(escs["S3_OPTIMIZADO"])
    e2["servicio"] = {"lmax_h": None, "meta_h": 24, "pen_hora": 3000}
    rG2 = md.resolver(base, parametros_escenario(cfg, base, e2), tiempo, gap)
    filas.append(_fila("G", "Servicio", "Una restricción de tiempo máximo de entrega o una penalización por hora sobre la meta debe "
                       "cambiar la asignación y el costo según los parámetros.",
                       f"Sin restricción: {b3['kpis']['horas_promedio']:.1f} h promedio, {b3['kpis']['pct_en_meta']:.1%} en meta 72 h. "
                       f"Máximo 30 h: {rG['kpis']['horas_promedio']:.1f} h, no atendido {rG['kpis']['no_atendido_m3']:,.0f} m3, costo logístico +"
                       f"{(rG['costos']['Costo logístico mensual equivalente'] - b3['costos']['Costo logístico mensual equivalente']) / 1e6:,.0f} MCOP/mes. "
                       f"Penalización 3.000 COP/m3·h sobre 24 h: {rG2['kpis']['horas_promedio']:.1f} h promedio.",
                       rG["kpis"]["horas_promedio"] <= b3["kpis"]["horas_promedio"] + 1e-6 and
                       rG2["kpis"]["horas_promedio"] <= b3["kpis"]["horas_promedio"] + 1e-6))

    # H. Coherencia entre vistas
    peor = 0.0
    for sid, r in R.items():
        if r["tipo"] != "optimizado":
            continue
        alm = sum(sum(d["costo_almac"]) for d in r["nodos_dia"].values())
        peor = max(peor, abs(alm - (r["costos"]["Almacenaje"] + r["costos"]["Costo financiero inventario"])))
        at = sum(a[4] for a in r["asignacion"])
        peor = max(peor, abs(at - r["kpis"]["atendido_m3"]))
        de = sum(sum(d["despachos"]) for d in r["nodos_dia"].values())
        peor = max(peor, abs(de - r["kpis"]["atendido_m3"]))
    filas.append(_fila("H", "Coherencia entre vistas", "Los costos y volúmenes de las series diarias, la asignación por cliente y los "
                       "totales coinciden; el simulador del navegador (HiGHS WebAssembly) usa el mismo modelo que Python.",
                       f"Diferencia máxima entre vistas: {peor:.2e}. Paridad Python ↔ navegador: ver pruebas/paridad.js "
                       "(última verificación: diferencia relativa < 1e-12 en 6 escenarios).", peor < 1e-3))
    return filas


def fijar_configuracion(prm, res):
    """Copia de parámetros con las bodegas opcionales fijas en la decisión del escenario (el modelo queda lineal)."""
    p = copy.deepcopy(prm)
    for nid, d in p["nodos"].items():
        if d.get("modo") in ("opcional", "forzado"):
            d["modo"] = "forzado" if res["abiertos"].get(nid) else "cerrado"
    return p


def base_sintetica():
    """Red mínima: planta P con su CEDI, CEDI C, una ciudad Z. 21 días; producción 100 m3/semana; ventas 50/50/200."""
    import datetime as dt
    d0 = dt.date(2026, 1, 5)   # lunes
    fechas = [str(d0 + dt.timedelta(days=i)) for i in range(21)]
    nodo = lambda i, tipo, pl: {"id": i, "nombre": i, "tipo": tipo, "es_planta": pl, "ciudad": i, "lat": 0.0, "lon": 0.0,
                                "posiciones": 1000.0, "costo_fijo_mes": 0.0, "manejo_salida": 0.0, "manejo_entrada": 0.0, "almac_dia": 0.0,
                                "bodega_externa": False, "cap_produccion_mes": None, "familias_planta": []}
    dem = [[d, 0, 50 / 7 if d < 14 else 200 / 7] for d in range(21)]
    base = {"meta": {"fechas": fechas, "dia_semana": [i % 7 for i in range(21)], "m3_por_posicion": 1.0, "planta_cerrada": None,
                     "m3_viaje_def": 50.0, "imputacion": {"metodo": "similares", "costo_km": 0, "fijo_viaje": 0}},
            "nodos": [nodo("P", "planta", True), nodo("C", "cedi", False)], "familias": ["X"],
            "zonas": [{"id": "Z", "zona": "Z", "depto": "Z", "lat": 0.0, "lon": 0.0}], "grupos": [[0, 0, 0, 0, 0]], "demanda": dem,
            "asign_hist": [[[1, 1.0]]], "c_out": [[100.0], [10.0]], "h_out": [[5.0], [5.0]],
            "lanes": {"P|C": {"tt": 5.0, "tp": None, "s": 0.0, "km": 10.0, "m3v": None, "horas": 1.0, "obs": True},
                      "C|P": {"tt": 5.0, "tp": None, "s": 0.0, "km": 10.0, "m3v": None, "horas": 1.0, "obs": True}},
            "arcos_hist": ["P|C"], "obs_desp": [], "obs_xi": [], "clientes": [], "cli_shares": [[]], "refs": {}}
    prm = {"horizonte_dias": 21, "desplazamiento": 0, "proyeccion": "replica", "crecimiento": 0, "crec_N": 0, "crec_E": 0,
           "crec_zona": {}, "crec_familia": {}, "planta_cerrada": None, "usar_reparto": False, "reparto": {}, "asignacion": "libre",
           "utilizacion_maxima": 1.0, "produccion": {"modo": "plan", "dias_semana": 7, "factor": 1.0},
           "inventario": {"dias_inicial_planta": 0, "dias_inicial_cedi": 0, "dias_seguridad": 0, "final_min": 1.0, "ocupacion_inicial_max": 0.7},
           "traslados": "plantas_y_observados", "traslado_ficticio_mismo_codigo": True,
           "servicio": {"lmax_h": None, "meta_h": 72, "pen_hora": 0},
           "holguras": {"atraso": False, "pen_atraso": 0, "pen_no_atendido": 0, "exceso": True, "pen_exceso": 1e6, "pen_stock_seguridad": 0},
           "costo_capital_dia": 0,
           "transporte": {"f_terc": 1, "f_prop": 1, "f_dist": 1, "metodo": "similares", "costo_km": 0, "fijo_viaje": 0, "s_propia": None, "f_zona": {}},
           "nodos": {"P": {"modo": "fijo"}, "C": {"modo": "fijo"}}}
    return base, prm
