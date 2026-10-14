"""Recomendación y hallazgos del resumen ejecutivo, calculados a partir de los resultados del modelo diario."""
from __future__ import annotations

from collections import defaultdict


def _m(v):  # millones de COP por mes
    return f"{v / 1e6:,.0f}".replace(",", ".") + " MCOP"


def _n(v, dec=0):
    return f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _mes(r, k):
    return r["costos"].get(k, 0.0) * 30 / r["H"]


def _principal_por_ciudad(base, r):
    acc = defaultdict(lambda: defaultdict(float))
    if r.get("tipo") == "historico":
        for a in r.get("asignacion_grupos", []):
            acc[base["grupos"][a[2]][0]][a[0]] += a[4]
    else:
        for a in r.get("asignacion", []):
            if not a[8]:
                acc[a[1]][a[0]] += a[4]
    return {z: max(d, key=d.get) for z, d in acc.items()}, {z: sum(d.values()) for z, d in acc.items()}


def construir_resumen(base: dict, cfg: dict, corrida: dict, combos: list, pruebas: list, info: dict) -> dict:
    R = corrida["resultados"]
    cerrada = base["meta"]["planta_cerrada"]
    A, S2, O, S4, S1 = R.get("S0_ACTUAL"), R.get("S2_SIN_PLANTA"), R.get("S3_OPTIMIZADO"), R.get("S4_BE_OBLIGATORIA"), R.get("S1_ACTUAL_OPT")
    log = "Costo logístico (sin penalizaciones)"
    rec, hall = [], []
    if O and A:
        d = _mes(A, log) - _mes(O, log)
        rec.append(f"Con {cerrada} cerrada y la red re-optimizada, el costo logístico es {_m(_mes(O, log))}/mes frente a {_m(_mes(A, log))}/mes de la operación "
                   f"actual observada: {'ahorro' if d > 0 else 'sobrecosto'} de {_m(abs(d))}/mes ({_m(abs(d) * 12)}/año), incluido el costo fijo de {cerrada} que desaparece.")
    if combos:
        reales = [c for c in combos if c["combinacion"] != "El optimizador decide"]
        mejor = min(reales, key=lambda c: c["total_mes"])
        otras = sorted([c for c in reales if c is not mejor], key=lambda c: c["total_mes"])
        rec.append(f"Bodega externa vs traslado a CEDI: la combinación de menor costo es «{mejor['combinacion']}» ({_m(mejor['total_mes'])}/mes). "
                   + (f"La siguiente, «{otras[0]['combinacion']}», cuesta {_m(otras[0]['vs_mejor_mes'])}/mes más. " if otras else "")
                   + ("Ninguna bodega externa ahorra transporte ni manejo suficiente para pagar su costo fijo: los CEDI con espacio absorben el volumen."
                      if mejor["bodegas_abiertas"] in ("-", "") else f"Se justifica abrir {mejor['bodegas_abiertas']}."))
    if S2 and O:
        rec.append(f"Cerrar {cerrada} manteniendo los CEDI históricos de cada cliente cuesta {_m(_mes(S2, log))}/mes: {_m(_mes(S2, log) - _mes(O, log))}/mes más que re-optimizar. "
                   "Ese es el valor de rediseñar desde qué nodo se atiende cada cliente.")
    if S4 and O:
        rec.append(f"Obligar el uso de bodegas externas cuesta {_m(_mes(S4, log) - _mes(O, log))}/mes adicionales frente al óptimo.")
    if O:
        pen = _mes(O, "Total penalizaciones")
        comp = {k: _mes(O, k) for k in O["costos"] if k.startswith("Penalización") and _mes(O, k) > 1e6}
        rec.append(f"Servicio del escenario optimizado: {O['kpis']['fill_rate']:.1%} de la demanda atendida, {O['kpis']['pct_en_meta']:.0%} del volumen dentro de la meta de "
                   f"{cfg['modelo']['servicio'].get('meta_h')} h" +
                   (f"; penalizaciones {_m(pen)}/mes (" + ", ".join(f"{k.replace('Penalización ', '')} {_m(v)}" for k, v in comp.items()) +
                    "): son precios de incumplimiento, no costos contables." if pen > 1e6 else "; sin penalizaciones relevantes."))
    rec.append("Las cifras dependen de supuestos que no están en los datos (plan de producción, inventario inicial, tarifas de bodegas candidatas): "
               "valídelos en Simulación antes de decidir.")

    if A:
        hall.append(f"Operación actual (horizonte {A['H']} días): {_n(A['kpis']['demanda_m3'] / A['H'])} m3/día despachados, transporte {_m(_mes(A, 'Total transporte'))}/mes "
                    f"(XI terceros {_m(_mes(A, 'Traslado XI terceros'))}, XI flota propia {_m(_mes(A, 'Traslado XI flota propia'))}, distribución {_m(_mes(A, 'Distribución'))}), "
                    f"almacenamiento {_m(_mes(A, 'Total almacenamiento'))}/mes.")
        exc = [(n, max(d["exceso"])) for n, d in A["nodos_dia"].items() if max(d["exceso"]) > 0.5]
        if exc:
            hall.append("Con la operación observada, superan su capacidad estimada: " + ", ".join(f"{n} (máx. {_n(v)} m3)" for n, v in exc) + ".")
    if A and O:
        pa, va = _principal_por_ciudad(base, A)
        po, _ = _principal_por_ciudad(base, O)
        camb = [z for z in pa if z in po and pa[z] != po[z]]
        vol = sum(va[z] for z in camb)
        hall.append(f"{len(camb)} de {len(pa)} ciudades cambian de CEDI principal en el optimizado ({vol / max(sum(va.values()), 1):.0%} del volumen). "
                    "Detalle por cliente en Escenarios › ¿Cómo cambia la atención de los clientes?")
        if base.get("obs_out"):
            tot = obs = 0.0
            for a in O.get("asignacion", []):
                if not a[8]:
                    tot += a[4]
                    obs += a[4] * base["obs_out"][a[0]][a[1]]
            if tot:
                hall.append(f"{1 - obs / tot:.0%} del volumen del optimizado usa tarifas de distribución estimadas por distancia (rutas nodo→ciudad sin historial en SAP), "
                            "frente a menos del 1 % en la operación actual. Conviene cotizar esas rutas antes de implementar.")
        hall.append(f"Traslados XI: {_n(A['kpis']['traslados_m3'] / A['H'])} m3/día en la operación actual vs {_n(O['kpis']['traslados_m3'] / O['H'])} m3/día en el optimizado.")
        ocm = max((max(d["ocupacion"]), n) for n, d in O["nodos_dia"].items() if d["pos_m3"] > 0)
        hall.append(f"Ocupación máxima en el optimizado: {ocm[1]} con {ocm[0]:.0%} de sus posiciones (límite de utilización {cfg['modelo']['utilizacion_maxima']:.0%}).")
    if S1 and O:
        hall.append(f"Con {cerrada} operando pero la red optimizada, el costo sería {_m(_mes(S1, log))}/mes: la diferencia con el optimizado sin planta "
                    f"({_m(_mes(S1, log) - _mes(O, log))}/mes) es el efecto del cierre una vez descontada la mejora de red.")
    imp = info.get("imputacion_xi") or {}
    if imp:
        hall.append(f"Traslados XI de flota propia (flete 0/1 COP): {_n(imp.get('viajes_propia_base', 0))} viajes en el periodo base; flete imputado "
                    f"{_m(imp.get('imp_similar_mes', 0))}/mes por viajes similares o {_m(imp.get('imp_km_mes', 0))}/mes por km "
                    f"({_n(imp.get('costo_fijo_viaje_cop', 0))} COP fijo + {_n(imp.get('costo_km_cop', 0))} COP/km). Método usado: {imp.get('metodo')}.")
    ok = sum(p["estado"] == "OK" for p in pruebas)
    hall.append(f"Pruebas de consistencia A–H: {ok} de {len(pruebas)} correctas (ver Metodología).")
    return {"reco": rec, "hallazgos": hall}
