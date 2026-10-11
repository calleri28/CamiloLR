"""Construye la recomendación, los hallazgos y los valores base del simulador a partir de los resultados."""
from __future__ import annotations

import pandas as pd

from .preparacion import Preparado


def _m(v):  # millones de COP
    return f"{v / 1e6:,.0f} MCOP".replace(",", ".")


def _n(v, dec=0):
    return f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def base_simulador(prep: Preparado, cfg: dict, sim: dict) -> dict:
    cerrada = cfg["produccion"]["planta_cerrada"]
    dem = prep.demanda[~prep.demanda.descontinuada]
    mov = dem[(dem.planta_asis == cerrada) & (dem.planta_tobe != cerrada)]
    planta = mov.groupby("planta_tobe").vol_mes.sum().idxmax() if len(mov) else sim["plantas"][0]
    mv = mov[mov.planta_tobe == planta]
    reparto = (mv.groupby("zona").vol_mes.sum() / mv.vol_mes.sum()).to_dict() if len(mv) else {}
    base = prep.sap[prep.sap.en_base & ~prep.sap.es_xi & (prep.sap.planta_asis != cerrada) & ~prep.sap.descontinuada]
    ocup = (base.groupby("CEDI").vol.sum() / prep.meses_base).to_dict()
    xi = prep.xi_asis.dropna(subset=["destino"])
    for n in sim["almacenes"]:
        if n["tipo"] == "bodega_externa_actual" and ocup.get(n["id"], 0) == 0:
            ocup[n["id"]] = float(xi[(xi.destino == n["id"]) & (xi.origen != n["id"])].vol_mes.sum())
    return {"doi": sim["defaults"]["dias_inventario"], "util": sim["defaults"]["utilizacion_maxima"], "planta": planta,
            "volumen": float(mv.vol_mes.sum()), "reparto": {r: float(reparto.get(r, 0.0)) for r in sim["regiones"]},
            "ocupacion_base": {k: float(v) for k, v in ocup.items()}}


def construir_resumen(prep: Preparado, cfg: dict, res: dict, sens: pd.DataFrame, calidad: pd.DataFrame,
                      equilibrio: dict, sim: dict) -> dict:
    cerrada = cfg["produccion"]["planta_cerrada"]
    B, A, O = res.get("E1_TRASLADO_CEDI"), res.get("E2_BODEGA_EXTERNA"), res.get("E3_OPTIMO")
    E0, E4 = res.get("E0_ASIS_OPT"), res.get("E4_OPTIMO_FLEX")
    doi = O["params"]["dias_inventario"] if O else cfg["almacenamiento"]["dias_inventario_efectivo"]
    rec = []
    if A and B:
        ta, tb = A["costos"]["Total"], B["costos"]["Total"]
        mejor, peor = ("traslado a CEDI de otra ciudad (Opción B)", "bodega externa (Opción A)") if tb <= ta else \
            ("bodega externa (Opción A)", "traslado a CEDI de otra ciudad (Opción B)")
        rec.append(f"Con {_n(doi, 1)} días de inventario, la alternativa de menor costo es {mejor}: "
                   f"{_m(min(ta, tb))}/mes frente a {_m(max(ta, tb))}/mes de {peor}; diferencia {_m(abs(ta - tb))}/mes "
                   f"({_m(abs(ta - tb) * 12)}/año).")
        if equilibrio:
            fm = equilibrio["fijo_maximo_bodega_para_empatar"]
            if fm <= 0:
                rec.append(f"Aun con costo fijo cero, la bodega externa quedaría {_m(-fm)}/mes por encima del traslado: "
                           "limitar los traslados y manejar el producto en un nodo adicional cuesta más que llevarlo a un "
                           "CEDI con espacio.")
            else:
                rec.append(f"La bodega externa sería competitiva solo si su costo fijo mensual fuera menor a {_m(fm)} "
                           f"(hoy {_m(equilibrio['fijo_bodega_actual'])}).")
    if O:
        ab = O["kpis"]["bodegas_abiertas"]
        rec.append("El óptimo libre " + (f"abre/mantiene: {', '.join(ab)}." if ab else
                                         "no usa bodegas externas: absorbe el volumen con el CEDI de la planta y "
                                         "CEDIs con capacidad libre, cerrando la bodega externa actual.")
                   + f" Costo {_m(O['costos']['Total'])}/mes.")
    if E0 and O:
        d = O["costos"]["Total"] - E0["costos"]["Total"]
        rec.append(f"Frente a la red AS-IS optimizada con {cerrada} operando ({_m(E0['costos']['Total'])}/mes), el TO-BE "
                   f"{'ahorra' if d < 0 else 'cuesta'} {_m(abs(d))}/mes, incluyendo el costo fijo de {cerrada} que desaparece.")
    if E4 and O:
        d = O["costos"]["Total"] - E4["costos"]["Total"]
        if d > 0:
            rec.append(f"Reasignar producción entre plantas con familias compatibles ahorraría {_m(d)}/mes adicionales "
                       f"(validar capacidades: capacidad_plantas muestra filas idénticas para CO02 y CO04).")
    if len(sens):
        sd = sens[sens.variable == "dias_inventario"].pivot_table(index="valor", columns="escenario", values="costo_total_cop")
        if {"E1_TRASLADO_CEDI", "E2_BODEGA_EXTERNA"} <= set(sd.columns):
            gana_a = sd[sd["E2_BODEGA_EXTERNA"] < sd["E1_TRASLADO_CEDI"]]
            if len(gana_a):
                rec.append(f"Sensibilidad: la bodega externa pasa a ser mejor desde ~{_n(gana_a.index.min())} días de inventario.")
            else:
                rec.append(f"Sensibilidad: en todo el rango evaluado ({_n(sd.index.min())}-{_n(sd.index.max())} días) "
                           f"el traslado a CEDI sigue siendo más económico que la bodega externa.")

    hall = []
    info = prep.info
    inval = calidad[calidad["Validación"].str.startswith("IMPORTE SAP <=")]
    if len(inval):
        hall.append(f"{inval['% afectado'].iat[0]:.0%} de las líneas SAP tienen flete en 0/1 peso: se excluyeron de las tarifas "
                    f"(solo {_n(prep.sap.flete_valido.sum())} líneas válidas).")
    hall.append(f"El cruce SAP-TMS por viaje sube de {_n(info['viajes_match_crudo'])} a {_n(info['viajes_match_normalizado'])} "
                f"viajes ({info['viajes_match_normalizado'] / info['viajes_sap']:.1%}) al quitar ceros a la izquierda del viaje en SAP.")
    hall.append("El maestro de referencias usa códigos de planta que no existen en la homologación (CO05, CO06, CO12, ...): "
                "se homologaron por volumen SAP (ej. CO06->CO01 Cajicá, CO12->CO02 Rionegro).")
    pl = prep.plantas
    if "CO01" in pl.index:
        hall.append(f"Con el traslado de referencias, CO01 pasa de {pl.at['CO01', 'util_asis']:.0%} a {pl.at['CO01', 'util_tobe']:.0%} "
                    f"de su capacidad de producción ({_n(pl.at['CO01', 'produccion_tobe_m3_mes'])} m3/mes).")
    hall.append(f"Días de inventario implícitos en Cajicá (CEDI lleno + uso de bodega externa CO99): {_n(info['doi_implicito'], 1)} días; "
                f"m3 por posición de estiba según maestro: {_n(prep.m3_por_posicion, 2)}.")
    if B:
        rn = B["nodos"].set_index("almacen")
        libres = rn[rn.utilizacion < 0.6].sort_values("utilizacion")
        if len(libres):
            hall.append("CEDIs con capacidad libre en TO-BE: " + ", ".join(f"{i} ({r.utilizacion:.0%})" for i, r in libres.head(4).iterrows()) + ".")
    if E0 and O:
        hall.append(f"Traslados entre nodos: {_n(E0['kpis']['traslados_m3_mes'])} m3/mes AS-IS optimizado vs "
                    f"{_n(O['kpis']['traslados_m3_mes'])} m3/mes en el óptimo TO-BE (~{_n(O['kpis']['viajes_traslado_mes'])} viajes/mes).")
    ar = res["ASIS_REAL"]
    hall.append(f"Flete registrado en SAP (líneas válidas): {_m(ar['kpis']['flete_registrado_sap_cop_mes'])}/mes; valorizando todo el "
                f"volumen a tarifa de ruta: {_m(ar['costos']['Flete traslado (XI)'] + ar['costos']['Flete distribución'])}/mes.")
    return {"recomendacion": " ".join(rec), "recomendacion_lista": rec, "hallazgos": hall,
            "base_simulador": base_simulador(prep, cfg, sim)}
