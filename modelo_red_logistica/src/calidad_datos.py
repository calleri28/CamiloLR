"""Validaciones de calidad de la información (SAP, TMS y maestros).

Cada validación devuelve: fuente, validación, resultado, registros afectados, % afectado,
severidad (Alta / Media / Baja / OK) y el tratamiento aplicado en el modelo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .preparacion import Preparado
from .utilidades import a_numero, normalizar_serie_texto


def _fila(fuente, validacion, afectados, total, severidad, tratamiento, resultado=None):
    pct = afectados / total if total else 0.0
    if resultado is None:
        resultado = f"{afectados:,.0f} de {total:,.0f} ({pct:.1%})"
    return {"Fuente": fuente, "Validación": validacion, "Resultado": resultado,
            "Registros afectados": float(afectados), "% afectado": pct,
            "Severidad": severidad if afectados else "OK", "Tratamiento en el modelo": tratamiento}


def validar(d: dict, prep: Preparado, cfg: dict) -> pd.DataFrame:
    sap = prep.sap
    info = prep.info
    filas = []
    n = info["sap_filas_original"]
    umbral = cfg["limpieza"]["umbral_importe_invalido"]

    # ------------------------------------------------------------------ SAP
    filas.append(_fila("SAP", "Registros duplicados exactos (todas las columnas iguales)", info["sap_duplicados"], n,
                       "Baja", "Eliminados" if cfg["limpieza"].get("eliminar_duplicados_exactos", True) else "Conservados"))
    inval = int((~sap.flete_valido).sum())
    filas.append(_fila("SAP", f"IMPORTE SAP <= {umbral} COP (fletes en 0 / 1 peso)", inval, len(sap), "Alta",
                       "Omitidos del cálculo de tarifas (numerador y denominador). Su volumen sí cuenta como demanda"))
    dec = int(((sap.imp > 0) & (sap.imp < 1)).sum())
    filas.append(_fila("SAP", "IMPORTE SAP con decimales entre 0 y 1 (0,8 / 0,9 COP)", dec, len(sap), "Media",
                       "Tratados como importe inválido (misma regla de 0/1 peso)"))
    xi = sap.es_xi
    filas.append(_fila("SAP", "Traslados XI con flete inválido", int((xi & ~sap.flete_valido).sum()), int(xi.sum()), "Alta",
                       "Son viajes de flota propia: se imputa su flete (viajes similares o por km); la tarifa de "
                       "terceros usa solo líneas válidas"))
    filas.append(_fila("SAP", "Líneas con tarifa COP/m3 atípica (fuera de percentiles configurados)",
                       int(sap.outlier_tarifa.sum()), int(sap.flete_valido.sum()), "Media",
                       f"Excluidas de tarifas. Rango XI {info['outlier_rango_xi'][0]:,.0f}-{info['outlier_rango_xi'][1]:,.0f}; "
                       f"distribución {info['outlier_rango_dist'][0]:,.0f}-{info['outlier_rango_dist'][1]:,.0f} COP/m3"))
    filas.append(_fila("SAP", "Viaje vacío", int(sap.Viaje.isna().sum()), len(sap), "Media",
                       "No se cruzan con TMS; se usan para demanda y tarifas"))
    lz = int(sap.Viaje.astype("string").str.startswith("0").sum())
    filas.append(_fila("SAP", "Viaje con ceros a la izquierda (formato distinto a TMS)", lz, len(sap), "Alta",
                       f"Normalizado. Match de viajes pasa de {info['viajes_match_crudo']:,} a {info['viajes_match_normalizado']:,}"))
    filas.append(_fila("SAP", "Centro productor vacío", int(d["sap"]["Centro productor"].isna().sum()), n, "Media",
                       "Completado con planta del maestro de referencias (homologada)"))
    filas.append(_fila("SAP", "Población vacía", int(sap.Poblacion.isna().sum()), len(sap), "Media",
                       f"La demanda usa la ciudad del maestro de clientes; {info['poblacion_completada_tms']:,} completadas con TMS"))
    filas.append(_fila("SAP", "CorredorTransporte vacío", int(sap.CorredorTransporte.isna().sum()), len(sap), "Baja",
                       "No se usa para el modelo (se usa origen-destino)"))
    filas.append(_fila("SAP", "Cantidad no entera (cajas fraccionadas)",
                       int(d["sap"]["Cantidad"].astype("string").str.contains(r"[.,]").sum()), n, "Baja", "Se conserva"))
    r = sap.vol / (a_numero(sap["Cantidad"]) * sap.vcaja)
    desv = int(((r < 0.95) | (r > 1.05)).sum())
    filas.append(_fila("SAP", "Volumen SAP difiere > 5 % de Cantidad x Volumen_Caja del maestro", desv, len(sap), "Media",
                       "Se usa el volumen SAP (fuente principal)"))
    filas.append(_fila("SAP", "Referencias SAP sin maestro de referencias", int(sap.Cod_Ref.isna().sum()), len(sap), "Alta",
                       "Sin planta TO-BE: se asigna por familia"))
    fam = int((sap.Familia.notna() & (sap.Familia != sap["Unidad Negocio"])).sum())
    filas.append(_fila("SAP", "Unidad Negocio SAP distinta a Familia del maestro", fam, len(sap), "Baja", "Se usa Unidad Negocio SAP"))
    ult = sap.fecha.max()
    parcial = int((sap.mes == str(ult.to_period("M"))).sum())
    filas.append(_fila("SAP", f"Mes parcial al final de la serie (datos hasta {ult.date()})", parcial, len(sap), "Baja",
                       f"Periodo base {prep.meses_lista[0]} a {prep.meses_lista[-1]} ({prep.meses_base} meses completos)"))
    nodos = d["maestro_nodos"]
    cerr = nodos[nodos.Estado.str.strip().str.lower() == "cerrado"]
    for _, rr in cerr.iterrows():
        fc = pd.to_datetime(rr["Fecha_Cierre"], dayfirst=True, errors="coerce")
        post = sap[(sap["Centro productor"] == rr["ID_Nodo"]) & (sap.fecha > fc)]
        filas.append(_fila("SAP", f"Producción de {rr['ID_Nodo']} registrada después de su cierre ({fc.date()})",
                           len(post), int((sap["Centro productor"] == rr["ID_Nodo"]).sum()), "Media",
                           f"{post.vol.sum():,.0f} m3 tras el cierre: transición gradual. El TO-BE asume cierre total"))
    cedi_prod = sap[(sap["Centro productor"] == "CO14")]
    filas.append(_fila("SAP", "CEDI CO14 (Cali) aparece como Centro productor sin capacidad de planta definida",
                       len(cedi_prod), len(sap), "Baja", "Se mantiene como origen con su volumen histórico"))
    filas.append(_fila("SAP", "Traslados XI cuyo origen real (TMS) difiere del campo CEDI",
                       info["xi_origen_corregido_tms"], int(xi.sum()), "Media",
                       "Origen corregido con Población Origen de TMS (ej. CO99 sale desde Cajicá)"))
    filas.append(_fila("SAP", "Traslados XI cuyo cliente destino no corresponde a ningún nodo",
                       info["xi_sin_nodo_destino"], int(xi.sum()), "Baja", "Excluidos de tarifas XI (destinos Funza/Bogotá)"))
    c720 = sap[(sap.Cliente == "C720")]
    filas.append(_fila("SAP", "Cliente C720 (Tocancipá) registrado con Poblacion 'ANTIOQUIA'",
                       int((c720.Poblacion == "ANTIOQUIA").sum()), len(c720), "Media",
                       "Homologado al nodo CO99 (bodega externa Tocancipá) por parametros.yaml"))
    multi = sap[sap.viaje_n.notna()].groupby("viaje_n").CEDI.nunique()
    filas.append(_fila("SAP", "Viajes con más de un CEDI", int((multi > 1).sum()), len(multi), "Baja",
                       "Se toma el primer CEDI para el cruce con TMS"))
    im = info["imputacion_xi"]
    filas.append(_fila("SAP", "Traslados XI de flota propia: viajes con todas sus líneas en 0/1 peso", im["viajes_propia"],
                       int(sap[sap.es_xi].viaje_n.nunique()), "Alta",
                       f"Flete imputado ({im['metodo']}): {im['imp_similar_mes'] / 1e6:,.0f} MCOP/mes por viajes similares / "
                       f"{im['imp_km_mes'] / 1e6:,.0f} MCOP/mes por km. {im['viajes_mismo_nodo']:,} viajes dentro del mismo nodo sin costo"))
    xa = sap[sap.Tipo == "XA"]
    filas.append(_fila("SAP", "Tipo XA (clientes regulares, no nodos)", len(xa), len(sap), "Baja",
                       "Tratado como distribución, no como traslado"))

    # ------------------------------------------------------------------ TMS
    tms = prep.tms
    nt = len(tms)
    filas.append(_fila("TMS", "Viajes duplicados", int(tms.viaje_n.duplicated().sum()), nt, "Alta", "Se conserva el primero"))
    filas.append(_fila("TMS", "Fechas centinela (año 1953)", int(tms.fecha_centinela.fillna(False).sum()), nt, "Media",
                       "Tiempo de tránsito no calculado para esos viajes"))
    malo = tms.transito_h.notna() & ~tms.transito_valido
    filas.append(_fila("TMS", "Tiempo de tránsito negativo, < 0,5 h o > 15 días", int(malo.sum()), nt, "Media",
                       "Excluidos del cálculo de tiempos"))
    filas.append(_fila("TMS", "Fechas reales vacías", int(tms["Fecha llegada real"].isna().sum()), nt, "Baja", "Sin tiempo de tránsito"))
    filas.append(_fila("TMS", "Volumen = 0", int((tms.vol_tms == 0).sum()), nt, "Media", "No se usan para m3 por viaje"))
    filas.append(_fila("TMS", "Peso = 0", int((tms.peso_tms == 0).sum()), nt, "Baja", "Peso no se usa en el modelo"))
    vv = a_numero(tms["Volumen viaje"])
    filas.append(_fila("TMS", "Volumen distinto de 'Volumen viaje'", int((abs(vv - tms.vol_tms) > 0.01).sum()), nt, "Baja",
                       "Se usa 'Volumen'"))
    filas.append(_fila("TMS", "'Usuario Creación' vacío", int(tms["Usuario Creación"].isna().sum()), nt, "Baja", "Campo no usado"))
    filas.append(_fila("TMS", "Centro TMS numérico (1..6) sin homologación a código SAP", nt, nt, "Baja",
                       "Origen se toma de 'Población Origen' homologada a nodo"))
    vs, vm = info["viajes_sap"], info["viajes_match_normalizado"]
    filas.append(_fila("SAP-TMS", "Viajes SAP sin match en TMS (después de normalizar)", vs - vm, vs, "Media",
                       "Para estos viajes se usa solo SAP (fuente principal)"))
    solo_tms = len(set(tms.viaje_n.dropna()) - set(sap.viaje_n.dropna()))
    filas.append(_fila("SAP-TMS", f"Viajes TMS sin SAP (TMS cubre {info['tms_periodo'][0][:10]} a {info['tms_periodo'][1][:10]})",
                       solo_tms, nt, "Baja", "No se usan (SAP es la fuente principal)"))
    filas.append(_fila("SAP-TMS", "Volumen del viaje SAP vs TMS dentro de ±10 %",
                       0, 1, "OK", "Validación de consistencia", resultado=f"{info['viajes_vol_sap_tms_10pct']:.1%} de los viajes cruzados"))

    # ------------------------------------------------------------------ Maestros
    ref = d["maestro_referencias"]
    nr = len(ref)
    filas.append(_fila("Maestro referencias", "Referencias sin familia / planta / volumen", int(ref.Familia.isna().sum()), nr, "Media",
                       "Planta asignada por familia"))
    vc = a_numero(ref.Volumen_Caja_m3)
    filas.append(_fila("Maestro referencias", "Volumen_Caja_m3 = 0", int((vc == 0).sum()), nr, "Media",
                       "Excluidas del cálculo de m3 por posición"))
    codigos = set(ref.Planta_Produccion_Antes.dropna()) | set(ref.Planta_Produccion_Nueva.dropna())
    homol = set(d["homologacion_centros"].Codigo_SAP.str.strip())
    sin_h = sorted(codigos - homol)
    filas.append(_fila("Maestro referencias", "Códigos de planta (CO05, CO06, CO12...) no existen en homologacion_centros",
                       len(sin_h), len(codigos), "Alta",
                       "Homologación inferida por volumen SAP: " + ", ".join(f"{r.codigo_maestro}->{r.nodo_sap} ({r.participacion:.0%})"
                                                                         for r in prep.homolog_plantas.itertuples())))
    fe = pd.to_numeric(ref.Fecha_Efectiva_Cambio, errors="coerce")
    fe_dt = pd.to_datetime(fe, unit="D", origin="1899-12-30")
    if fe.notna().any():
        filas.append(_fila("Maestro referencias", f"Fecha_Efectiva_Cambio como número serial Excel ({int(fe.dropna().iloc[0])} = "
                                                  f"{fe_dt.dropna().iloc[0].date()}) distinta al cierre del maestro de nodos",
                           int(fe.notna().sum()), nr, "Media", "Se toma la fecha del maestro de nodos para el cierre"))
    filas.append(_fila("Maestro referencias", "Trasladadas/sin cambio sin Planta_Produccion_Nueva",
                       int((ref.Planta_Produccion_Nueva.isna() & ~ref.Situacion_Post_Cambio.fillna("").str.startswith("Descontinuada")).sum()),
                       nr, "Media", "Se mantiene la planta AS-IS"))
    filas.append(_fila("Maestro referencias", "Campos logísticos vacíos (dimensiones, peso, ABC, vida útil)",
                       int(ref.Peso_Caja_kg.isna().sum()), nr, "Baja", "No se usan"))
    cl = d["maestro_clientes"]
    filas.append(_fila("Maestro clientes", "Clientes sin coordenadas", int(cl.Latitud.isna().sum()), len(cl), "Media",
                       "Coordenadas de la ciudad / departamento"))
    filas.append(_fila("Maestro clientes", "Clientes sin Zona_Comercial", int(cl.Zona_Comercial.isna().sum()), len(cl), "Baja",
                       "Zona inferida por departamento"))
    filas.append(_fila("Maestro clientes", "Campos de servicio vacíos (frecuencia, ventana, lead time, prioridad)",
                       int(cl.Lead_Time_Comprometido_Dias.isna().sum()), len(cl), "Baja", "El modelo no impone restricción de servicio"))
    nn = d["maestro_nodos"]
    coords = nn[["Latitud", "Longitud"]].duplicated(keep=False)
    filas.append(_fila("Maestro nodos", "Nodos de ciudades distintas con coordenadas idénticas (CO01 Cajicá / CO99 Tocancipá)",
                       int(coords.sum()), len(nn), "Media", "CO99 toma coordenadas de Tocancipá (maestro de clientes)"))
    filas.append(_fila("Maestro nodos", "Departamento mal escrito ('Cuandinamarca')",
                       int(nn.Departamento.str.contains("Cuandinamarca", na=False).sum()), len(nn), "Baja", "No afecta el modelo"))
    ic = d["info_cedi"]
    vacios = [c for c in ["Volumen_Util_m3", "Ocupacion_Promedio_Pct", "Ocupacion_Pico_Pct", "Utilizacion_Max_Operable_Pct", "Area_m2"]
              if c in ic.columns and ic[c].isna().all()]
    filas.append(_fila("Info CEDI", "Campos de capacidad 100 % vacíos: " + ", ".join(vacios), len(vacios), 5, "Alta",
                       "Capacidad = posiciones x m3/posición (maestro) x utilización máx. (parámetro)"))
    filas.append(_fila("Info CEDI", "Niveles en texto ('3, 4, y 5')", int(ic.Niveles.astype(str).str.contains(",").sum()), len(ic), "Baja",
                       "No se usa"))
    cp = d["capacidad_plantas"]
    dupc = cp.duplicated(["Familia", "Capacidad_M3_Mes"], keep=False)
    filas.append(_fila("Capacidad plantas", "Plantas con familias y capacidad idénticas (posible copia: "
                       + ", ".join(cp.loc[dupc, "ID_Nodo"]) + ")", int(dupc.sum()), len(cp), "Media",
                       "Se usan tal cual; validar antes de usar el escenario de producción flexible"))
    fams = set(sap["Unidad Negocio"].dropna())
    con_cap = set(x for l in prep.plantas.familias.dropna() for x in l)
    filas.append(_fila("Capacidad plantas", "Familias vendidas sin planta con capacidad declarada: " + ", ".join(sorted(fams - con_cap)),
                       len(fams - con_cap), len(fams), "Media", "Se producen en su planta actual"))
    pp = d["parametros_plantas"]
    filas.append(_fila("Parámetros plantas", "Factor_Costo_Variable y Max_Lineas_Recibir vacíos",
                       int(pp.Factor_Costo_Variable.isna().sum()), len(pp), "Media", "Costos variables por modalidad (parametros.yaml)"))
    cf = a_numero(pp.Costo_Fijo_COP_Mes)
    rep = int(cf.duplicated(keep=False).sum())
    filas.append(_fila("Parámetros plantas", "Costo fijo idéntico en varios nodos (445 MCOP: posible valor por defecto)", rep, len(pp),
                       "Media", "Se usa tal cual"))
    nod = prep.nodos[~prep.nodos.candidata]
    sobre = nod[nod.cap_despacho_m3_mes.notna() & (nod.despacho_asis_m3_mes + nod.traslado_salida_asis_m3_mes > nod.cap_despacho_m3_mes)]
    filas.append(_fila("Info CEDI", "Nodos cuyo despacho real supera la capacidad de despacho declarada (estibas/día): "
                       + ", ".join(sobre.index), len(sobre), len(nod), "Media",
                       "Capacidad de muelles reportada como indicador, no como restricción"))
    return pd.DataFrame(filas)
