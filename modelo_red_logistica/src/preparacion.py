"""Limpieza, homologación y construcción de parámetros del modelo a partir de SAP, TMS y maestros.

Reglas de negocio principales
-----------------------------
* SAP es la fuente principal. TMS se usa como complemento cuando el viaje hace match
  (origen real del traslado, tiempos de tránsito, m3 por viaje, población faltante).
* Tipo XI = traslado entre planta/CEDI y otro CEDI. En XI el campo CEDI es el origen y el
  Cliente representa el nodo destino (homologado por ciudad o por parametros.yaml).
* Las líneas con IMPORTE SAP <= umbral (0 ó 1 peso) se omiten del cálculo de fletes
  (numerador y denominador); su volumen sí cuenta como demanda real.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .utilidades import LOG, a_numero, haversine_km, normalizar_serie_texto, normalizar_texto, normalizar_viaje


@dataclass
class Preparado:
    sap: pd.DataFrame
    tms: pd.DataFrame
    nodos: pd.DataFrame
    ciudades: pd.DataFrame
    plantas: pd.DataFrame
    demanda: pd.DataFrame
    tarifas_dist: pd.DataFrame
    tarifas_xi: pd.DataFrame
    regresion: dict
    factor_cedi: dict
    xi_asis: pd.DataFrame
    dist_asis: pd.DataFrame
    homolog_plantas: pd.DataFrame
    homolog_clientes: pd.DataFrame
    tiempos_lane: pd.DataFrame
    m3_por_posicion: float
    meses_base: int
    meses_lista: list
    series: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Maestros
# ---------------------------------------------------------------------------
def _preparar_clientes(cl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cl = cl.copy()
    cl["ciudad_n"] = normalizar_serie_texto(cl["Ciudad"])
    cl["depto_n"] = normalizar_serie_texto(cl["Departamento"])
    cl["zona"] = cl["Zona_Comercial"].astype("string").str.strip()
    zona_por_depto = cl.dropna(subset=["zona"]).groupby("depto_n")["zona"].agg(lambda s: s.mode().iat[0])
    cl["zona"] = cl["zona"].fillna(cl["depto_n"].map(zona_por_depto)).fillna(cl["Departamento"].str.title())
    cl["lat"] = a_numero(cl["Latitud"])
    cl["lon"] = a_numero(cl["Longitud"])
    for nivel in ("ciudad_n", "depto_n"):
        for c in ("lat", "lon"):
            cl[c] = cl[c].fillna(cl.groupby(nivel)[c].transform("mean"))
    ciudades = (cl.groupby("ciudad_n")
                .agg(lat=("lat", "mean"), lon=("lon", "mean"),
                     depto=("Departamento", lambda s: s.mode().iat[0]),
                     zona=("zona", lambda s: s.mode().iat[0]))
                .reset_index().rename(columns={"ciudad_n": "ciudad"}))
    return cl, ciudades


def _preparar_nodos(d: dict, cfg: dict, clientes: pd.DataFrame) -> pd.DataFrame:
    n = d["maestro_nodos"].copy()
    n["ID_Nodo"] = n["ID_Nodo"].str.strip()
    n["ciudad_n"] = normalizar_serie_texto(n["Ciudad"])
    n["lat"] = a_numero(n["Latitud"])
    n["lon"] = a_numero(n["Longitud"])
    n["coord_corregida"] = False
    # nodos de ciudades distintas con coordenadas idénticas -> se usan coordenadas de la ciudad (maestro clientes)
    dup = n.duplicated(["lat", "lon"], keep=False)
    coords_ciudad = clientes.groupby("ciudad_n")[["lat", "lon"]].mean()
    for i in n.index[dup]:
        c = n.at[i, "ciudad_n"]
        otros = n[(n.lat == n.at[i, "lat"]) & (n.lon == n.at[i, "lon"]) & (n.ciudad_n != c)]
        if len(otros) and c in coords_ciudad.index and n.at[i, "Tipo_Nodo"] == "CEDI":
            n.loc[i, ["lat", "lon"]] = coords_ciudad.loc[c, ["lat", "lon"]].values
            n.at[i, "coord_corregida"] = True
    ic = d["info_cedi"].copy()
    ic["ID_Nodo"] = ic["ID_Nodo"].str.strip()
    ic["posiciones"] = a_numero(ic["Posiciones_Estiba"])
    ic["dias_semana"] = a_numero(ic["Dias_Operacion_Semana"])
    ic["cap_despacho_estibas_dia"] = a_numero(ic["Cap_Despacho_Estibas_Dia"])
    ic["cap_recibo_estibas_dia"] = a_numero(ic["Cap_Recibo_Estibas_Dia"])
    pp = d["parametros_plantas"].copy()
    pp["ID_Nodo"] = pp["ID_Nodo"].str.strip()
    pp["costo_fijo"] = a_numero(pp["Costo_Fijo_COP_Mes"])
    n = (n.merge(ic[["ID_Nodo", "posiciones", "dias_semana", "cap_despacho_estibas_dia", "cap_recibo_estibas_dia"]],
                 on="ID_Nodo", how="left")
         .merge(pp[["ID_Nodo", "costo_fijo"]], on="ID_Nodo", how="left"))
    n["costo_fijo"] = n["costo_fijo"].fillna(0.0)
    n["candidata"] = False
    n["nombre"] = n["ID_Nodo"] + " " + n["Ciudad"].str.title()
    return n.set_index("ID_Nodo")


def _preparar_plantas(d: dict) -> pd.DataFrame:
    cp = d["capacidad_plantas"].copy()
    cp["ID_Nodo"] = cp["ID_Nodo"].str.strip()
    cp["capacidad_m3_mes"] = a_numero(cp["Capacidad_M3_Mes"])
    cp["familias"] = cp["Familia"].str.split(";").apply(lambda l: sorted({x.strip() for x in l if x.strip()}))
    return cp.set_index("ID_Nodo")[["capacidad_m3_mes", "familias"]]


# ---------------------------------------------------------------------------
# Regresión de tarifas por distancia
# ---------------------------------------------------------------------------
def _ajustar_regresion(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["km", "tarifa"])
    if len(df) < 5:
        med = float(df["tarifa"].median()) if len(df) else 40000.0
        return {"a": med, "b": 0.0, "r2": 0.0, "n": int(len(df))}
    w = np.sqrt(df["vol"].clip(lower=1).values)
    b, a = np.polyfit(df["km"].values, df["tarifa"].values, 1, w=w)
    if b < 0:  # sin sentido físico: se usa tarifa media ponderada
        a, b = float(np.average(df["tarifa"], weights=df["vol"])), 0.0
    pred = a + b * df["km"].values
    ww = df["vol"].values
    ss_res = np.sum(ww * (df["tarifa"].values - pred) ** 2)
    ss_tot = np.sum(ww * (df["tarifa"].values - np.average(df["tarifa"].values, weights=ww)) ** 2)
    return {"a": float(a), "b": float(b), "r2": float(1 - ss_res / ss_tot) if ss_tot else 0.0, "n": int(len(df))}


# ---------------------------------------------------------------------------
# Proceso principal
# ---------------------------------------------------------------------------
def preparar(d: dict, cfg: dict) -> Preparado:
    lim = cfg["limpieza"]
    info: dict = {}

    # ----------------------------------------------------------------- SAP
    sap = d["sap"]
    info["sap_filas_original"] = len(sap)
    dup_mask = sap.duplicated()
    info["sap_duplicados"] = int(dup_mask.sum())
    if lim.get("eliminar_duplicados_exactos", True):
        sap = sap[~dup_mask]
    sap = sap.copy()
    sap["imp"] = a_numero(sap["IMPORTE SAP"]).fillna(0.0)
    sap["vol"] = a_numero(sap["Volumen"]).fillna(0.0)
    sap["cant"] = a_numero(sap["Cantidad"])
    sap["fecha"] = pd.to_datetime(sap["Periodo"].str[:10], errors="coerce")
    sap["mes"] = sap["fecha"].dt.to_period("M").astype(str)
    sap["viaje_n"] = normalizar_viaje(sap["Viaje"])
    sap["es_xi"] = sap["Tipo"].isin(lim["tipos_traslado"])
    sap["flete_valido"] = sap["imp"] > float(lim["umbral_importe_invalido"])
    for c in ("Centro productor", "CEDI", "Cliente", "Referencia", "Tipo"):
        sap[c] = sap[c].astype("string").str.strip()
    LOG.info("SAP: %s líneas, %s con flete válido (> %s COP)", f"{len(sap):,}",
             f"{int(sap.flete_valido.sum()):,}", lim["umbral_importe_invalido"])

    # periodo base
    pb = cfg["periodo_base"]
    meses = pd.period_range(pb["desde"], pb["hasta"], freq="M").astype(str).tolist()
    sap["en_base"] = sap["mes"].isin(meses)
    n_meses = len(meses)

    # ---------------------------------------------------------- Clientes / nodos
    clientes, ciudades = _preparar_clientes(d["maestro_clientes"])
    nodos = _preparar_nodos(d, cfg, clientes)
    plantas_cap = _preparar_plantas(d)

    cli = clientes.set_index("Nombre_Cliente")
    sap["ciudad"] = sap["Cliente"].map(cli["ciudad_n"])
    sap["zona"] = sap["Cliente"].map(cli["zona"])

    # ciudad -> nodo
    ciudad_nodo = {c: i for i, c in nodos["ciudad_n"].items()}
    for c, nodo in (cfg["homologacion"].get("ciudades_nodo") or {}).items():
        ciudad_nodo[normalizar_texto(c)] = nodo

    # ---------------------------------------------------------- Referencias
    ref = d["maestro_referencias"].copy()
    ref["vcaja"] = a_numero(ref["Volumen_Caja_m3"])
    ref["cxe"] = a_numero(ref["Cajas_x_Estiba"])
    ref["m3_estiba"] = ref["vcaja"] * ref["cxe"]
    sap = sap.merge(ref[["Cod_Ref", "Familia", "Planta_Produccion_Antes", "Planta_Produccion_Nueva",
                         "Situacion_Post_Cambio", "vcaja", "m3_estiba"]],
                    left_on="Referencia", right_on="Cod_Ref", how="left")

    # homologación planta del maestro de referencias -> nodo SAP (inferida por volumen)
    obs = sap.dropna(subset=["Centro productor", "Planta_Produccion_Antes"])
    tab = obs.groupby(["Planta_Produccion_Antes", "Centro productor"])["vol"].sum().reset_index()
    tot = tab.groupby("Planta_Produccion_Antes")["vol"].transform("sum")
    tab["participacion"] = tab["vol"] / tot
    homolog_plantas = (tab.sort_values("vol", ascending=False).drop_duplicates("Planta_Produccion_Antes")
                       .rename(columns={"Planta_Produccion_Antes": "codigo_maestro", "Centro productor": "nodo_sap",
                                        "vol": "vol_m3"})
                       .sort_values("codigo_maestro").reset_index(drop=True))
    homolog_plantas["origen"] = "inferido SAP"
    mapa_plantas = dict(zip(homolog_plantas.codigo_maestro, homolog_plantas.nodo_sap))
    for k, v in (cfg["homologacion"].get("plantas_referencia") or {}).items():
        mapa_plantas[k] = v
        homolog_plantas.loc[homolog_plantas.codigo_maestro == k, ["nodo_sap", "origen"]] = [v, "parametros.yaml"]
    for cod in set(ref["Planta_Produccion_Nueva"].dropna()) - set(mapa_plantas):
        mapa_plantas[cod] = cod
    LOG.info("Homologación plantas maestro->SAP: %s", mapa_plantas)

    prod = cfg["produccion"]
    cerrada = prod["planta_cerrada"]
    sap["planta_asis"] = sap["Centro productor"].fillna(sap["Planta_Produccion_Antes"].map(mapa_plantas))
    sit = sap["Situacion_Post_Cambio"].fillna("")
    sap["descontinuada"] = sit.str.startswith("Descontinuada")
    nueva = sap["Planta_Produccion_Nueva"].map(mapa_plantas)
    sap["planta_tobe"] = np.where(sit.str.startswith("Trasladada"), nueva, sap["planta_asis"])
    sap.loc[sap["descontinuada"], "planta_tobe"] = None
    # familias -> planta destino principal de lo que sale de la planta cerrada
    movidas = sap[(sap.planta_asis == cerrada) & sap.planta_tobe.notna() & (sap.planta_tobe != cerrada)]
    destino_familia = movidas.groupby(["Unidad Negocio", "planta_tobe"])["vol"].sum().reset_index() \
        .sort_values("vol", ascending=False).drop_duplicates("Unidad Negocio") \
        .set_index("Unidad Negocio")["planta_tobe"]
    destino_def = movidas.groupby("planta_tobe")["vol"].sum().idxmax() if len(movidas) else None
    sin_destino = (sap.planta_tobe == cerrada) & ~sap.descontinuada
    info["lineas_tobe_en_planta_cerrada"] = int(sin_destino.sum())
    info["vol_tobe_en_planta_cerrada_base"] = float(sap.loc[sin_destino & sap.en_base, "vol"].sum())
    sap.loc[sin_destino, "planta_tobe"] = sap.loc[sin_destino, "Unidad Negocio"].map(destino_familia).fillna(destino_def)
    # líneas sin planta conocida -> planta con más volumen de la familia
    fam_planta = sap.dropna(subset=["planta_asis"]).groupby(["Unidad Negocio", "planta_asis"])["vol"].sum() \
        .reset_index().sort_values("vol", ascending=False).drop_duplicates("Unidad Negocio") \
        .set_index("Unidad Negocio")["planta_asis"]
    sin_planta = sap.planta_asis.isna()
    info["lineas_sin_planta"] = int(sin_planta.sum())
    sap.loc[sin_planta, "planta_asis"] = sap.loc[sin_planta, "Unidad Negocio"].map(fam_planta)
    m = sap.planta_tobe.isna() & ~sap.descontinuada
    sap.loc[m, "planta_tobe"] = sap.loc[m, "planta_asis"]

    # ---------------------------------------------------------- TMS
    tms = d["tms"].copy()
    tms["viaje_n"] = normalizar_viaje(tms["Viaje"])
    for c in ("Fecha salida real", "Fecha llegada real", "Fecha creación de viaje"):
        tms[c + "_dt"] = pd.to_datetime(tms[c].str[:19], errors="coerce")
    tms["fecha_centinela"] = (tms["Fecha salida real_dt"].dt.year < 2000) | (tms["Fecha llegada real_dt"].dt.year < 2000)
    tms["transito_h"] = (tms["Fecha llegada real_dt"] - tms["Fecha salida real_dt"]).dt.total_seconds() / 3600
    tms["transito_valido"] = tms["transito_h"].between(0.5, 360) & ~tms["fecha_centinela"]
    tms["vol_tms"] = a_numero(tms["Volumen"])
    tms["peso_tms"] = a_numero(tms["Peso"])
    tms["origen_n"] = normalizar_serie_texto(tms["Población Origen"])
    tms["destino_n"] = normalizar_serie_texto(tms["Población Destino"])
    tms["nodo_origen_tms"] = tms["origen_n"].map(ciudad_nodo)
    tms_v = tms.drop_duplicates("viaje_n").set_index("viaje_n")

    sap["match_tms"] = sap["viaje_n"].isin(tms_v.index)
    info["viajes_sap"] = int(sap["viaje_n"].nunique())
    info["viajes_match_crudo"] = int(len(set(sap["Viaje"].dropna()) & set(tms["Viaje"].dropna())))
    info["viajes_match_normalizado"] = int(sap.loc[sap.match_tms, "viaje_n"].nunique())
    LOG.info("Match SAP-TMS por viaje: %s de %s (%.1f %%)", f"{info['viajes_match_normalizado']:,}",
             f"{info['viajes_sap']:,}", 100 * info["viajes_match_normalizado"] / max(info["viajes_sap"], 1))

    # completar población faltante en SAP con destino TMS
    falta_pob = sap["Poblacion"].isna() & sap.match_tms
    sap["poblacion_final"] = sap["Poblacion"]
    sap.loc[falta_pob, "poblacion_final"] = sap.loc[falta_pob, "viaje_n"].map(tms_v["Población Destino"])
    info["poblacion_completada_tms"] = int(falta_pob.sum())

    # ---------------------------------------------------------- Traslados XI: origen / destino
    homolog_cli = {}
    for cliente, ciudad in cli["ciudad_n"].items():
        if ciudad in ciudad_nodo:
            homolog_cli[cliente] = ciudad_nodo[ciudad]
    for k, v in (cfg["homologacion"].get("clientes_nodo") or {}).items():
        homolog_cli[k] = v
    xi = sap.es_xi
    sap["nodo_destino"] = None
    sap.loc[xi, "nodo_destino"] = sap.loc[xi, "Cliente"].map(homolog_cli)
    sap["nodo_origen"] = sap["CEDI"]
    origen_tms = sap.loc[xi & sap.match_tms, "viaje_n"].map(tms_v["nodo_origen_tms"])
    corregir = origen_tms.notna() & (origen_tms != sap.loc[origen_tms.index, "CEDI"])
    sap.loc[corregir[corregir].index, "nodo_origen"] = origen_tms[corregir]
    info["xi_origen_corregido_tms"] = int(corregir.sum())
    info["xi_sin_nodo_destino"] = int((xi & sap.nodo_destino.isna()).sum())
    homolog_clientes = (sap[xi].groupby(["Cliente", "nodo_destino"], dropna=False)
                        .agg(lineas=("vol", "size"), vol_m3=("vol", "sum")).reset_index())
    homolog_clientes["ciudad_maestro"] = homolog_clientes["Cliente"].map(cli["Ciudad"])
    homolog_clientes["origen"] = np.where(homolog_clientes["Cliente"].isin(list((cfg["homologacion"].get("clientes_nodo") or {}).keys())),
                                          "parametros.yaml", "ciudad del nodo")
    homolog_clientes.loc[homolog_clientes.nodo_destino.isna(), "origen"] = "sin nodo (destino externo)"

    # ---------------------------------------------------------- Outliers de tarifa
    sap["tarifa_linea"] = np.where(sap.flete_valido & (sap.vol > 0), sap.imp / sap.vol.where(sap.vol > 0), np.nan)
    p_lo, p_hi = lim["percentil_outlier_tarifa"]
    sap["outlier_tarifa"] = False
    for flag in (True, False):
        s = sap.loc[sap.es_xi == flag, "tarifa_linea"].dropna()
        lo, hi = np.percentile(s, [p_lo, p_hi]) if len(s) else (0, np.inf)
        idx = sap.index[(sap.es_xi == flag) & sap.tarifa_linea.notna() & ~sap.tarifa_linea.between(lo, hi)]
        sap.loc[idx, "outlier_tarifa"] = True
        info[f"outlier_rango_{'xi' if flag else 'dist'}"] = (float(lo), float(hi))
    sap["usar_tarifa"] = sap.flete_valido & ~sap.outlier_tarifa & (sap.vol > 0)

    # ---------------------------------------------------------- Coordenadas y distancias
    coord_nodo = nodos[["lat", "lon"]]
    coord_ciudad = ciudades.set_index("ciudad")[["lat", "lon"]]
    fr = float(lim["factor_ruta_km"])

    def km_nodo_ciudad(o, c):
        if o not in coord_nodo.index or c not in coord_ciudad.index:
            return np.nan
        a, b = coord_nodo.loc[o], coord_ciudad.loc[c]
        return float(haversine_km(a.lat, a.lon, b.lat, b.lon)) * fr

    def km_nodo_nodo(o, dd):
        if o not in coord_nodo.index or dd not in coord_nodo.index:
            return np.nan
        a, b = coord_nodo.loc[o], coord_nodo.loc[dd]
        return float(haversine_km(a.lat, a.lon, b.lat, b.lon)) * fr

    vmin = float(lim["volumen_minimo_lane_m3"])
    base = sap[sap.en_base]

    # tarifas de distribución observadas (CEDI -> ciudad cliente)
    dist = base[~base.es_xi & base.ciudad.notna()]
    td = (dist[dist.usar_tarifa].groupby(["CEDI", "ciudad"])
          .agg(imp=("imp", "sum"), vol=("vol", "sum"), lineas=("imp", "size")).reset_index())
    td["tarifa"] = td.imp / td.vol
    td["km"] = [km_nodo_ciudad(o, c) for o, c in zip(td.CEDI, td.ciudad)]
    td["confiable"] = td.vol >= vmin
    td = td.rename(columns={"CEDI": "origen", "ciudad": "destino"})

    # tarifas de traslado observadas (nodo -> nodo)
    xib = base[base.es_xi & base.nodo_destino.notna() & (base.nodo_origen != base.nodo_destino)]
    tx = (xib[xib.usar_tarifa].groupby(["nodo_origen", "nodo_destino"])
          .agg(imp=("imp", "sum"), vol=("vol", "sum"), lineas=("imp", "size")).reset_index())
    tx["tarifa"] = tx.imp / tx.vol
    tx["km"] = [km_nodo_nodo(o, x) for o, x in zip(tx.nodo_origen, tx.nodo_destino)]
    tx["confiable"] = tx.vol >= vmin
    tx = tx.rename(columns={"nodo_origen": "origen", "nodo_destino": "destino"})

    reg = {"distribucion": _ajustar_regresion(td[td.confiable]),
           "traslado": _ajustar_regresion(tx[tx.confiable])}
    LOG.info("Regresión tarifa distribución: %.0f + %.1f*km (R2=%.2f, n=%s)", reg["distribucion"]["a"],
             reg["distribucion"]["b"], reg["distribucion"]["r2"], reg["distribucion"]["n"])
    LOG.info("Regresión tarifa traslado:     %.0f + %.1f*km (R2=%.2f, n=%s)", reg["traslado"]["a"],
             reg["traslado"]["b"], reg["traslado"]["r2"], reg["traslado"]["n"])
    # factor por CEDI (prima de cada origen sobre la curva general)
    tdc = td[td.confiable & td.km.notna()].copy()
    tdc["pred"] = reg["distribucion"]["a"] + reg["distribucion"]["b"] * tdc.km
    factor_cedi = (tdc.assign(r=tdc.tarifa / tdc.pred).groupby("origen")
                   .apply(lambda g: float(np.average(g.r, weights=g.vol)), include_groups=False)
                   .clip(0.5, 2.0).to_dict())

    # flujos AS-IS observados (m3/mes) — todo el volumen, con o sin flete válido
    dist_asis = (dist.groupby(["CEDI", "ciudad"])["vol"].sum() / n_meses).reset_index() \
        .rename(columns={"CEDI": "origen", "ciudad": "destino", "vol": "vol_mes"})
    xi_all = base[base.es_xi]
    xi_asis = (xi_all.groupby(["nodo_origen", "nodo_destino"], dropna=False)["vol"].sum() / n_meses).reset_index() \
        .rename(columns={"nodo_origen": "origen", "nodo_destino": "destino", "vol": "vol_mes"})

    # ---------------------------------------------------------- Tiempos de tránsito y m3/viaje (TMS)
    viajes = (base[base.match_tms].groupby("viaje_n")
              .agg(es_xi=("es_xi", "max"), origen=("nodo_origen", "first"), cedi=("CEDI", "first"),
                   destino_nodo=("nodo_destino", "first"), ciudad=("ciudad", "first"), vol_sap=("vol", "sum")))
    viajes = viajes.join(tms_v[["transito_h", "transito_valido", "vol_tms"]], how="inner")
    viajes["destino"] = np.where(viajes.es_xi, viajes.destino_nodo, viajes.ciudad)
    viajes["origen"] = np.where(viajes.es_xi, viajes.origen, viajes.cedi)
    tv = viajes[viajes.transito_valido]
    tiempos_lane = (tv.groupby(["es_xi", "origen", "destino"])
                    .agg(viajes=("transito_h", "size"), transito_h_mediana=("transito_h", "median"),
                         m3_por_viaje=("vol_tms", "median")).reset_index())
    info["viajes_con_transito_valido"] = int(len(tv))
    ratio = (viajes.vol_sap / viajes.vol_tms.where(viajes.vol_tms > 0))
    info["viajes_vol_sap_tms_10pct"] = float(ratio.between(0.9, 1.1).mean()) if len(ratio) else 0.0

    # ---------------------------------------------------------- Demanda (m3/mes)
    crec = 1.0 + float(prod.get("crecimiento_demanda", 0.0))
    dem = (dist.groupby(["ciudad", "zona", "Unidad Negocio", "planta_asis", "planta_tobe", "descontinuada"],
                        dropna=False)["vol"].sum() / n_meses * crec).reset_index() \
        .rename(columns={"Unidad Negocio": "familia", "vol": "vol_mes"})
    dem["planta_tobe"] = dem["planta_tobe"].astype(object).where(dem["planta_tobe"].notna(), None)
    info["demanda_m3_mes"] = float(dem.vol_mes.sum())
    info["exportacion_m3_mes"] = float(dist[dist.Mercado == "E"].vol.sum() / n_meses)

    # ---------------------------------------------------------- m3 por posición
    alm = cfg["almacenamiento"]
    if alm.get("m3_por_posicion"):
        m3_pos = float(alm["m3_por_posicion"])
    else:
        v = base[(base.m3_estiba > 0) & (base.m3_estiba < 5)]
        m3_pos = float(v.vol.sum() / (v.vol / v.m3_estiba).sum())
    LOG.info("m3 por posición de estiba (ponderado por volumen): %.2f", m3_pos)

    # ---------------------------------------------------------- Capacidad y costos de nodos
    util = float(alm["utilizacion_maxima"])
    cmod = alm["costos_por_modalidad"]
    nodos["manejo_cop_m3"] = nodos["Modalidad"].map(lambda x: cmod.get(x, cmod["Propio"])["manejo_cop_m3"])
    nodos["posicion_cop_mes"] = nodos["Modalidad"].map(lambda x: cmod.get(x, cmod["Propio"])["posicion_cop_mes"])
    nodos["posiciones_adicionales"] = 0.0
    nodos["tarifas_salida_como"] = None
    nodos["tarifa_entrada_como"] = None
    for nodo, aj in (alm.get("ajustes_nodo") or {}).items():
        for k, v in aj.items():
            nodos.at[nodo, k] = v
    for bc in cfg.get("bodegas_candidatas") or []:
        if not bc.get("activa", True):
            continue
        with warnings.catch_warnings():  # pandas avisa por columnas vacías al agregar la fila
            warnings.simplefilter("ignore", FutureWarning)
            nodos.loc[bc["id"]] = pd.Series({
                "Nombre_Nodo": bc["nombre"], "Tipo_Nodo": "Bodega externa candidata", "Estado": "Candidata",
                "Modalidad": "Operador logístico (3PL)", "Ciudad": bc["ciudad"].title(),
                "ciudad_n": normalizar_texto(bc["ciudad"]), "lat": bc["lat"], "lon": bc["lon"],
                "posiciones": bc["posiciones"], "costo_fijo": bc["costo_fijo_cop_mes"],
                "manejo_cop_m3": bc["manejo_cop_m3"], "posicion_cop_mes": bc["posicion_cop_mes"],
                "candidata": True, "nombre": f"{bc['id']} {bc['ciudad'].title()}", "posiciones_adicionales": 0.0,
                "tarifas_salida_como": bc.get("tarifas_salida_como"), "tarifa_entrada_como": bc.get("tarifa_entrada_como"),
                "coord_corregida": False,
            })
    nodos["posiciones_total"] = nodos["posiciones"].fillna(0) + nodos["posiciones_adicionales"].fillna(0)
    nodos["cap_inv_m3"] = nodos["posiciones_total"] * m3_pos * util
    dias = nodos["dias_semana"].fillna(6)
    nodos["cap_despacho_m3_mes"] = nodos["cap_despacho_estibas_dia"] * dias * 52 / 12 * m3_pos
    despacho_asis = dist.groupby("CEDI")["vol"].sum() / n_meses
    xi_salida = xi_all.groupby("nodo_origen")["vol"].sum() / n_meses
    nodos["despacho_asis_m3_mes"] = despacho_asis.reindex(nodos.index).fillna(0)
    nodos["traslado_salida_asis_m3_mes"] = xi_salida.reindex(nodos.index).fillna(0)

    # DOI implícito: el nodo de calibración (CEDI de la planta que usa bodega externa) se asume lleno en AS-IS
    nodo_cal = alm.get("nodo_calibracion_doi")
    if nodo_cal in nodos.index and nodos.at[nodo_cal, "despacho_asis_m3_mes"] > 0:
        info["doi_implicito"] = float(30 * nodos.at[nodo_cal, "cap_inv_m3"] / nodos.at[nodo_cal, "despacho_asis_m3_mes"])
    else:
        info["doi_implicito"] = 20.0
    if str(alm["dias_inventario"]).lower() == "auto":
        alm["dias_inventario_efectivo"] = round(info["doi_implicito"], 1)
    else:
        alm["dias_inventario_efectivo"] = float(alm["dias_inventario"])
    LOG.info("Días de inventario: %.1f (implícito AS-IS en %s = %.1f)", alm["dias_inventario_efectivo"], nodo_cal,
             info["doi_implicito"])

    # ---------------------------------------------------------- Plantas
    plantas_tobe = set(dem.planta_tobe.dropna())
    plantas_asis = set(dem.planta_asis.dropna())
    plantas = pd.DataFrame(index=sorted(plantas_tobe | plantas_asis))
    plantas["capacidad_m3_mes"] = plantas_cap["capacidad_m3_mes"].reindex(plantas.index)
    plantas["familias"] = plantas_cap["familias"].reindex(plantas.index)
    plantas["produccion_asis_m3_mes"] = dem.groupby("planta_asis").vol_mes.sum().reindex(plantas.index).fillna(0)
    tobe = dem[~dem.descontinuada] if prod.get("excluir_descontinuadas", True) else dem
    plantas["produccion_tobe_m3_mes"] = tobe.groupby("planta_tobe").vol_mes.sum().reindex(plantas.index).fillna(0)
    plantas.loc[cerrada, "produccion_tobe_m3_mes"] = 0.0 if cerrada in plantas.index else 0.0
    plantas["util_asis"] = plantas.produccion_asis_m3_mes / plantas.capacidad_m3_mes
    plantas["util_tobe"] = plantas.produccion_tobe_m3_mes / plantas.capacidad_m3_mes

    # ---------------------------------------------------------- Series para gráficos
    sap_full = sap[sap.mes.notna()]
    series = {
        "vol_planta_mes": sap_full[~sap_full.es_xi].pivot_table(index="mes", columns="planta_asis", values="vol",
                                                                aggfunc="sum").fillna(0),
        "vol_tipo_mes": sap_full.pivot_table(index="mes", columns="es_xi", values="vol", aggfunc="sum").fillna(0),
        "flete_valido_tipo": sap.groupby("Tipo").agg(lineas=("imp", "size"),
                                                    pct_invalido=("flete_valido", lambda s: 1 - s.mean()),
                                                    vol_m3=("vol", "sum")).reset_index(),
        "meses_datos": sorted(sap_full.mes.unique().tolist()),
    }
    info["periodo_datos"] = (str(sap.fecha.min().date()), str(sap.fecha.max().date()))
    info["tms_periodo"] = (str(tms["Fecha creación de viaje_dt"].min()), str(tms["Fecha creación de viaje_dt"].max()))
    info["ciudad_nodo"] = ciudad_nodo
    info["homolog_cli"] = homolog_cli

    return Preparado(sap=sap, tms=tms, nodos=nodos, ciudades=ciudades, plantas=plantas, demanda=dem,
                     tarifas_dist=td, tarifas_xi=tx, regresion=reg, factor_cedi=factor_cedi,
                     xi_asis=xi_asis, dist_asis=dist_asis, homolog_plantas=homolog_plantas,
                     homolog_clientes=homolog_clientes, tiempos_lane=tiempos_lane, m3_por_posicion=m3_pos,
                     meses_base=n_meses, meses_lista=meses, series=series, info=info)
