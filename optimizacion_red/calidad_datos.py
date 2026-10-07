"""
Validación y corrección de los extractos SAP (fuente principal) y TMS
(fuente complementaria, unida por número de viaje cuando hay match).

Cada regla deja trazabilidad: cuántos registros toca, cuánto peso (t) y cuánto
importe de flete representa, y qué acción se tomó (corregir / excluir del
cálculo de tarifa / excluir del todo).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import parametros as P
from datos_prueba import km_vial


@dataclass
class Bitacora:
    reglas: list = field(default_factory=list)

    def registrar(self, regla, descripcion, accion, mascara, df):
        self.reglas.append(dict(
            regla=regla, descripcion=descripcion, accion=accion,
            registros=int(mascara.sum()),
            toneladas=float(df.loc[mascara, "peso_kg"].fillna(0).clip(lower=0).sum() / 1000) if "peso_kg" in df else 0.0,
            importe=float(df.loc[mascara, "importe_flete"].fillna(0).clip(lower=0).sum()) if "importe_flete" in df else 0.0,
        ))

    def tabla(self):
        return pd.DataFrame(self.reglas)


def numero_flexible(v):
    """Convierte '1.234.567,89', '1,234,567.89', 1234567 o '' a float."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = re.sub(r"[^\d,.\-]", "", str(v))
    if not s:
        return np.nan
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        partes = s.split(",")
        s = s.replace(",", ".") if len(partes) == 2 and len(partes[1]) <= 2 else s.replace(",", "")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def normalizar_viaje(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = re.sub(r"\D", "", str(v).split(".")[0]).lstrip("0")
    return s or None


def limpiar_sap(sap_raw: pd.DataFrame, maestro_mat: pd.DataFrame | None = None):
    bit = Bitacora()
    df = sap_raw.rename(columns={v: k for k, v in P.COLUMNAS_SAP.items()}).copy()
    faltantes = [c for c in P.COLUMNAS_SAP if c not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas SAP faltantes (revise COLUMNAS_SAP en parametros.py): {faltantes}")
    total_inicial = len(df)

    # --- Normalización de tipos y textos ---------------------------------
    for c in ["tipo_mov", "origen", "destino", "material", "familia", "unidad", "transportista"]:
        df[c] = df[c].astype("string").str.strip().str.upper()
    for c in ["importe_flete", "peso_kg", "cantidad"]:
        es_texto = df[c].map(lambda v: isinstance(v, str))
        if c == "importe_flete" and es_texto.any():
            bit.registrar("R01", "Importe con formato texto (separadores de miles/decimales)", "Corregido: convertido a número",
                          es_texto, df.assign(importe_flete=df[c].map(numero_flexible), peso_kg=df["peso_kg"].map(numero_flexible)))
        df[c] = df[c].map(numero_flexible)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["viaje_norm"] = df["viaje"].map(normalizar_viaje)
    tipo_sucio = sap_raw[P.COLUMNAS_SAP["tipo_mov"]].astype(str) != df["tipo_mov"].astype(str)
    bit.registrar("R02", "Tipo de movimiento / centro con espacios o minúsculas", "Corregido: trim + mayúsculas",
                  tipo_sucio | (sap_raw[P.COLUMNAS_SAP["destino"]].astype(str) != df["destino"].astype(str)), df)

    # --- Filtro de alcance: solo traslados XI ------------------------------
    m = df["tipo_mov"] != P.TIPO_MOV_TRASLADO
    bit.registrar("R03", "Movimientos distintos a XI (ventas, devoluciones, otros)", "Excluido: fuera de alcance", m, df)
    df = df[~m]

    # --- Duplicados ---------------------------------------------------------
    m = df.duplicated(subset=["documento", "posicion", "tipo_mov"], keep="first")
    bit.registrar("R04", "Registros duplicados (documento + posición)", "Excluido", m, df)
    df = df[~m]

    # --- Fechas -------------------------------------------------------------
    m = df["fecha"].isna() | (df["fecha"] < P.FECHA_MIN) | (df["fecha"] > P.FECHA_MAX)
    bit.registrar("R05", "Fecha nula o fuera de rango", "Excluido", m, df)
    df = df[~m]

    # --- Reversiones / cantidades no positivas ------------------------------
    m = (df["cantidad"] <= 0) | (df["peso_kg"] < 0)
    bit.registrar("R06", "Cantidad negativa o cero (reversión / anulación)", "Excluido", m, df)
    df = df[~m]

    # --- Centros ------------------------------------------------------------
    validos = set(P.PLANTAS) | set(P.CEDIS)
    m = ~df["origen"].isin(validos) | ~df["destino"].isin(validos)
    bit.registrar("R07", "Centro origen/destino no existe en el maestro", "Excluido", m, df)
    df = df[~m]
    m = df["origen"] == df["destino"]
    bit.registrar("R08", "Origen igual a destino", "Excluido", m, df)
    df = df[~m]

    # --- Peso faltante: imputación ------------------------------------------
    m = df["peso_kg"].isna() | (df["peso_kg"] == 0)
    if m.any():
        kg_u = maestro_mat.set_index("Material")["Kg_Unidad"] if maestro_mat is not None else pd.Series(dtype=float)
        kg_fam = (df.loc[~m, "peso_kg"] / df.loc[~m, "cantidad"]).groupby(df.loc[~m, "familia"]).median()
        factor = df.loc[m, "material"].map(kg_u).fillna(df.loc[m, "familia"].map(kg_fam))
        df.loc[m, "peso_kg"] = df.loc[m, "cantidad"] * factor
        bit.registrar("R09", "Peso faltante", "Corregido: cantidad × kg/unidad del maestro de materiales", m, df)

    # --- Flete 0 / 1 peso: se conserva el volumen, se excluye del costo ------
    df["valido_tarifa"] = True
    df["motivo_exclusion_tarifa"] = ""
    m = df["importe_flete"].isna() | (df["importe_flete"] <= P.IMPORTE_MINIMO_VALIDO)
    bit.registrar("R10", f"Importe de flete en 0, 1 peso o vacío", "Excluido del cálculo de tarifa (el volumen sí cuenta para demanda)", m, df)
    df.loc[m, ["valido_tarifa", "motivo_exclusion_tarifa"]] = [False, "importe<=1"]

    # --- Atípicos de costo unitario por ruta (IQR) --------------------------
    df["cop_kg"] = df["importe_flete"] / df["peso_kg"]
    df["ruta"] = df["origen"] + "→" + df["destino"]
    ok = df["valido_tarifa"]
    df["log_cop_kg"] = np.log(df["cop_kg"].where(df["cop_kg"] > 0))
    q = df[ok].groupby("ruta")["log_cop_kg"].quantile([0.25, 0.75]).unstack()
    q.columns = ["q1", "q3"]
    lim = df[["ruta"]].join(q, on="ruta")
    iqr = lim["q3"] - lim["q1"]
    m = ok & ((df["log_cop_kg"] < lim["q1"] - P.OUTLIER_IQR_FACTOR * iqr) | (df["log_cop_kg"] > lim["q3"] + P.OUTLIER_IQR_FACTOR * iqr))
    bit.registrar("R11", f"Costo COP/kg atípico en la ruta (IQR × {P.OUTLIER_IQR_FACTOR} sobre log)", "Excluido del cálculo de tarifa", m, df)
    df.loc[m, ["valido_tarifa", "motivo_exclusion_tarifa"]] = [False, "atipico"]

    m = df["viaje_norm"].isna()
    bit.registrar("R12", "Registro sin número de viaje", "Se conserva; no puede enriquecerse con TMS", m, df)

    resumen = dict(registros_iniciales=total_inicial, registros_xi_limpios=len(df),
                   registros_validos_tarifa=int(df["valido_tarifa"].sum()))
    return df.reset_index(drop=True), bit, resumen


def limpiar_tms(tms_raw: pd.DataFrame):
    t = tms_raw.rename(columns={v: k for k, v in P.COLUMNAS_TMS.items()}).copy()
    t["viaje_norm"] = t["viaje"].map(normalizar_viaje)
    for c in ["km", "capacidad_kg", "costo_tms"]:
        t[c] = t[c].map(numero_flexible)
    t["tipo_vehiculo"] = t["tipo_vehiculo"].astype("string").str.strip().str.upper()
    t["km_invalido"] = t["km"].isna() | (t["km"] <= 0)
    t.loc[t["km_invalido"], "km"] = np.nan
    t = t.drop_duplicates("viaje_norm")
    return t


def cruzar_sap_tms(sap: pd.DataFrame, tms: pd.DataFrame, bit: Bitacora):
    """SAP manda; TMS complementa (km, vehículo, capacidad) cuando el viaje cruza."""
    cols = ["viaje_norm", "tipo_vehiculo", "capacidad_kg", "km", "costo_tms", "km_invalido"]
    df = sap.merge(tms[cols], on="viaje_norm", how="left")
    df["match_tms"] = df["tipo_vehiculo"].notna()
    # km esperado por distancia geodésica para validar el km del TMS
    coords = {**{k: (v["lat"], v["lon"]) for k, v in P.PLANTAS.items()}, **{k: (v["lat"], v["lon"]) for k, v in P.CEDIS.items()}}
    df["km_teorico"] = [km_vial(*coords[o], *coords[d]) for o, d in zip(df["origen"], df["destino"])]
    m = df["match_tms"] & df["km"].notna() & ((df["km"] > 2.5 * df["km_teorico"]) | (df["km"] < 0.4 * df["km_teorico"]))
    df.loc[m, "km"] = np.nan
    bit.registrar("R13", "Km del TMS incoherente con la distancia de la ruta", "Corregido: se usa km teórico", m, df)
    m = df["match_tms"] & df["km_invalido"].fillna(False).astype(bool)
    bit.registrar("R14", "Km del TMS en cero o vacío", "Corregido: se usa km teórico", m, df)
    m = ~df["match_tms"] & df["viaje_norm"].notna()
    bit.registrar("R15", "Viaje SAP sin match en TMS", "Se conserva SAP; km teórico, sin dato de vehículo", m, df)

    viajes = df.groupby("viaje_norm").agg(peso_viaje=("peso_kg", "sum"))
    df = df.join(viajes, on="viaje_norm")
    df["utilizacion"] = np.where(df["match_tms"], df["peso_viaje"] / df["capacidad_kg"], np.nan)
    df["km_usado"] = df["km"].fillna(df["km_teorico"])

    v_sap = df["viaje_norm"].dropna().unique()
    match = dict(
        viajes_sap=int(len(v_sap)),
        viajes_con_match=int(df.loc[df["match_tms"], "viaje_norm"].nunique()),
        lineas_sin_viaje=int(df["viaje_norm"].isna().sum()),
        viajes_tms_sin_sap=int((~tms["viaje_norm"].isin(v_sap)).sum()),
    )
    match["tasa_match"] = match["viajes_con_match"] / max(1, match["viajes_sap"])
    return df, match
