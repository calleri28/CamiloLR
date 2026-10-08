"""
Lectura, validación y corrección de la información.

SAP es la fuente principal. TMS complementa por número de viaje cuando hay
match. Los maestros (referencias, nodos, clientes, Info_CEDI) se validan y se
usan para corregir y enriquecer SAP.

Cada regla deja trazabilidad en la bitácora: tabla, cuántos registros toca,
cuánto volumen (m3) e importe representa y qué acción se tomó.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import parametros as P
from datos_prueba import km_vial


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm_encabezado(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", sin_tildes(str(s)).upper())


def norm_texto(serie: pd.Series) -> pd.Series:
    """Mayúsculas, sin tildes, sin espacios dobles. Vacíos -> NA."""
    s = serie.astype("string").str.strip().map(lambda v: sin_tildes(v).upper() if isinstance(v, str) else v)
    s = s.str.replace(r"\s+", " ", regex=True)
    return s.mask(s.isin(["", "NAN", "NONE", "NULL"]))


def norm_codigo(v):
    """Códigos SAP (viaje, referencia): solo alfanuméricos, sin ceros a la izquierda."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    s = re.sub(r"[^0-9A-Za-z]", "", str(v).split(".")[0] if re.fullmatch(r"\d+\.0", str(v)) else str(v)).upper().lstrip("0")
    return s or None


def numero_flexible(v):
    """'1.234.567,89' · '1,234,567.89' · '2,9E-02' · 1234567 · '' -> float."""
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = re.sub(r"[^\d,.\-eE+]", "", str(v).strip())
    if not s or not re.search(r"\d", s):
        return np.nan
    if "e" in s.lower():
        try:
            return float(s.replace(",", "."))
        except ValueError:
            return np.nan
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        partes = s.split(",")
        s = s.replace(",", ".") if len(partes) == 2 else s.replace(",", "")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def leer_tabla(ruta: Path) -> pd.DataFrame | None:
    """Lee CSV (separador ; , o tab, UTF-8 o Latin-1) o Excel con el mismo nombre."""
    ruta = Path(ruta)
    for alterna in (ruta, ruta.with_suffix(".xlsx"), ruta.with_suffix(".xls")):
        if alterna.exists():
            ruta = alterna
            break
    else:
        return None
    if ruta.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(ruta, dtype=str)
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(ruta, encoding=enc) as fh:
                muestra = fh.read(20_000)
            sep = csv.Sniffer().sniff(muestra.splitlines()[0], delimiters=";,\t|").delimiter
            return pd.read_csv(ruta, sep=sep, dtype=str, encoding=enc, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo leer {ruta}")


def renombrar(df: pd.DataFrame, mapeo: dict, tabla: str, obligatorias=()) -> pd.DataFrame:
    """Renombra encabezados reales -> nombres internos ignorando tildes, mayúsculas y espacios."""
    reales = {norm_encabezado(c): c for c in df.columns}
    ren, faltan = {}, []
    for interno, externo in mapeo.items():
        c = reales.get(norm_encabezado(externo))
        if c is not None:
            ren[c] = interno
        else:
            faltan.append(externo)
    obl = [mapeo[k] for k in obligatorias if mapeo[k] in faltan]
    if obl:
        raise ValueError(f"[{tabla}] faltan columnas obligatorias: {obl}. Revise el mapeo en parametros.py. "
                         f"Columnas encontradas: {list(df.columns)}")
    out = df.rename(columns=ren)
    for interno, externo in mapeo.items():
        if externo in faltan:
            out[interno] = pd.NA
    return out


@dataclass
class Bitacora:
    reglas: list = field(default_factory=list)

    def registrar(self, tabla, regla, descripcion, accion, mascara, df=None, vol="volumen", imp="importe"):
        m = pd.Series(mascara).fillna(False).astype(bool)
        d = df if df is not None else pd.DataFrame(index=m.index)
        self.reglas.append(dict(
            tabla=tabla, regla=regla, descripcion=descripcion, accion=accion, registros=int(m.sum()),
            volumen_m3=float(pd.to_numeric(d.loc[m, vol], errors="coerce").clip(lower=0).sum()) if vol in d else 0.0,
            importe=float(pd.to_numeric(d.loc[m, imp], errors="coerce").clip(lower=0).sum()) if imp in d else 0.0))

    def tabla(self):
        return pd.DataFrame(self.reglas)


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def cargar_entradas(carpeta: Path) -> dict:
    datos = {k: leer_tabla(Path(carpeta) / nombre) for k, nombre in P.ARCHIVOS.items()}
    obligatorios = ["sap", "referencias", "nodos", "homologacion", "capacidad", "param_plantas", "param_familias"]
    faltan = [P.ARCHIVOS[k] for k in obligatorios if datos[k] is None]
    if faltan:
        raise FileNotFoundError(f"No se encontraron estos archivos en {carpeta}: {faltan}. "
                                "Ejecute `python modelo_red.py --generar-prueba` para crear datos de ejemplo.")
    return datos


# ---------------------------------------------------------------------------
# Maestros
# ---------------------------------------------------------------------------
def coord_valida(lat, lon):
    (a, b), (c, d) = P.LIMITES_COLOMBIA["lat"], P.LIMITES_COLOMBIA["lon"]
    return lat.between(a, b) & lon.between(c, d)


def limpiar_clientes(raw, bit: Bitacora):
    if raw is None:
        return None
    c = renombrar(raw, P.COLUMNAS_CLIENTES, "Clientes", ["nombre", "ciudad"])
    c["nombre_n"], c["ciudad_n"] = norm_texto(c["nombre"]), norm_texto(c["ciudad"])
    c["lat"], c["lon"] = c["lat"].map(numero_flexible), c["lon"].map(numero_flexible)
    m = c["lon"] > 0
    bit.registrar("Clientes", "C01", "Longitud positiva (signo perdido)", "Corregido: se cambia el signo", m)
    c.loc[m, "lon"] = -c.loc[m, "lon"]
    m = c["lat"].isna() | c["lon"].isna()
    bit.registrar("Clientes", "C02", "Cliente sin coordenadas", "Se completa con la ciudad de otros clientes / nodos", m)
    centro = c[~m].groupby("ciudad_n")[["lat", "lon"]].median()
    c.loc[m, "lat"] = c.loc[m, "ciudad_n"].map(centro["lat"])
    c.loc[m, "lon"] = c.loc[m, "ciudad_n"].map(centro["lon"])
    m = c["lat"].notna() & ~coord_valida(c["lat"], c["lon"])
    bit.registrar("Clientes", "C03", "Coordenadas fuera de Colombia (exportación u error)", "Se conserva; se marca como internacional", m)
    c["internacional"] = m
    return c


def limpiar_nodos(raw_nodos, raw_homolog, clientes, bit: Bitacora):
    n = renombrar(raw_nodos, P.COLUMNAS_NODOS, "Nodos", ["id_nodo", "tipo", "lat", "lon"])
    n["id_nodo"] = norm_texto(n["id_nodo"])
    n["tipo_n"], n["estado_n"], n["ciudad_n"] = norm_texto(n["tipo"]), norm_texto(n["estado"]), norm_texto(n["ciudad"])
    n["clase"] = np.where(n["tipo_n"].fillna("").str.contains(P.PALABRA_PLANTA), "PLANTA",
                          np.where(n["tipo_n"].fillna("").map(lambda t: any(p in t for p in P.PALABRAS_CEDI)), "CEDI", "OTRO"))
    n["lat"], n["lon"] = n["lat"].map(numero_flexible), n["lon"].map(numero_flexible)
    n["fecha_cierre"] = pd.to_datetime(n["fecha_cierre"].replace("", pd.NA), errors="coerce")
    m = n["id_nodo"].duplicated()
    bit.registrar("Nodos", "N01", "ID_Nodo duplicado", "Excluido (se conserva el primero)", m)
    n = n[~m].copy()
    m = n["lon"] > 0
    bit.registrar("Nodos", "N02", "Longitud positiva (en Colombia es negativa)", "Corregido: se cambia el signo", m)
    n.loc[m, "lon"] = -n.loc[m, "lon"]
    m = n["lat"].isna() | n["lon"].isna()
    if m.any() and clientes is not None:
        ref = clientes.groupby("ciudad_n")[["lat", "lon"]].median()
        n.loc[m, "lat"] = n.loc[m, "ciudad_n"].map(ref["lat"])
        n.loc[m, "lon"] = n.loc[m, "ciudad_n"].map(ref["lon"])
    bit.registrar("Nodos", "N03", "Nodo sin coordenadas", "Corregido con la ubicación de clientes de la misma ciudad", m)
    m = n["lat"].isna() | ~coord_valida(n["lat"], n["lon"])
    bit.registrar("Nodos", "N04", "Coordenadas inválidas tras corrección", "Nodo excluido del modelo (revisar maestro)", m)
    n = n[~m]
    m = n["clase"] == "OTRO"
    bit.registrar("Nodos", "N05", "Nodos que no son planta ni CEDI (maquiladores, proveedores)", "Informativo: no entran a la red", m)

    h = renombrar(raw_homolog, P.COLUMNAS_HOMOLOGACION, "Homologación", ["codigo_sap", "id_nodo"])
    h["codigo_sap"], h["id_nodo"] = norm_texto(h["codigo_sap"]), norm_texto(h["id_nodo"])
    homolog = dict(zip(h["codigo_sap"], h["id_nodo"]))
    homolog.update({i: i for i in n["id_nodo"]})        # si SAP ya usa el ID_Nodo, también sirve
    m = ~h["id_nodo"].isin(n["id_nodo"])
    bit.registrar("Homologación", "H01", "Código homologado a un ID_Nodo que no existe en el maestro", "Ese código no se podrá usar", m)
    return n.set_index("id_nodo"), homolog


def limpiar_referencias(raw, nodos, homolog, bit: Bitacora):
    r = renombrar(raw, P.COLUMNAS_REFERENCIAS, "Referencias", ["cod_ref", "familia", "vol_caja_m3"])
    r["cod_n"] = r["cod_ref"].map(norm_codigo)
    r["familia"] = norm_texto(r["familia"])
    for c in ["planta_antes", "planta_nueva", "situacion", "control"]:
        r[c] = norm_texto(r[c])
    for c in ["planta_antes", "planta_nueva"]:          # acepta código SAP (CO45) o ID_Nodo (PL01)
        r[c] = r[c].map(lambda v: homolog.get(v, v) if isinstance(v, str) else v)
    r["ref_sust_n"] = r["ref_sustituta"].map(norm_codigo)
    for c in ["unid_caja", "cajas_estiba", "largo_cm", "ancho_cm", "alto_cm", "vol_caja_m3", "peso_caja_kg"]:
        r[c] = r[c].map(numero_flexible)

    m = r["cod_n"].duplicated()
    bit.registrar("Referencias", "RF01", "Cod_Ref duplicado", "Excluido (se conserva el primero)", m)
    r = r[~m].copy()

    vol_dim = r["largo_cm"] * r["ancho_cm"] * r["alto_cm"] / 1e6
    ratio = r["vol_caja_m3"] / vol_dim
    m = vol_dim.notna() & (r["vol_caja_m3"].isna() | (ratio < 0.7) | (ratio > 1.4))
    bit.registrar("Referencias", "RF02", "Volumen_Caja_m3 vacío o inconsistente con largo × ancho × alto (p. ej. en cm³)",
                  "Corregido: se recalcula con las dimensiones", m)
    r.loc[m, "vol_caja_m3"] = vol_dim[m]
    r["m3_estiba"] = r["vol_caja_m3"] * r["cajas_estiba"]

    sit = r["situacion"].fillna("")
    tras = sit.str.contains("TRASLAD")
    desc = sit.str.contains("DESCONTINU")
    plantas = set(nodos.index[nodos["clase"] == "PLANTA"])
    m1 = tras & (r["planta_nueva"].isna() | (r["planta_nueva"] == r["planta_antes"]) | ~r["planta_nueva"].isin(plantas))
    bit.registrar("Referencias", "RF03", "Trasladada sin planta nueva válida (vacía, igual a la anterior o inexistente)",
                  "No se usa para el plan de traslado", m1)
    m2 = desc & r["planta_nueva"].notna()
    bit.registrar("Referencias", "RF04", "Descontinuada con planta nueva diligenciada", "Se ignora la planta nueva", m2)
    r.loc[m2, "planta_nueva"] = pd.NA
    m3 = r["ref_sust_n"].notna() & ~r["ref_sust_n"].isin(r["cod_n"])
    bit.registrar("Referencias", "RF05", "Referencia sustituta que no existe en el maestro", "Se trata como descontinuada sin sustituta", m3)
    r.loc[m3, "ref_sust_n"] = pd.NA
    m4 = (m1 | m2) & (r["control"] == "OK")
    bit.registrar("Referencias", "RF06", "Control_Consistencia = OK pero la regla detecta inconsistencia", "Informativo: revisar el control", m4)
    en_cierre = set(nodos.index[(nodos["clase"] == "PLANTA") & nodos["fecha_cierre"].notna()])
    m5 = r["planta_antes"].isin(en_cierre) & sit.str.contains("SIN CAMBIO")
    bit.registrar("Referencias", "RF07", "Referencia de la planta que cierra marcada «Sin cambio»",
                  "Informativo: su volumen igual debe reasignarse; el modelo lo absorbe", m5)
    r["tras_valida"] = tras & ~m1
    r["descontinuada"] = desc
    return r.set_index("cod_n")


def resumir_info_cedi(raw, nodos, bit: Bitacora):
    if raw is None:
        return {}
    ic = renombrar(raw, P.COLUMNAS_INFO_CEDI, "Info_CEDI")
    if ic["id_nodo"].isna().all():
        bit.registrar("Info_CEDI", "IC01", "La hoja Info_CEDI no trae código de CEDI (ID_Nodo)",
                      "No se puede usar la capacidad de recibo; agregue la columna ID_Nodo", pd.Series([True]))
        return {}
    ic["id_nodo"] = norm_texto(ic["id_nodo"])
    for c in ["posiciones", "cap_recibo_estibas_dia", "util_max_operable", "dias_semana"]:
        ic[c] = ic[c].map(numero_flexible)
    m = ~ic["id_nodo"].isin(nodos.index)
    bit.registrar("Info_CEDI", "IC02", "Registro de Info_CEDI con ID_Nodo inexistente", "Excluido", m)
    ic = ic[~m]
    g = ic.groupby("id_nodo").agg(posiciones=("posiciones", "sum"), cap_recibo_estibas_dia=("cap_recibo_estibas_dia", "max"),
                                  util_max=("util_max_operable", "max"), dias_semana=("dias_semana", "max"))
    out = {}
    for k, v in g.iterrows():
        u = 100.0 if pd.isna(v.util_max) else float(v.util_max)
        out[k] = dict(posiciones=float(np.nan_to_num(v.posiciones)), cap_recibo_estibas_dia=float(np.nan_to_num(v.cap_recibo_estibas_dia)),
                      util_max=u / 100 if u > 1 else u)
    return out


# ---------------------------------------------------------------------------
# SAP
# ---------------------------------------------------------------------------
def limpiar_sap(raw, refs, homolog, nodos, clientes, bit: Bitacora):
    df = renombrar(raw, P.COLUMNAS_SAP, "SAP", ["viaje", "referencia", "centro_productor", "cedi", "periodo", "tipo", "cantidad", "importe", "volumen"])
    inicial = len(df)
    # --- Normalización ----------------------------------------------------
    df["importe_txt"] = df["importe"].astype("string")
    df["volumen_txt"] = df["volumen"].astype("string")
    for c in ["importe", "volumen", "cantidad"]:
        df[c] = df[c].map(numero_flexible)
    t = df["importe_txt"].fillna("")
    bit.registrar("SAP", "S01", "IMPORTE SAP o Volumen en texto (separadores de miles, notación científica)",
                  "Corregido: convertido a número", t.str.contains(r"\.\d{3}") | df["volumen_txt"].fillna("").str.contains("E", case=False), df)
    crudo_tipo = df["tipo"].astype("string")
    for c in ["tipo", "centro_productor", "cedi", "unidad_negocio", "corredor", "poblacion", "mercado", "cliente"]:
        df[c] = norm_texto(df[c])
    bit.registrar("SAP", "S02", "Tipo, centros o población con espacios, minúsculas o tildes (BOGOTÁ/BOGOTA)",
                  "Corregido: mayúsculas sin tildes", crudo_tipo.fillna("") != df["tipo"].fillna(""), df)
    df["periodo"] = pd.to_datetime(df["periodo"], errors="coerce")
    df["viaje_n"] = df["viaje"].map(norm_codigo)
    df["ref_n"] = df["referencia"].map(norm_codigo)

    # --- Alcance: solo traslados XI --------------------------------------
    m = df["tipo"] != P.TIPO_MOV_TRASLADO
    resumen_tipos = df.loc[m, "tipo"].fillna("(vacío)").value_counts().to_dict()
    bit.registrar("SAP", "S03", f"Movimientos distintos a {P.TIPO_MOV_TRASLADO} ({', '.join(f'{k}: {v}' for k, v in resumen_tipos.items())})",
                  "Excluido: fuera de alcance (no es traslado primario)", m, df)
    no_xi = df[m].copy()
    df = df[~m].copy()

    cols_dup = ["viaje", "cliente", "referencia", "centro_productor", "cedi", "periodo", "cantidad", "importe", "volumen"]
    m = df.duplicated(subset=cols_dup, keep="first")
    bit.registrar("SAP", "S04", "Registros duplicados (todas las columnas iguales)", "Excluido", m, df)
    df = df[~m]

    m = df["periodo"].isna() | (df["periodo"] < P.FECHA_MIN) | (df["periodo"] > P.FECHA_MAX)
    bit.registrar("SAP", "S05", "Periodo vacío o fuera de rango", "Excluido", m, df)
    df = df[~m]

    m = (df["cantidad"] <= 0) | (df["volumen"] < 0)
    bit.registrar("SAP", "S06", "Cantidad negativa o cero (reversión / anulación)", "Excluido", m, df)
    df = df[~m]

    df["origen"] = df["centro_productor"].map(homolog)
    df["destino"] = df["cedi"].map(homolog)
    validos = set(nodos.index)
    m = ~df["origen"].isin(validos) | ~df["destino"].isin(validos)
    sin_h = sorted(set(df.loc[~df["origen"].isin(validos), "centro_productor"].dropna()) | set(df.loc[~df["destino"].isin(validos), "cedi"].dropna()))
    bit.registrar("SAP", "S07", f"Centro productor o CEDI sin homologación en el maestro de nodos ({', '.join(sin_h[:6])})",
                  "Excluido: agregue el código a homologacion_centros.csv", m, df)
    df = df[~m]
    m = df["origen"] == df["destino"]
    bit.registrar("SAP", "S08", "Centro productor igual al CEDI destino", "Excluido", m, df)
    df = df[~m]
    m = nodos.loc[df["destino"], "clase"].values != "CEDI"
    bit.registrar("SAP", "S09", "Destino XI que no es CEDI en el maestro de nodos", "Excluido", pd.Series(m, index=df.index), df)
    df = df[~m]

    # --- Referencias -----------------------------------------------------
    df["en_maestro"] = df["ref_n"].isin(refs.index)
    m = ~df["en_maestro"]
    bit.registrar("SAP", "S10", "Referencia que no existe en el maestro de referencias", "Se conserva con la Unidad Negocio de SAP", m, df)
    fam_m = df["ref_n"].map(refs["familia"])
    m = df["unidad_negocio"].isna()
    bit.registrar("SAP", "S11", "Unidad Negocio vacía", "Corregido con la Familia del maestro de referencias", m & fam_m.notna(), df)
    df["familia"] = df["unidad_negocio"].fillna(fam_m)
    m = df["unidad_negocio"].notna() & fam_m.notna() & (df["unidad_negocio"] != fam_m)
    bit.registrar("SAP", "S12", "Unidad Negocio distinta a la Familia del maestro", "Se usa la familia del maestro", m, df)
    df.loc[m, "familia"] = fam_m[m]
    m = df["familia"].isna()
    bit.registrar("SAP", "S13", "Sin familia (ni en SAP ni en maestro)", "Excluido", m, df)
    df = df[~m]

    # --- Volumen ---------------------------------------------------------
    # m3 por caja: del maestro; si la referencia no está, mediana de su familia en el maestro
    vol_caja = df["ref_n"].map(refs["vol_caja_m3"]).fillna(df["familia"].map(refs.groupby("familia")["vol_caja_m3"].median()))
    vol_teo = df["cantidad"] * vol_caja
    m = (df["volumen"].isna() | (df["volumen"] == 0)) & vol_teo.notna()
    bit.registrar("SAP", "S14", "Volumen vacío o en cero", "Corregido: Cantidad × Volumen_Caja_m3 del maestro (o mediana de la familia)", m, df.assign(volumen=vol_teo))
    df.loc[m, "volumen"] = vol_teo[m]
    ratio = df["volumen"] / vol_teo
    lo, hi = P.TOLERANCIA_VOLUMEN
    m = vol_teo.notna() & ((ratio < lo) | (ratio > hi))
    bit.registrar("SAP", "S15", f"Volumen inconsistente con Cantidad × Volumen_Caja_m3 (fuera de {lo}–{hi}×; p. ej. en litros)",
                  "Corregido con el volumen teórico del maestro", m, df.assign(volumen=vol_teo))
    df.loc[m, "volumen"] = vol_teo[m]
    m = df["volumen"].isna() | (df["volumen"] <= 0)
    bit.registrar("SAP", "S16", "Volumen no recuperable", "Excluido", m, df)
    df = df[~m]

    # --- Consistencia con maestro de referencias ------------------------
    es_planta = nodos.loc[df["origen"], "clase"].values == "PLANTA"
    pa = df["ref_n"].map(refs["planta_antes"])
    m = pd.Series(es_planta, index=df.index) & pa.notna() & (pa != df["origen"])
    bit.registrar("SAP", "S17", "Centro productor distinto a Planta_Produccion_Antes del maestro", "Informativo (se usa SAP)", m, df)

    # --- Flete -----------------------------------------------------------
    df["valido_tarifa"] = True
    df["motivo_exclusion_tarifa"] = ""
    m = df["importe"].isna() | (df["importe"] <= P.IMPORTE_MINIMO_VALIDO)
    bit.registrar("SAP", "S18", "IMPORTE SAP en 0, 1 peso o vacío",
                  "Excluido del cálculo de tarifa (el volumen sí cuenta para demanda)", m, df)
    df.loc[m, ["valido_tarifa", "motivo_exclusion_tarifa"]] = [False, "importe<=1"]

    df["cop_m3"] = df["importe"] / df["volumen"]
    df["ruta"] = df["origen"] + "→" + df["destino"]
    df["log_cop"] = np.log(df["cop_m3"].where(df["cop_m3"] > 0))
    ok = df["valido_tarifa"]
    q = df[ok].groupby("ruta")["log_cop"].quantile([0.25, 0.75]).unstack()
    q.columns = ["q1", "q3"]
    lim = df[["ruta"]].join(q, on="ruta")
    iqr = lim["q3"] - lim["q1"]
    k = P.OUTLIER_IQR_FACTOR
    m = ok & ((df["log_cop"] < lim["q1"] - k * iqr) | (df["log_cop"] > lim["q3"] + k * iqr))
    bit.registrar("SAP", "S19", "COP/m3 atípico dentro de la ruta (dígito de más, viaje casi vacío)", "Excluido del cálculo de tarifa", m, df)
    df.loc[m, ["valido_tarifa", "motivo_exclusion_tarifa"]] = [False, "atipico"]

    m = df["viaje_n"].isna()
    bit.registrar("SAP", "S20", "Registro sin número de viaje", "Se conserva; no puede cruzarse con TMS", m, df)

    # --- Clientes (informativo sobre todos los movimientos) --------------
    info_clientes = None
    if clientes is not None:
        todo = pd.concat([df[["cliente", "poblacion"]], no_xi[["cliente", "poblacion"]]])
        partes = todo["cliente"].fillna("").str.rsplit(" - ", n=1, expand=True).reindex(columns=[0, 1])
        todo["nom"], todo["ciu"] = partes[0].str.strip(), partes[1].str.strip()
        claves = set(zip(clientes["nombre_n"], clientes["ciudad_n"]))
        dist = no_xi["tipo"] != P.TIPO_MOV_TRASLADO
        sub = todo.loc[no_xi.index[dist]] if len(no_xi) else todo.iloc[:0]
        match = [(a, b) in claves for a, b in zip(sub["nom"], sub["ciu"])]
        info_clientes = dict(lineas_distribucion=int(len(sub)), match_maestro=float(np.mean(match)) if len(sub) else None,
                             sin_formato=int((sub["ciu"].isna() | (sub["ciu"] == "")).sum()))
        bit.registrar("SAP", "S21", "Cliente de distribución (no XI) que no cruza con el maestro de clientes por nombre + ciudad",
                      "Informativo: el campo Cliente se separa en nombre y ciudad por « - »", pd.Series([not x for x in match]))

    resumen = dict(registros_iniciales=int(inicial), registros_xi_limpios=int(len(df)),
                   registros_validos_tarifa=int(df["valido_tarifa"].sum()), tipos_excluidos=resumen_tipos,
                   clientes=info_clientes)
    return df.reset_index(drop=True), resumen


# ---------------------------------------------------------------------------
# TMS y cruce
# ---------------------------------------------------------------------------
def limpiar_tms(raw):
    if raw is None:
        return None
    t = renombrar(raw, P.COLUMNAS_TMS, "TMS", ["viaje"])
    t["viaje_n"] = t["viaje"].map(norm_codigo)
    for c in ["km", "capacidad_m3", "capacidad_kg", "costo_tms"]:
        t[c] = t[c].map(numero_flexible)
    t["tipo_vehiculo"] = norm_texto(t["tipo_vehiculo"])
    t["km_invalido"] = t["km"].isna() | (t["km"] <= 0)
    t.loc[t["km_invalido"], "km"] = np.nan
    return t.dropna(subset=["viaje_n"]).drop_duplicates("viaje_n")


def cruzar_sap_tms(sap, tms, nodos, bit: Bitacora):
    """SAP manda; el TMS complementa km, vehículo y capacidad cuando el viaje cruza."""
    df = sap.copy()
    if tms is not None:
        cols = ["viaje_n", "tipo_vehiculo", "capacidad_m3", "km", "costo_tms", "km_invalido"]
        df = df.merge(tms[cols], on="viaje_n", how="left")
        df["match_tms"] = df["tipo_vehiculo"].notna() & df["viaje_n"].notna()
    else:
        for c in ["tipo_vehiculo", "capacidad_m3", "km", "costo_tms"]:
            df[c] = np.nan
        df["km_invalido"], df["match_tms"] = False, False
    df["km_teorico"] = [km_vial(nodos.at[o, "lat"], nodos.at[o, "lon"], nodos.at[d, "lat"], nodos.at[d, "lon"])
                        for o, d in zip(df["origen"], df["destino"])]
    m = df["match_tms"] & df["km"].notna() & ((df["km"] > 2.5 * df["km_teorico"]) | (df["km"] < 0.4 * df["km_teorico"]))
    bit.registrar("TMS", "T01", "Km del TMS incoherente con la distancia de la ruta", "Corregido: se usa km teórico", m, df)
    df.loc[m, "km"] = np.nan
    m = df["match_tms"] & df["km_invalido"].fillna(False).astype(bool)
    bit.registrar("TMS", "T02", "Km del TMS en cero o vacío", "Corregido: se usa km teórico", m, df)
    m = ~df["match_tms"] & df["viaje_n"].notna()
    bit.registrar("TMS", "T03", "Viaje SAP sin match en TMS", "Se conserva SAP; km teórico, sin dato de vehículo", m, df)

    vol_viaje = df.groupby("viaje_n")["volumen"].transform("sum")
    df["utilizacion"] = np.where(df["match_tms"], vol_viaje / df["capacidad_m3"], np.nan)
    df["km_usado"] = df["km"].fillna(df["km_teorico"])
    if P.COMPLETAR_IMPORTE_CON_TMS:
        m = ~df["valido_tarifa"] & (df["motivo_exclusion_tarifa"] == "importe<=1") & df["match_tms"] & df["costo_tms"].notna()
        df.loc[m, "importe"] = df.loc[m, "costo_tms"] * df.loc[m, "volumen"] / vol_viaje[m]
        df.loc[m, ["valido_tarifa", "motivo_exclusion_tarifa"]] = [True, "completado_tms"]
        bit.registrar("TMS", "T04", "Importe 0/1 completado con costo TMS prorrateado por volumen", "Corregido", m, df)

    v_sap = df["viaje_n"].dropna().unique()
    match = dict(viajes_sap=int(len(v_sap)), viajes_con_match=int(df.loc[df["match_tms"], "viaje_n"].nunique()),
                 lineas_sin_viaje=int(df["viaje_n"].isna().sum()),
                 viajes_tms_sin_sap=int((~tms["viaje_n"].isin(v_sap)).sum()) if tms is not None else 0,
                 hay_tms=tms is not None)
    match["tasa_match"] = match["viajes_con_match"] / max(1, match["viajes_sap"])
    return df, match
