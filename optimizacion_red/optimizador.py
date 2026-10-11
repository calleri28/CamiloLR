"""
Parámetros del modelo a partir de los datos limpios y modelo MILP (PuLP).

Unidad base: m3 por mes (el volumen es lo que llena el vehículo en productos
de consumo masivo voluminosos, y es el campo que trae SAP).

Modelo:
  Conjuntos: i plantas, j CEDIs, f familias (Unidad Negocio).
  Variables:
    x[i,j,f] >= 0   m3/mes de la familia f de la planta i al CEDI j
    he[i,f]  >= 0   m3/mes con horas extra / turno adicional
    q[j,f]   >= 0   m3/mes de maquila entregados en j
    u[j,f]   >= 0   m3/mes de demanda no atendida
    w[i,f]   ∈ {0,1} trasladar la línea f de la planta que cierra a la planta i
                    (en el escenario «plan» w es la fracción fija que indica el maestro de referencias)
  Min  Σ (producción + inventario en planta + flete efectivo + manejo CEDI + inventario CEDI)·x
       + Σ cHE·he + Σ (maquila entregada)·q + Σ pen·u + Σ costo_fijo[i] + Σ CAPEX[f]/meses · w[i,f]
       (+ desborde a bodega externa, calculado sobre la solución)
  s.a. Σ_i x[i,j,f] + q[j,f] + u[j,f] = D[j,f]                         demanda
       Σ_j x[i,j,f] <= cap[i,f]·abierta[i] + capT[f]·w[i,f] + he[i,f]  capacidad por línea
       he[i,f] <= %HE · (cap[i,f]·abierta[i] + capT[f]·w[i,f])
       Σ_i w[i,f] <= 1 ;  Σ_f w[i,f] <= máx. líneas a recibir[i] ;  Σ q <= cap. maquila
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pulp

import parametros as P
from calidad_datos import norm_texto, numero_flexible, renombrar
from datos_prueba import km_vial


# ---------------------------------------------------------------------------
# Estimaciones
# ---------------------------------------------------------------------------
def estimar_demanda(df: pd.DataFrame, nodos: pd.DataFrame, refs: pd.DataFrame):
    """Demanda neta por CEDI-familia = entradas XI - salidas XI a otros CEDI (m3/mes),
    y la parte que corresponde a referencias descontinuadas sin sustituta."""
    df = df.assign(mes=df["periodo"].dt.to_period("M"))
    n_meses = max(1, df["mes"].nunique())
    cedis = set(nodos.index[nodos["clase"] == "CEDI"])
    desc = df["ref_n"].map(refs["descontinuada"]).fillna(False).astype(bool) & df["ref_n"].map(refs["ref_sust_n"]).isna()
    ent = df[df["destino"].isin(cedis)].groupby(["destino", "familia"])["volumen"].sum()
    sal = df[df["origen"].isin(cedis)].groupby(["origen", "familia"])["volumen"].sum()
    sal.index.names = ent.index.names
    neto = ent.sub(sal, fill_value=0).clip(lower=0) / n_meses
    ent_desc = df[df["destino"].isin(cedis) & desc].groupby(["destino", "familia"])["volumen"].sum()
    frac_desc = (ent_desc / ent).reindex(ent.index).fillna(0)
    d = pd.DataFrame({"demanda_m3_mes": neto, "descontinuada_m3_mes": neto * frac_desc}).reset_index()
    return d.rename(columns={"destino": "cedi"}), n_meses


def estimar_tarifas(df: pd.DataFrame, plantas, cedis, nodos):
    """Tarifa de terceros COP/m3 por ruta (histórica si hay suficientes viajes, si no regresión por km),
    participación histórica de flota propia por ruta y costo histórico con flete imputado."""
    v = df[df["valido_tarifa"]]
    por_ruta = v.groupby(["origen", "destino"]).agg(
        importe=("importe", "sum"), m3=("volumen", "sum"), viajes=("viaje_n", "nunique"),
        km=("km_usado", "median"), utilizacion=("utilizacion", "mean"), match_tms=("match_tms", "mean")).reset_index()
    por_ruta["cop_m3"] = por_ruta["importe"] / por_ruta["m3"]
    todo = df.groupby(["origen", "destino"]).agg(m3_total=("volumen", "sum"), imp_total=("importe", "sum"),
                                                 m3_propia=("volumen", lambda x: x[df.loc[x.index, "es_propia"]].sum()))
    por_ruta = por_ruta.join(todo, on=["origen", "destino"])
    base = por_ruta[por_ruta["viajes"] >= P.MIN_VIAJES_TARIFA]
    if len(base) < 3:
        raise ValueError("Hay menos de 3 rutas con historia suficiente para estimar tarifas. Revise IMPORTE SAP o baje MIN_VIAJES_TARIFA.")
    w = np.sqrt(base["m3"])
    b, a = np.polyfit(base["km"], base["cop_m3"], 1, w=w)
    pred = a + b * base["km"]
    media = np.average(base["cop_m3"], weights=w)
    r2 = 1 - ((base["cop_m3"] - pred) ** 2 * w).sum() / (((base["cop_m3"] - media) ** 2) * w).sum()
    en_plantas = todo.reset_index()
    en_plantas = en_plantas[en_plantas["origen"].isin(plantas)]
    share_global = float(en_plantas["m3_propia"].sum() / en_plantas["m3_total"].sum()) if len(en_plantas) else 0.0

    hist = por_ruta.set_index(["origen", "destino"])
    filas = []
    for i in plantas:
        for j in cedis:
            km_t = km_vial(nodos.at[i, "lat"], nodos.at[i, "lon"], nodos.at[j, "lat"], nodos.at[j, "lon"])
            h = hist.loc[(i, j)] if (i, j) in hist.index else None
            tiene = h is not None and h["viajes"] >= P.MIN_VIAJES_TARIFA
            share = float(h["m3_propia"] / h["m3_total"]) if h is not None and h["m3_total"] > 0 else share_global
            filas.append(dict(
                origen=i, destino=j, km=float(h["km"]) if tiene else km_t,
                tarifa_cop_m3=float(h["cop_m3"]) if tiene else float(max(a + b * km_t, a)),
                fuente="Histórico SAP" if tiene else "Estimada (regresión km)", historia=bool(tiene),
                viajes=0 if h is None else int(h["viajes"]), share_propia=share,
                costo_hist_con_imputado=float(h["imp_total"] / h["m3_total"]) if h is not None and h["m3_total"] > 0 else None,
                utilizacion=None if h is None or pd.isna(h["utilizacion"]) else float(h["utilizacion"])))
    reg = dict(intercepto=float(a), pendiente=float(b), r2=float(r2), n_rutas=int(len(base)), share_propia_global=share_global)
    return pd.DataFrame(filas), por_ruta, reg


def plan_desde_referencias(df, refs, cierre):
    """Reparto del traslado por familia según el maestro de referencias, ponderado por el volumen
    histórico que despachaba la planta que cierra. Devuelve {familia: {planta: fracción}} y el detalle."""
    r = refs[(refs["planta_antes"] == cierre)].copy()
    vol = df[df["origen"] == cierre].groupby("ref_n")["volumen"].sum()
    r["vol_hist_m3"] = r.index.map(vol).fillna(0)
    r["peso"] = np.where(r["vol_hist_m3"] > 0, r["vol_hist_m3"], 1e-6)
    t = r[r["tras_valida"]]
    plan = {}
    for f, g in t.groupby("familia"):
        s = g.groupby("planta_nueva")["peso"].sum()
        plan[f] = {k: float(v / s.sum()) for k, v in s.items()}
    det = r.groupby(["familia", r["situacion"].fillna("SIN DATO"), r["planta_nueva"].fillna("—")]).agg(
        referencias=("familia", "size"), vol_hist_m3=("vol_hist_m3", "sum")).reset_index()
    return plan, det




def _tabla_param(entradas, clave, mapeo, nombre, obligatorias=()):
    raw = entradas.get(clave)
    if raw is None:
        return pd.DataFrame(columns=list(mapeo)).set_index("id_nodo" if "id_nodo" in mapeo else "familia")
    d = renombrar(raw, mapeo, nombre, obligatorias)
    llave = "id_nodo" if "id_nodo" in mapeo else "familia"
    d[llave] = norm_texto(d[llave])
    for c in mapeo:
        if c != llave:
            d[c] = d[c].map(numero_flexible)
    return d.set_index(llave)


def construir_insumos(entradas, nodos, refs, demanda, tarifas, plan, info_cedi, n_meses, info_flete, reg):
    """Paquete único que usan el MILP (Python) y el simulador (HTML)."""
    cap = renombrar(entradas["capacidad"], P.COLUMNAS_CAPACIDAD, "Capacidad", ["id_nodo", "familia", "capacidad_m3_mes"])
    cap["id_nodo"], cap["familia"] = norm_texto(cap["id_nodo"]), norm_texto(cap["familia"])
    cap["capacidad_m3_mes"] = cap["capacidad_m3_mes"].map(numero_flexible)
    pp = _tabla_param(entradas, "param_plantas", P.COLUMNAS_PARAM_PLANTAS, "Parámetros plantas", ["id_nodo", "costo_fijo_mes"])
    pf = _tabla_param(entradas, "param_familias", P.COLUMNAS_PARAM_FAMILIAS, "Parámetros familias", ["familia", "costo_var_m3"])
    pc = _tabla_param(entradas, "param_cedis", P.COLUMNAS_PARAM_CEDIS, "Parámetros CEDI")

    plantas = [i for i in nodos.index[nodos["clase"] == "PLANTA"] if i in pp.index]
    cedis = sorted(demanda["cedi"].unique())
    familias = sorted(demanda["familia"].unique())
    faltan = [f for f in familias if f not in pf.index]
    if faltan:
        raise ValueError(f"Familias sin parámetros en {P.ARCHIVOS['param_familias']}: {faltan}")

    def val(df, i, c, defecto):
        v = df.at[i, c] if (i in df.index and c in df.columns) else np.nan
        return float(defecto if pd.isna(v) else v)

    cierre = P.PLANTA_CIERRE or next((i for i in plantas if pd.notna(nodos.at[i, "fecha_cierre"])
                                      or "CIERRE" in str(nodos.at[i, "estado_n"]) or "CERRAD" in str(nodos.at[i, "estado_n"])), None)
    capd = {f"{r.id_nodo}|{r.familia}": float(r.capacidad_m3_mes) for r in cap.itertuples() if r.id_nodo in plantas and r.familia in familias}
    cv = {f"{i}|{f}": val(pf, f, "costo_var_m3", 0) * val(pp, i, "factor_costo_var", 1.0) for i in plantas for f in familias}
    pct_t = {f: val(pf, f, "pct_trasladable", 1.0) for f in familias}
    pct_t = {f: v / 100 if v > 1 else v for f, v in pct_t.items()}
    cap_tras = {f: capd.get(f"{cierre}|{f}", 0.0) * pct_t[f] for f in familias} if cierre else {f: 0.0 for f in familias}
    m3_estiba = refs.groupby("familia")["m3_estiba"].median().reindex(familias).fillna(1.7).to_dict()
    clave = lambda r: f"{r.origen}|{r.destino}"
    tar = {clave(r): round(float(r.tarifa_cop_m3), 1) for r in tarifas.itertuples()}
    nodo = lambda i: dict(nombre=str(nodos.at[i, "nombre"]), ciudad=str(nodos.at[i, "ciudad"]),
                          lat=float(nodos.at[i, "lat"]), lon=float(nodos.at[i, "lon"]))
    cedi_d = {}
    for j in cedis:
        rc = info_cedi.get(j) or {}
        cedi_d[j] = dict(**nodo(j), recibo=info_cedi.get(j),
                         posiciones=float(rc.get("posiciones") or 0) or None, util_max=float(rc.get("util_max") or 1.0),
                         manejo=val(pc, j, "costo_manejo", P.COSTO_MANEJO_COP_ESTIBA),
                         cpos=val(pc, j, "costo_posicion", P.COSTO_POSICION_CEDI_COP_MES),
                         cext=val(pc, j, "costo_externo", P.COSTO_POSICION_EXTERNA_COP_MES),
                         dias_base=val(pc, j, "dias_base", P.DIAS_COBERTURA_BASE))
    return dict(
        plantas={i: dict(**nodo(i), costo_fijo=val(pp, i, "costo_fijo_mes", 0), max_lineas=int(val(pp, i, "max_lineas", 2)),
                         cpos=val(pp, i, "costo_posicion", P.COSTO_POSICION_PLANTA_COP_MES),
                         dias_inv=val(pp, i, "dias_inventario", P.DIAS_INVENTARIO_PLANTA)) for i in plantas},
        cedis=cedi_d, familias=familias, planta_cierre=cierre, unidad="m³",
        cap=capd, cv=cv, cv_base={f: val(pf, f, "costo_var_m3", 0) for f in familias},
        demanda={f"{r.cedi}|{r.familia}": round(float(r.demanda_m3_mes), 2) for r in demanda.itertuples()},
        descontinuada={f"{r.cedi}|{r.familia}": round(float(r.descontinuada_m3_mes), 2) for r in demanda.itertuples() if r.descontinuada_m3_mes > 0},
        tarifa=tar, km={clave(r): round(float(r.km), 1) for r in tarifas.itertuples()},
        historia={clave(r): bool(r.historia) for r in tarifas.itertuples()},
        share_propia={clave(r): round(float(r.share_propia), 4) for r in tarifas.itertuples()},
        share_propia_global=reg["share_propia_global"],
        cop_km_flota=info_flete["cop_km"], cop_km_derivado=info_flete["cop_km_derivado"],
        vol_viaje_propio=info_flete["vol_viaje_propio"],
        cap_traslado=cap_tras, capex={f: val(pf, f, "capex_traslado", 0) for f in familias},
        plan=plan, m3_estiba=m3_estiba, dias_mes=P.DIAS_OPERACION_MES, meses_historia=int(n_meses),
        flete_medio_cedi={j: float(np.mean([tar[f"{i}|{j}"] for i in plantas])) for j in cedis},
    )


def cfg_base(ins):
    """Supuestos por defecto. El HTML usa exactamente las mismas claves."""
    I, J, F = list(ins["plantas"]), list(ins["cedis"]), ins["familias"]
    return dict(
        abiertas={i: True for i in I}, cap_pct={i: 1.0 for i in I}, cv_pct={i: 1.0 for i in I}, fijo_pct={i: 1.0 for i in I},
        residual_cierre=0.0, traslados={}, aplicar_desc=False,
        g=0.0, g_fam={f: 0.0 for f in F}, g_cedi={j: 0.0 for j in J},
        ff=1.0, fp=1.0, he_pct=P.HORAS_EXTRA_MAX_PCT, he_sobre=P.SOBRECOSTO_HORAS_EXTRA_PCT,
        maq_cap=P.MAQUILA_MAX_M3_MES, maq_pct=P.COSTO_MAQUILA_PCT, penalidad=P.PENALIDAD_NO_ATENDIDA_COP_M3,
        meses=P.MESES_AMORTIZACION, capex_idx=1.0, costo_cierre_unico=0.0,
        metodo="km" if P.METODO_IMPUTACION_FLETE == "km" else "mixto", cop_km=ins["cop_km_flota"],
        fr=P.FACTOR_RETORNO_FLOTA_PROPIA, share_mode="hist", share=ins["share_propia_global"], prima_nuevas=0.0,
        manejo_idx=1.0, pos_idx=1.0, ext_idx=1.0, dias_delta=0.0, dias_1000=P.DIAS_COBERTURA_POR_1000KM,
        pos_cap_pct=1.0, dias_planta_delta=0.0,
    )


# ---------------------------------------------------------------------------
# Costos unitarios (idénticos en el simulador JS)
# ---------------------------------------------------------------------------
def tarifa_efectiva(ins, cfg, i, j):
    k = f"{i}|{j}"
    hist = ins["historia"][k]
    terc = ins["tarifa"][k] * (1 + (0 if hist else cfg["prima_nuevas"]))
    own_km = cfg["cop_km"] * ins["km"][k] * cfg["fr"] / ins["vol_viaje_propio"]
    own = own_km if (cfg["metodo"] == "km" or not hist) else terc
    s = ins["share_propia"][k] if cfg["share_mode"] == "hist" else cfg["share"]
    return ((1 - s) * terc + s * own) * cfg["ff"]


def dias_cobertura(ins, cfg, j, km):
    return max(0.0, ins["cedis"][j]["dias_base"] + cfg["dias_delta"] + cfg["dias_1000"] * km / 1000)


def unitarios(ins, cfg):
    I, J, F = list(ins["plantas"]), list(ins["cedis"]), ins["familias"]
    m3e = ins["m3_estiba"]
    u = dict(cvv={}, alm_planta={}, flete={}, manejo={}, inv={}, dias={}, maq_prod={}, maq_flete={}, maq_dias={})
    for i in I:
        p = ins["plantas"][i]
        for f in F:
            u["cvv"][i, f] = ins["cv"][f"{i}|{f}"] * cfg["fp"] * cfg["cv_pct"][i]
            u["alm_planta"][i, f] = max(0.0, p["dias_inv"] + cfg["dias_planta_delta"]) / 30 * p["cpos"] * cfg["pos_idx"] / m3e[f]
        for j in J:
            u["flete"][i, j] = tarifa_efectiva(ins, cfg, i, j)
            u["dias"][i, j] = dias_cobertura(ins, cfg, j, ins["km"][f"{i}|{j}"])
    for j in J:
        c = ins["cedis"][j]
        u["maq_dias"][j] = dias_cobertura(ins, cfg, j, 0)
        u["maq_flete"][j] = ins["flete_medio_cedi"][j] * cfg["ff"]
        for f in F:
            u["manejo"][j, f] = c["manejo"] * cfg["manejo_idx"] / m3e[f]
            u["maq_prod"][f] = ins["cv_base"][f] * (1 + cfg["maq_pct"]) * cfg["fp"]
            for i in I:
                u["inv"][i, j, f] = u["dias"][i, j] / 30 * c["cpos"] * cfg["pos_idx"] / m3e[f]
            u["inv"]["MQ", j, f] = u["maq_dias"][j] / 30 * c["cpos"] * cfg["pos_idx"] / m3e[f]
    return u


def demanda_cfg(ins, cfg):
    D = {}
    for j in ins["cedis"]:
        for f in ins["familias"]:
            k = f"{j}|{f}"
            d = ins["demanda"].get(k, 0.0) - (ins["descontinuada"].get(k, 0.0) if cfg["aplicar_desc"] else 0.0)
            D[j, f] = max(0.0, d) * (1 + cfg["g"]) * (1 + cfg["g_fam"][f]) * (1 + cfg["g_cedi"][j])
    return D


# ---------------------------------------------------------------------------
# MILP
# ---------------------------------------------------------------------------
def _solver():
    disp = pulp.listSolvers(onlyAvailable=True)
    if "PULP_CBC_CMD" in disp:
        return pulp.PULP_CBC_CMD(msg=False, gapRel=1e-7)
    if "HiGHS" in disp:
        return pulp.HiGHS(msg=False, gapRel=1e-7)
    if "HiGHS_CMD" in disp:
        return pulp.HiGHS_CMD(msg=False)
    raise RuntimeError("No hay solver disponible. Ejecute: pip install -r requirements.txt")


def escenario(eid, nombre, corto, descripcion, ins, cerrar=True, traslado="optimizar", aplicar_desc=True,
              sensibilidad=False, **cambios):
    """traslado: 'ninguno' | 'optimizar' | 'plan' (maestro de referencias) | {familia: {planta: fracción}}"""
    cfg = cfg_base(ins)
    if cerrar and ins["planta_cierre"]:
        cfg["abiertas"][ins["planta_cierre"]] = False
    cfg["aplicar_desc"] = aplicar_desc and cerrar
    cfg.update(cambios)
    return dict(id=eid, nombre=nombre, nombre_corto=corto, descripcion=descripcion, sensibilidad=sensibilidad,
                traslado=traslado, cfg=cfg)


GRUPOS = {"Producción": ["produccion", "horas_extra", "fijo", "capex"],
          "Transporte": ["flete"],
          "Almacenamiento": ["manejo", "inventario_cedi", "desborde", "inventario_planta"],
          "Maquila y no atendida": ["maquila", "penalidad"]}


def resolver(ins: dict, esc: dict) -> dict:
    cfg = esc["cfg"]
    I, J, F = list(ins["plantas"]), list(ins["cedis"]), ins["familias"]
    cierre = ins["planta_cierre"]
    ab = cfg["abiertas"]
    D = demanda_cfg(ins, cfg)
    U = unitarios(ins, cfg)
    cap = {(i, f): ins["cap"].get(f"{i}|{f}", 0.0) * cfg["cap_pct"][i] * ab[i] for i in I for f in F}
    capT = ins["cap_traslado"]

    mdl = pulp.LpProblem(esc["id"], pulp.LpMinimize)
    tr = esc["traslado"]
    if tr == "plan":
        tr = ins["plan"]
    w = {}
    hay_cierre = cierre is not None and not ab[cierre]
    if hay_cierre and tr != "ninguno":
        for f in F:
            if capT.get(f, 0) <= 0:
                continue
            for i in I:
                if i == cierre or not ab[i]:
                    continue
                if tr == "optimizar":
                    w[i, f] = pulp.LpVariable(f"w_{i}_{f}".replace(" ", "_"), cat="Binary")
                elif tr.get(f, {}).get(i, 0) > 0:
                    w[i, f] = float(tr[f][i])
    cap_tot = {(i, f): cap[i, f] + capT.get(f, 0) * w.get((i, f), 0) for i in I for f in F}
    activos = [(i, f) for i in I for f in F if cap[i, f] > 0 or (i, f) in w]

    x = {(i, j, f): pulp.LpVariable(f"x_{i}_{j}_{f}".replace(" ", "_"), lowBound=0) for (i, f) in activos for j in J}
    he = {(i, f): pulp.LpVariable(f"he_{i}_{f}".replace(" ", "_"), lowBound=0) for (i, f) in activos}
    q = {(j, f): pulp.LpVariable(f"q_{j}_{f}".replace(" ", "_"), lowBound=0) for j in J for f in F}
    un = {(j, f): pulp.LpVariable(f"u_{j}_{f}".replace(" ", "_"), lowBound=0) for j in J for f in F}

    S = pulp.lpSum
    e = dict(
        produccion=S(U["cvv"][i, f] * x[i, j, f] for (i, j, f) in x),
        horas_extra=S(U["cvv"][i, f] * cfg["he_sobre"] * he[i, f] for (i, f) in he),
        flete=S(U["flete"][i, j] * x[i, j, f] for (i, j, f) in x) + S(U["maq_flete"][j] * q[j, f] for (j, f) in q),
        manejo=S(U["manejo"][j, f] * x[i, j, f] for (i, j, f) in x) + S(U["manejo"][j, f] * q[j, f] for (j, f) in q),
        inventario_cedi=S(U["inv"][i, j, f] * x[i, j, f] for (i, j, f) in x) + S(U["inv"]["MQ", j, f] * q[j, f] for (j, f) in q),
        inventario_planta=S(U["alm_planta"][i, f] * x[i, j, f] for (i, j, f) in x),
        maquila=S(U["maq_prod"][f] * q[j, f] for (j, f) in q),
        penalidad=S(cfg["penalidad"] * un[j, f] for (j, f) in un),
        capex=S(ins["capex"][f] * cfg["capex_idx"] / cfg["meses"] * w[i, f] for (i, f) in w),
    )
    fijo = sum(ins["plantas"][i]["costo_fijo"] * cfg["fijo_pct"][i] for i in I if ab[i])
    if hay_cierre:
        fijo += ins["plantas"][cierre]["costo_fijo"] * cfg["residual_cierre"]
    mdl += S(e.values()) + fijo

    for j in J:
        for f in F:
            mdl += S(x[i, j, f] for i in I if (i, f) in he) + q[j, f] + un[j, f] == D[j, f]
    for (i, f) in activos:
        mdl += S(x[i, j, f] for j in J) <= cap_tot[i, f] + he[i, f]
        mdl += he[i, f] <= cfg["he_pct"] * cap_tot[i, f]
    if tr == "optimizar":
        for f in F:
            vs = [w[i, f] for i in I if (i, f) in w]
            if vs:
                mdl += S(vs) <= 1
        for i in I:
            vs = [w[i, f] for f in F if (i, f) in w]
            if vs:
                mdl += S(vs) <= ins["plantas"][i]["max_lineas"]
    mdl += S(q.values()) <= cfg["maq_cap"]

    estado = mdl.solve(_solver())
    v = lambda z: float(pulp.value(z) or 0.0)
    costos = {k: v(z) for k, z in e.items()}
    costos["fijo"] = fijo
    wv = {k: (z.value() if hasattr(z, "value") else z) or 0.0 for k, z in w.items()}
    traslados = {}
    for (i, f), z in wv.items():
        if z > 1e-6:
            traslados.setdefault(f, {})[i] = float(z)
    xv = {k: z.value() or 0.0 for k, z in x.items()}
    qv = {k: z.value() or 0.0 for k, z in q.items()}

    # Desborde de posiciones en CEDI (sobre la solución)
    det_cedi, desborde = {}, 0.0
    m3e = ins["m3_estiba"]
    for j in J:
        c = ins["cedis"][j]
        inv = sum(xv[i, jj, f] * U["dias"][i, j] / 30 / m3e[f] for (i, jj, f) in xv if jj == j) + \
            sum(qv[j, f] * U["maq_dias"][j] / 30 / m3e[f] for f in F)
        capj = c["posiciones"] * c["util_max"] * cfg["pos_cap_pct"] if c["posiciones"] else None
        exc = max(0.0, inv - capj) if capj is not None else 0.0
        costo = exc * max(0.0, c["cext"] * cfg["ext_idx"] - c["cpos"] * cfg["pos_idx"])
        desborde += costo
        det_cedi[j] = dict(inventario_estibas=inv, capacidad_estibas=capj, exceso_estibas=exc, costo_desborde=costo)
    costos["desborde"] = desborde
    costos["total"] = sum(costos.values())

    flujos = [dict(origen=i, destino=j, familia=f, m3=val, flete=U["flete"][i, j], km=ins["km"][f"{i}|{j}"])
              for (i, j, f), val in xv.items() if val > 1e-6]
    cap_val = {(i, f): cap[i, f] + capT.get(f, 0) * wv.get((i, f), 0) for i in I for f in F}
    uso = {}
    for i in I:
        c_i = sum(cap_val[i, f] for f in F)
        prod = sum(fl["m3"] for fl in flujos if fl["origen"] == i)
        uso[i] = dict(capacidad=c_i, produccion=prod, horas_extra=sum(he[i, f].value() or 0 for f in F if (i, f) in he),
                      utilizacion=prod / c_i if c_i else 0.0)
    vol = sum(fl["m3"] for fl in flujos)
    dem = sum(D.values())
    return dict(estado=pulp.LpStatus[estado], costos=costos, flujos=flujos, traslados=traslados, uso=uso, cedis=det_cedi,
                maquila_m3=sum(qv.values()), no_atendida_m3=sum(z.value() or 0 for z in un.values()),
                demanda_m3=dem, km_promedio=sum(fl["m3"] * fl["km"] for fl in flujos) / vol if vol else 0.0,
                costo_por_m3=costos["total"] / dem if dem else 0.0)
