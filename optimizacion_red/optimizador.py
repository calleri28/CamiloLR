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
  Min  Σ (cv[i,f]·fp + tarifa[i,j]·ff)·x + Σ cHE·he + Σ cM·q + Σ pen·u
       + Σ costo_fijo[i] (plantas abiertas) + Σ CAPEX[f]/meses · w[i,f]
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
    """COP/m3 por ruta: histórico si hay suficientes viajes válidos; si no, regresión por km."""
    v = df[df["valido_tarifa"]]
    por_ruta = v.groupby(["origen", "destino"]).agg(
        importe=("importe", "sum"), m3=("volumen", "sum"), viajes=("viaje_n", "nunique"),
        km=("km_usado", "median"), utilizacion=("utilizacion", "mean"), match_tms=("match_tms", "mean")).reset_index()
    por_ruta["cop_m3"] = por_ruta["importe"] / por_ruta["m3"]
    base = por_ruta[por_ruta["viajes"] >= P.MIN_VIAJES_TARIFA]
    if len(base) < 3:
        raise ValueError("Hay menos de 3 rutas con historia suficiente para estimar tarifas. Revise IMPORTE SAP o baje MIN_VIAJES_TARIFA.")
    w = np.sqrt(base["m3"])
    b, a = np.polyfit(base["km"], base["cop_m3"], 1, w=w)
    pred = a + b * base["km"]
    media = np.average(base["cop_m3"], weights=w)
    r2 = 1 - ((base["cop_m3"] - pred) ** 2 * w).sum() / (((base["cop_m3"] - media) ** 2) * w).sum()

    hist = por_ruta.set_index(["origen", "destino"])
    filas = []
    for i in plantas:
        for j in cedis:
            km_t = km_vial(nodos.at[i, "lat"], nodos.at[i, "lon"], nodos.at[j, "lat"], nodos.at[j, "lon"])
            h = hist.loc[(i, j)] if (i, j) in hist.index else None
            if h is not None and h["viajes"] >= P.MIN_VIAJES_TARIFA:
                filas.append(dict(origen=i, destino=j, km=float(h["km"]), tarifa_cop_m3=float(h["cop_m3"]), fuente="Histórico SAP",
                                  viajes=int(h["viajes"]), utilizacion=None if pd.isna(h["utilizacion"]) else float(h["utilizacion"])))
            else:
                filas.append(dict(origen=i, destino=j, km=km_t, tarifa_cop_m3=float(max(a + b * km_t, a)), fuente="Estimada (regresión km)",
                                  viajes=0 if h is None else int(h["viajes"]), utilizacion=None))
    return pd.DataFrame(filas), por_ruta, dict(intercepto=float(a), pendiente=float(b), r2=float(r2), n_rutas=int(len(base)))


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


def construir_insumos(entradas, nodos, refs, demanda, tarifas, plan, info_cedi, n_meses):
    """Paquete único que usan el MILP (Python) y el simulador (HTML)."""
    cap = renombrar(entradas["capacidad"], P.COLUMNAS_CAPACIDAD, "Capacidad", ["id_nodo", "familia", "capacidad_m3_mes"])
    pp = renombrar(entradas["param_plantas"], P.COLUMNAS_PARAM_PLANTAS, "Parámetros plantas", ["id_nodo", "costo_fijo_mes"])
    pf = renombrar(entradas["param_familias"], P.COLUMNAS_PARAM_FAMILIAS, "Parámetros familias", ["familia", "costo_var_m3"])
    for d, cols in ((cap, ["id_nodo", "familia"]), (pp, ["id_nodo"]), (pf, ["familia"])):
        for c in cols:
            d[c] = norm_texto(d[c])
    cap["capacidad_m3_mes"] = cap["capacidad_m3_mes"].map(numero_flexible)
    pp = pp.set_index("id_nodo")
    pf = pf.set_index("familia")
    for c in ["costo_fijo_mes", "factor_costo_var", "max_lineas"]:
        pp[c] = pp[c].map(numero_flexible)
    for c in ["costo_var_m3", "capex_traslado", "pct_trasladable"]:
        pf[c] = pf[c].map(numero_flexible)

    plantas = [i for i in nodos.index[nodos["clase"] == "PLANTA"] if i in pp.index]
    cedis = sorted(demanda["cedi"].unique())
    familias = sorted(demanda["familia"].unique())
    faltan = [f for f in familias if f not in pf.index]
    if faltan:
        raise ValueError(f"Familias sin parámetros en {P.ARCHIVOS['param_familias']}: {faltan}")

    cierre = P.PLANTA_CIERRE or next((i for i in plantas if pd.notna(nodos.at[i, "fecha_cierre"])
                                      or "CIERRE" in str(nodos.at[i, "estado_n"]) or "CERRAD" in str(nodos.at[i, "estado_n"])), None)
    capd = {f"{r.id_nodo}|{r.familia}": float(r.capacidad_m3_mes) for r in cap.itertuples() if r.id_nodo in plantas and r.familia in familias}
    cv = {f"{i}|{f}": float(pf.at[f, "costo_var_m3"] * (pp.at[i, "factor_costo_var"] if pd.notna(pp.at[i, "factor_costo_var"]) else 1.0))
          for i in plantas for f in familias}
    pct_t = {f: float(pf.at[f, "pct_trasladable"]) if pd.notna(pf.at[f, "pct_trasladable"]) else 1.0 for f in familias}
    pct_t = {f: v / 100 if v > 1 else v for f, v in pct_t.items()}
    cap_tras = {f: capd.get(f"{cierre}|{f}", 0.0) * pct_t[f] for f in familias} if cierre else {f: 0.0 for f in familias}
    m3_estiba = refs.groupby("familia")["m3_estiba"].median().reindex(familias).fillna(1.7).to_dict()
    tar = {f"{r.origen}|{r.destino}": round(float(r.tarifa_cop_m3), 1) for r in tarifas.itertuples()}
    nodo = lambda i: dict(nombre=str(nodos.at[i, "nombre"]), ciudad=str(nodos.at[i, "ciudad"]),
                          lat=float(nodos.at[i, "lat"]), lon=float(nodos.at[i, "lon"]))
    return dict(
        plantas={i: dict(**nodo(i), costo_fijo=float(pp.at[i, "costo_fijo_mes"]),
                         max_lineas=int(pp.at[i, "max_lineas"]) if pd.notna(pp.at[i, "max_lineas"]) else 2) for i in plantas},
        cedis={j: dict(**nodo(j), recibo=info_cedi.get(j)) for j in cedis},
        familias=familias, planta_cierre=cierre, unidad="m³",
        cap=capd, cv=cv, cv_base={f: float(pf.at[f, "costo_var_m3"]) for f in familias},
        demanda={f"{r.cedi}|{r.familia}": round(float(r.demanda_m3_mes), 2) for r in demanda.itertuples()},
        descontinuada={f"{r.cedi}|{r.familia}": round(float(r.descontinuada_m3_mes), 2) for r in demanda.itertuples() if r.descontinuada_m3_mes > 0},
        tarifa=tar, km={f"{r.origen}|{r.destino}": round(float(r.km), 1) for r in tarifas.itertuples()},
        cap_traslado=cap_tras, capex={f: float(np.nan_to_num(pf.at[f, "capex_traslado"])) for f in familias},
        plan=plan, meses_amortizacion=P.MESES_AMORTIZACION,
        he_max_pct=P.HORAS_EXTRA_MAX_PCT, he_sobrecosto=P.SOBRECOSTO_HORAS_EXTRA_PCT,
        maquila_pct=P.COSTO_MAQUILA_PCT, maquila_cap=P.MAQUILA_MAX_M3_MES,
        flete_medio_cedi={j: float(np.mean([tar[f"{i}|{j}"] for i in plantas])) for j in cedis},
        penalidad=P.PENALIDAD_NO_ATENDIDA_COP_M3, m3_estiba=m3_estiba, dias_mes=P.DIAS_OPERACION_MES,
        meses_historia=int(n_meses),
    )


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
              crec_demanda=0.0, indice_flete=1.0, indice_produccion=1.0, sensibilidad=False):
    """traslado: 'ninguno' | 'optimizar' | 'plan' (maestro de referencias) | {familia: {planta: fracción}}"""
    abiertas = {i: True for i in ins["plantas"]}
    if cerrar and ins["planta_cierre"]:
        abiertas[ins["planta_cierre"]] = False
    return dict(id=eid, nombre=nombre, nombre_corto=corto, descripcion=descripcion, sensibilidad=sensibilidad,
                abiertas=abiertas, cap_pct={i: 1.0 for i in ins["plantas"]}, traslado=traslado,
                aplicar_desc=aplicar_desc and cerrar, crec_demanda=crec_demanda,
                indice_flete=indice_flete, indice_produccion=indice_produccion)


def costo_he(ins, i, f, fp):
    return ins["he_sobrecosto"] * ins["cv"][f"{i}|{f}"] * fp


def costo_maquila(ins, j, f, fp, ff):
    return ins["cv_base"][f] * (1 + ins["maquila_pct"]) * fp + ins["flete_medio_cedi"][j] * ff


def resolver(ins: dict, esc: dict) -> dict:
    I, J, F = list(ins["plantas"]), list(ins["cedis"]), ins["familias"]
    cierre = ins["planta_cierre"]
    fp, ff, g = esc["indice_produccion"], esc["indice_flete"], esc["crec_demanda"]
    ab = esc["abiertas"]
    D = {(j, f): max(0.0, ins["demanda"].get(f"{j}|{f}", 0.0) - (ins["descontinuada"].get(f"{j}|{f}", 0.0) if esc["aplicar_desc"] else 0.0)) * (1 + g)
         for j in J for f in F}
    cap = {(i, f): ins["cap"].get(f"{i}|{f}", 0.0) * esc["cap_pct"][i] * ab[i] for i in I for f in F}
    capT = ins["cap_traslado"]

    mdl = pulp.LpProblem(esc["id"], pulp.LpMinimize)
    # Traslados
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
    u = {(j, f): pulp.LpVariable(f"u_{j}_{f}".replace(" ", "_"), lowBound=0) for j in J for f in F}

    c_fle = {(i, j): ins["tarifa"][f"{i}|{j}"] * ff for i in I for j in J}
    e_prod = pulp.lpSum(ins["cv"][f"{i}|{f}"] * fp * x[i, j, f] for (i, j, f) in x)
    e_flete = pulp.lpSum(c_fle[i, j] * x[i, j, f] for (i, j, f) in x)
    e_he = pulp.lpSum(costo_he(ins, i, f, fp) * he[i, f] for (i, f) in he)
    e_maq = pulp.lpSum(costo_maquila(ins, j, f, fp, ff) * q[j, f] for (j, f) in q)
    e_pen = pulp.lpSum(ins["penalidad"] * u[j, f] for (j, f) in u)
    fijo = sum(ins["plantas"][i]["costo_fijo"] for i in I if ab[i])
    e_capex = pulp.lpSum(ins["capex"][f] / ins["meses_amortizacion"] * w[i, f] for (i, f) in w)
    mdl += e_prod + e_flete + e_he + e_maq + e_pen + e_capex + fijo

    for j in J:
        for f in F:
            mdl += pulp.lpSum(x[i, j, f] for i in I if (i, f) in he) + q[j, f] + u[j, f] == D[j, f]
    for (i, f) in activos:
        mdl += pulp.lpSum(x[i, j, f] for j in J) <= cap_tot[i, f] + he[i, f]
        mdl += he[i, f] <= ins["he_max_pct"] * cap_tot[i, f]
    if tr == "optimizar":
        for f in F:
            vs = [w[i, f] for i in I if (i, f) in w]
            if vs:
                mdl += pulp.lpSum(vs) <= 1
        for i in I:
            vs = [w[i, f] for f in F if (i, f) in w]
            if vs:
                mdl += pulp.lpSum(vs) <= ins["plantas"][i]["max_lineas"]
    mdl += pulp.lpSum(q.values()) <= ins["maquila_cap"]

    estado = mdl.solve(_solver())
    val = lambda e: float(pulp.value(e) or 0.0)
    wv = {k: (v.value() if hasattr(v, "value") else v) or 0.0 for k, v in w.items()}
    traslados = {}
    for (i, f), v in wv.items():
        if v > 1e-6:
            traslados.setdefault(f, {})[i] = float(v)
    flujos = [dict(origen=i, destino=j, familia=f, m3=float(v.value()), tarifa=c_fle[i, j], km=ins["km"][f"{i}|{j}"])
              for (i, j, f), v in x.items() if (v.value() or 0) > 1e-6]
    cap_val = {(i, f): cap[i, f] + capT.get(f, 0) * wv.get((i, f), 0) for i in I for f in F}
    uso = {}
    for i in I:
        c_i = sum(cap_val[i, f] for f in F)
        prod = sum(fl["m3"] for fl in flujos if fl["origen"] == i)
        uso[i] = dict(capacidad=c_i, produccion=prod, horas_extra=sum(he[i, f].value() or 0 for f in F if (i, f) in he),
                      utilizacion=prod / c_i if c_i else 0.0,
                      por_familia={f: dict(capacidad=cap_val[i, f], produccion=sum(fl["m3"] for fl in flujos if fl["origen"] == i and fl["familia"] == f))
                                   for f in F if cap_val[i, f] > 0})
    costos = dict(produccion=val(e_prod), flete=val(e_flete), horas_extra=val(e_he), maquila=val(e_maq),
                  penalidad=val(e_pen), fijo=fijo, capex=val(e_capex))
    costos["total"] = sum(costos.values())
    vol = sum(fl["m3"] for fl in flujos)
    dem = sum(D.values())
    return dict(estado=pulp.LpStatus[estado], costos=costos, flujos=flujos, traslados=traslados, uso=uso,
                maquila_m3=sum(v.value() or 0 for v in q.values()), no_atendida_m3=sum(v.value() or 0 for v in u.values()),
                demanda_m3=dem, km_promedio=sum(fl["m3"] * fl["km"] for fl in flujos) / vol if vol else 0.0,
                costo_por_m3=costos["total"] / dem if dem else 0.0)
