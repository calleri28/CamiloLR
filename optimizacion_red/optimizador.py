"""
Construcción de parámetros del modelo (demanda, tarifas, distancias) a partir
de los datos limpios, y modelo de optimización MILP (PuLP + HiGHS).

Modelo (mensual, en toneladas):
  Conjuntos: i plantas, j CEDIs, f familias.
  Variables:
    x[i,j,f] >= 0  t/mes enviadas de la planta i al CEDI j de la familia f
    he[i]    >= 0  t/mes producidas con horas extra / turno adicional
    q[j,f]   >= 0  t/mes tercerizadas (maquila) entregadas en j
    u[j,f]   >= 0  t/mes de demanda no atendida
    z[i,f]   ∈ {0,1}  trasladar la línea de la familia f de la planta que cierra a i
  Min  Σ (cv[i,f]·fp[i] + tarifa[i,j]·ff) x + Σ cHE[i] he + Σ cM[j,f] q + Σ pen u
       + Σ costo_fijo[i]·abierta[i] + Σ capex[f]/meses · z[i,f]
  s.a. Σ_i x[i,j,f] + q[j,f] + u[j,f] = D[j,f]·(1+g)                (demanda)
       Σ_{j,f} x[i,j,f] <= cap[i]·abierta[i] + Σ_f capL[f]·z[i,f] + he[i]  (capacidad)
       he[i] <= %HE · (cap[i]·abierta[i] + Σ_f capL[f]·z[i,f])
       x[i,j,f] <= M · (hace[i,f] + z[i,f])                         (habilitación)
       Σ_i z[i,f] <= 1 ;  Σ_f z[i,f] <= máx. líneas por planta ;  Σ q <= cap. maquila
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pulp

import parametros as P
from datos_prueba import km_vial


# ---------------------------------------------------------------------------
# Parámetros a partir de los datos
# ---------------------------------------------------------------------------
def coordenadas():
    return {**{k: (v["lat"], v["lon"]) for k, v in P.PLANTAS.items()},
            **{k: (v["lat"], v["lon"]) for k, v in P.CEDIS.items()}}


def estimar_demanda(df: pd.DataFrame) -> pd.DataFrame:
    """Demanda neta mensual por CEDI y familia = entradas XI - salidas XI a otros CEDI."""
    df = df.assign(mes=df["fecha"].dt.to_period("M"))
    n_meses = df["mes"].nunique()
    ent = df[df["destino"].isin(P.CEDIS)].groupby(["destino", "familia"])["peso_kg"].sum()
    sal = df[df["origen"].isin(P.CEDIS)].groupby(["origen", "familia"])["peso_kg"].sum()
    sal.index.names = ent.index.names
    neto = ent.sub(sal, fill_value=0).clip(lower=0) / 1000 / n_meses
    d = neto.rename("demanda_t_mes").reset_index().rename(columns={"destino": "cedi"})
    return d


def estimar_tarifas(df: pd.DataFrame):
    """Tarifa COP/t por ruta: histórica si hay suficientes viajes válidos, si no, regresión por km."""
    v = df[df["valido_tarifa"]]
    por_ruta = v.groupby(["origen", "destino"]).agg(
        importe=("importe_flete", "sum"), kg=("peso_kg", "sum"),
        viajes=("viaje_norm", "nunique"), km=("km_usado", "median"),
        utilizacion=("utilizacion", "mean"), match_tms=("match_tms", "mean")).reset_index()
    por_ruta["cop_t"] = por_ruta["importe"] / por_ruta["kg"] * 1000

    # Regresión ponderada COP/t = a + b·km sobre rutas con historia suficiente
    base = por_ruta[por_ruta["viajes"] >= P.MIN_REGISTROS_TARIFA]
    w = np.sqrt(base["kg"])
    b, a = np.polyfit(base["km"], base["cop_t"], 1, w=w)
    pred = a + b * base["km"]
    r2 = 1 - ((base["cop_t"] - pred) ** 2 * w).sum() / (((base["cop_t"] - np.average(base["cop_t"], weights=w)) ** 2) * w).sum()

    coords = coordenadas()
    filas = []
    hist = por_ruta.set_index(["origen", "destino"])
    for i in P.PLANTAS:
        for j in P.CEDIS:
            km_t = km_vial(*coords[i], *coords[j])
            if (i, j) in hist.index and hist.loc[(i, j), "viajes"] >= P.MIN_REGISTROS_TARIFA:
                h = hist.loc[(i, j)]
                filas.append(dict(origen=i, destino=j, km=float(h["km"]), tarifa_cop_t=float(h["cop_t"]),
                                  fuente="Histórico SAP", viajes=int(h["viajes"]),
                                  utilizacion=float(h["utilizacion"]) if pd.notna(h["utilizacion"]) else None))
            else:
                filas.append(dict(origen=i, destino=j, km=km_t, tarifa_cop_t=float(a + b * km_t),
                                  fuente="Estimada (regresión km)", viajes=int(hist.loc[(i, j), "viajes"]) if (i, j) in hist.index else 0,
                                  utilizacion=None))
    tarifas = pd.DataFrame(filas)
    reg = dict(intercepto=float(a), pendiente=float(b), r2=float(r2), n_rutas=int(len(base)))
    return tarifas, por_ruta, reg


def construir_insumos(demanda: pd.DataFrame, tarifas: pd.DataFrame) -> dict:
    """Paquete único de insumos que usan tanto el MILP (Python) como el simulador (JS)."""
    cv_min = {f: min(p["costo_var"][f] for p in P.PLANTAS.values()) for f in P.FAMILIAS}
    flete_medio_cedi = tarifas.groupby("destino")["tarifa_cop_t"].mean().to_dict()
    return dict(
        plantas={k: dict(nombre=v["nombre"], ciudad=v["ciudad"], lat=v["lat"], lon=v["lon"],
                         capacidad_t=v["capacidad_t"], costo_fijo=v["costo_fijo"],
                         familias=v["familias"], costo_var=v["costo_var"]) for k, v in P.PLANTAS.items()},
        cedis={k: dict(nombre=v["nombre"], ciudad=v["ciudad"], lat=v["lat"], lon=v["lon"]) for k, v in P.CEDIS.items()},
        familias=P.FAMILIAS,
        planta_cierre=P.PLANTA_CIERRE,
        demanda={f"{r.cedi}|{r.familia}": round(float(r.demanda_t_mes), 2) for r in demanda.itertuples()},
        tarifa={f"{r.origen}|{r.destino}": round(float(r.tarifa_cop_t), 0) for r in tarifas.itertuples()},
        km={f"{r.origen}|{r.destino}": round(float(r.km), 1) for r in tarifas.itertuples()},
        cap_linea=P.CAPACIDAD_LINEA_TRASLADADA_T,
        capex_linea=P.CAPEX_TRASLADO_LINEA,
        meses_amortizacion=P.MESES_AMORTIZACION,
        max_lineas=P.MAX_LINEAS_RECIBIDAS_POR_PLANTA,
        he_max_pct=P.HORAS_EXTRA_MAX_PCT,
        he_sobrecosto=P.SOBRECOSTO_HORAS_EXTRA_PCT,
        maquila_pct=P.COSTO_MAQUILA_PCT,
        maquila_cap=P.MAQUILA_MAX_T_MES,
        cv_min=cv_min,
        flete_medio_cedi=flete_medio_cedi,
        penalidad=P.PENALIDAD_DEMANDA_NO_ATENDIDA,
    )


def _solver():
    """HiGHS (pip install highspy) por defecto; CBC si está disponible."""
    disp = pulp.listSolvers(onlyAvailable=True)
    if "HiGHS" in disp:
        return pulp.HiGHS(msg=False, gapRel=1e-6)
    if "PULP_CBC_CMD" in disp:
        return pulp.PULP_CBC_CMD(msg=False, gapRel=1e-6)
    if "COIN_CMD" in disp:
        return pulp.COIN_CMD(msg=False)
    raise RuntimeError("No hay solver MILP disponible: instale highspy (pip install highspy)")


# ---------------------------------------------------------------------------
# Escenario y MILP
# ---------------------------------------------------------------------------
def escenario(nombre, descripcion, abiertas=None, traslado="optimizar", crec_demanda=0.0,
              indice_flete=1.0, indice_produccion=1.0, cap_pct=None):
    """traslado: 'ninguno' | 'optimizar' | dict familia -> planta."""
    return dict(nombre=nombre, descripcion=descripcion,
                abiertas=abiertas or {k: True for k in P.PLANTAS}, traslado=traslado,
                crec_demanda=crec_demanda, indice_flete=indice_flete,
                indice_produccion=indice_produccion, cap_pct=cap_pct or {k: 1.0 for k in P.PLANTAS})


def costo_he(ins, i, fp):
    p = ins["plantas"][i]
    return ins["he_sobrecosto"] * np.mean([p["costo_var"][f] for f in p["familias"]]) * fp


def costo_maquila(ins, j, f, fp, ff):
    return ins["cv_min"][f] * (1 + ins["maquila_pct"]) * fp + ins["flete_medio_cedi"][j] * ff


def resolver(ins: dict, esc: dict) -> dict:
    I, J, F = list(ins["plantas"]), list(ins["cedis"]), ins["familias"]
    cierre = ins["planta_cierre"]
    fp, ff, g = esc["indice_produccion"], esc["indice_flete"], esc["crec_demanda"]
    D = {(j, f): ins["demanda"].get(f"{j}|{f}", 0.0) * (1 + g) for j in J for f in F}
    ab = esc["abiertas"]
    cap = {i: ins["plantas"][i]["capacidad_t"] * esc["cap_pct"][i] for i in I}
    hace = {(i, f): ab[i] and f in ins["plantas"][i]["familias"] for i in I for f in F}

    # Candidatos de traslado de línea
    fam_cierre = ins["plantas"][cierre]["familias"]
    cand = [(i, f) for i in I for f in fam_cierre if i != cierre and ab[i]] if not ab[cierre] else []
    mdl = pulp.LpProblem(esc["nombre"].replace(" ", "_"), pulp.LpMinimize)
    x = {(i, j, f): pulp.LpVariable(f"x_{i}_{j}_{f}", lowBound=0) for i in I for j in J for f in F}
    he = {i: pulp.LpVariable(f"he_{i}", lowBound=0) for i in I}
    q = {(j, f): pulp.LpVariable(f"q_{j}_{f}", lowBound=0) for j in J for f in F}
    u = {(j, f): pulp.LpVariable(f"u_{j}_{f}", lowBound=0) for j in J for f in F}
    z = {(i, f): pulp.LpVariable(f"z_{i}_{f}", cat="Binary") for (i, f) in cand}
    tr = esc["traslado"]
    for (i, f), var in z.items():
        if tr == "ninguno":
            var.upBound = 0
        elif isinstance(tr, dict):
            var.lowBound = var.upBound = 1 if tr.get(f) == i else 0

    c_prod = {(i, f): ins["plantas"][i]["costo_var"][f] * fp for i in I for f in F}
    c_fle = {(i, j): ins["tarifa"][f"{i}|{j}"] * ff for i in I for j in J}
    cap_amort = {f: ins["capex_linea"][f] / ins["meses_amortizacion"] for f in F}

    e_prod = pulp.lpSum(c_prod[i, f] * x[i, j, f] for i in I for j in J for f in F)
    e_flete = pulp.lpSum(c_fle[i, j] * x[i, j, f] for i in I for j in J for f in F)
    e_he = pulp.lpSum(costo_he(ins, i, fp) * he[i] for i in I)
    e_maq = pulp.lpSum(costo_maquila(ins, j, f, fp, ff) * q[j, f] for j in J for f in F)
    e_pen = pulp.lpSum(ins["penalidad"] * u[j, f] for j in J for f in F)
    fijo = sum(ins["plantas"][i]["costo_fijo"] for i in I if ab[i])
    e_capex = pulp.lpSum(cap_amort[f] * z[i, f] for (i, f) in z)
    mdl += e_prod + e_flete + e_he + e_maq + e_pen + e_capex + fijo

    for j in J:
        for f in F:
            mdl += pulp.lpSum(x[i, j, f] for i in I) + q[j, f] + u[j, f] == D[j, f], f"dem_{j}_{f}"
    for i in I:
        cap_lineas = pulp.lpSum(ins["cap_linea"][f] * z[i, f] for f in F if (i, f) in z)
        mdl += pulp.lpSum(x[i, j, f] for j in J for f in F) <= cap[i] * ab[i] + cap_lineas + he[i], f"cap_{i}"
        mdl += he[i] <= ins["he_max_pct"] * (cap[i] * ab[i] + cap_lineas), f"he_{i}"
        mdl += pulp.lpSum(z[i, f] for f in F if (i, f) in z) <= ins["max_lineas"], f"maxlin_{i}"
        for f in F:
            M = sum(D[j, f] for j in J)
            hab = (1 if hace[i, f] else 0) + (z[i, f] if (i, f) in z else 0)
            for j in J:
                mdl += x[i, j, f] <= M * hab
    for f in F:
        if any((i, f) in z for i in I):
            mdl += pulp.lpSum(z[i, f] for i in I if (i, f) in z) <= 1
    mdl += pulp.lpSum(q.values()) <= ins["maquila_cap"], "maquila"

    estado = mdl.solve(_solver())
    val = lambda e: float(pulp.value(e) or 0.0)
    flujos = [dict(origen=i, destino=j, familia=f, t=x[i, j, f].value(),
                   tarifa=c_fle[i, j], km=ins["km"][f"{i}|{j}"])
              for i in I for j in J for f in F if (x[i, j, f].value() or 0) > 1e-6]
    traslados = {f: i for (i, f), v in z.items() if (v.value() or 0) > 0.5}
    uso = {}
    for i in I:
        cap_tot = cap[i] * ab[i] + sum(ins["cap_linea"][f] for f, ii in traslados.items() if ii == i)
        prod = sum(fl["t"] for fl in flujos if fl["origen"] == i)
        uso[i] = dict(capacidad=cap_tot, produccion=prod, horas_extra=he[i].value() or 0.0,
                      utilizacion=prod / cap_tot if cap_tot else 0.0)
    costos = dict(produccion=val(e_prod), flete=val(e_flete), horas_extra=val(e_he), maquila=val(e_maq),
                  penalidad=val(e_pen), fijo=fijo, capex=val(e_capex))
    costos["total"] = sum(costos.values())
    tk = sum(fl["t"] * fl["km"] for fl in flujos)
    t_tot = sum(fl["t"] for fl in flujos)
    return dict(escenario=esc, estado=pulp.LpStatus[estado], costos=costos, flujos=flujos,
                traslados=traslados, uso=uso,
                maquila_t=sum(v.value() or 0 for v in q.values()),
                no_atendida_t=sum(v.value() or 0 for v in u.values()),
                demanda_t=sum(D.values()), km_promedio=tk / t_tot if t_tot else 0.0,
                costo_por_t=costos["total"] / sum(D.values()))
