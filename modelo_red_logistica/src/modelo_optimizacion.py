"""Modelo de optimización de red (MILP) — flujo multiproducto planta -> almacén -> ciudad.

Formulación (horizonte: 1 mes promedio, unidades m3)
---------------------------------------------------
Conjuntos: P plantas, W almacenes (CEDI propios, 3PL, bodega externa actual y candidatas),
           Z ciudades de demanda, K productos (planta de origen, o familia si la producción es flexible).

Variables: x[p,w,k] >= 0  m3 de k producidos en p y trasladados a w (si p = w no hay flete)
           y[w,z,k] >= 0  m3 de k despachados desde w a la ciudad z
           a[w] in {0,1}  abrir / mantener un almacén opcional (bodega externa)
           d[w] >= 0      m3 de inventario que no caben en w (déficit, penalizado)

min  sum c_in[p,w]*f_xi*x + sum (c_out[w,z]*f_dist + v[w])*y + sum F[w]*a[w] + pen*sum d[w]
s.a. sum_w y[w,z,k] = D[z,k]                                  (demanda)
     sum_p x[p,w,k] = sum_z y[w,z,k]                          (balance en almacén)
     (DOI/30) * sum_{z,k} y[w,z,k] <= Cap[w]*a[w] + d[w]       (capacidad de almacenamiento)
     sum_{w,k} x[p,w,k] <= CapPlanta[p]                       (solo producción flexible)
     sum_k x[p,w,k] <= XI_asis[p,w]                           (opción A: traslados a otros CEDI limitados)
con v[w] = manejo[w] + costo_posición[w] * DOI/30 / m3_por_posición.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from .preparacion import Preparado
from .utilidades import LOG, haversine_km


# ---------------------------------------------------------------------------
# Costos unitarios de rutas
# ---------------------------------------------------------------------------
class Tarifario:
    """Devuelve COP/m3 para traslados (nodo->nodo) y distribución (nodo->ciudad)."""

    def __init__(self, prep: Preparado, cfg: dict):
        self.p = prep
        self.fr = float(cfg["limpieza"]["factor_ruta_km"])
        td = prep.tarifas_dist[prep.tarifas_dist.confiable]
        tx = prep.tarifas_xi[prep.tarifas_xi.confiable]
        self.obs_dist = {(o, d): t for o, d, t in zip(td.origen, td.destino, td.tarifa)}
        self.obs_xi = {(o, d): t for o, d, t in zip(tx.origen, tx.destino, tx.tarifa)}
        self.coord_n = prep.nodos[["lat", "lon"]].astype(float)
        self.coord_c = prep.ciudades.set_index("ciudad")[["lat", "lon"]].astype(float)
        self.reg = prep.regresion
        self._cache_in: dict = {}
        self._cache_out: dict = {}
        self.fuente: dict = {}

    def km_nn(self, a, b):
        if a not in self.coord_n.index or b not in self.coord_n.index:
            return np.nan
        A, B = self.coord_n.loc[a], self.coord_n.loc[b]
        return float(haversine_km(A.lat, A.lon, B.lat, B.lon)) * self.fr

    def km_nc(self, a, c):
        if a not in self.coord_n.index or c not in self.coord_c.index:
            return np.nan
        A, B = self.coord_n.loc[a], self.coord_c.loc[c]
        if np.isnan(B.lat):
            return np.nan
        return float(haversine_km(A.lat, A.lon, B.lat, B.lon)) * self.fr

    def c_in(self, p, w) -> float:
        if p == w:
            return 0.0
        key = (p, w)
        if key in self._cache_in:
            return self._cache_in[key]
        como = self.p.nodos.at[w, "tarifa_entrada_como"] if w in self.p.nodos.index else None
        val, fuente = None, None
        if isinstance(como, str) and como:
            if (p, como) in self.obs_xi:
                val, fuente = self.obs_xi[(p, como)], f"observada {p}->{como}"
            elif p == como:
                val, fuente = 0.0, "mismo sitio"
        if val is None and key in self.obs_xi:
            val, fuente = self.obs_xi[key], "observada SAP"
        if val is None:
            km = self.km_nn(p, w)
            r = self.reg["traslado"]
            val = r["a"] + r["b"] * (km if not np.isnan(km) else 400.0)
            fuente = "estimada por distancia"
        self._cache_in[key] = float(val)
        self.fuente[("in",) + key] = fuente
        return float(val)

    def c_out(self, w, z) -> float:
        key = (w, z)
        if key in self._cache_out:
            return self._cache_out[key]
        como = self.p.nodos.at[w, "tarifas_salida_como"] if w in self.p.nodos.index else None
        src = como if isinstance(como, str) and como else w
        if (src, z) in self.obs_dist:
            val, fuente = self.obs_dist[(src, z)], "observada SAP" if src == w else f"observada {src}"
        else:
            km = self.km_nc(src, z)
            r = self.reg["distribucion"]
            if np.isnan(km):
                # ciudad sin coordenadas: tarifa media observada del origen
                sub = [t for (o, _), t in self.obs_dist.items() if o == src]
                val = float(np.mean(sub)) if sub else r["a"] + r["b"] * 400
            else:
                val = (r["a"] + r["b"] * km) * self.p.factor_cedi.get(src, 1.0)
            fuente = "estimada por distancia"
        self._cache_out[key] = float(val)
        self.fuente[("out",) + key] = fuente
        return float(val)


# ---------------------------------------------------------------------------
# Instancia de un escenario
# ---------------------------------------------------------------------------
@dataclass
class Instancia:
    esc: dict
    params: dict
    D: dict                 # (z,k) -> m3/mes
    P_k: dict               # k -> plantas que pueden producir k
    W: list
    nodos: pd.DataFrame     # atributos de W
    binarios: list
    cap_planta: dict
    lim_xi: dict            # (p,w) -> m3/mes máximo
    constantes_fijas: float
    zona_ciudad: dict


def parametros_escenario(cfg: dict, esc: dict, overrides: dict | None = None) -> dict:
    alm = cfg["almacenamiento"]
    p = {
        "dias_inventario": float(alm.get("dias_inventario_efectivo", alm["dias_inventario"])),
        "utilizacion_maxima": float(alm["utilizacion_maxima"]),
        "penalidad": float(alm["penalidad_deficit_cop_m3"]),
        "factor_flete_traslado": 1.0,
        "factor_flete_distribucion": 1.0,
        "crecimiento_extra": 0.0,
    }
    p.update({k: v for k, v in esc.items() if k in p})
    if overrides:
        p.update(overrides)
    return p


def construir_instancia(prep: Preparado, cfg: dict, esc: dict, params: dict) -> Instancia:
    prod = cfg["produccion"]
    cerrada = prod["planta_cerrada"] if esc.get("cerrar_planta") else None
    modo = esc.get("produccion", "tobe")
    dem = prep.demanda.copy()
    if prod.get("excluir_descontinuadas", True):   # misma demanda en todos los escenarios -> comparables
        dem = dem[~dem.descontinuada]
    dem["vol_mes"] = dem["vol_mes"].astype(float) * (1 + params.get("crecimiento_extra", 0.0))
    if modo == "asis":
        dem["k"] = dem["planta_asis"]
    elif modo == "tobe":
        dem["k"] = dem["planta_tobe"]
    else:
        dem["k"] = dem["familia"]
    dem = dem.dropna(subset=["k", "ciudad"])
    D = dem.groupby(["ciudad", "k"])["vol_mes"].sum()
    D = {kz: float(v) for kz, v in D.items() if v > 1e-6}
    zona_ciudad = dem.groupby("ciudad")["zona"].first().to_dict()

    # plantas por producto
    P_k: dict = {}
    cap_planta: dict = {}
    if modo in ("asis", "tobe"):
        for (_, k) in D:
            P_k[k] = [k]
    else:
        pl = prep.plantas
        tobe = prep.demanda[~prep.demanda.descontinuada]
        prod_fam = tobe.groupby(["familia", "planta_tobe"])["vol_mes"].sum().reset_index()
        for (_, f) in D:
            if f in P_k:
                continue
            capaces = [p for p in pl.index if p != cerrada and isinstance(pl.at[p, "familias"], list)
                       and f in pl.at[p, "familias"]]
            actuales = prod_fam[(prod_fam.familia == f) & (prod_fam.planta_tobe != cerrada)].planta_tobe.tolist()
            P_k[f] = sorted(set(capaces) | set(actuales))
        util_p = float(prod.get("utilizacion_maxima_planta", 1.0))
        for p in pl.index:
            if p != cerrada and pd.notna(pl.at[p, "capacidad_m3_mes"]):
                cap_planta[p] = float(pl.at[p, "capacidad_m3_mes"]) * util_p

    # almacenes
    n = prep.nodos.copy()
    opcionales = set(cfg["almacenamiento"].get("nodos_opcionales_tobe") or [])
    activos = n[(n.posiciones_total > 0)]
    if cerrada:
        activos = activos[activos.index != cerrada]
    if not esc.get("bodegas_candidatas"):
        activos = activos[~activos.candidata]
    if esc.get("cerrar_opcionales"):
        activos = activos[~activos.index.isin(opcionales)]
    activos = activos.copy()
    activos["cap_inv_m3"] = activos["posiciones_total"] * prep.m3_por_posicion * params["utilizacion_maxima"]
    doi = params["dias_inventario"]
    activos["costo_var_m3"] = activos["manejo_cop_m3"] + activos["posicion_cop_mes"] * doi / 30 / prep.m3_por_posicion
    binarios = [w for w in activos.index if activos.at[w, "candidata"] or
                (w in opcionales and not esc.get("opcionales_fijos_abiertos"))]
    constantes = float(activos.loc[~activos.index.isin(binarios), "costo_fijo"].sum())
    # plantas que no son almacén (no aplica en estos datos) también suman costo fijo
    for p in set(x for l in P_k.values() for x in l):
        if p not in activos.index and p in n.index:
            constantes += float(n.at[p, "costo_fijo"])

    lim_xi = {}
    if esc.get("limitar_traslados_asis"):
        xa = prep.xi_asis.dropna(subset=["destino"])
        obs = {(o, d): v for o, d, v in zip(xa.origen, xa.destino, xa.vol_mes)}
        plantas = set(x for l in P_k.values() for x in l)
        holgura = float(esc.get("holgura_traslados", 1.0))
        for p in plantas:
            for w in activos.index:
                if w != p and not activos.at[w, "candidata"] and w not in opcionales:
                    lim_xi[(p, w)] = obs.get((p, w), 0.0) * holgura

    return Instancia(esc=esc, params=params, D=D, P_k=P_k, W=list(activos.index), nodos=activos,
                     binarios=binarios, cap_planta=cap_planta, lim_xi=lim_xi, constantes_fijas=constantes,
                     zona_ciudad=zona_ciudad)


# ---------------------------------------------------------------------------
# Constructor de modelos lineales sobre scipy.optimize.milp (HiGHS incluido en SciPy)
# ---------------------------------------------------------------------------
class ModeloLineal:
    def __init__(self):
        self.c, self.lb, self.ub, self.entera = [], [], [], []
        self.filas, self.cols, self.vals, self.lo, self.hi = [], [], [], [], []
        self.n_restr = 0
        self.valores = None

    def var(self, costo=0.0, lb=0.0, ub=np.inf, entera=False) -> int:
        self.c.append(float(costo)); self.lb.append(lb); self.ub.append(ub); self.entera.append(1 if entera else 0)
        return len(self.c) - 1

    def restr(self, idx, coef, lo=-np.inf, hi=np.inf):
        r = self.n_restr
        self.filas.extend([r] * len(idx)); self.cols.extend(idx); self.vals.extend(coef)
        self.lo.append(lo); self.hi.append(hi)
        self.n_restr += 1

    def resolver(self, tiempo=300, gap=1e-3):
        A = sparse.csr_matrix((self.vals, (self.filas, self.cols)), shape=(self.n_restr, len(self.c)))
        res = milp(c=np.array(self.c), constraints=LinearConstraint(A, self.lo, self.hi),
                   integrality=np.array(self.entera), bounds=Bounds(self.lb, self.ub),
                   options={"time_limit": tiempo, "mip_rel_gap": gap, "disp": False})
        self.valores = res.x if res.x is not None else np.zeros(len(self.c))
        estado = {0: "Optimal", 1: "Limite tiempo/iter", 2: "Infactible", 3: "No acotado", 4: "Otro"}.get(res.status, str(res.status))
        if res.status == 1 and res.x is not None:
            estado = "Factible (límite de tiempo)"
        return estado, float(res.fun) if res.fun is not None else float("nan")

    def v(self, i) -> float:
        return float(self.valores[i])


def resolver(inst: Instancia, tar: Tarifario, prep: Preparado, cfg: dict) -> dict:
    t0 = time.time()
    prm = inst.params
    fxi, fdi = prm["factor_flete_traslado"], prm["factor_flete_distribucion"]
    doi = prm["dias_inventario"]
    W = inst.W
    nd = inst.nodos
    m = ModeloLineal()
    var = nd["costo_var_m3"].to_dict()

    ks = sorted({k for (_, k) in inst.D})
    Z_k = {k: [z for (z, kk) in inst.D if kk == k] for k in ks}
    x, y, c_in, c_out = {}, {}, {}, {}
    for k in ks:
        for p in inst.P_k[k]:
            for w in W:
                c_in[(p, w)] = tar.c_in(p, w)
                x[(p, w, k)] = m.var(c_in[(p, w)] * fxi)
        for z in Z_k[k]:
            for w in W:
                if (w, z) not in c_out:
                    c_out[(w, z)] = tar.c_out(w, z)
                y[(w, z, k)] = m.var(c_out[(w, z)] * fdi + var[w])
    a = {w: m.var(float(nd.at[w, "costo_fijo"]), 0, 1, entera=True) for w in inst.binarios}
    dfc = {w: m.var(prm["penalidad"]) for w in W if w not in inst.binarios}
    dpl = {p: m.var(prm["penalidad"]) for p in inst.cap_planta}

    for (z, k), dval in inst.D.items():                       # demanda
        m.restr([y[(w, z, k)] for w in W], [1.0] * len(W), dval, dval)
    for k in ks:                                              # balance en almacén
        for w in W:
            ix = [x[(p, w, k)] for p in inst.P_k[k]]
            iy = [y[(w, z, k)] for z in Z_k[k]]
            m.restr(ix + iy, [1.0] * len(ix) + [-1.0] * len(iy), 0.0, 0.0)
    y_por_w = {w: [] for w in W}
    for (w, z, k), i in y.items():
        y_por_w[w].append(i)
    for w in W:                                               # capacidad de almacenamiento
        cap = float(nd.at[w, "cap_inv_m3"])
        idx, coef = y_por_w[w], [doi / 30.0] * len(y_por_w[w])
        if w in a:
            m.restr(idx + [a[w]], coef + [-cap], hi=0.0)
        else:
            m.restr(idx + [dfc[w]], coef + [-1.0], hi=cap)
    if inst.cap_planta:                                       # capacidad de planta (producción flexible)
        x_por_p: dict = {}
        for (p, w, k), i in x.items():
            x_por_p.setdefault(p, []).append(i)
        for p, cap in inst.cap_planta.items():
            if p in x_por_p:
                m.restr(x_por_p[p] + [dpl[p]], [1.0] * len(x_por_p[p]) + [-1.0], hi=cap)
    for (p, w), lim in inst.lim_xi.items():                   # opción A: traslados limitados al AS-IS
        vs = [i for (pp, ww, _), i in x.items() if pp == p and ww == w]
        if vs:
            m.restr(vs, [1.0] * len(vs), hi=lim)

    sc = cfg.get("solver", {})
    estado_txt, fobj = m.resolver(sc.get("tiempo_maximo_seg", 300), sc.get("gap_relativo", 1e-3))
    t = time.time() - t0
    LOG.info("  %-18s %-10s z=%s MCOP  (%d vars, %d restr, %.1fs)", inst.esc["id"], estado_txt,
             f"{(fobj + inst.constantes_fijas) / 1e6:,.0f}", len(m.c), m.n_restr, t)

    # ------------------------------------------------------------ resultados
    fx = pd.DataFrame([(p, w, k, m.v(i)) for (p, w, k), i in x.items()],
                      columns=["planta", "almacen", "producto", "m3_mes"])
    fx = fx[fx.m3_mes > 1e-6].copy()
    fx["tarifa_cop_m3"] = [c_in[(p, w)] * fxi for p, w in zip(fx.planta, fx.almacen)]
    fx["costo_cop_mes"] = fx.m3_mes * fx.tarifa_cop_m3
    fx["fuente_tarifa"] = [tar.fuente.get(("in", p, w), "mismo sitio") for p, w in zip(fx.planta, fx.almacen)]
    fy = pd.DataFrame([(w, z, k, m.v(i)) for (w, z, k), i in y.items()],
                      columns=["almacen", "ciudad", "producto", "m3_mes"])
    fy = fy[fy.m3_mes > 1e-6].copy()
    fy["zona"] = fy.ciudad.map(inst.zona_ciudad)
    fy["tarifa_cop_m3"] = [c_out[(w, z)] * fdi for w, z in zip(fy.almacen, fy.ciudad)]
    fy["costo_cop_mes"] = fy.m3_mes * fy.tarifa_cop_m3
    fy["fuente_tarifa"] = [tar.fuente.get(("out", w, z)) for w, z in zip(fy.almacen, fy.ciudad)]

    thr = fy.groupby("almacen").m3_mes.sum().reindex(W).fillna(0)
    rn = pd.DataFrame(index=W)
    rn["nombre"] = nd["nombre"]
    rn["tipo"] = np.where(nd["candidata"], "Bodega externa candidata", nd["Modalidad"])
    rn["abierto"] = [bool(round(m.v(a[w]))) if w in a else True for w in W]
    rn["decision"] = ["binaria" if w in a else "fija" for w in W]
    rn["throughput_m3_mes"] = thr
    rn["inventario_m3"] = thr * doi / 30
    rn["capacidad_m3"] = nd["cap_inv_m3"]
    rn["posiciones_requeridas"] = rn.inventario_m3 / prep.m3_por_posicion
    rn["posiciones_disponibles"] = nd["posiciones_total"]
    rn["utilizacion"] = np.where(rn.capacidad_m3 > 0, rn.inventario_m3 / (nd["posiciones_total"] * prep.m3_por_posicion), 0)
    rn["deficit_m3"] = [m.v(dfc[w]) if w in dfc else 0.0 for w in W]
    rn["costo_manejo_cop"] = thr * nd["manejo_cop_m3"]
    rn["costo_posiciones_cop"] = rn.posiciones_requeridas * nd["posicion_cop_mes"]
    rn["costo_fijo_cop"] = [float(nd.at[w, "costo_fijo"]) if rn.at[w, "abierto"] else 0.0 for w in W]
    rn["entrada_traslado_m3_mes"] = fx[fx.planta != fx.almacen].groupby("almacen").m3_mes.sum().reindex(W).fillna(0)
    rn.index.name = "almacen"

    costos = {
        "Flete traslado (XI)": float(fx.costo_cop_mes.sum()),
        "Flete distribución": float(fy.costo_cop_mes.sum()),
        "Manejo en almacén": float(rn.costo_manejo_cop.sum()),
        "Posiciones bodega externa": float(rn.costo_posiciones_cop.sum()),
        "Fijo almacenes opcionales": float(sum(nd.at[w, "costo_fijo"] for w in a if rn.at[w, "abierto"])),
        "Fijo red (nodos fijos)": inst.constantes_fijas,
        "Penalización déficit capacidad": float(prm["penalidad"] * (sum(m.v(i) for i in dfc.values()) +
                                                                   sum(m.v(i) for i in dpl.values()))),
    }
    costos["Total"] = sum(costos.values())
    costos["Total variable + fijo opcional"] = costos["Total"] - costos["Fijo red (nodos fijos)"]

    # KPIs
    tl = prep.tiempos_lane
    t_xi = {(o, d): h for o, d, h in zip(tl[tl.es_xi].origen, tl[tl.es_xi].destino, tl[tl.es_xi].transito_h_mediana)}
    t_di = {(o, d): h for o, d, h in zip(tl[~tl.es_xi].origen, tl[~tl.es_xi].destino, tl[~tl.es_xi].transito_h_mediana)}
    m3v = {(o, d): v for o, d, v in zip(tl[tl.es_xi].origen, tl[tl.es_xi].destino, tl[tl.es_xi].m3_por_viaje)}
    m3v_def = float(tl[tl.es_xi].m3_por_viaje.median()) if (tl.es_xi.any()) else 50.0
    tras = fx[fx.planta != fx.almacen].copy()
    tras["viajes_mes"] = [m / max(m3v.get((p, w), m3v_def), 1.0) for p, w, m in zip(tras.planta, tras.almacen, tras.m3_mes)]
    fx.loc[tras.index, "viajes_mes"] = tras["viajes_mes"]
    fx["transito_h"] = [t_xi.get((p, w), (tar.km_nn(p, w) / 45 + 3) if p != w else 0.0) for p, w in zip(fx.planta, fx.almacen)]
    fy["transito_h"] = [t_di.get((w, z), (tar.km_nc(w, z) / 45 + 3) if not np.isnan(tar.km_nc(w, z)) else np.nan)
                        for w, z in zip(fy.almacen, fy.ciudad)]
    vol_total = float(fy.m3_mes.sum())
    kpis = {
        "estado_solver": estado_txt,
        "tiempo_solucion_s": round(t, 1),
        "variables": len(m.c),
        "restricciones": m.n_restr,
        "deficit_planta_m3": float(sum(m.v(i) for i in dpl.values())),
        "demanda_m3_mes": vol_total,
        "traslados_m3_mes": float(tras.m3_mes.sum()),
        "viajes_traslado_mes": float(tras.viajes_mes.sum()) if len(tras) else 0.0,
        "pct_volumen_trasladado": float(tras.m3_mes.sum()) / vol_total if vol_total else 0,
        "costo_por_m3": costos["Total"] / vol_total if vol_total else 0,
        "costo_variable_por_m3": costos["Total variable + fijo opcional"] / vol_total if vol_total else 0,
        "transito_promedio_h": float(np.nansum(fy.transito_h * fy.m3_mes) / fy.loc[fy.transito_h.notna(), "m3_mes"].sum()
                                     + np.nansum(fx.transito_h * fx.m3_mes) / vol_total),
        "deficit_m3": float(rn.deficit_m3.sum()),
        "bodegas_abiertas": [w for w in a if rn.at[w, "abierto"]],
        "bodegas_cerradas": [w for w in a if not rn.at[w, "abierto"]],
        "objetivo_cop": fobj,
    }
    return {"id": inst.esc["id"], "nombre": inst.esc.get("nombre", inst.esc["id"]), "params": dict(prm),
            "flujos_traslado": fx.reset_index(drop=True), "flujos_distribucion": fy.reset_index(drop=True),
            "nodos": rn.reset_index(), "costos": costos, "kpis": kpis}


# ---------------------------------------------------------------------------
# Evaluación de la red AS-IS real (flujos observados x tarifas)
# ---------------------------------------------------------------------------
def evaluar_asis_real(prep: Preparado, tar: Tarifario, cfg: dict) -> dict:
    prm = parametros_escenario(cfg, {})
    doi = prm["dias_inventario"]
    n = prep.nodos[~prep.nodos.candidata].copy()
    da = prep.dist_asis.copy()
    da["tarifa_cop_m3"] = [tar.c_out(o, z) if o in n.index else np.nan for o, z in zip(da.origen, da.destino)]
    da["costo_cop_mes"] = da.vol_mes * da.tarifa_cop_m3
    xa = prep.xi_asis.dropna(subset=["destino"])
    xa = xa[xa.origen != xa.destino].copy()
    xa["tarifa_cop_m3"] = [tar.c_in(o, d) for o, d in zip(xa.origen, xa.destino)]
    xa["costo_cop_mes"] = xa.vol_mes * xa.tarifa_cop_m3
    thr = da.groupby("origen").vol_mes.sum().reindex(n.index).fillna(0)
    # la bodega externa actual (CO99) recibe pero no registra despachos: su throughput = entradas
    entradas = xa.groupby("destino").vol_mes.sum().reindex(n.index).fillna(0)
    thr = np.maximum(thr, np.where(thr == 0, entradas, 0))
    costos = {
        "Flete traslado (XI)": float(xa.costo_cop_mes.sum()),
        "Flete distribución": float(da.costo_cop_mes.sum()),
        "Manejo en almacén": float((thr * n.manejo_cop_m3).sum()),
        "Posiciones bodega externa": float((thr * doi / 30 / prep.m3_por_posicion * n.posicion_cop_mes).sum()),
        "Fijo almacenes opcionales": 0.0,
        "Fijo red (nodos fijos)": float(n.costo_fijo.sum()),
        "Penalización déficit capacidad": 0.0,
    }
    costos["Total"] = sum(costos.values())
    costos["Total variable + fijo opcional"] = costos["Total"] - costos["Fijo red (nodos fijos)"]
    base = prep.sap[prep.sap.en_base]
    flete_sap = float(base.loc[base.flete_valido, "imp"].sum()) / prep.meses_base
    rn = pd.DataFrame({"nombre": n.nombre, "throughput_m3_mes": thr, "inventario_m3": thr * doi / 30,
                       "capacidad_m3": n.posiciones_total * prep.m3_por_posicion * prm["utilizacion_maxima"]})
    rn["utilizacion"] = rn.inventario_m3 / (n.posiciones_total * prep.m3_por_posicion)
    vol = float(da.vol_mes.sum())
    return {"id": "ASIS_REAL", "nombre": "AS-IS real (flujos observados SAP x tarifas)", "costos": costos,
            "flujos_distribucion": da, "flujos_traslado": xa, "nodos": rn.reset_index().rename(columns={"ID_Nodo": "almacen", "index": "almacen"}),
            "kpis": {"demanda_m3_mes": vol, "traslados_m3_mes": float(xa.vol_mes.sum()),
                     "costo_por_m3": costos["Total"] / vol if vol else 0,
                     "flete_registrado_sap_cop_mes": flete_sap,
                     "costo_variable_por_m3": costos["Total variable + fijo opcional"] / vol if vol else 0},
            "params": prm}
