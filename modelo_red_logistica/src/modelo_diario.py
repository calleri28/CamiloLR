"""Modelo multiperiodo DIARIO de la red logística (MILP).

Réplica exacta de `recursos/motor.js` (mismo orden de cálculo, mismas reglas). Cualquier cambio aquí
debe replicarse allá; `pruebas/paridad.js` compara ambos.

Notación (ver pestaña Metodología del HTML):
  t ∈ T días del horizonte, n ∈ N nodos activos, k ∈ K productos (familia × planta productora),
  z ∈ Z ciudades de demanda, a=(u,v) ∈ A arcos de traslado con tiempo de tránsito L_a días.
Variables:
  s[n,z,k] ∈ [0,1]   fracción de la demanda (z,k) que atiende el nodo n (política de atención)
  x[a,k,t] ≥ 0       m3 despachados por el arco a el día t (llegan el día t + L_a)
  I[n,k,t] ≥ 0       inventario al cierre del día t
  B[n,k,t] ≥ 0       demanda asignada a n aún no despachada (atraso / faltante)
  E[n,t] ≥ 0         m3 por encima de la capacidad física (incumplimiento visible)
  F[n,k,t] ≥ 0       faltante frente al stock de seguridad
  q[n,k] ≥ 0         stock de seguridad objetivo (función de la asignación)
  y[n] ∈ {0,1}       nodo opcional abierto
  P[k,t] ≥ 0         producción (solo en modo flexible; en modo plan es dato)
  u[z,k] ∈ [0,1]     fracción de la demanda sin nodo que la pueda atender (p. ej. por restricción de servicio)
"""
from __future__ import annotations

import math
import time

import numpy as np

from .modelo_optimizacion import ModeloLineal

ESCALA = 1e-6     # el objetivo se resuelve en millones de COP (estabilidad numérica)

COMP = ["Traslado XI terceros", "Traslado XI flota propia", "Distribución", "Manejo", "Almacenaje", "Costo financiero inventario",
        "Fijo bodegas externas", "Fijo nodos (plantas y CEDI)", "Penalización atraso", "Penalización no atendido",
        "Penalización exceso capacidad", "Penalización stock seguridad", "Penalización servicio"]
GRUPO = {"Traslado XI terceros": "Transporte", "Traslado XI flota propia": "Transporte", "Distribución": "Transporte",
         "Manejo": "Almacenamiento", "Almacenaje": "Almacenamiento", "Costo financiero inventario": "Almacenamiento",
         "Fijo bodegas externas": "Almacenamiento", "Fijo nodos (plantas y CEDI)": "Fijo nodos",
         "Penalización atraso": "Penalizaciones", "Penalización no atendido": "Penalizaciones",
         "Penalización exceso capacidad": "Penalizaciones", "Penalización stock seguridad": "Penalizaciones",
         "Penalización servicio": "Penalizaciones"}


# ---------------------------------------------------------------------------
# 1. Instancia: datos del escenario listos para el modelo
# ---------------------------------------------------------------------------
def instancia(base: dict, prm: dict) -> dict:
    nodos = base["nodos"]
    NN = len(nodos)
    fechas = base["meta"]["fechas"]
    dow = base["meta"]["dia_semana"]
    W = len(fechas)
    off = max(0, min(int(prm.get("desplazamiento", 0)), W - 1))
    H = max(1, min(int(prm["horizonte_dias"]), W - off))
    dias = list(range(off, off + H))
    cerr = prm.get("planta_cerrada") or None
    pn = prm["nodos"]
    activo = [pn.get(n["id"], {}).get("modo", "fijo") != "cerrado" and n["id"] != cerr for n in nodos]
    modo = [pn.get(n["id"], {}).get("modo", "fijo") for n in nodos]
    idn = {n["id"]: i for i, n in enumerate(nodos)}
    grupos = base["grupos"]
    G = len(grupos)
    zonas = base["zonas"]

    # demanda por grupo y día (proyección)
    D0 = np.zeros((G, W))
    for d, g, m3 in base["demanda"]:
        D0[g, d] += m3
    if prm.get("proyeccion") == "perfil":
        prof = np.zeros((G, 7))
        cnt = np.zeros(7)
        for d in range(W):
            prof[:, dow[d]] += D0[:, d]
            cnt[dow[d]] += 1
        prof = prof / np.where(cnt > 0, cnt, 1)
        Dg0 = np.array([prof[:, dow[d]] for d in dias]).T
    else:
        Dg0 = D0[:, dias]
    fac = np.ones(G)
    for g, (z, f, pa, pt, me) in enumerate(grupos):
        fac[g] = ((1 + prm.get("crecimiento", 0)) * (1 + (prm.get("crec_E", 0) if me else prm.get("crec_N", 0))) *
                  (1 + prm.get("crec_zona", {}).get(zonas[z]["zona"] or "", 0)) *
                  (1 + prm.get("crec_familia", {}).get(base["familias"][f], 0)))
    Dg = Dg0 * fac[:, None]

    # producto de cada grupo: planta productora según escenario
    rep = {k: v for k, v in (prm.get("reparto") or {}).items() if v and k in idn and activo[idn[k]] and nodos[idn[k]]["es_planta"]}
    tot_rep = sum(rep.values())
    gk = []          # por grupo: lista (k_key, fracción)
    for g, (z, f, pa, pt, me) in enumerate(grupos):
        if cerr is None:
            partes = [(pa, 1.0)]
        elif nodos[pa]["id"] == cerr:
            partes = [(idn[p], v / tot_rep) for p, v in rep.items()] if (prm.get("usar_reparto") and tot_rep > 0) else [(pt, 1.0)]
        else:
            partes = [(pa if nodos[pt]["id"] == cerr else pt, 1.0)]
        gk.append([(f * 1000 + p, fr) for p, fr in partes])
    K = sorted({k for lst in gk for k, _ in lst})
    idk = {k: i for i, k in enumerate(K)}
    planta_k = [k % 1000 for k in K]
    fam_k = [k // 1000 for k in K]

    # demanda por (z,k): diaria, total y fracción exportación
    zk = {}
    for g, (z, f, pa, pt, me) in enumerate(grupos):
        for k, fr in gk[g]:
            e = zk.setdefault((z, idk[k]), {"z": z, "k": idk[k], "D": np.zeros(H), "D0": np.zeros(H), "E": 0.0, "grupos": []})
            e["D"] += Dg[g] * fr
            e["D0"] += Dg0[g] * fr
            e["E"] += Dg[g].sum() * fr * me
            e["grupos"].append((g, fr))
    ex = prm.get("extra")
    if ex and ex.get("m3", 0) > 0:       # demanda adicional del validador de rutas
        key = ex["familia"] * 1000 + ex["planta"]
        if key not in idk:
            K.append(key)
            idk[key] = len(K) - 1
            planta_k.append(ex["planta"])
            fam_k.append(ex["familia"])
        De = np.zeros(H)
        De[max(0, min(H - 1, int(ex.get("t", 0))))] = ex["m3"]
        zk[(10 ** 9, idk[key])] = {"z": ex["z"], "k": idk[key], "D": De, "D0": np.zeros(H), "E": 0.0, "grupos": [],
                                   "extra": True, "forzado": ex.get("nodo")}
    pares = [p for p in sorted(zk) if zk[p]["D"].sum() > 1e-9]

    # capacidades y costos de nodos
    m3p = base["meta"]["m3_por_posicion"]
    util = prm["utilizacion_maxima"]
    cap = np.array([pn.get(n["id"], {}).get("posiciones", n["posiciones"]) * m3p * util for n in nodos])
    pos_m3 = np.array([pn.get(n["id"], {}).get("posiciones", n["posiciones"]) * m3p for n in nodos])
    man_s = np.array([pn.get(n["id"], {}).get("manejo_salida", n["manejo_salida"]) for n in nodos])
    man_e = np.array([pn.get(n["id"], {}).get("manejo_entrada", n["manejo_entrada"]) for n in nodos])
    alm = np.array([pn.get(n["id"], {}).get("almac_dia", n["almac_dia"]) for n in nodos])
    fijo = np.array([pn.get(n["id"], {}).get("costo_fijo_mes", n["costo_fijo_mes"]) for n in nodos])
    N = [i for i in range(NN) if activo[i] and (pos_m3[i] > 0 or nodos[i]["es_planta"])]

    # arcos de traslado
    tr = prm["transporte"]
    pol = prm.get("traslados", "plantas_y_observados")
    hist = set(base["arcos_hist"])
    arcos = []
    for u in N:
        for v in N:
            if u == v:
                continue
            key = f"{nodos[u]['id']}|{nodos[v]['id']}"
            ok = (pol == "todos" or nodos[u]["es_planta"] or
                  (pol == "plantas_y_observados" and (key in hist or nodos[u]["bodega_externa"] or nodos[v]["bodega_externa"])))
            if not ok:
                continue
            L = base["lanes"][key]
            s_ = L["s"] if tr.get("s_propia") is None else tr["s_propia"]
            if tr.get("metodo") == "km":
                tp = (tr["fijo_viaje"] + tr["costo_km"] * L["km"]) / (L["m3v"] or base["meta"]["m3_viaje_def"]) if L["km"] > 0 else 0.0
            else:
                tp = L["tt"] if L["tp"] is None else L["tp"]
            arcos.append({"u": u, "v": v, "L": int(math.floor(L["horas"] / 24)), "horas": L["horas"],
                          "c_terc": (1 - s_) * L["tt"] * tr["f_terc"], "c_prop": s_ * tp * tr["f_prop"],
                          "c_man": man_s[u] + man_e[v]})

    # servicio: nodos que pueden atender cada ciudad
    sv = prm["servicio"]
    lmax = sv.get("lmax_h")
    puede = {}
    for pk in pares:
        e = zk[pk]
        if e.get("extra") and e.get("forzado") is not None:
            puede[pk] = [e["forzado"]] if e["forzado"] in N else []
        else:
            puede[pk] = [n for n in N if lmax is None or base["h_out"][n][e["z"]] <= lmax]

    # inventario inicial (sin crecimiento: es el stock físico al inicio)
    inv = prm["inventario"]
    I0 = np.zeros((NN, len(K)))
    D0k = np.zeros(len(K))
    for e in zk.values():
        if not e.get("extra"):
            D0k[e["k"]] += e["D0"].sum() / H
    for k in range(len(K)):
        if activo[planta_k[k]]:
            I0[planta_k[k], k] += inv["dias_inicial_planta"] * D0k[k]
    for g in range(G):
        avg = Dg0[g].sum() / H
        for kk, fr in gk[g]:
            for n, sh in base["asign_hist"][g]:
                if n in N and not nodos[n]["bodega_externa"]:
                    I0[n, idk[kk]] += inv["dias_inicial_cedi"] * avg * fr * sh

    # el inventario inicial de la regla no puede superar la ocupación inicial máxima del nodo
    oc0 = inv.get("ocupacion_inicial_max", 0.7)
    for n in range(NN):
        lim = oc0 * pos_m3[n]
        tot0 = I0[n].sum()
        if tot0 > lim > 0:
            I0[n] *= lim / tot0
    # producción
    pr = prm["produccion"]
    work = np.array([1.0 if dow[d] < pr.get("dias_semana", 6) else 0.0 for d in dias])
    if work.sum() == 0:
        work[:] = 1.0
    Dk = np.zeros((len(K), H))
    for e in zk.values():
        if not e.get("extra"):
            Dk[e["k"]] += e["D"]
    plan = np.array([pr.get("factor", 1.0) * Dk[k].sum() / work.sum() * work for k in range(len(K))])
    plan_extra = np.zeros_like(plan)
    if ex and ex.get("m3", 0) > 0:
        plan_extra[idk[ex["familia"] * 1000 + ex["planta"]], max(0, min(H - 1, int(ex.get("t", 0))))] += ex["m3"]
    if ex and ex.get("m3", 0) > 0:      # la demanda adicional se produce en su planta el mismo día
        plan[idk[ex["familia"] * 1000 + ex["planta"]], max(0, min(H - 1, int(ex.get("t", 0))))] += ex["m3"]
    capdia = {}
    for p in set(planta_k):
        cm = nodos[p]["cap_produccion_mes"]
        if cm:
            capdia[p] = cm / (pr.get("dias_semana", 6) * 52 / 12) * work

    # asignación histórica (escenario "red actual")
    fija = {}
    if prm.get("asignacion") == "historica":
        for pk in pares:
            if zk[pk].get("extra"):
                continue
            acc = {}
            tot = 0.0
            for g, fr in zk[pk]["grupos"]:
                w8 = Dg0[g].sum() * fr
                for n, sh in base["asign_hist"][g]:
                    if n in N:
                        acc[n] = acc.get(n, 0.0) + w8 * sh
                        tot += w8 * sh
            if tot > 0:
                fija[pk] = {n: acc[n] / tot for n in sorted(acc)}
    return {"H": H, "dias": dias, "fechas": [fechas[d] for d in dias], "N": N, "K": K, "planta_k": planta_k, "fam_k": fam_k,
            "zk": zk, "pares": pares, "cap": cap, "pos_m3": pos_m3, "man_s": man_s, "man_e": man_e, "alm": alm, "fijo": fijo,
            "modo": modo, "arcos": arcos, "puede": puede, "I0": I0, "plan": plan, "plan_extra": plan_extra, "work": work, "capdia": capdia,
            "fija": fija, "gk": gk, "Dg": Dg, "nodos": nodos, "prm": prm}


# ---------------------------------------------------------------------------
# 2. Modelo de optimización
# ---------------------------------------------------------------------------
def construir(base: dict, ins: dict) -> tuple[ModeloLineal, dict]:
    prm = ins["prm"]
    H, N, K, pares = ins["H"], ins["N"], ins["K"], ins["pares"]
    nodos = ins["nodos"]
    hol = prm["holguras"]
    sv = prm["servicio"]
    tr = prm["transporte"]
    m = ModeloLineal()
    V = {"s": {}, "x": {}, "I": {}, "B": {}, "E": {}, "F": {}, "q": {}, "y": {}, "P": {}, "u": {}}
    flex = prm["produccion"].get("modo") == "flexible"
    ss = prm["inventario"].get("dias_seguridad", 0) or 0
    capcost = prm.get("costo_capital_dia", 0) or 0
    fz = tr.get("f_zona", {})
    zonas = base["zonas"]

    # variables de apertura
    for n in N:
        if ins["modo"][n] in ("opcional", "forzado"):
            fc = ins["fijo"][n] * H / 30
            lo = 1.0 if ins["modo"][n] == "forzado" else 0.0
            V["y"][n] = m.var(fc * ESCALA, lo, 1.0, entera=True)
    # asignación
    for pk in pares:
        e = ins["zk"][pk]
        z = e["z"]
        tot = float(e["D"].sum())
        fija = ins["fija"].get(pk)
        for n in (ins["puede"][pk] if not fija else list(fija)):
            h = base["h_out"][n][z]
            c = tot * (base["c_out"][n][z] * tr["f_dist"] * fz.get(zonas[z]["zona"] or "", 1.0) + ins["man_s"][n] +
                       sv.get("pen_hora", 0) * max(0.0, h - sv.get("meta_h", 1e9)))
            if fija:
                V["s"][(n, pk)] = m.var(c * ESCALA, fija[n], fija[n])
            else:
                V["s"][(n, pk)] = m.var(c * ESCALA, 0.0, 1.0)
        if hol.get("atraso") or not ins["puede"][pk]:
            V["u"][pk] = m.var(tot * hol["pen_no_atendido"] * ESCALA, 0.0, 1.0)
    # traslados
    for ai, a in enumerate(ins["arcos"]):
        c = a["c_terc"] + a["c_prop"] + a["c_man"]
        for k in range(len(K)):
            for t in range(H):
                V["x"][(ai, k, t)] = m.var(c * ESCALA)
    # inventario, atraso, stock de seguridad
    for n in N:
        for k in range(len(K)):
            for t in range(H):
                V["I"][(n, k, t)] = m.var((ins["alm"][n] + capcost) * ESCALA)
                if hol.get("atraso"):
                    V["B"][(n, k, t)] = m.var((hol["pen_no_atendido"] if t == H - 1 else hol["pen_atraso"]) * ESCALA)
        if hol.get("exceso"):
            for t in range(H):
                V["E"][(n, t)] = m.var(hol["pen_exceso"] * ESCALA)
    sirve = {}
    for (n, pk) in V["s"]:
        sirve.setdefault((n, ins["zk"][pk]["k"]), []).append(pk)
    sirve = {key: sirve[key] for key in sorted(sirve)}
    if ss > 0:
        for (n, k) in sirve:
            V["q"][(n, k)] = m.var(0.0)
            for t in range(H):
                V["F"][(n, k, t)] = m.var(hol.get("pen_stock_seguridad", 0) * ESCALA)
    if flex:
        for k in range(len(K)):
            for t in range(H):
                V["P"][(k, t)] = m.var(0.0)

    # R1 asignación completa
    for pk in pares:
        idx = [V["s"][(n, pk)] for n in N if (n, pk) in V["s"]]
        coef = [1.0] * len(idx)
        if pk in V["u"]:
            idx.append(V["u"][pk])
            coef.append(1.0)
        m.restr(idx, coef, 1.0, 1.0)
    # R2 balance diario
    ent = {}
    sal = {}
    for ai, a in enumerate(ins["arcos"]):
        ent.setdefault(a["v"], []).append(ai)
        sal.setdefault(a["u"], []).append(ai)
    for n in N:
        for k in range(len(K)):
            es_pl = ins["planta_k"][k] == n
            for t in range(H):
                idx, coef = [V["I"][(n, k, t)]], [1.0]
                if hol.get("atraso"):
                    idx.append(V["B"][(n, k, t)])
                    coef.append(-1.0)
                if t > 0:
                    idx.append(V["I"][(n, k, t - 1)])
                    coef.append(-1.0)
                    if hol.get("atraso"):
                        idx.append(V["B"][(n, k, t - 1)])
                        coef.append(1.0)
                for ai in ent.get(n, []):
                    t0 = t - ins["arcos"][ai]["L"]
                    if t0 >= 0:
                        idx.append(V["x"][(ai, k, t0)])
                        coef.append(-1.0)
                for ai in sal.get(n, []):
                    idx.append(V["x"][(ai, k, t)])
                    coef.append(1.0)
                for pk in sirve.get((n, k), []):
                    d = float(ins["zk"][pk]["D"][t])
                    if d != 0:
                        idx.append(V["s"][(n, pk)])
                        coef.append(d)
                rhs = ins["I0"][n, k] if t == 0 else 0.0
                if es_pl:
                    if flex:
                        idx.append(V["P"][(k, t)])
                        coef.append(-1.0)
                        rhs += float(ins["plan_extra"][k, t])
                    else:
                        rhs += float(ins["plan"][k, t])
                m.restr(idx, coef, rhs, rhs)
    # R3 capacidad física diaria
    for n in N:
        for t in range(H):
            idx = [V["I"][(n, k, t)] for k in range(len(K))]
            coef = [1.0] * len(idx)
            if hol.get("exceso"):
                idx.append(V["E"][(n, t)])
                coef.append(-1.0)
            if n in V["y"]:
                idx.append(V["y"][n])
                coef.append(-float(ins["cap"][n]))
                m.restr(idx, coef, hi=0.0)
            else:
                m.restr(idx, coef, hi=float(ins["cap"][n]))
    # R4 stock de seguridad
    if ss > 0:
        for (n, k), zs in sirve.items():
            idx = [V["q"][(n, k)]] + [V["s"][(n, pk)] for pk in zs]
            coef = [1.0] + [-ss * float(ins["zk"][pk]["D"].sum()) / H for pk in zs]
            m.restr(idx, coef, 0.0, 0.0)
            for t in range(H):
                m.restr([V["I"][(n, k, t)], V["F"][(n, k, t)], V["q"][(n, k)]], [1.0, 1.0, -1.0], lo=0.0)
    # R5 nodos opcionales: solo atienden / reciben si están abiertos
    for n, yi in V["y"].items():
        for (nn, pk), si in V["s"].items():
            if nn == n:
                m.restr([si, yi], [1.0, -1.0], hi=0.0)
        for t in range(H):
            idx = [V["x"][(ai, k, t)] for ai in ent.get(n, []) for k in range(len(K))]
            if idx:
                m.restr(idx + [yi], [1.0] * len(idx) + [-float(ins["cap"][n])], hi=0.0)
    # R6 inventario final de la red (incluye producto en tránsito)
    #    modo "nodo" (por defecto): cada nodo termina con la cobertura que le corresponde según la demanda que atiende
    #    en el nuevo esquema (proporcional a s); el total de la red es el mismo que en el modo "red".
    fm = prm["inventario"].get("final_min", 1.0)
    por_nodo = prm["inventario"].get("final_por_nodo", True)
    dpl, dce = prm["inventario"]["dias_inicial_planta"], prm["inventario"]["dias_inicial_cedi"]
    D0k = np.zeros(len(K))
    for pk in pares:
        if not ins["zk"][pk].get("extra"):
            D0k[ins["zk"][pk]["k"]] += float(ins["zk"][pk]["D0"].sum()) / H
    for k in range(len(K)):
        tot0 = float(ins["I0"][:, k].sum())
        W = (dpl + dce) * D0k[k]
        if not por_nodo or W <= 0:
            idx = [V["I"][(n, k, H - 1)] for n in N]
            coef = [1.0] * len(idx)
            for ai, a in enumerate(ins["arcos"]):
                for t in range(max(0, H - a["L"]), H):
                    if t + a["L"] >= H:
                        idx.append(V["x"][(ai, k, t)])
                        coef.append(1.0)
            m.restr(idx, coef, lo=fm * tot0)
            continue
        f6 = fm * tot0 / W
        for n in N:
            idx, coef = [V["I"][(n, k, H - 1)]], [1.0]
            for ai in ent.get(n, []):
                a = ins["arcos"][ai]
                for t in range(max(0, H - a["L"]), H):
                    idx.append(V["x"][(ai, k, t)])
                    coef.append(1.0)
            for pk in sirve.get((n, k), []):
                e = ins["zk"][pk]
                if not e.get("extra"):
                    idx.append(V["s"][(n, pk)])
                    coef.append(-f6 * dce * float(e["D0"].sum()) / H)
            rhs = f6 * dpl * D0k[k] if ins["planta_k"][k] == n else 0.0
            m.restr(idx, coef, lo=rhs)
    # R7 capacidad de producción (modo flexible)
    if flex:
        for p, cd in ins["capdia"].items():
            ks = [k for k in range(len(K)) if ins["planta_k"][k] == p]
            for t in range(H):
                m.restr([V["P"][(k, t)] for k in ks], [1.0] * len(ks), hi=float(cd[t]))
    # constante: costos fijos de nodos que no son opcionales
    const = sum(ins["fijo"][n] * H / 30 for n in N if n not in V["y"])
    return m, {"V": V, "const": const, "ent": ent, "sal": sal, "sirve": sirve}


def resolver(base: dict, prm: dict, tiempo=300, gap=1e-3) -> dict:
    t0 = time.time()
    ins = instancia(base, prm)
    m, meta = construir(base, ins)
    estado, fobj = m.resolver(tiempo, gap)
    res = extraer(base, ins, m, meta)
    res.update({"estado": estado, "tiempo_s": round(time.time() - t0, 1), "variables": len(m.c), "restricciones": m.n_restr,
                "objetivo": (fobj / ESCALA + meta["const"]) if not math.isnan(fobj) else None})
    return res


# ---------------------------------------------------------------------------
# 3. Resultados (idénticos a motor.js)
# ---------------------------------------------------------------------------
def _costos_vacios():
    return {c: 0.0 for c in COMP}


def totales(c: dict, H: int) -> dict:
    out = dict(c)
    for g in ("Transporte", "Almacenamiento", "Fijo nodos", "Penalizaciones"):
        out["Total " + g.lower()] = sum(v for k, v in c.items() if GRUPO.get(k) == g)
    out["Costo logístico (sin penalizaciones)"] = out["Total transporte"] + out["Total almacenamiento"] + out["Total fijo nodos"]
    out["Total"] = out["Costo logístico (sin penalizaciones)"] + out["Total penalizaciones"]
    out["Total mensual equivalente"] = out["Total"] * 30 / H
    out["Costo logístico mensual equivalente"] = out["Costo logístico (sin penalizaciones)"] * 30 / H
    return out


def extraer(base, ins, m, meta) -> dict:
    v = m.v
    V = meta["V"]
    prm = ins["prm"]
    H, N, K = ins["H"], ins["N"], ins["K"]
    nodos = ins["nodos"]
    sv = prm["servicio"]
    tr = prm["transporte"]
    hol = prm["holguras"]
    zonas = base["zonas"]
    fz = tr.get("f_zona", {})
    c = _costos_vacios()
    abiertos = {n: (v(V["y"][n]) > 0.5 if n in V["y"] else True) for n in N}
    # asignación y distribución
    asign = []
    desp = np.zeros((len(nodos), len(K), H))
    m3_meta = m3_tot = mh = 0.0
    for (n, pk), i in V["s"].items():
        sh = v(i)
        if sh < 1e-7:
            continue
        e = ins["zk"][pk]
        z, k = e["z"], e["k"]
        D = e["D"] * sh
        tot = float(D.sum())
        desp[n, k] += D
        cd = tot * base["c_out"][n][z] * tr["f_dist"] * fz.get(zonas[z]["zona"] or "", 1.0)
        h = base["h_out"][n][z]
        c["Distribución"] += cd
        c["Manejo"] += tot * ins["man_s"][n]
        c["Penalización servicio"] += tot * sv.get("pen_hora", 0) * max(0.0, h - sv.get("meta_h", 1e9))
        m3_tot += tot
        mh += tot * h
        if h <= sv.get("meta_h", 1e9):
            m3_meta += tot
        asign.append([n, z, k, round(sh, 6), tot, cd, h, e["E"] * sh / max(e["D"].sum(), 1e-9) * tot,
                      1 if e.get("extra") else 0, ("X" if e.get("extra") else str(pk[0])) + "|" + str(k)])
    no_at = 0.0
    for pk, i in V["u"].items():
        val = v(i)
        if val > 1e-7:
            q = float(ins["zk"][pk]["D"].sum()) * val
            no_at += q
            c["Penalización no atendido"] += q * hol["pen_no_atendido"]
    # traslados
    tras = {}
    llega = np.zeros((len(nodos), len(K), H))
    sale = np.zeros((len(nodos), len(K), H))
    xdia = {}
    for (ai, k, t), i in V["x"].items():
        q = v(i)
        if q < 1e-7:
            continue
        a = ins["arcos"][ai]
        sale[a["u"], k, t] += q
        if t + a["L"] < H:
            llega[a["v"], k, t + a["L"]] += q
        c["Traslado XI terceros"] += q * a["c_terc"]
        c["Traslado XI flota propia"] += q * a["c_prop"]
        c["Manejo"] += q * a["c_man"]
        r = tras.setdefault((a["u"], a["v"], k), [a["u"], a["v"], k, 0.0, 0.0, 0.0, a["L"], a["horas"]])
        r[3] += q
        r[4] += q * a["c_terc"]
        r[5] += q * a["c_prop"]
        xdia.setdefault((a["u"], a["v"]), np.zeros(H))[t] += q
    # inventarios por nodo y día
    inv = np.zeros((len(nodos), len(K), H))
    bkl = np.zeros((len(nodos), len(K), H))
    for (n, k, t), i in V["I"].items():
        inv[n, k, t] = v(i)
    for (n, k, t), i in V["B"].items():
        bkl[n, k, t] = v(i)
    prod = np.zeros((len(K), H))
    for k in range(len(K)):
        prod[k] = [v(V["P"][(k, t)]) + ins["plan_extra"][k, t] for t in range(H)] if V["P"] else ins["plan"][k]
    cap_alm = (prm.get("costo_capital_dia", 0) or 0)
    nodos_dia = {}
    for n in N:
        I = inv[n].sum(0)
        Ii = np.concatenate([[ins["I0"][n].sum()], I[:-1]])
        pr = np.array([prod[k] for k in range(len(K)) if ins["planta_k"][k] == n]).sum(0) if any(
            ins["planta_k"][k] == n for k in range(len(K))) else np.zeros(H)
        E = np.array([v(V["E"][(n, t)]) for t in range(H)]) if V["E"] else np.zeros(H)
        nodos_dia[nodos[n]["id"]] = {
            "inv_ini": Ii.tolist(), "produccion": pr.tolist(), "entradas": llega[n].sum(0).tolist(), "salidas_tr": sale[n].sum(0).tolist(),
            "despachos": desp[n].sum(0).tolist(), "inv_fin": I.tolist(), "atraso": bkl[n].sum(0).tolist(),
            "cap": float(ins["cap"][n]), "pos_m3": float(ins["pos_m3"][n]),
            "ocupacion": (I / ins["pos_m3"][n]).tolist() if ins["pos_m3"][n] > 0 else [0.0] * H,
            "exceso": E.tolist(), "abierto": abiertos[n],
            "costo_almac": (I * (ins["alm"][n] + cap_alm)).tolist(),
            "costo_manejo": (desp[n].sum(0) * ins["man_s"][n]).tolist()}
        c["Almacenaje"] += float(I.sum() * ins["alm"][n])
        c["Costo financiero inventario"] += float(I.sum() * cap_alm)
        c["Penalización exceso capacidad"] += float(E.sum() * hol.get("pen_exceso", 0))
        b = bkl[n]
        c["Penalización atraso"] += float(b[:, :H - 1].sum() * hol.get("pen_atraso", 0)) if H > 1 else 0.0
        c["Penalización no atendido"] += float(b[:, H - 1].sum() * hol.get("pen_no_atendido", 0))
        fc = ins["fijo"][n] * H / 30 if abiertos[n] else 0.0
        if nodos[n]["bodega_externa"]:
            c["Fijo bodegas externas"] += fc
        else:
            c["Fijo nodos (plantas y CEDI)"] += fc
    for (n, k, t), i in V["F"].items():
        c["Penalización stock seguridad"] += v(i) * hol.get("pen_stock_seguridad", 0)
    # balance (Caso A): residuo máximo de la ecuación de conservación
    res_bal = 0.0
    for n in N:
        for k in range(len(K)):
            prev = ins["I0"][n, k]
            prevb = 0.0
            for t in range(H):
                entra = llega[n, k, t] + (prod[k, t] if ins["planta_k"][k] == n else 0.0)
                r = (inv[n, k, t] - bkl[n, k, t]) - (prev - prevb + entra - sale[n, k, t] - desp[n, k, t])
                res_bal = max(res_bal, abs(r))
                prev, prevb = inv[n, k, t], bkl[n, k, t]
    produccion = {}
    for k in range(len(K)):
        p = nodos[ins["planta_k"][k]]["id"]
        produccion.setdefault(p, np.zeros(H))
        produccion[p] = produccion[p] + prod[k]
    cap_prod = {nodos[p]["id"]: cd.tolist() for p, cd in ins["capdia"].items()}
    atraso_m3dia = float(bkl[:, :, :H - 1].sum()) if H > 1 else 0.0
    fin_bkl = float(bkl[:, :, H - 1].sum())
    tot_dem = float(sum(e["D"].sum() for e in ins["zk"].values()))
    kpis = {"demanda_m3": tot_dem, "atendido_m3": m3_tot, "no_atendido_m3": no_at + fin_bkl, "atraso_m3_dia": atraso_m3dia,
            "fill_rate": 1 - (no_at + fin_bkl) / tot_dem if tot_dem else 1.0,
            "pct_en_meta": m3_meta / m3_tot if m3_tot else 0.0, "horas_promedio": mh / m3_tot if m3_tot else 0.0,
            "traslados_m3": float(sum(r[3] for r in tras.values())),
            "exceso_max_m3": max([max(d["exceso"]) for d in nodos_dia.values()] + [0.0]),
            "residuo_balance": res_bal, "bodegas_abiertas": [nodos[n]["id"] for n in V["y"] if abiertos[n]]}
    diag = diagnostico(ins, nodos_dia, kpis, produccion, cap_prod, V, v)
    return {"H": H, "fechas": ins["fechas"], "K": [[ins["fam_k"][k], ins["planta_k"][k]] for k in range(len(K))],
            "costos": totales(c, H), "kpis": kpis, "nodos_dia": nodos_dia, "asignacion": asign,
            "traslados": [list(r) for r in tras.values()], "traslados_dia": {f"{nodos[u]['id']}|{nodos[w]['id']}": a.tolist()
                                                                            for (u, w), a in xdia.items()},
            "produccion": {p: a.tolist() for p, a in produccion.items()}, "cap_produccion": cap_prod,
            "abiertos": {nodos[n]["id"]: abiertos[n] for n in N}, "diagnostico": diag,
            "demanda_dia": np.array([e["D"] for e in ins["zk"].values() if not e.get("extra")]).sum(0).tolist(), "tipo": "optimizado"}


def diagnostico(ins, nodos_dia, kpis, produccion, cap_prod, V, v) -> list:
    out = []
    for nid, d in nodos_dia.items():
        ex = np.array(d["exceso"])
        if ex.max() > 0.5:
            out.append({"tipo": "Capacidad", "nodo": nid, "detalle": f"{nid} supera su capacidad {int((ex > 0.5).sum())} días "
                        f"(máximo {ex.max():,.0f} m3). Falta capacidad de almacenamiento: amplíe posiciones o habilite una bodega externa."})
        oc = np.array(d["inv_fin"]) / d["cap"] if d["cap"] > 0 else np.zeros(1)
        dias = int((oc > 0.98).sum())
        if dias and ex.max() <= 0.5:
            out.append({"tipo": "Restricción activa", "nodo": nid, "detalle": f"{nid} opera al límite de capacidad {dias} días: "
                        "la capacidad está condicionando la solución."})
        at = np.array(d["atraso"])
        if at.max() > 0.5:
            out.append({"tipo": "Servicio", "nodo": nid, "detalle": f"{nid} acumula atrasos (máx. {at.max():,.0f} m3): falta producto "
                        "disponible a tiempo (producción, inventario inicial o tiempos de tránsito)."})
    for p, a in produccion.items():
        if p in cap_prod:
            ex = np.array(a) - np.array(cap_prod[p])
            if ex.max() > 0.5:
                out.append({"tipo": "Producción", "nodo": p, "detalle": f"El plan de {p} supera su capacidad diaria {int((ex > 0.5).sum())} "
                            f"días (máx. +{ex.max():,.0f} m3/día). Revise el reparto de producción o use el modo flexible."})
    sin = [1 for pk in ins["pares"] if not ins["puede"][pk]]
    if sin:
        out.append({"tipo": "Servicio", "nodo": "-", "detalle": f"{len(sin)} combinaciones ciudad-producto no tienen ningún nodo dentro "
                    "del tiempo máximo de entrega: la restricción de servicio las hace infactibles (quedan como no atendidas)."})
    if kpis["no_atendido_m3"] > 0.5:
        out.append({"tipo": "Servicio", "nodo": "-", "detalle": f"{kpis['no_atendido_m3']:,.0f} m3 quedan sin atender al final del horizonte."})
    return out


# ---------------------------------------------------------------------------
# 4. Escenario ACTUAL: reconstrucción de la operación observada (no es optimización)
# ---------------------------------------------------------------------------
def reconstruir_actual(base: dict, prm: dict) -> dict:
    """Flujos observados en SAP (despachos y traslados XI) + producción estimada (tasa estable que equilibra las
    salidas de cada planta). El inventario inicial es el mayor entre la regla de días de inventario y el mínimo
    físicamente necesario para que la secuencia observada no genere inventario negativo."""
    p = dict(prm)
    p["planta_cerrada"] = None
    p["asignacion"] = "libre"
    ins = instancia(base, p)
    nodos = base["nodos"]
    NN = len(nodos)
    H, dias = ins["H"], ins["dias"]
    d0 = dias[0]
    F = len(base["familias"])
    KK = F * 1000 + NN
    # productos AS-IS: (familia, planta actual)
    keys = sorted({g[1] * 1000 + g[2] for g in base["grupos"]} | {r[3] * 1000 + r[4] for r in base["obs_xi"]})
    idk = {k: i for i, k in enumerate(keys)}
    nk = len(keys)
    desp = np.zeros((NN, nk, H))
    ent = np.zeros((NN, nk, H))
    sal = np.zeros((NN, nk, H))
    tr = prm["transporte"]
    c = _costos_vacios()
    m3_meta = m3_tot = mh = 0.0
    sv = prm["servicio"]
    zonas = base["zonas"]
    fz = tr.get("f_zona", {})
    flujos_cli = []
    for dia, n, g, m3 in base["obs_desp"]:
        t = dia - d0
        if not (0 <= t < H):
            continue
        z, f, pa, pt, me = base["grupos"][g]
        k = idk[f * 1000 + pa]
        desp[n, k, t] += m3
        cd = m3 * base["c_out"][n][z] * tr["f_dist"] * fz.get(zonas[z]["zona"] or "", 1.0)
        c["Distribución"] += cd
        c["Manejo"] += m3 * ins["man_s"][n]
        h = base["h_out"][n][z]
        m3_tot += m3
        mh += m3 * h
        if h <= sv.get("meta_h", 1e9):
            m3_meta += m3
        flujos_cli.append((n, g, m3, cd))
    ficticio = prm.get("traslado_ficticio_mismo_codigo", True)
    tras = {}
    for dia, u, w, f, pa, m3, costo, propia in base["obs_xi"]:
        t = dia - d0
        if not (0 <= t < H) or (ficticio and u == w):
            continue
        k = idk[f * 1000 + pa]
        sal[u, k, t] += m3
        L = int(math.floor(base["lanes"][f"{nodos[u]['id']}|{nodos[w]['id']}"]["horas"] / 24)) if w >= 0 else 0
        if w >= 0 and t + L < H:
            ent[w, k, t + L] += m3
        if propia:
            c["Traslado XI flota propia"] += costo
        else:
            c["Traslado XI terceros"] += costo
        c["Manejo"] += m3 * (ins["man_s"][u] + (ins["man_e"][w] if w >= 0 else 0.0))
        r = tras.setdefault((u, w, k), [u, w, k, 0.0, 0.0, 0.0, L, base["lanes"][f"{nodos[u]['id']}|{nodos[w]['id']}"]["horas"] if w >= 0 else 0])
        r[3] += m3
        r[4 if not propia else 5] += costo
    # producción estimada: tasa estable en días hábiles que equilibra las salidas netas de la planta
    work = ins["work"]
    prod = np.zeros((nk, H))
    for k in keys:
        pl = k % 1000
        kk = idk[k]
        neto = desp[pl, kk].sum() + sal[pl, kk].sum() - ent[pl, kk].sum()
        prod[kk] = max(0.0, neto) / work.sum() * work
    # inventario inicial: regla de días vs mínimo físico
    inv = prm["inventario"]
    avg_d = desp.sum(2) / H
    avg_tot = np.zeros(nk)
    for k in keys:
        avg_tot[idk[k]] = desp[:, idk[k]].sum() / H
    REGLA = np.zeros((NN, nk))
    for n in range(NN):
        for kk in range(nk):
            REGLA[n, kk] = inv["dias_inicial_cedi"] * avg_d[n, kk] + (inv["dias_inicial_planta"] * avg_tot[kk] if keys[kk] % 1000 == n else 0.0)
        lim = inv.get("ocupacion_inicial_max", 0.7) * ins["pos_m3"][n]
        if REGLA[n].sum() > lim > 0:
            REGLA[n] *= lim / REGLA[n].sum()
    I0 = np.zeros((NN, nk))
    nodos_dia = {}
    I0_min = {}
    for n in range(NN):
        es_pl = nodos[n]["es_planta"]
        for kk in range(nk):
            regla = REGLA[n, kk]
            neto = (prod[kk] if keys[kk] % 1000 == n else 0.0) + ent[n, kk] - sal[n, kk] - desp[n, kk]
            minimo = max(0.0, -float(np.cumsum(neto).min())) if H else 0.0
            I0[n, kk] = max(regla, minimo)
            if minimo > regla + 1e-6:
                I0_min[nodos[n]["id"]] = I0_min.get(nodos[n]["id"], 0.0) + minimo - regla
    m3p = base["meta"]["m3_por_posicion"]
    for n in range(NN):
        flujo = desp[n].sum() + sal[n].sum() + ent[n].sum()
        if flujo <= 0 and not nodos[n]["es_planta"] and nodos[n]["tipo"] == "candidata":
            continue
        if nodos[n]["tipo"] == "candidata":
            continue
        pr = np.array([prod[kk] for kk in range(nk) if keys[kk] % 1000 == n]).sum(0) if any(keys[kk] % 1000 == n for kk in range(nk)) else np.zeros(H)
        I = I0[n].sum() + np.cumsum(pr + ent[n].sum(0) - sal[n].sum(0) - desp[n].sum(0))
        Ii = np.concatenate([[I0[n].sum()], I[:-1]])
        capn = ins["cap"][n]
        pos = ins["pos_m3"][n]
        nodos_dia[nodos[n]["id"]] = {
            "inv_ini": Ii.tolist(), "produccion": pr.tolist(), "entradas": ent[n].sum(0).tolist(), "salidas_tr": sal[n].sum(0).tolist(),
            "despachos": desp[n].sum(0).tolist(), "inv_fin": I.tolist(), "atraso": [0.0] * H, "cap": float(capn), "pos_m3": float(pos),
            "ocupacion": (I / pos).tolist() if pos > 0 else [0.0] * H, "exceso": np.maximum(0, I - capn).tolist(), "abierto": True,
            "costo_almac": (I * (ins["alm"][n] + (prm.get("costo_capital_dia", 0) or 0))).tolist(),
            "costo_manejo": (desp[n].sum(0) * ins["man_s"][n]).tolist()}
        c["Almacenaje"] += float(I.sum() * ins["alm"][n])
        c["Costo financiero inventario"] += float(I.sum() * (prm.get("costo_capital_dia", 0) or 0))
        fc = ins["fijo"][n] * H / 30
        if nodos[n]["bodega_externa"]:
            c["Fijo bodegas externas"] += fc
        else:
            c["Fijo nodos (plantas y CEDI)"] += fc
    produccion = {}
    for kk, k in enumerate(keys):
        p_ = nodos[k % 1000]["id"]
        produccion[p_] = (np.array(produccion.get(p_, np.zeros(H))) + prod[kk]).tolist()
    exceso_max = max([max(d["exceso"]) for d in nodos_dia.values()] + [0.0])
    tot = desp.sum()
    kpis = {"demanda_m3": float(tot), "atendido_m3": float(tot), "no_atendido_m3": 0.0, "atraso_m3_dia": 0.0, "fill_rate": 1.0,
            "pct_en_meta": m3_meta / m3_tot if m3_tot else 0.0, "horas_promedio": mh / m3_tot if m3_tot else 0.0,
            "traslados_m3": float(sal.sum()), "exceso_max_m3": exceso_max, "residuo_balance": 0.0,
            "bodegas_abiertas": [n["id"] for n in nodos if n["tipo"] == "bodega_externa"],
            "inventario_inicial_ajustado": I0_min}
    diag = [{"tipo": "Dato estimado", "nodo": "-", "detalle": "La producción diaria no existe en los datos: se estima como una tasa estable "
             "en días hábiles que equilibra las salidas observadas de cada planta."}]
    for nid, q in I0_min.items():
        diag.append({"tipo": "Dato estimado", "nodo": nid, "detalle": f"La secuencia observada en {nid} requiere {q:,.0f} m3 más de inventario "
                     "inicial que la regla de días para no ser negativa; se usó el mínimo físico."})
    for nid, d in nodos_dia.items():
        ex = np.array(d["exceso"])
        if ex.max() > 0.5:
            diag.append({"tipo": "Capacidad", "nodo": nid, "detalle": f"Con la operación observada {nid} supera su capacidad estimada "
                         f"{int((ex > 0.5).sum())} días (máx. {ex.max():,.0f} m3)."})
    asign = {}
    for n, g, m3, cd in flujos_cli:
        z = base["grupos"][g][0]
        r = asign.setdefault((n, z, g), [n, z, g, 1.0, 0.0, 0.0, base["h_out"][n][z], 0.0])
        r[4] += m3
        r[5] += cd
        r[7] += m3 * base["grupos"][g][4]
    return {"H": H, "fechas": ins["fechas"], "K": [[k // 1000, k % 1000] for k in keys], "costos": totales(c, H), "kpis": kpis,
            "nodos_dia": nodos_dia, "asignacion_grupos": [list(r) for r in asign.values()],
            "traslados": [list(r) for r in tras.values()], "produccion": produccion, "cap_produccion": {},
            "abiertos": {nid: True for nid in nodos_dia}, "diagnostico": diag, "estado": "Observado (reconstruido)",
            "demanda_dia": desp.sum((0, 1)).tolist(), "tipo": "historico", "objetivo": None, "tiempo_s": 0, "variables": 0,
            "restricciones": 0}
