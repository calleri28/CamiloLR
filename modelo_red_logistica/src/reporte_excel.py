"""Genera el Excel consolidado del modelo diario: comparación de escenarios, series diarias por nodo, asignación, traslados,
combinaciones de bodegas, pruebas A-H, calidad de datos y un validador de rutas con fórmulas vivas."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name

from .modelo_diario import COMP


def _sin_inf(df: pd.DataFrame) -> pd.DataFrame:
    d2 = df.copy()
    num = d2.select_dtypes(include=[np.number]).columns
    d2[num] = d2[num].astype(float).where(np.isfinite(d2[num].astype(float)))
    return d2


class _Libro:
    def __init__(self, ruta: Path):
        b = xlsxwriter.Workbook(str(ruta), {"nan_inf_to_errors": True})
        self.b = b
        self.f = {
            "titulo": b.add_format({"bold": True, "font_size": 16, "font_color": "#184f95"}),
            "sub": b.add_format({"italic": True, "font_color": "#52514e"}),
            "h": b.add_format({"bold": True, "bg_color": "#184f95", "font_color": "#ffffff", "border": 1,
                               "text_wrap": True, "valign": "vcenter"}),
            "input": b.add_format({"bg_color": "#fff2b3", "border": 1, "num_format": "#,##0.00"}),
            "input_txt": b.add_format({"bg_color": "#fff2b3", "border": 1}),
            "input_pct": b.add_format({"bg_color": "#fff2b3", "border": 1, "num_format": "0.0%"}),
            "input_cop": b.add_format({"bg_color": "#fff2b3", "border": 1, "num_format": "#,##0"}),
            "cop": b.add_format({"num_format": "#,##0", "border": 1}),
            "num": b.add_format({"num_format": "#,##0.0", "border": 1}),
            "pct": b.add_format({"num_format": "0.0%", "border": 1}),
            "txt": b.add_format({"border": 1}),
            "bold": b.add_format({"bold": True}),
            "res": b.add_format({"bold": True, "bg_color": "#cde2fb", "border": 2, "num_format": "#,##0", "font_size": 12}),
            "res_txt": b.add_format({"bold": True, "bg_color": "#cde2fb", "border": 2, "font_size": 12}),
            "wrap": b.add_format({"text_wrap": True, "valign": "top"}),
            "mcop": b.add_format({"num_format": "#,##0", "border": 1}),
        }

    def tabla(self, df: pd.DataFrame, hoja: str, titulo: str | None = None, fila: int = 0, formatos: dict | None = None,
              anchos: dict | None = None):
        df = df.copy()
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].apply(lambda v: ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else v)
            if str(df[c].dtype) in ("string", "Float64", "Int64", "boolean"):
                df[c] = df[c].astype(object).where(df[c].notna(), None)
        ws = self.b.get_worksheet_by_name(hoja) or self.b.add_worksheet(hoja)
        r0 = fila
        if titulo:
            ws.write(r0, 0, titulo, self.f["titulo"])
            r0 += 2
        if len(df) == 0:
            ws.write(r0, 0, "(sin datos)")
            return ws
        d2 = _sin_inf(df)
        datos = d2.astype(object).where(d2.notna(), None).values.tolist()
        cols = []
        for c in df.columns:
            fmt = (formatos or {}).get(c)
            if fmt is None:
                if pd.api.types.is_bool_dtype(df[c]) or df[c].dtype == object:
                    fmt = self.f["txt"]
                elif pd.api.types.is_numeric_dtype(df[c]):
                    fmt = self.f["cop"] if d2[c].abs().max() > 1000 else self.f["num"]
                else:
                    fmt = self.f["txt"]
            cols.append({"header": str(c), "format": fmt})
        ws.add_table(r0, 0, r0 + len(df), len(df.columns) - 1,
                     {"data": datos, "columns": cols, "style": "Table Style Medium 2", "autofilter": True})
        for j, c in enumerate(df.columns):
            w = (anchos or {}).get(c, min(max(len(str(c)), 10) + 2, 45))
            ws.set_column(j, j, w)
        ws.freeze_panes(r0 + 1, 0)
        return ws

    def cerrar(self):
        self.b.close()



FILAS_COSTO = COMP + ["Total transporte", "Total almacenamiento", "Total fijo nodos", "Total penalizaciones",
                      "Costo logístico (sin penalizaciones)", "Total"]


def _comparacion(R: dict, orden: list) -> pd.DataFrame:
    filas = []
    for k in FILAS_COSTO:
        filas.append({"Concepto": k, "Unidad": "MCOP/mes", **{R[s]["nombre"]: R[s]["costos"].get(k, 0) * 30 / R[s]["H"] / 1e6 for s in orden}})
    kp = [("fill_rate", "Demanda atendida", "%"), ("pct_en_meta", "Volumen en meta de entrega", "%"), ("horas_promedio", "Horas promedio de entrega", "h"),
          ("traslados_m3", "Traslados XI", "m3/día"), ("exceso_max_m3", "Exceso máximo de capacidad", "m3"), ("no_atendido_m3", "No atendido", "m3")]
    for k, nom, u in kp:
        filas.append({"Concepto": nom, "Unidad": u, **{R[s]["nombre"]: (R[s]["kpis"].get(k, 0) / R[s]["H"] if u == "m3/día" else R[s]["kpis"].get(k, 0))
                                                       for s in orden}})
    filas.append({"Concepto": "Bodegas externas abiertas", "Unidad": "", **{R[s]["nombre"]: ", ".join(R[s]["kpis"].get("bodegas_abiertas", [])) or "-" for s in orden}})
    filas.append({"Concepto": "Tipo de resultado", "Unidad": "", **{R[s]["nombre"]: "Histórico + estimado" if R[s].get("tipo") == "historico" else "Optimizado"
                                                                    for s in orden}})
    filas.append({"Concepto": "Estado del solver", "Unidad": "", **{R[s]["nombre"]: R[s].get("estado") for s in orden}})
    return pd.DataFrame(filas)


def _nodos_dia(R, orden):
    filas = []
    cols = ["inv_ini", "produccion", "entradas", "salidas_tr", "despachos", "inv_fin", "atraso", "ocupacion", "exceso", "costo_almac", "costo_manejo"]
    for s in orden:
        r = R[s]
        for n, d in r["nodos_dia"].items():
            for t, fe in enumerate(r["fechas"]):
                fila = {"escenario": r["nombre"], "nodo": n, "fecha": fe}
                for k in cols:
                    if isinstance(d.get(k), list):
                        fila[k] = d[k][t]
                fila["capacidad_m3"], fila["posiciones_m3"] = d["cap"], d["pos_m3"]
                filas.append(fila)
    return pd.DataFrame(filas)


def _asignacion(base, R, orden):
    N, Z, F = base["nodos"], base["zonas"], base["familias"]
    filas = []
    for s in orden:
        r = R[s]
        if r.get("tipo") == "historico":
            for a in r.get("asignacion_grupos", []):
                g = base["grupos"][a[2]]
                filas.append([r["nombre"], N[a[0]]["id"], Z[g[0]]["id"], Z[g[0]]["zona"], F[g[1]], N[g[2]]["id"], a[4], a[5], a[6]])
        else:
            for a in r.get("asignacion", []):
                if a[8]:
                    continue
                fa, pl = r["K"][a[2]]
                filas.append([r["nombre"], N[a[0]]["id"], Z[a[1]]["id"], Z[a[1]]["zona"], F[fa], N[pl]["id"], a[4], a[5], a[6]])
    df = pd.DataFrame(filas, columns=["escenario", "nodo_despacho", "ciudad", "zona", "familia", "planta", "m3_horizonte", "costo_distribucion_cop", "horas"])
    df["mh"] = df.m3_horizonte * df.horas
    g = df.groupby(["escenario", "nodo_despacho", "ciudad", "zona", "familia", "planta"], as_index=False, sort=False, dropna=False).agg(
        m3_horizonte=("m3_horizonte", "sum"), costo_distribucion_cop=("costo_distribucion_cop", "sum"), mh=("mh", "sum"))
    g["horas"] = g.mh / g.m3_horizonte.where(g.m3_horizonte > 0)
    return g.drop(columns="mh")


def _traslados(base, R, orden):
    N, F = base["nodos"], base["familias"]
    filas = []
    for s in orden:
        r = R[s]
        for x in r.get("traslados", []):
            fa, pl = r["K"][x[2]]
            filas.append([r["nombre"], N[x[0]]["id"], N[x[1]]["id"] if x[1] >= 0 else "Externo", F[fa], N[pl]["id"], x[3], x[4], x[5], x[6], x[7]])
    return pd.DataFrame(filas, columns=["escenario", "origen", "destino", "familia", "planta", "m3_horizonte", "costo_terceros_cop",
                                        "costo_flota_propia_cop", "transito_dias", "horas"])


def _parametros(prm, pref=""):
    out = []
    for k, v in prm.items():
        if k == "nodos":
            for n, d in v.items():
                out.append({"parámetro": f"nodos.{n}", "valor": ", ".join(f"{a}={b}" for a, b in d.items())})
        elif isinstance(v, dict):
            out += _parametros(v, pref + k + ".")
        else:
            out.append({"parámetro": pref + k, "valor": "" if v is None else (str(v) if isinstance(v, (list, dict, bool)) else v)})
    return out


def _validador(lb, base: dict, prm: dict):
    """Hoja con fórmulas: costo variable de atender V m3 en una ciudad desde una planta, directo o vía cada nodo."""
    b, f = lb.b, lb.f
    N, Z = base["nodos"], base["zonas"]
    tr = prm["transporte"]
    ids = [n["id"] for n in N]
    ciud = [z["id"] for z in Z]
    nN, nZ = len(ids), len(ciud)
    wt = b.add_worksheet("Aux_tarifas")

    def bloque(r0, titulo, cols, valor):
        wt.write(r0, 0, titulo, f["bold"])
        for j, c in enumerate(cols):
            wt.write(r0 + 1, j + 1, c, f["h"])
        for i, u in enumerate(ids):
            wt.write(r0 + 2 + i, 0, u, f["h"])
            for j in range(len(cols)):
                v = valor(i, j)
                if v is None:
                    wt.write_blank(r0 + 2 + i, j + 1, None)
                else:
                    wt.write_number(r0 + 2 + i, j + 1, round(float(v), 2))
        last = xl_col_to_name(len(cols))
        return (f"Aux_tarifas!$B${r0 + 3}:${last}${r0 + 2 + nN}", f"Aux_tarifas!$A${r0 + 3}:$A${r0 + 2 + nN}",
                f"Aux_tarifas!$B${r0 + 2}:${last}${r0 + 2}")

    def xi(i, j):
        if i == j:
            return 0.0
        L = base["lanes"].get(f"{ids[i]}|{ids[j]}")
        if not L:
            return None
        s = L["s"] if tr.get("s_propia") is None else tr["s_propia"]
        if tr.get("metodo") == "km":
            tp = (tr["fijo_viaje"] + tr["costo_km"] * L["km"]) / (L["m3v"] or base["meta"]["m3_viaje_def"]) if L["km"] else 0.0
        else:
            tp = L["tt"] if L["tp"] is None else L["tp"]
        return (1 - s) * L["tt"] * tr["f_terc"] + s * tp * tr["f_prop"]

    def hx(i, j):
        L = base["lanes"].get(f"{ids[i]}|{ids[j]}")
        return 0.0 if i == j else (L["horas"] if L else 999)

    XI, XIr, XIc = bloque(0, "Tarifa XI ponderada terceros/flota propia (COP/m3), origen fila → destino columna", ids, xi)
    HX, HXr, HXc = bloque(nN + 3, "Horas de tránsito XI (999 = sin ruta registrada)", ids, hx)
    DI, DIr, DIc = bloque(2 * (nN + 3), "Tarifa de distribución (COP/m3), nodo fila → ciudad columna", ciud, lambda i, j: base["c_out"][i][j] * tr["f_dist"])
    DH, DHr, DHc = bloque(3 * (nN + 3), "Horas de distribución, nodo fila → ciudad columna", ciud, lambda i, j: base["h_out"][i][j])
    wt.hide()

    ws = b.add_worksheet("Validador_rutas")
    ws.write(0, 0, "Validador de rutas (fórmulas vivas)", f["titulo"])
    ws.write(1, 0, "Cambie las celdas amarillas. Costo variable de atender el volumen; no incluye costos fijos ni el efecto de red "
                   "(el validador del HTML re-optimiza para obtener el costo marginal exacto).", f["sub"])
    entradas = [("Planta de origen", "CO01", "input_txt"), ("Ciudad destino", "MEDELLIN", "input_txt"), ("Volumen (m3)", 10, "input"),
                ("Días en almacén intermedio", 1, "input"), ("Costo financiero (COP/m3·día)", prm.get("costo_capital_dia") or 0, "input_cop")]
    for i, (t, v, fm) in enumerate(entradas):
        ws.write(3 + i, 0, t, f["bold"])
        ws.write(3 + i, 1, v, f[fm])
    ws.data_validation(3, 1, 3, 1, {"validate": "list", "source": [n["id"] for n in N if n["es_planta"]]})
    ws.data_validation(4, 1, 4, 1, {"validate": "list", "source": "=" + DIc})
    O, Zc, V, Dd, Cf = "$B$4", "$B$5", "$B$6", "$B$7", "$B$8"
    hdr = ["Vía nodo", "Manejo salida COP/m3", "Manejo entrada COP/m3", "Almacenaje COP/m3·día", "Flete XI tramo 1", "Manejo tramo 1",
           "Almacenaje intermedio", "Flete distribución tramo 2", "Manejo tramo 2", "Costo variable total", "COP/m3", "Horas totales",
           "Diferencia vs directo"]
    for j, h in enumerate(hdr):
        ws.write(10, j, h, f["h"])
    r1, r2 = 12, 11 + nN
    col = lambda c: f"${c}${r1}:${c}${r2}"
    for i, n in enumerate(N):
        r = 11 + i
        p = prm["nodos"].get(n["id"], {})
        ws.write(r, 0, n["id"], f["txt"])
        ws.write_number(r, 1, p.get("manejo_salida", n["manejo_salida"]), f["input_cop"])
        ws.write_number(r, 2, p.get("manejo_entrada", n["manejo_entrada"]), f["input_cop"])
        ws.write_number(r, 3, p.get("almac_dia", n["almac_dia"]), f["input_cop"])
        A, R_ = f"$A{r + 1}", r + 1
        dif = f"{A}<>{O}"
        ms_o = f"INDEX({col('B')},MATCH({O},{col('A')},0))"
        ws.write_formula(r, 4, f"=IF({dif},{V}*INDEX({XI},MATCH({O},{XIr},0),MATCH({A},{XIc},0)),0)", f["cop"])
        ws.write_formula(r, 5, f"=IF({dif},{V}*({ms_o}+C{R_}),0)", f["cop"])
        ws.write_formula(r, 6, f"=IF({dif},{V}*{Dd}*(D{R_}+{Cf}),0)", f["cop"])
        ws.write_formula(r, 7, f"={V}*INDEX({DI},MATCH({A},{DIr},0),MATCH({Zc},{DIc},0))", f["cop"])
        ws.write_formula(r, 8, f"={V}*B{R_}", f["cop"])
        ws.write_formula(r, 9, f"=SUM(E{R_}:I{R_})", f["res"])
        ws.write_formula(r, 10, f"=J{R_}/{V}", f["cop"])
        ws.write_formula(r, 11, f"=IF({dif},INDEX({HX},MATCH({O},{HXr},0),MATCH({A},{HXc},0)),0)+INDEX({DH},MATCH({A},{DHr},0),MATCH({Zc},{DHc},0))", f["num"])
        ws.write_formula(r, 12, f"=J{R_}-INDEX({col('J')},MATCH({O},{col('A')},0))", f["cop"])
    ws.set_column(0, 0, 30)
    ws.set_column(1, 12, 17)
    ws.write(13 + nN, 0, "Horas ≥ 999 = ruta XI sin registro (no factible sin habilitar la ruta). La tarifa XI pondera terceros y flota propia imputada "
                         "con la participación histórica de la ruta y los factores del escenario optimizado.", f["sub"])


def generar_excel(ruta: Path, base: dict, cfg: dict, corrida: dict, combos: list, pruebas: list, calidad: pd.DataFrame,
                  resumen: dict, viajes_xi: list):
    from .reporte_html import ORDEN
    R, P = corrida["resultados"], corrida["parametros"]
    orden = [k for k in ORDEN if k in R] + [k for k in R if k not in ORDEN]
    lb = _Libro(ruta)
    b, f = lb.b, lb.f
    ws = b.add_worksheet("Resumen")
    ws.write(0, 0, f"Red logística · cierre de planta {base['meta']['planta_cerrada']} · modelo diario", f["titulo"])
    ws.write(1, 0, "Costos en millones de COP por mes equivalente (horizonte × 30 / días). Las penalizaciones miden incumplimientos; no son costos contables.", f["sub"])
    ws.write(3, 0, "Recomendación", f["bold"])
    r = 4
    for t in resumen["reco"]:
        ws.write(r, 0, "• " + t, f["wrap"])
        ws.set_row(r, 36)
        r += 1
    ws.write(r + 1, 0, "Hallazgos", f["bold"])
    r += 2
    for t in resumen["hallazgos"]:
        ws.write(r, 0, "• " + t, f["wrap"])
        ws.set_row(r, 32)
        r += 1
    ws.set_column(0, 0, 160)
    lb.tabla(_comparacion(R, orden), "Comparacion", "Comparación de escenarios (misma demanda), MCOP por mes equivalente")
    if combos:
        lb.tabla(pd.DataFrame(combos), "Combinaciones_BE", "Bodegas externas: todas las combinaciones (cada fila es una optimización completa)")
    lb.tabla(_nodos_dia(R, orden), "Nodos_diario", "Inventario, producción, traslados, despachos y ocupación por nodo y día")
    lb.tabla(_asignacion(base, R, orden), "Asignacion", "Desde qué nodo se atiende cada ciudad y familia (m3 del horizonte)")
    lb.tabla(_traslados(base, R, orden), "Traslados", "Traslados XI por ruta y producto (horizonte)")
    lb.tabla(pd.DataFrame(pruebas), "Pruebas", "Pruebas de consistencia A–H")
    lb.tabla(calidad, "Calidad_datos", "Validaciones de calidad de datos")
    if viajes_xi:
        lb.tabla(pd.DataFrame(viajes_xi), "Flota_propia_XI", "Traslados XI de flota propia: flete imputado por ruta (promedio mensual)")
    prm = P.get("S3_OPTIMIZADO") or P[orden[-1]]
    lb.tabla(pd.DataFrame(_parametros(prm)), "Parametros", "Parámetros del escenario optimizado")
    _validador(lb, base, prm)
    lb.cerrar()
