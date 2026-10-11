"""Genera el Excel consolidado con resultados, tablas filtrables y una hoja 'Simulador' con fórmulas vivas."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name, xl_rowcol_to_cell

def _sin_inf(df: pd.DataFrame) -> pd.DataFrame:
    d2 = df.copy()
    num = d2.select_dtypes(include=[np.number]).columns
    d2[num] = d2[num].astype(float).where(np.isfinite(d2[num].astype(float)))
    return d2


from .preparacion import Preparado

COMPONENTES = ["Flete traslado (XI)", "Flete distribución", "Manejo en almacén", "Posiciones bodega externa",
               "Fijo almacenes opcionales", "Fijo red (nodos fijos)", "Penalización déficit capacidad"]

TIPO_TXT = {"propio": "CEDI propio", "3pl": "CEDI 3PL", "bodega_externa_actual": "Bodega externa actual",
            "candidata": "Bodega externa candidata"}


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


def _hojas_tarifas(lb: _Libro, sim: dict):
    b, f = lb.b, lb.f
    W = sim["almacenes"]
    K, R = sim["plantas"], sim["regiones"]
    wt = b.add_worksheet("Tarifas_traslado")
    wt.write(0, 0, "Planta \\ Almacén", f["h"])
    for j, n in enumerate(W):
        wt.write(0, j + 1, n["id"], f["h"])
    for i, k in enumerate(K):
        wt.write(i + 1, 0, k, f["h"])
        for j, n in enumerate(W):
            wt.write_number(i + 1, j + 1, sim["c_in"][k][n["id"]], f["cop"])
    wt.set_column(0, len(W), 14)
    wt.write(len(K) + 2, 0, "COP/m3. Observada en SAP (XI, flete válido) o estimada por distancia (regresión).", f["sub"])
    wz = b.add_worksheet("Tarifas_zona")
    wz.write(0, 0, "Zona \\ Almacén", f["h"])
    for j, n in enumerate(W):
        wz.write(0, j + 1, n["id"], f["h"])
    for i, r in enumerate(R):
        wz.write(i + 1, 0, r, f["h"])
        for j, n in enumerate(W):
            wz.write_number(i + 1, j + 1, sim["c_out"][n["id"]][r], f["cop"])
    wz.set_column(0, 0, 28)
    wz.set_column(1, len(W), 14)
    wz.write(len(R) + 2, 0, "COP/m3 promedio ponderado por la demanda de cada ciudad de la zona.", f["sub"])


def _hoja_simulador(lb: _Libro, prep: Preparado, sim: dict, cfg: dict, base_sim: dict):
    b, f = lb.b, lb.f
    W = sim["almacenes"]
    K, R = sim["plantas"], sim["regiones"]
    ult_col_t = xl_col_to_name(len(W))
    rng_t = f"Tarifas_traslado!$B$2:${ult_col_t}${len(K) + 1}"
    rng_t_pl = f"Tarifas_traslado!$A$2:$A${len(K) + 1}"
    rng_t_al = f"Tarifas_traslado!$B$1:${ult_col_t}$1"
    rng_z = f"Tarifas_zona!$B$2:${ult_col_t}${len(R) + 1}"
    rng_z_al = f"Tarifas_zona!$B$1:${ult_col_t}$1"

    ws = b.add_worksheet("Simulador")
    ws.write(0, 0, "Simulador de decisión: ¿dónde ubicar el volumen que deja la planta cerrada?", f["titulo"])
    ws.write(1, 0, "Celdas amarillas = editables. Compara bodega externa vs traslado a CEDI de otra ciudad para un volumen "
                   "mensual, con restricción de posiciones libres. Las fórmulas se recalculan al instante.", f["sub"])
    entradas = [
        ("Días de inventario (cobertura)", base_sim["doi"], f["input"]),
        ("Utilización máxima de posiciones", base_sim["util"], f["input_pct"]),
        ("m3 por posición de estiba", sim["m3_por_posicion"], f["input"]),
        ("Factor flete traslado (1 = tarifa actual)", 1.0, f["input"]),
        ("Factor flete distribución (1 = tarifa actual)", 1.0, f["input"]),
        ("Planta origen del volumen", base_sim["planta"], f["input_txt"]),
        ("Volumen a ubicar (m3/mes)", base_sim["volumen"], f["input_cop"]),
        ("Crecimiento del volumen", 0.0, f["input_pct"]),
    ]
    for i, (lab, val, fmt) in enumerate(entradas):
        ws.write(3 + i, 0, lab, f["bold"])
        ws.write(3 + i, 1, val, fmt)
    ws.data_validation(8, 1, 8, 1, {"validate": "list", "source": K})
    ws.write(11, 0, "Volumen efectivo (m3/mes)", f["bold"])
    ws.write_formula(11, 1, "=B10*(1+B11)", f["cop"])
    DOI, UT, M3P, FXI, FDI, PL, VOL = "$B$4", "$B$5", "$B$6", "$B$7", "$B$8", "$B$9", "$B$12"

    # reparto por zona
    ws.write(3, 3, "Zona comercial", f["h"])
    ws.write(3, 4, "% del volumen", f["h"])
    ws.write(3, 5, "m3/mes", f["h"])
    z0 = 4
    for i, r in enumerate(R):
        ws.write(z0 + i, 3, r, f["txt"])
        ws.write_number(z0 + i, 4, base_sim["reparto"].get(r, 0.0), f["input_pct"])
        ws.write_formula(z0 + i, 5, f"={VOL}*E{z0 + i + 1}", f["cop"])
    zf = z0 + len(R)
    ws.write(zf, 3, "Total (debe ser 100 %)", f["bold"])
    ws.write_formula(zf, 4, f"=SUM(E{z0 + 1}:E{zf})", f["pct"])
    ws.write_formula(zf, 5, f"=SUM(F{z0 + 1}:F{zf})", f["cop"])
    rng_zm3 = f"$F${z0 + 1}:$F${zf}"

    # tabla de opciones
    t0 = max(14, zf + 3)
    hdr = ["Almacén", "Nombre", "Tipo", "Posiciones totales", "Ocupación base (m3/mes despachados)",
           "Posiciones libres", "Posiciones requeridas", "¿Cabe?", "Manejo COP/m3", "Costo posición COP/mes",
           "Costo fijo incremental COP/mes", "Tarifa traslado desde planta COP/m3", "Distribución ponderada COP/m3",
           "Costo unitario total COP/m3", "Costo variable COP/mes", "Costo total COP/mes", "Ranking (si cabe)"]
    ws.write(t0 - 1, 0, "Opciones de almacenamiento", f["titulo"])
    for j, h in enumerate(hdr):
        ws.write(t0, j, h, f["h"])
    ws.set_row(t0, 45)
    n0, n1 = t0 + 1, t0 + len(W)
    for i, n in enumerate(W):
        r = n0 + i
        R1 = r + 1
        ws.write(r, 0, n["id"], f["txt"])
        ws.write(r, 1, n["nombre"], f["txt"])
        ws.write(r, 2, TIPO_TXT[n["tipo"]], f["txt"])
        ws.write_number(r, 3, n["posiciones"], f["input_cop"])
        ws.write_number(r, 4, base_sim["ocupacion_base"].get(n["id"], 0.0), f["input_cop"])
        ws.write_formula(r, 5, f"=MAX(0,D{R1}*{UT}-E{R1}*{DOI}/30/{M3P})", f["cop"])
        ws.write_formula(r, 6, f"={VOL}*{DOI}/30/{M3P}", f["cop"])
        ws.write_formula(r, 7, f'=IF(F{R1}>=G{R1},"Sí","No")', f["txt"])
        ws.write_number(r, 8, n["manejo"], f["input_cop"])
        ws.write_number(r, 9, n["posicion_cop"], f["input_cop"])
        fijo_inc = n["fijo"] if n["tipo"] in ("candidata", "bodega_externa_actual") else 0.0
        ws.write_number(r, 10, fijo_inc, f["input_cop"])
        ws.write_formula(r, 11, f"=IF({PL}=A{R1},0,INDEX({rng_t},MATCH({PL},{rng_t_pl},0),MATCH(A{R1},{rng_t_al},0)))*{FXI}", f["cop"])
        ws.write_formula(r, 12, f"=IF({VOL}>0,SUMPRODUCT({rng_zm3},INDEX({rng_z},0,MATCH(A{R1},{rng_z_al},0)))/{VOL},0)*{FDI}", f["cop"])
        ws.write_formula(r, 13, f"=L{R1}+I{R1}+J{R1}*{DOI}/30/{M3P}+M{R1}", f["cop"])
        ws.write_formula(r, 14, f"=N{R1}*{VOL}", f["cop"])
        ws.write_formula(r, 15, f"=O{R1}+K{R1}", f["cop"])
        ws.write_formula(r, 16, f'=IF(H{R1}="Sí",COUNTIFS($H${n0 + 1}:$H${n1 + 1},"Sí",$P${n0 + 1}:$P${n1 + 1},"<"&P{R1})+1,"-")', f["txt"])
    ws.set_column(0, 0, 34)
    ws.set_column(1, 1, 26)
    ws.set_column(2, 2, 24)
    ws.set_column(3, 16, 16)
    ws.set_column(3, 3, 16)
    rA, rC, rH, rP, rQ = (f"$A${n0 + 1}:$A${n1 + 1}", f"$C${n0 + 1}:$C${n1 + 1}", f"$H${n0 + 1}:$H${n1 + 1}",
                          f"$P${n0 + 1}:$P${n1 + 1}", f"$Q${n0 + 1}:$Q${n1 + 1}")
    # resultado
    q0 = n1 + 3
    ws.write(q0, 0, "Resultado", f["titulo"])
    filas = [
        ("Mejor opción (que cabe)", f'=IFERROR(INDEX({rA},MATCH(1,{rQ},0)),"Ninguna opción tiene capacidad")', f["res_txt"]),
        ("Costo total mejor opción (COP/mes)", f"=IFERROR(INDEX({rP},MATCH(1,{rQ},0)),0)", f["res"]),
        ("Segunda mejor opción", f'=IFERROR(INDEX({rA},MATCH(2,{rQ},0)),"-")', f["res_txt"]),
        ("Ahorro vs. segunda opción (COP/mes)", f"=IFERROR(INDEX({rP},MATCH(2,{rQ},0))-INDEX({rP},MATCH(1,{rQ},0)),0)", f["res"]),
        ("Mejor bodega externa (COP/mes)", f'=IF(COUNTIFS({rC},"Bodega externa*",{rH},"Sí")=0,"Sin capacidad",'
                                            f'_xlfn.MINIFS({rP},{rC},"Bodega externa*",{rH},"Sí"))', f["res"]),
        ("Mejor traslado a CEDI de otra ciudad (COP/mes)", f'=IF(COUNTIFS({rC},"CEDI*",{rH},"Sí",{rA},"<>"&{PL})=0,"Sin capacidad",'
                                                           f'_xlfn.MINIFS({rP},{rC},"CEDI*",{rH},"Sí",{rA},"<>"&{PL}))', f["res"]),
        ("Decisión: ¿bodega externa o traslado a CEDI?", f'=IF(AND(ISNUMBER(B{q0 + 6}),ISNUMBER(B{q0 + 7})),IF(B{q0 + 6}<B{q0 + 7},'
                                                         f'"BODEGA EXTERNA","TRASLADO A CEDI"),IF(ISNUMBER(B{q0 + 6}),"BODEGA EXTERNA",'
                                                         f'IF(ISNUMBER(B{q0 + 7}),"TRASLADO A CEDI","AMPLIAR CAPACIDAD")))', f["res_txt"]),
        ("Costo fijo máximo de bodega externa para empatar (COP/mes)",
         f'=IF(AND(ISNUMBER(B{q0 + 6}),ISNUMBER(B{q0 + 7})),B{q0 + 7}-(B{q0 + 6}-INDEX($K${n0 + 1}:$K${n1 + 1},'
         f'MATCH(B{q0 + 6},{rP},0))),"-")', f["res"]),
    ]
    for i, (lab, fo, fmt) in enumerate(filas):
        ws.write(q0 + 1 + i, 0, lab, f["bold"])
        ws.write_formula(q0 + 1 + i, 1, fo, fmt)
    ws.write(q0 + 9, 2, "Si una opción dice 'Sin capacidad', amplíe sus posiciones (columna D) o reduzca el volumen (B10).", f["sub"])
    ws.write(q0 + 10, 0, "Nota: la ocupación base es el despacho AS-IS del almacén sin el volumen de la planta cerrada. "
                         "El costo fijo incremental de CEDIs existentes es 0 (ya se paga). El modelo completo (MILP) "
                         "reparte el volumen entre varios almacenes; este simulador evalúa cada opción por separado.", f["sub"])
    ch = b.add_chart({"type": "bar"})
    ch.add_series({"name": "Costo total COP/mes", "categories": ["Simulador", n0, 0, n1, 0],
                   "values": ["Simulador", n0, 15, n1, 15], "fill": {"color": "#2a78d6"}, "gap": 60})
    ch.set_title({"name": "Costo mensual por opción (se actualiza con los parámetros)"})
    ch.set_legend({"none": True})
    ch.set_x_axis({"num_format": "#,##0", "major_gridlines": {"visible": True, "line": {"color": "#e1e0d9"}}})
    ch.set_y_axis({"reverse": True})
    ch.set_size({"width": 760, "height": 420})
    ws.insert_chart(q0 + 12, 0, ch)

    # mejor almacén por zona (sin restricción de capacidad)
    m0 = q0 + 35
    ws.write(m0, 0, "Costo unitario por zona (COP/m3) y mejor almacén por zona, sin restricción de capacidad", f["titulo"])
    ws.write(m0 + 1, 0, "Zona", f["h"])
    for j, n in enumerate(W):
        ws.write(m0 + 1, j + 1, n["id"], f["h"])
    ws.write(m0 + 1, len(W) + 1, "Mejor almacén", f["h"])
    for i, r in enumerate(R):
        rr = m0 + 2 + i
        ws.write(rr, 0, r, f["txt"])
        for j, n in enumerate(W):
            fila_op = n0 + j + 1
            ws.write_formula(rr, j + 1, f"=L{fila_op}+I{fila_op}+J{fila_op}*{DOI}/30/{M3P}+"
                                        f"INDEX({rng_z},MATCH($A{rr + 1},Tarifas_zona!$A$2:$A${len(R) + 1},0),"
                                        f"MATCH({xl_rowcol_to_cell(m0 + 1, j + 1, row_abs=True)},{rng_z_al},0))*{FDI}", f["cop"])
        c1, c2 = xl_rowcol_to_cell(rr, 1), xl_rowcol_to_cell(rr, len(W))
        ws.write_formula(rr, len(W) + 1, f"=INDEX(${xl_col_to_name(1)}${m0 + 2}:${xl_col_to_name(len(W))}${m0 + 2},"
                                         f"MATCH(MIN({c1}:{c2}),{c1}:{c2},0))", f["txt"])
    ws.conditional_format(m0 + 2, 1, m0 + 1 + len(R), len(W),
                          {"type": "3_color_scale", "min_color": "#cde2fb", "mid_color": "#f0efec", "max_color": "#f4b6b5"})


def generar_excel(ruta: Path, prep: Preparado, cfg: dict, res: dict, calidad: pd.DataFrame, sens: pd.DataFrame,
                  sim: dict, equilibrio: dict, resumen: dict):
    lb = _Libro(ruta)
    b, f = lb.b, lb.f
    ids = [k for k in res if k != "ASIS_REAL"]

    # ------------------------------------------------------------ Resumen
    ws = b.add_worksheet("Resumen")
    ws.write(0, 0, "Modelo de optimización de red logística — cierre de planta " + cfg["produccion"]["planta_cerrada"], f["titulo"])
    ws.write(1, 0, f"Periodo base {prep.meses_lista[0]} a {prep.meses_lista[-1]} · costos en COP por mes promedio · "
                   f"días de inventario {res[ids[0]]['params']['dias_inventario']:.1f}", f["sub"])
    ws.write(3, 0, "Recomendación", f["bold"])
    ws.merge_range(4, 0, 9, 8, resumen["recomendacion"], f["wrap"])
    ws.write(11, 0, "Hallazgos", f["bold"])
    for i, h in enumerate(resumen["hallazgos"]):
        ws.merge_range(12 + i, 0, 12 + i, 8, f"• {h}", f["wrap"])
        ws.set_row(12 + i, 30)
    r0 = 13 + len(resumen["hallazgos"])
    ws.write(r0, 0, "Comparación de escenarios (COP/mes)", f["bold"])
    cols = ["ASIS_REAL"] + ids
    ws.write(r0 + 1, 0, "Componente", f["h"])
    for j, k in enumerate(cols):
        ws.write(r0 + 1, j + 1, res[k]["nombre"], f["h"])
    ws.set_row(r0 + 1, 60)
    for i, comp in enumerate(COMPONENTES + ["Total"]):
        ws.write(r0 + 2 + i, 0, comp, f["bold"] if comp == "Total" else f["txt"])
        for j, k in enumerate(cols):
            ws.write_number(r0 + 2 + i, j + 1, res[k]["costos"].get(comp, 0.0), f["cop"])
    ws.set_column(0, 0, 34)
    ws.set_column(1, len(cols), 22)
    ch = b.add_chart({"type": "bar", "subtype": "stacked"})
    paleta = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#898781", "#e34948"]
    for i, comp in enumerate(COMPONENTES):
        ch.add_series({"name": ["Resumen", r0 + 2 + i, 0], "categories": ["Resumen", r0 + 1, 1, r0 + 1, len(cols)],
                       "values": ["Resumen", r0 + 2 + i, 1, r0 + 2 + i, len(cols)], "fill": {"color": paleta[i]},
                       "gap": 50, "border": {"color": "#ffffff", "width": 1.5}})
    ch.set_title({"name": "Costo mensual por escenario y componente"})
    ch.set_x_axis({"num_format": "#,##0,,\" M\""})
    ch.set_y_axis({"reverse": True})
    ch.set_size({"width": 980, "height": 460})
    ch.set_legend({"position": "bottom"})
    ws.insert_chart(r0 + 11, 0, ch)

    # ------------------------------------------------------------ Simulador (fórmulas)
    _hoja_simulador(lb, prep, sim, cfg, resumen["base_simulador"])

    # ------------------------------------------------------------ Escenarios KPI
    filas = []
    for k in cols:
        r = res[k]
        kp = r["kpis"]
        filas.append({"Escenario": k, "Nombre": r["nombre"], "Costo total COP/mes": r["costos"]["Total"],
                      "Costo variable + fijo opcional COP/mes": r["costos"]["Total variable + fijo opcional"],
                      "Demanda m3/mes": kp.get("demanda_m3_mes"), "Traslados m3/mes": kp.get("traslados_m3_mes"),
                      "Viajes de traslado/mes": kp.get("viajes_traslado_mes"), "Costo COP/m3": kp.get("costo_por_m3"),
                      "Tránsito promedio h": kp.get("transito_promedio_h"), "Déficit capacidad m3": kp.get("deficit_m3"),
                      "Bodegas abiertas": ", ".join(kp.get("bodegas_abiertas", [])) or "-",
                      "Bodegas cerradas": ", ".join(kp.get("bodegas_cerradas", [])) or "-",
                      "Días inventario": r["params"]["dias_inventario"], "Estado solver": kp.get("estado_solver", "evaluación")})
    lb.tabla(pd.DataFrame(filas), "Escenarios", "Indicadores por escenario")

    # ------------------------------------------------------------ Sensibilidad
    if len(sens):
        ws = lb.tabla(sens, "Sensibilidad", "Sensibilidad: costo total por opción")
        piv = sens[sens.variable == "dias_inventario"].pivot_table(index="valor", columns="escenario", values="costo_total_cop")
        c0 = len(sens.columns) + 2
        ws.write(2, c0, "Días inventario", f["h"])
        for j, e in enumerate(piv.columns):
            ws.write(2, c0 + 1 + j, e, f["h"])
        for i, (v, row) in enumerate(piv.iterrows()):
            ws.write_number(3 + i, c0, v, f["num"])
            for j, e in enumerate(piv.columns):
                ws.write_number(3 + i, c0 + 1 + j, row[e], f["cop"])
        ch = b.add_chart({"type": "line"})
        for j, e in enumerate(piv.columns):
            ch.add_series({"name": ["Sensibilidad", 2, c0 + 1 + j], "categories": ["Sensibilidad", 3, c0, 2 + len(piv), c0],
                           "values": ["Sensibilidad", 3, c0 + 1 + j, 2 + len(piv), c0 + 1 + j],
                           "line": {"width": 2.25, "color": paleta[j]}, "marker": {"type": "circle", "size": 6}})
        ch.set_title({"name": "Costo total vs días de inventario"})
        ch.set_y_axis({"num_format": "#,##0,,\" M\""})
        ch.set_x_axis({"name": "Días de inventario"})
        ch.set_size({"width": 760, "height": 380})
        ws.insert_chart(4 + len(piv), c0, ch)

    # ------------------------------------------------------------ Detalle de flujos y almacenes
    tx, ty, tn = [], [], []
    for k in cols:
        a = res[k]["flujos_traslado"].copy().rename(columns={"origen": "planta", "destino": "almacen", "vol_mes": "m3_mes"})
        a.insert(0, "Escenario", k)
        tx.append(a)
        bb = res[k]["flujos_distribucion"].copy().rename(columns={"origen": "almacen", "destino": "ciudad", "vol_mes": "m3_mes"})
        bb.insert(0, "Escenario", k)
        ty.append(bb)
        c = res[k]["nodos"].copy()
        c.insert(0, "Escenario", k)
        tn.append(c)
    fx = pd.concat(tx, ignore_index=True)
    lb.tabla(fx, "Flujos_traslado", "Traslados planta -> almacén (m3/mes)")
    fy = pd.concat(ty, ignore_index=True)
    lb.tabla(fy, "Flujos_distribucion", "Distribución almacén -> ciudad (m3/mes)")
    lb.tabla(pd.concat(tn, ignore_index=True), "Almacenes", "Uso de almacenes por escenario",
             formatos={"utilizacion": f["pct"]})

    # ------------------------------------------------------------ Calidad y datos base
    lb.tabla(calidad, "Calidad_datos", "Validaciones de calidad de la información",
             formatos={"% afectado": f["pct"], "Registros afectados": f["cop"]},
             anchos={"Validación": 70, "Resultado": 30, "Tratamiento en el modelo": 80})
    ws = b.get_worksheet_by_name("Calidad_datos")
    for sev, col in (("Alta", "#f4b6b5"), ("Media", "#ffe3a3"), ("Baja", "#e1e0d9"), ("OK", "#c9ecd6")):
        ws.conditional_format(3, 5, 3 + len(calidad), 5, {"type": "cell", "criteria": "==", "value": f'"{sev}"',
                                                         "format": b.add_format({"bg_color": col})})
    pl = prep.plantas.reset_index().rename(columns={"index": "planta"})
    lb.tabla(pl, "Plantas", "Producción vs capacidad de planta (m3/mes)", formatos={"util_asis": f["pct"], "util_tobe": f["pct"]})
    nd = prep.nodos.reset_index().rename(columns={"index": "ID_Nodo"})
    keep = ["ID_Nodo", "nombre", "Tipo_Nodo", "Modalidad", "Estado", "Ciudad", "lat", "lon", "posiciones_total", "cap_inv_m3",
            "cap_despacho_m3_mes", "despacho_asis_m3_mes", "traslado_salida_asis_m3_mes", "costo_fijo", "manejo_cop_m3",
            "posicion_cop_mes", "coord_corregida"]
    lb.tabla(nd[[c for c in keep if c in nd.columns]], "Nodos", "Nodos y capacidades")
    td = prep.tarifas_dist.copy()
    td.insert(0, "tipo", "Distribución")
    tx_ = prep.tarifas_xi.copy()
    tx_.insert(0, "tipo", "Traslado XI")
    lb.tabla(pd.concat([tx_, td], ignore_index=True), "Tarifas_observadas",
             "Tarifas observadas SAP (solo líneas con flete válido y sin atípicos)")
    lb.tabla(prep.tiempos_lane, "Tiempos_TMS", "Tiempos de tránsito y m3 por viaje (TMS, viajes cruzados con SAP)")
    hp = prep.homolog_plantas.copy()
    lb.tabla(hp, "Homologaciones", "Homologación planta maestro de referencias -> centro SAP",
             formatos={"participacion": f["pct"]})
    ws = b.get_worksheet_by_name("Homologaciones")
    hc = prep.homolog_clientes.copy()
    hc["nodo_destino"] = hc["nodo_destino"].astype(object).where(hc["nodo_destino"].notna(), "(sin nodo)")
    r1 = len(hp) + 6
    ws.write(r1 - 2, 0, "Homologación cliente destino de traslados XI -> nodo", f["titulo"])
    ws.add_table(r1, 0, r1 + len(hc), len(hc.columns) - 1,
                 {"data": hc.astype(object).where(hc.notna(), None).values.tolist(),
                  "columns": [{"header": c} for c in hc.columns], "style": "Table Style Medium 2"})
    dem = prep.demanda.copy()
    dem["planta_tobe"] = dem["planta_tobe"].fillna("(descontinuada)")
    lb.tabla(dem, "Demanda", "Demanda base por ciudad, familia y planta (m3/mes)")
    par = [("Periodo base", f"{prep.meses_lista[0]} a {prep.meses_lista[-1]} ({prep.meses_base} meses)"),
           ("Días de inventario usados", res[ids[0]]["params"]["dias_inventario"]),
           ("DOI implícito AS-IS (calibración)", prep.info["doi_implicito"]),
           ("m3 por posición", prep.m3_por_posicion),
           ("Utilización máxima", cfg["almacenamiento"]["utilizacion_maxima"]),
           ("Umbral importe inválido", cfg["limpieza"]["umbral_importe_invalido"]),
           ("Regresión distribución (COP/m3 = a + b*km)", f"a={prep.regresion['distribucion']['a']:,.0f}  b={prep.regresion['distribucion']['b']:,.1f}  R2={prep.regresion['distribucion']['r2']:.2f}"),
           ("Regresión traslado (COP/m3 = a + b*km)", f"a={prep.regresion['traslado']['a']:,.0f}  b={prep.regresion['traslado']['b']:,.1f}  R2={prep.regresion['traslado']['r2']:.2f}"),
           ("Factor km carretera", cfg["limpieza"]["factor_ruta_km"]),
           ("Penalidad déficit COP/m3", cfg["almacenamiento"]["penalidad_deficit_cop_m3"])]
    for k, v in (equilibrio or {}).items():
        par.append((k, v))
    lb.tabla(pd.DataFrame(par, columns=["Parámetro", "Valor"]).astype({"Valor": str}), "Parametros", "Parámetros y supuestos usados",
             anchos={"Parámetro": 50, "Valor": 60})
    _hojas_tarifas(lb, sim)
    b.get_worksheet_by_name("Resumen").activate()
    lb.cerrar()
