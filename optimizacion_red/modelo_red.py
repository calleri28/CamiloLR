"""
Modelo de optimización de red logística: cierre de planta y traslado de líneas.

Flujo:
  1. Carga extractos SAP (principal), TMS (complementario) y maestro de
     materiales; si no se pasan archivos, genera DATOS DE PRUEBA.
  2. Valida y corrige la información (bitácora de reglas R01..R15).
  3. Estima demanda neta por CEDI-familia y tarifas COP/t por ruta
     (histórica o por regresión con km para rutas nuevas).
  4. Resuelve escenarios con un MILP (PuLP).
  5. Genera un HTML autocontenido con resultados y un simulador interactivo.

Uso:
  python modelo_red.py                                   # datos de prueba
  python modelo_red.py --sap sap.xlsx --tms tms.csv --materiales mat.xlsx
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

import calidad_datos as cd
import datos_prueba
import optimizador as op
import parametros as P

AQUI = Path(__file__).resolve().parent


def leer(ruta: str | None):
    if not ruta:
        return None
    p = Path(ruta)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, dtype=str)
    return pd.read_csv(p, dtype=str, sep=None, engine="python", encoding_errors="replace")


def definir_escenarios():
    todas = {k: True for k in P.PLANTAS}
    cierre = {**todas, P.PLANTA_CIERRE: False}
    nom = P.PLANTAS[P.PLANTA_CIERRE]["ciudad"]
    plan = ", ".join(f"{f} → {P.PLANTAS[p]['ciudad']}" for f, p in P.PLAN_TRASLADO_PROPUESTO.items())
    return [
        ("E0", "Red actual optimizada", "Red actual", op.escenario("E0", "Todas las plantas abiertas, asignación planta→CEDI óptima.", todas, "ninguno"), False),
        ("E1", f"Cierre {nom} sin traslado de líneas", "Cierre sin traslado", op.escenario("E1", "Se cierra la planta y la demanda se absorbe con capacidad existente, horas extra y maquila.", cierre, "ninguno"), False),
        ("E2", f"Cierre {nom} + traslado óptimo", "Cierre + traslado óptimo", op.escenario("E2", "El modelo decide qué líneas trasladar y a qué planta.", cierre, "optimizar"), False),
        ("E3", f"Cierre {nom} + plan propuesto", "Cierre + plan propuesto", op.escenario("E3", f"Traslado definido por la compañía: {plan}.", cierre, dict(P.PLAN_TRASLADO_PROPUESTO)), False),
        ("E4", "E2 con demanda +10 %", "E2 · demanda +10 %", op.escenario("E4", "Sensibilidad: crecimiento de demanda; se re-optimizan traslados.", cierre, "optimizar", crec_demanda=0.10), True),
        ("E5", "E2 con flete +15 %", "E2 · flete +15 %", op.escenario("E5", "Sensibilidad: alza de ACPM/peajes; se re-optimizan traslados.", cierre, "optimizar", indice_flete=1.15), True),
    ]


def _es(x, dec=0):
    """Formato numérico colombiano: 1.234.567,8"""
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def hallazgos(bit: pd.DataFrame, match: dict, resumen: dict, reg: dict, tarifas: pd.DataFrame, historico: dict):
    b = bit.set_index("regla")
    h = []
    if "R10" in b.index:
        h.append(f"<b>{_es(b.loc['R10','registros'])} líneas XI con flete en 0, 1 peso o vacío</b> ({_es(b.loc['R10','toneladas'])} t). "
                 "Se excluyen del cálculo de tarifa para no subestimar el COP/t, pero su volumen sí cuenta como demanda: el producto se movió.")
    if "R11" in b.index:
        h.append(f"{_es(b.loc['R11','registros'])} líneas con COP/kg atípico dentro de su ruta (errores de digitación ×10 o ÷10, viajes casi vacíos). "
                 f"Representan {_es(b.loc['R11','importe']/1e6)} M COP que distorsionarían la tarifa.")
    h.append(f"Match SAP↔TMS por viaje: <b>{_es(match['tasa_match']*100)} %</b>. Se normaliza el número de viaje (ceros a la izquierda, caracteres no numéricos) antes del cruce. "
             f"Cuando no hay match se conserva el registro SAP y el km se toma de la distancia teórica (geodésica × {_es(P.FACTOR_SINUOSIDAD_VIAL, 2)}).")
    if "R04" in b.index:
        h.append(f"{_es(b.loc['R04','registros'])} duplicados por documento+posición y {_es(b['registros'].get('R06', 0))} reversiones/cantidades negativas eliminados.")
    n_est = int((tarifas["fuente"] != "Histórico SAP").sum())
    h.append(f"{n_est} de {len(tarifas)} rutas planta→CEDI no tienen historia suficiente y se valoran con la regresión por km (R² {_es(reg['r2'], 2)}). "
             "Son justamente las rutas nuevas que aparecen al trasladar producción: conviene validarlas con cotizaciones de transportadores.")
    h.append(f"El {_es(historico['pct_reexpedicion']*100)} % del flete XI histórico corresponde a re-expediciones CEDI→CEDI (doble manipulación); "
             "el modelo evalúa abastecer esos CEDI directamente desde planta.")
    return h


METODOLOGIA = f"""
<h2>Metodología</h2>
<h3>1. Datos y validaciones</h3>
<p>La tabla SAP es la fuente principal; se filtran los movimientos <code>XI</code> (traslado planta/CEDI → CEDI).
El TMS complementa por número de viaje (km, tipo de vehículo, capacidad) solo cuando hay match. Reglas aplicadas:
normalización de textos y números en formato colombiano, duplicados, fechas fuera de rango, reversiones, centros fuera del maestro,
origen = destino, imputación de peso con el maestro de materiales, exclusión de importes ≤ {P.IMPORTE_MINIMO_VALIDO:.0f} peso del cálculo
de tarifas, atípicos de COP/kg por ruta (IQR sobre log, factor {P.OUTLIER_IQR_FACTOR}) y km del TMS incoherentes.</p>
<h3>2. Parámetros estimados</h3>
<p><b>Demanda</b>: entradas XI a cada CEDI menos salidas XI a otros CEDI, promedio mensual por familia.
<b>Tarifa</b>: COP/t histórico de la ruta si tiene ≥ {P.MIN_REGISTROS_TARIFA} viajes válidos; si no, regresión ponderada COP/t = a + b·km.</p>
<h3>3. Modelo MILP (Python · PuLP)</h3>
<p>Minimiza el costo mensual = producción variable + flete primario + horas extra + maquila + penalidad por demanda no atendida
+ costo fijo de plantas abiertas + CAPEX de traslado de líneas amortizado a {P.MESES_AMORTIZACION} meses. Restricciones: demanda por CEDI y familia,
capacidad de planta (más la capacidad que aporta cada línea trasladada), horas extra ≤ {P.HORAS_EXTRA_MAX_PCT:.0%} de la capacidad,
una planta solo produce las familias que tiene o que recibe por traslado, cada línea se traslada a lo sumo a una planta,
máximo {P.MAX_LINEAS_RECIBIDAS_POR_PLANTA} líneas recibidas por planta y tope de maquila de {_es(P.MAQUILA_MAX_T_MES)} t/mes.</p>
<h3>4. Simulador</h3>
<p>Con las decisiones de traslado fijas, el problema es un flujo de costo mínimo; el navegador lo resuelve de forma exacta
(caminos más cortos sucesivos) en cada cambio de variable. «Optimizar traslados» enumera todas las combinaciones de líneas
y plantas receptoras, lo que equivale a las variables binarias del MILP. Los resultados del simulador coinciden con los de Python
para los mismos supuestos.</p>
<h3>5. Limitaciones</h3>
<ul><li>La capacidad de planta es agregada en toneladas (no por línea/SKU) y la demanda es un promedio mensual: para picos estacionales, ejecute un escenario con crecimiento de demanda.</li>
<li>El flete secundario CEDI → cliente no cambia con el cierre y no se modela.</li>
<li>Costos de cierre (liquidaciones, desmantelamiento) e ingresos por venta de activos deben sumarse aparte al caso de negocio.</li>
<li>Los costos de producción, capacidades y CAPEX están en <code>parametros.py</code>; en este reporte son valores de prueba.</li></ul>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sap"), ap.add_argument("--tms"), ap.add_argument("--materiales")
    ap.add_argument("--salida", default=str(AQUI / "salida"))
    a = ap.parse_args()
    warnings.simplefilter("ignore", category=FutureWarning)
    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)

    prueba = not a.sap
    if prueba:
        print("» Generando datos de prueba (estructura SAP / TMS)…")
        sap_raw, tms_raw, mat = datos_prueba.generar()
        (out / "datos_prueba").mkdir(exist_ok=True)
        sap_raw.to_csv(out / "datos_prueba" / "sap_traslados.csv", index=False)
        tms_raw.to_csv(out / "datos_prueba" / "tms_viajes.csv", index=False)
        mat.to_csv(out / "datos_prueba" / "maestro_materiales.csv", index=False)
    else:
        sap_raw, tms_raw = leer(a.sap), leer(a.tms)
        mat = leer(a.materiales)
        if mat is not None:
            mat["Kg_Unidad"] = mat["Kg_Unidad"].map(cd.numero_flexible)
        if tms_raw is None:
            tms_raw = pd.DataFrame(columns=list(P.COLUMNAS_TMS.values()))

    print("» Validando y corrigiendo datos…")
    sap, bit, resumen = cd.limpiar_sap(sap_raw, mat)
    tms = cd.limpiar_tms(tms_raw)
    sap, match = cd.cruzar_sap_tms(sap, tms, bit)
    bitacora = bit.tabla()
    bitacora.to_csv(out / "bitacora_calidad.csv", index=False)
    sap.to_csv(out / "sap_xi_depurado.csv", index=False)

    print("» Estimando demanda y tarifas…")
    demanda = op.estimar_demanda(sap)
    tarifas, por_ruta, reg = op.estimar_tarifas(sap)
    demanda.to_csv(out / "demanda_cedi_familia.csv", index=False)
    tarifas.to_csv(out / "tarifas_planta_cedi.csv", index=False)
    ins = op.construir_insumos(demanda, tarifas)

    meses = sap["fecha"].dt.to_period("M").nunique()
    v = sap[sap["valido_tarifa"]]
    cop_kg = v["importe_flete"].sum() / v["peso_kg"].sum()
    re_exp = sap["origen"].isin(P.CEDIS)
    historico = dict(meses=int(meses), flete_mes=float(cop_kg * sap["peso_kg"].sum() / meses),
                     toneladas_mes=float(sap["peso_kg"].sum() / 1000 / meses),
                     pct_reexpedicion=float(v.loc[v["origen"].isin(P.CEDIS), "importe_flete"].sum() / v["importe_flete"].sum()),
                     t_reexpedicion_mes=float(sap.loc[re_exp, "peso_kg"].sum() / 1000 / meses))

    print("» Optimizando escenarios (MILP)…")
    escenarios = []
    for eid, nombre, corto, esc, sens in definir_escenarios():
        r = op.resolver(ins, esc)
        print(f"   {eid} {nombre:<45} {r['estado']:<8} total {r['costos']['total']/1e6:>10,.0f} M COP/mes  traslados={r['traslados']}")
        escenarios.append(dict(id=eid, nombre=nombre, nombre_corto=corto, descripcion=esc["descripcion"], sensibilidad=sens,
                               estado=r["estado"], costos=r["costos"], flujos=r["flujos"], traslados=r["traslados"], uso=r["uso"],
                               maquila_t=r["maquila_t"], no_atendida_t=r["no_atendida_t"], demanda_t=r["demanda_t"],
                               km_promedio=r["km_promedio"], costo_por_t=r["costo_por_t"],
                               config=dict(abiertas=esc["abiertas"], cap_pct=esc["cap_pct"], crec_demanda=esc["crec_demanda"],
                                           indice_flete=esc["indice_flete"], indice_produccion=esc["indice_produccion"])))
    pd.DataFrame([dict(escenario=e["id"], nombre=e["nombre"], estado=e["estado"], **{f"costo_{k}": v for k, v in e["costos"].items()},
                       traslados=json.dumps(e["traslados"], ensure_ascii=False), maquila_t=e["maquila_t"],
                       no_atendida_t=e["no_atendida_t"], costo_por_t=e["costo_por_t"]) for e in escenarios]
                 ).to_csv(out / "resultados_escenarios.csv", index=False)
    pd.DataFrame([dict(escenario=e["id"], **f) for e in escenarios for f in e["flujos"]]).to_csv(out / "flujos_optimos.csv", index=False)

    rutas = por_ruta[por_ruta["origen"].isin(P.PLANTAS)].copy()
    rutas["t_mes"] = rutas["kg"] / 1000 / meses
    datos = dict(
        generado=datetime.now().strftime("%Y-%m-%d %H:%M"), datos_prueba=prueba, insumos=ins, escenarios=escenarios,
        historico=historico, min_registros=P.MIN_REGISTROS_TARIFA,
        calidad=dict(bitacora=bitacora.to_dict("records"), resumen=resumen, match=match,
                     hallazgos=hallazgos(bitacora, match, resumen, reg, tarifas, historico)),
        tarifas=dict(regresion=reg,
                     rutas=[dict(origen=r.origen, destino=r.destino, km=float(r.km), cop_t=float(r.cop_t), viajes=int(r.viajes),
                                 t_mes=float(r.t_mes), utilizacion=None if pd.isna(r.utilizacion) else float(r.utilizacion))
                            for r in rutas.itertuples()],
                     estimadas=[f"{r.origen}|{r.destino}" for r in tarifas.itertuples() if r.fuente != "Histórico SAP"]),
        metodologia=METODOLOGIA,
    )
    html = (AQUI / "plantilla_reporte.html").read_text(encoding="utf-8")
    js = json.dumps(datos, ensure_ascii=False, default=float).replace("</", "<\\/")
    destino = out / "reporte_optimizacion.html"
    destino.write_text(html.replace("/*__DATA__*/null", js), encoding="utf-8")
    print(f"» Reporte: {destino}")
    return destino


if __name__ == "__main__":
    main()
