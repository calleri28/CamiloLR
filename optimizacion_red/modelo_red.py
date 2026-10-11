"""
Modelo de optimización de red logística: cierre de planta y traslado de líneas.

Pasos:
  1. Lee los archivos de `datos_entrada/` (SAP, TMS, maestros y parámetros).
  2. Valida y corrige la información (bitácora de reglas).
  3. Estima la demanda por CEDI y familia (m3/mes), las tarifas COP/m3 por ruta
     y el plan de traslado que indica el maestro de referencias.
  4. Resuelve los escenarios con un MILP (PuLP).
  5. Genera `salida/reporte_optimizacion.html` (resultados + simulador) y
     `salida/resultados_modelo.xlsx`.

Uso:
  python modelo_red.py                      # usa datos_entrada/ (si está vacía, crea datos de prueba)
  python modelo_red.py --generar-prueba     # regenera los CSV de prueba y corre el modelo
  python modelo_red.py --entrada C:/ruta/a/mis_datos --abrir
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd

import calidad_datos as cd
import datos_prueba
import optimizador as op
import parametros as P

AQUI = Path(__file__).resolve().parent


def _es(x, dec=0):
    """Formato numérico colombiano: 1.234.567,8"""
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def definir_escenarios(ins):
    c = ins["planta_cierre"]
    nom = ins["plantas"][c]["ciudad"] if c else ""
    plan = "; ".join(f"{f.title()} → " + ", ".join(f"{ins['plantas'][k]['ciudad']} {v:.0%}" for k, v in d.items())
                     for f, d in ins["plan"].items()) or "sin traslados válidos en el maestro"
    esc = [op.escenario("E0", "Red actual optimizada", "Red actual",
                        "Todas las plantas abiertas; asignación planta→CEDI óptima. Es la referencia.", ins, cerrar=False, traslado="ninguno")]
    if c:
        esc += [
            op.escenario("E1", f"Cierre {nom} sin traslado de líneas", "Cierre sin traslado",
                         "Se cierra la planta; la demanda se absorbe con capacidad existente, horas extra y maquila.", ins, traslado="ninguno"),
            op.escenario("E2", f"Cierre {nom} + traslado óptimo", "Cierre + traslado óptimo",
                         "El modelo decide qué líneas trasladar y a qué planta.", ins, traslado="optimizar"),
            op.escenario("E3", f"Cierre {nom} + plan del maestro de referencias", "Cierre + plan referencias",
                         f"Traslado según Situacion_Post_Cambio / Planta_Produccion_Nueva: {plan}.", ins, traslado="plan"),
            op.escenario("E4", "E2 con demanda +10 %", "E2 · demanda +10 %",
                         "Sensibilidad: crecimiento de demanda; se re-optimizan traslados.", ins, g=0.10, sensibilidad=True),
            op.escenario("E5", "E2 con flete +15 %", "E2 · flete +15 %",
                         "Sensibilidad: alza de ACPM y peajes; se re-optimizan traslados.", ins, ff=1.15, sensibilidad=True),
        ]
    return esc


def hallazgos(bit, match, resumen, reg, tarifas, historico, ins):
    b = bit.set_index("regla")
    get = lambda r, c="registros": b[c].get(r, 0)
    h = []
    fi = historico["flete_imputado"]
    h.append(f"<b>{_es(get('S18'))} líneas XI con IMPORTE SAP en 0, 1 peso, vacío o simbólico</b> ({_es(get('S18', 'volumen_m3'))} m³): "
             "son viajes de flota propia o sin liquidar y sí tienen costo. "
             f"Se imputó el flete a {_es(fi['lineas_similares'])} líneas con viajes similares de terceros (misma ruta y vehículo) "
             f"y a {_es(fi['lineas_km'])} por costo por km ({_es(fi['cop_km'])} COP/km × km × {_es(fi['factor_retorno'], 1)}), "
             "sobre todo en rutas que solo hace la flota propia.")
    h.append(f"Sin la imputación el flete histórico sería {_es(historico['flete_sap_sin_imputar_mes'] / 1e6)} M COP/mes en lugar de "
             f"{_es(historico['flete_mes'] / 1e6)} M: se subestimaría en {_es((1 - historico['flete_sap_sin_imputar_mes'] / historico['flete_mes']) * 100)} %.")
    h.append(f"{_es(get('S19'))} líneas con COP/m³ atípico dentro de su ruta (dígito de más, viajes casi vacíos), por "
             f"{_es(get('S19', 'importe') / 1e6)} M COP, se re-imputaron para no distorsionar la tarifa.")
    h.append(f"Volumen: {_es(get('S14'))} líneas vacías y {_es(get('S15'))} inconsistentes con Cantidad × Volumen_Caja_m3 "
             "(p. ej. en litros) se corrigieron con el maestro de referencias. Esto confirma que <i>Cantidad</i> está en cajas.")
    if match["hay_tms"]:
        h.append(f"Match SAP↔TMS por viaje: <b>{_es(match['tasa_match'] * 100)} %</b>. El número de viaje se normaliza (ceros a la izquierda) antes del cruce; "
                 "sin match se conserva SAP y el km se calcula con las coordenadas del maestro de nodos.")
    else:
        h.append("No se cargó TMS: los km salen de las coordenadas del maestro de nodos.")
    h.append(f"{_es(get('S07'))} líneas con centros sin homologación y {_es(get('S08'))} con origen = destino se excluyeron. "
             "Agregue los códigos faltantes a <code>homologacion_centros.csv</code>.")
    n_est = int((tarifas["fuente"] != "Histórico SAP").sum())
    h.append(f"{n_est} de {len(tarifas)} rutas planta→CEDI no tienen historia suficiente y se valoran con la regresión por km "
             f"(R² {_es(reg['r2'], 2)}). Son las rutas nuevas que aparecen al trasladar producción: valídelas con cotizaciones.")
    h.append(f"Maestro de referencias: {_es(get('RF03'))} referencias «Trasladada» sin planta nueva válida y {_es(get('RF04'))} "
             f"«Descontinuada» con planta nueva; {_es(get('RF06'))} de ellas tenían Control_Consistencia = OK.")
    if historico["pct_reexpedicion"] > 0:
        h.append(f"El {_es(historico['pct_reexpedicion'] * 100)} % del flete XI histórico es re-expedición CEDI→CEDI (doble manipulación); "
                 "el modelo evalúa abastecer esos CEDI directamente desde planta.")
    if resumen.get("clientes") and resumen["clientes"]["match_maestro"] is not None:
        h.append(f"Clientes: el {_es(resumen['clientes']['match_maestro'] * 100)} % de las líneas de distribución cruza con el maestro "
                 "por nombre + ciudad tras separar el campo Cliente. Conviene incluir Cod_Cliente en el extracto SAP.")
    return h


def metodologia(ins):
    return f"""
<h2>Metodología</h2>
<h3>1. Fuentes y validaciones</h3>
<p>SAP es la fuente principal. Se filtran los movimientos <code>{P.TIPO_MOV_TRASLADO}</code> (traslado planta/CEDI → CEDI); el
<i>Centro productor</i> se toma como origen y el <i>CEDI</i> como destino, ambos homologados al maestro de nodos. El TMS complementa por
número de viaje (km, vehículo, capacidad) solo cuando hay match. El maestro de referencias corrige familia y volumen y aporta el plan de
traslado; el maestro de nodos aporta coordenadas, tipo de nodo y la planta que cierra; Info_CEDI aporta la capacidad de recibo.</p>
<h3>2. Parámetros estimados</h3>
<p><b>Demanda</b> (m³/mes): entradas XI a cada CEDI menos salidas XI a otros CEDI, promedio de {ins['meses_historia']} meses.
En los escenarios de cierre se descuenta el volumen de referencias descontinuadas sin sustituta.
<b>Tarifa de terceros</b> (COP/m³): histórico de la ruta si tiene ≥ {P.MIN_VIAJES_TARIFA} viajes liquidados; si no, regresión ponderada COP/m³ = a + b·km.
<b>Flota propia</b>: los viajes con IMPORTE SAP en 0 o 1 peso se valoran con viajes similares (misma ruta y vehículo, ≥ {P.MIN_VIAJES_SIMILARES} viajes de terceros)
o, si no hay, con costo por km × km × factor de retorno ÷ m³ por viaje. La tarifa efectiva de cada ruta mezcla tercero y flota propia según su participación histórica.</p>
<p><b>Almacenamiento</b>: manejo en CEDI (COP/estiba de recibo + despacho); inventario en CEDI = m³ recibidos × días de cobertura ÷ 30 ÷ m³ por estiba × costo de posición,
donde los días de cobertura = días base + {P.DIAS_COBERTURA_POR_1000KM} días por cada 1.000 km a la planta que abastece (más lejos, más stock de seguridad);
el inventario que excede las posiciones de Info_CEDI × ocupación máxima operable se cobra a tarifa de bodega externa; inventario en planta = días de inventario × costo de posición.
La vista «Agrupado» suma transporte y almacenamiento como Logística; las otras vistas los separan.</p>
<h3>3. Modelo MILP (Python · PuLP)</h3>
<p>Minimiza el costo mensual: producción variable + horas extra + flete primario + manejo e inventario en CEDI + inventario en planta
+ maquila + penalidad por demanda no atendida + costo fijo de plantas abiertas + CAPEX de traslado de líneas amortizado a {P.MESES_AMORTIZACION} meses. Restricciones: demanda por CEDI y familia,
capacidad por planta y familia (líneas) más la capacidad que aporta una línea trasladada ({'; '.join(f'{f.title()} {_es(v)} m³' for f, v in ins['cap_traslado'].items() if v)}),
horas extra ≤ {P.HORAS_EXTRA_MAX_PCT:.0%}, cada línea se traslada a una sola planta, máximo de líneas a recibir por planta
y maquila ≤ {_es(P.MAQUILA_MAX_M3_MES)} m³/mes.</p>
<h3>4. Simulador</h3>
<p>Con los traslados fijos el problema es un flujo de costo mínimo y el navegador lo resuelve de forma exacta en cada cambio.
«Optimizar traslados» enumera todas las combinaciones de línea y planta receptora, que equivale a las variables binarias del MILP.
Los resultados del simulador coinciden con los de Python para los mismos supuestos.</p>
<h3>5. Limitaciones</h3>
<ul><li>La demanda es un promedio mensual: para picos estacionales, use el control de crecimiento de demanda.</li>
<li>La capacidad de recibo del CEDI se reporta como alerta; no restringe porque toda la demanda del CEDI debe llegar a él.</li>
<li>El desborde a bodega externa se calcula sobre la solución óptima (no cambia la asignación); el costo por días de cobertura sí entra a la optimización.</li>
<li>El costo por km de la flota propia no incluye costos fijos por viaje (cargue, esperas); ajústelo con el control del simulador.</li>
<li>El flete secundario CEDI → cliente no cambia con el cierre y no se modela.</li>
<li>Costos de cierre (liquidaciones, desmantelamiento) e ingresos por venta de activos se suman aparte al caso de negocio.</li></ul>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", default=str(P.CARPETA_ENTRADA), help="Carpeta con los archivos de entrada")
    ap.add_argument("--salida", default=str(P.CARPETA_SALIDA), help="Carpeta de resultados")
    ap.add_argument("--generar-prueba", action="store_true", help="Crea/sobrescribe los CSV de prueba en la carpeta de entrada")
    ap.add_argument("--abrir", action="store_true", help="Abre el reporte en el navegador al terminar")
    a = ap.parse_args(argv)
    for flujo in (sys.stdout, sys.stderr):          # consola de Windows: evita errores con tildes y símbolos
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    warnings.simplefilter("ignore", category=FutureWarning)
    entrada, out = Path(a.entrada), Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)

    if a.generar_prueba or not any((entrada / P.ARCHIVOS["sap"]).with_suffix(s).exists() for s in (".csv", ".xlsx", ".xls")):
        print(f"» Generando datos de prueba en {entrada} …")
        datos_prueba.generar(entrada)

    print("» Leyendo archivos …")
    e = cd.cargar_entradas(entrada)
    bit = cd.Bitacora()
    print("» Validando y corrigiendo datos …")
    clientes = cd.limpiar_clientes(e["clientes"], bit)
    nodos, homolog = cd.limpiar_nodos(e["nodos"], e["homologacion"], clientes, bit)
    refs = cd.limpiar_referencias(e["referencias"], nodos, homolog, bit)
    info_cedi = cd.resumir_info_cedi(e["info_cedi"], nodos, bit)
    sap, resumen = cd.limpiar_sap(e["sap"], refs, homolog, nodos, clientes, bit)
    tms = cd.limpiar_tms(e["tms"])
    sap, match = cd.cruzar_sap_tms(sap, tms, nodos, bit)
    sap, info_flete = cd.imputar_flete(sap, bit)
    print(f"   Flete imputado (flota propia / 0-1 peso / atípicos): {_es(info_flete['lineas_similares'])} líneas por viajes similares, "
          f"{_es(info_flete['lineas_km'])} por costo por km ({_es(info_flete['cop_km'])} COP/km)")
    bitacora = bit.tabla()
    print(f"   SAP: {_es(resumen['registros_iniciales'])} registros → {_es(resumen['registros_xi_limpios'])} traslados XI depurados "
          f"({_es(resumen['registros_validos_tarifa'])} válidos para tarifa) · match TMS {match['tasa_match']:.0%}")

    print("» Estimando demanda, tarifas y plan de traslado …")
    demanda, n_meses = op.estimar_demanda(sap, nodos, refs)
    plantas = [i for i in nodos.index[nodos["clase"] == "PLANTA"]]
    cedis = sorted(demanda["cedi"].unique())
    tarifas, por_ruta, reg = op.estimar_tarifas(sap, plantas, cedis, nodos)
    cierre_tmp = P.PLANTA_CIERRE or next((i for i in plantas if pd.notna(nodos.at[i, "fecha_cierre"])), None)
    plan, plan_det = op.plan_desde_referencias(sap, refs, cierre_tmp) if cierre_tmp else ({}, pd.DataFrame())
    ins = op.construir_insumos(e, nodos, refs, demanda, tarifas, plan, info_cedi, n_meses, info_flete, reg)
    if not ins["planta_cierre"]:
        print("   ⚠ No se identificó planta en cierre (Fecha_Cierre vacía en el maestro de nodos y PLANTA_CIERRE = None). Solo se corre E0.")
    else:
        print(f"   Planta que cierra: {ins['planta_cierre']} ({ins['plantas'][ins['planta_cierre']]['nombre']})")

    meses = sap["periodo"].dt.to_period("M").nunique()
    cedis_set = set(nodos.index[nodos["clase"] == "CEDI"])
    v = sap[sap["fuente_flete"] == "SAP tercero"]
    por_fuente = sap.groupby("fuente_flete").agg(lineas=("importe", "size"), m3=("volumen", "sum"), importe=("importe", "sum"))
    historico = dict(meses=int(meses), flete_mes=float(sap["importe"].sum() / meses),
                     flete_sap_sin_imputar_mes=float(v["importe"].sum() / meses),
                     m3_mes=float(sap["volumen"].sum() / meses),
                     pct_volumen_propia=float(sap.loc[sap["es_propia"], "volumen"].sum() / sap["volumen"].sum()),
                     por_fuente={k: dict(lineas=int(r.lineas), m3_mes=float(r.m3 / meses), importe_mes=float(r.importe / meses))
                                 for k, r in por_fuente.iterrows()},
                     flete_imputado=info_flete,
                     pct_reexpedicion=float(sap.loc[sap["origen"].isin(cedis_set), "importe"].sum() / sap["importe"].sum()))

    print("» Optimizando escenarios (MILP) …")
    escenarios = []
    for esc in definir_escenarios(ins):
        r = op.resolver(ins, esc)
        tr = ", ".join(f"{f.title()}→" + "/".join(d) for f, d in r["traslados"].items()) or "—"
        print(f"   {esc['id']} {esc['nombre']:<52} {r['estado']:<8} {r['costos']['total'] / 1e6:>10,.0f} M COP/mes · traslados: {tr}")
        escenarios.append(dict(id=esc["id"], nombre=esc["nombre"], nombre_corto=esc["nombre_corto"], descripcion=esc["descripcion"],
                               sensibilidad=esc["sensibilidad"], **r, config={**esc["cfg"], "traslados": r["traslados"]}))

    # ---------------- Salidas tabulares ---------------------------------
    res = pd.DataFrame([dict(escenario=x["id"], nombre=x["nombre"], estado=x["estado"],
                             **{f"costo_{k}_cop_mes": v for k, v in x["costos"].items()},
                             traslados=json.dumps(x["traslados"], ensure_ascii=False), maquila_m3=x["maquila_m3"],
                             no_atendida_m3=x["no_atendida_m3"], costo_por_m3=x["costo_por_m3"], km_promedio=x["km_promedio"]) for x in escenarios])
    flujos = pd.DataFrame([dict(escenario=x["id"], **f) for x in escenarios for f in x["flujos"]])
    hojas = {"Escenarios": res, "Flujos_optimos": flujos, "Bitacora_calidad": bitacora, "Demanda_CEDI_familia": demanda,
             "Tarifas_planta_CEDI": tarifas, "Plan_traslado_referencias": plan_det}
    for nombre, df in hojas.items():
        df.to_csv(out / f"{nombre.lower()}.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
    sap.drop(columns=["log_cop", "importe_txt", "volumen_txt"], errors="ignore").to_csv(
        out / "sap_xi_depurado.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
    try:
        with pd.ExcelWriter(out / "resultados_modelo.xlsx") as xw:
            for nombre, df in hojas.items():
                df.to_excel(xw, sheet_name=nombre[:31], index=False)
    except ImportError:
        print("   (instale openpyxl para obtener también resultados_modelo.xlsx)")

    # ---------------- Reporte HTML --------------------------------------
    rutas = por_ruta[por_ruta["origen"].isin(plantas)].copy()
    rutas["m3_mes"] = rutas["m3"] / meses
    datos = dict(
        generado=datetime.now().strftime("%Y-%m-%d %H:%M"), insumos=ins, escenarios=escenarios, historico=historico,
        min_viajes=P.MIN_VIAJES_TARIFA,
        calidad=dict(bitacora=bitacora.to_dict("records"), resumen=resumen, match=match,
                     hallazgos=hallazgos(bitacora, match, resumen, reg, tarifas, historico, ins)),
        tarifas=dict(regresion=reg, estimadas=[f"{r.origen}|{r.destino}" for r in tarifas.itertuples() if r.fuente != "Histórico SAP"],
                     rutas=[dict(origen=r.origen, destino=r.destino, km=float(r.km), cop_m3=float(r.cop_m3), viajes=int(r.viajes),
                                 m3_mes=float(r.m3_mes), utilizacion=None if pd.isna(r.utilizacion) else float(r.utilizacion))
                            for r in rutas.itertuples()]),
        plan_detalle=plan_det.to_dict("records"), metodologia=metodologia(ins),
        cfg_base=op.cfg_base(ins), grupos=op.GRUPOS)
    html = (AQUI / "plantilla_reporte.html").read_text(encoding="utf-8")
    js = json.dumps(datos, ensure_ascii=False, default=lambda o: None if pd.isna(o) else float(o)).replace("</", "<\\/")
    destino = out / "reporte_optimizacion.html"
    destino.write_text(html.replace("/*__DATA__*/null", js), encoding="utf-8")
    print(f"» Listo. Reporte: {destino}")
    print(f"         Excel:   {out / 'resultados_modelo.xlsx'}")
    if a.abrir:
        webbrowser.open(destino.as_uri())
    return destino


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as err:
        print(f"\n✖ {err}", file=sys.stderr)
        sys.exit(1)
