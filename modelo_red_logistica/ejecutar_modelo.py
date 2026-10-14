"""
EJECUTABLE PRINCIPAL — Modelo diario de optimización de la red logística (cierre de planta y reubicación de producción)

Uso (desde la carpeta del proyecto, con el entorno virtual activo):
    python ejecutar_modelo.py
    python ejecutar_modelo.py --datos "C:/ruta/a/Datos" --salidas salidas --sin-pruebas --sin-combinaciones

Genera en la carpeta de salidas:
    reporte_red_logistica.html     (dashboard de 7 vistas + simulación con el mismo optimizador en el navegador)
    resultados_red_logistica.xlsx  (resultados consolidados + validador de rutas con fórmulas)
    log_ejecucion.txt
"""
from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

import yaml  # noqa: E402

from src import pruebas as pr  # noqa: E402
from src.base_modelo import construir_base  # noqa: E402
from src.calidad_datos import validar  # noqa: E402
from src.carga_datos import cargar_todo  # noqa: E402
from src.escenarios_diarios import combinaciones_bodegas, correr  # noqa: E402
from src.flujos import clientes_geo  # noqa: E402
from src.modelo_optimizacion import Tarifario  # noqa: E402
from src.preparacion import preparar  # noqa: E402
from src.reporte_excel import generar_excel  # noqa: E402
from src.reporte_html import _viajes_xi, generar_html  # noqa: E402
from src.resumen import construir_resumen  # noqa: E402
from src.utilidades import configurar_log  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Modelo diario de optimización de red logística")
    ap.add_argument("--config", default=str(RAIZ / "parametros.yaml"), help="archivo de parámetros")
    ap.add_argument("--datos", default=None, help="carpeta con los CSV (por defecto la de parametros.yaml)")
    ap.add_argument("--salidas", default=None, help="carpeta de resultados")
    ap.add_argument("--sin-pruebas", action="store_true", help="omite las pruebas de consistencia A-H (~2 min)")
    ap.add_argument("--sin-combinaciones", action="store_true", help="omite las combinaciones de bodegas externas (~30 s)")
    ap.add_argument("--no-abrir", action="store_true", help="no abre el HTML al terminar")
    ap.add_argument("--guardar-corrida", default=None, help="(avanzado) guarda los resultados en un archivo pickle")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    datos = Path(args.datos or cfg["rutas"]["datos"])
    salidas = Path(args.salidas or cfg["rutas"]["salidas"])
    if not datos.is_absolute():
        datos = RAIZ / datos
    if not salidas.is_absolute():
        salidas = RAIZ / salidas
    salidas.mkdir(parents=True, exist_ok=True)
    log = configurar_log(salidas / "log_ejecucion.txt")
    if args.sin_combinaciones:
        cfg.setdefault("combinaciones_bodegas", {})["activa"] = False

    t0 = time.time()
    log.info("=== 1/7 Carga de datos desde %s", datos)
    d = cargar_todo(datos, salidas / "cache" if cfg["rutas"].get("usar_cache", True) else None)
    log.info("=== 2/7 Limpieza, homologación, imputación de fletes XI de flota propia")
    prep = preparar(d, cfg)
    log.info("=== 3/7 Validaciones de calidad de datos")
    calidad = validar(d, prep, cfg)
    log.info("    %d validaciones (%d severidad alta)", len(calidad), int((calidad.Severidad == "Alta").sum()))
    log.info("=== 4/7 Base del modelo diario (demanda por día, tarifas, capacidades)")
    base = construir_base(prep, cfg, Tarifario(prep, cfg))
    log.info("=== 5/7 Escenarios (Actual reconstruido + optimizaciones MILP con HiGHS)")
    corrida = correr(base, cfg)
    combos = combinaciones_bodegas(base, cfg)
    if args.sin_pruebas:
        pruebas = []
    else:
        log.info("=== 6/7 Pruebas de consistencia A-H")
        pruebas = pr.ejecutar(base, cfg, corrida, combos)
        for p in pruebas:
            log.info("    %s %-28s %s · %s", p["caso"], p["prueba"], p["estado"], p["resultado"])
    resumen = construir_resumen(base, cfg, corrida, combos, pruebas, prep.info)
    if args.guardar_corrida:
        import pickle
        with open(args.guardar_corrida, "wb") as fh:
            pickle.dump({"base": base, "cfg": cfg, "corrida": corrida, "combos": combos, "pruebas": pruebas, "calidad": calidad}, fh)
    for linea in resumen["reco"]:
        log.info("  > %s", linea)
    log.info("=== 7/7 Excel y HTML")
    ruta_xlsx = salidas / "resultados_red_logistica.xlsx"
    generar_excel(ruta_xlsx, base, cfg, corrida, combos, pruebas, calidad, resumen, _viajes_xi(prep))
    log.info("    %s", ruta_xlsx)
    ruta_html = salidas / "reporte_red_logistica.html"
    generar_html(ruta_html, prep, cfg, base, corrida, combos, pruebas, calidad, resumen, clientes_geo(prep, d))
    log.info("    %s", ruta_html)
    log.info("Listo en %.0f s", time.time() - t0)
    if not args.no_abrir:
        try:
            webbrowser.open(ruta_html.resolve().as_uri())
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
