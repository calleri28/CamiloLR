"""
EJECUTABLE PRINCIPAL — Modelo de optimización de red logística (cierre de planta y reubicación de producción)

Uso (desde la carpeta del proyecto, con el entorno virtual activo):
    python ejecutar_modelo.py
    python ejecutar_modelo.py --datos "C:/ruta/a/Datos" --salidas salidas --sin-sensibilidad

Genera en la carpeta de salidas:
    reporte_red_logistica.html     (dashboard interactivo + simulador)
    resultados_red_logistica.xlsx  (resultados consolidados + hoja Simulador con fórmulas)
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

from src.calidad_datos import validar  # noqa: E402
from src.carga_datos import cargar_todo  # noqa: E402
from src.escenarios import correr_escenarios, datos_simulador, punto_equilibrio, sensibilidad  # noqa: E402
from src.flujos import clientes_geo, flujos_cliente  # noqa: E402
from src.preparacion import preparar  # noqa: E402
from src.reporte_excel import generar_excel  # noqa: E402
from src.reporte_html import generar_html  # noqa: E402
from src.resumen import construir_resumen  # noqa: E402
from src.utilidades import configurar_log  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Modelo de optimización de red logística")
    ap.add_argument("--config", default=str(RAIZ / "parametros.yaml"), help="archivo de parámetros")
    ap.add_argument("--datos", default=None, help="carpeta con los CSV (por defecto la de parametros.yaml)")
    ap.add_argument("--salidas", default=None, help="carpeta de resultados")
    ap.add_argument("--sin-sensibilidad", action="store_true", help="omite el análisis de sensibilidad")
    ap.add_argument("--no-abrir", action="store_true", help="no abre el HTML al terminar")
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
    if args.sin_sensibilidad:
        cfg.setdefault("sensibilidad", {})["activa"] = False

    t0 = time.time()
    log.info("=== 1/6 Carga de datos desde %s", datos)
    d = cargar_todo(datos, salidas / "cache" if cfg["rutas"].get("usar_cache", True) else None)
    log.info("=== 2/6 Limpieza, homologación y parámetros")
    prep = preparar(d, cfg)
    log.info("=== 3/6 Validaciones de calidad de datos")
    calidad = validar(d, prep, cfg)
    log.info("    %d validaciones (%d severidad alta)", len(calidad), int((calidad.Severidad == "Alta").sum()))
    log.info("=== 4/6 Optimización de escenarios")
    res, tar = correr_escenarios(prep, cfg)
    sens = sensibilidad(prep, cfg, tar)
    equilibrio = punto_equilibrio(res)
    sim = datos_simulador(prep, cfg, tar, res)
    resumen = construir_resumen(prep, cfg, res, sens, calidad, equilibrio, sim)
    for linea in resumen["recomendacion_lista"]:
        log.info("  > %s", linea)
    log.info("=== 5/6 Excel")
    ruta_xlsx = salidas / "resultados_red_logistica.xlsx"
    generar_excel(ruta_xlsx, prep, cfg, res, calidad, sens, sim, equilibrio, resumen)
    log.info("    %s", ruta_xlsx)
    log.info("=== 6/6 HTML")
    ruta_html = salidas / "reporte_red_logistica.html"
    flujos = flujos_cliente(prep, cfg, res)
    generar_html(ruta_html, prep, cfg, res, calidad, sens, sim, equilibrio, resumen, flujos, clientes_geo(prep, d))
    log.info("    %s", ruta_html)
    log.info("Listo en %.0f s", time.time() - t0)
    if not args.no_abrir:
        try:
            webbrowser.open(ruta_html.resolve().as_uri())
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
