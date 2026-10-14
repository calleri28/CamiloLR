"""Exporta la base del modelo y varios casos resueltos en Python para compararlos con el motor del navegador (paridad.js)."""
import copy
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import yaml  # noqa: E402

from src.base_modelo import construir_base, parametros_escenario  # noqa: E402
from src.carga_datos import cargar_todo  # noqa: E402
from src.modelo_optimizacion import Tarifario  # noqa: E402
from src import modelo_diario as md  # noqa: E402
from src.preparacion import preparar  # noqa: E402

cfg = yaml.safe_load(open(RAIZ / "parametros.yaml", encoding="utf-8"))
datos = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / cfg["rutas"]["datos"]
d = cargar_todo(datos, RAIZ / cfg["rutas"]["salidas"] / "cache")
prep = preparar(d, cfg)
base = construir_base(prep, cfg, Tarifario(prep, cfg))
esc = {e["id"]: e for e in cfg["escenarios_diarios"]}
casos = {k: parametros_escenario(cfg, base, esc[k]) for k in ("S1_ACTUAL_OPT", "S2_SIN_PLANTA", "S3_OPTIMIZADO", "S5_SIN_BE")}
c = copy.deepcopy(casos["S3_OPTIMIZADO"])
c.update({"horizonte_dias": 14, "proyeccion": "perfil", "crec_E": 0.3})
c["transporte"]["metodo"] = "km"
c["servicio"]["lmax_h"] = 60
casos["T_KM_SERVICIO"] = c
c = copy.deepcopy(casos["S3_OPTIMIZADO"])
c["horizonte_dias"] = 14
c["extra"] = {"z": 5, "familia": 0, "planta": 0, "t": 3, "m3": 10, "nodo": 1}
casos["T_DEMANDA_EXTRA"] = c
out = {k: md.resolver(base, p, 600, 1e-7)["costos"]["Total"] for k, p in casos.items()}
S = Path(__file__).resolve().parent / "salida"
S.mkdir(exist_ok=True)
json.dump(base, open(S / "base.json", "w"))
json.dump(casos, open(S / "casos.json", "w"))
json.dump(out, open(S / "resultados_python.json", "w"))
print("Exportado en", S, "- ahora ejecute: node pruebas/paridad.js")
