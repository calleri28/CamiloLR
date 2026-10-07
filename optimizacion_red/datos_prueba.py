"""
Generador de DATOS DE PRUEBA con la estructura de los extractos SAP y TMS.

Simula 12 meses de traslados (XI) planta -> CEDI y CEDI -> CEDI, más otros
movimientos (ventas, devoluciones) y, a propósito, inyecta los problemas de
calidad típicos de la información real:
  * importes de flete en 0 o 1 peso
  * importes con formato texto colombiano ("1.234.567,00") y errores de digitación (x10, /10)
  * viaje con ceros a la izquierda en SAP y sin ellos en TMS; viajes sin match
  * centros con espacios / minúsculas, tipo de movimiento "xi " sucio
  * pesos faltantes, cantidades negativas (reversiones), duplicados
  * fechas imposibles, origen = destino, km en 0 en TMS
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import parametros as P

VEHICULOS = {
    # tipo: capacidad kg, costo base por viaje COP, costo por km COP
    "TRACTOMULA": (34_000, 900_000, 7_200),
    "DOBLETROQUE": (17_000, 600_000, 4_800),
    "SENCILLO": (9_000, 400_000, 3_300),
}

KG_POR_CAJA = {"BEBIDAS": 14.5, "SNACKS": 4.2, "ASEO_HOGAR": 11.0, "CUIDADO_PERSONAL": 6.8}

# Demanda base (t/mes) por familia y participación de cada CEDI
DEMANDA_FAMILIA_T = {"BEBIDAS": 10_000, "SNACKS": 2_500, "ASEO_HOGAR": 3_500, "CUIDADO_PERSONAL": 4_000}
PESO_CEDI = {"C01": 0.27, "C02": 0.16, "C03": 0.12, "C04": 0.08, "C05": 0.07, "C06": 0.05,
             "C07": 0.05, "C08": 0.04, "C09": 0.04, "C10": 0.04, "C11": 0.04, "C12": 0.04}

# Abastecimiento indirecto hoy: CEDI destino <- CEDI hub (fracción de su demanda)
REEXPEDICION = {"C08": ("C01", 0.6), "C09": ("C03", 0.5), "C12": ("C07", 0.4)}


def km_vial(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return max(15.0, 2 * r * math.asin(math.sqrt(a)) * P.FACTOR_SINUOSIDAD_VIAL)


def _coord(codigo):
    n = P.PLANTAS.get(codigo) or P.CEDIS[codigo]
    return n["lat"], n["lon"]


def _asignacion_actual():
    """Abastecimiento AS-IS (no óptimo): planta capaz más cercana 75 %, segunda 25 %."""
    asign = {}
    for f in P.FAMILIAS:
        capaces = [p for p, d in P.PLANTAS.items() if f in d["familias"]]
        for c in P.CEDIS:
            orden = sorted(capaces, key=lambda p: km_vial(*_coord(p), *_coord(c)))
            asign[(c, f)] = [(orden[0], 0.75), (orden[1], 0.25)] if len(orden) > 1 else [(orden[0], 1.0)]
    return asign


def generar(rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng(P.SEMILLA)
    materiales = []
    for f in P.FAMILIAS:
        for k in range(6):
            materiales.append(dict(Material=f"{f[:3]}{1000 + k}", Familia=f,
                                   Kg_Unidad=round(KG_POR_CAJA[f] * rng.uniform(0.7, 1.3), 2)))
    maestro_mat = pd.DataFrame(materiales)
    mats_fam = {f: maestro_mat[maestro_mat.Familia == f] for f in P.FAMILIAS}

    asign = _asignacion_actual()
    sap, tms = [], []
    doc, viaje_id = 4_900_000_000, 4_500_000
    meses = pd.date_range("2025-10-01", periods=P.MESES_HISTORIA, freq="MS")

    def emitir_viajes(origen, destino, toneladas, mes, f):
        nonlocal doc, viaje_id
        km = km_vial(*_coord(origen), *_coord(destino))
        kg_total = toneladas * 1000
        if kg_total < 300:
            return
        tipo = "TRACTOMULA" if km > 120 and rng.random() < 0.85 else str(rng.choice(["DOBLETROQUE", "SENCILLO", "TRACTOMULA"]))
        if kg_total < VEHICULOS[tipo][0] * 0.6:   # carga pequeña: vehículo más chico que alcance
            tipo = next((t for t in ("SENCILLO", "DOBLETROQUE", "TRACTOMULA") if VEHICULOS[t][0] * 0.95 >= kg_total), "TRACTOMULA")
        cap, base, por_km = VEHICULOS[tipo]
        n_viajes = math.ceil(kg_total / (cap * 0.90))
        for _ in range(n_viajes):
            carga = min(cap * 0.99, kg_total / n_viajes * rng.uniform(0.95, 1.05))
            km_real = km * rng.uniform(0.97, 1.08)
            costo = (base + por_km * km_real) * rng.lognormal(0, 0.07)
            viaje_id += 1
            fecha = mes + pd.Timedelta(days=int(rng.integers(0, 28)))
            n_lineas = int(rng.integers(1, 5))
            partes = rng.dirichlet(np.ones(n_lineas))
            doc += 1
            for pos, frac in enumerate(partes, start=1):
                m = mats_fam[f].iloc[int(rng.integers(len(mats_fam[f])))]
                kg = carga * frac
                sap.append({"Documento": str(doc), "Posicion": pos * 10, "Fecha_Contabilizacion": fecha,
                            "Tipo_Movimiento": "XI", "Centro_Origen": origen, "Centro_Destino": destino,
                            "Material": m.Material, "Familia": f, "Cantidad": round(kg / m.Kg_Unidad),
                            "Unidad": "CJ", "Peso_Kg": round(kg, 1), "Importe_Flete_COP": round(costo * frac),
                            "Viaje": f"{viaje_id:010d}", "Transportista": rng.choice(["TRANS-A", "TRANS-B", "TRANS-C", "TRANS-D"])})
            tms.append({"ID_Viaje": str(viaje_id), "Fecha_Despacho": fecha, "Placa": f"T{rng.integers(100, 999)}{rng.integers(10, 99)}",
                        "Tipo_Vehiculo": tipo, "Capacidad_Kg": cap, "Km_Recorridos": round(km_real, 1),
                        "Costo_Viaje_COP": round(costo * rng.uniform(0.98, 1.03)),
                        "Origen": origen, "Destino": destino})

    for mes in meses:
        estac = 1 + 0.12 * math.sin((mes.month - 3) / 12 * 2 * math.pi) + (0.15 if mes.month == 12 else 0)
        for (c, f), fuentes in asign.items():
            dem = DEMANDA_FAMILIA_T[f] * PESO_CEDI[c] * estac * rng.uniform(0.92, 1.08)
            hub, frac_hub = REEXPEDICION.get(c, (None, 0))
            for planta, w in fuentes:
                destino_directo = hub if hub else c
                if hub:
                    emitir_viajes(planta, hub, dem * w * frac_hub, mes, f)
                    emitir_viajes(planta, c, dem * w * (1 - frac_hub), mes, f)
                else:
                    emitir_viajes(planta, destino_directo, dem * w, mes, f)
            if hub:
                emitir_viajes(hub, c, dem * frac_hub, mes, f)

    sap = pd.DataFrame(sap)
    tms = pd.DataFrame(tms)

    # Otros movimientos (distribución secundaria / devoluciones) que deben filtrarse
    otros = sap.sample(frac=0.25, random_state=1).copy()
    otros["Tipo_Movimiento"] = rng.choice(["VE", "DV", "Z01"], len(otros), p=[0.7, 0.2, 0.1])
    otros["Documento"] = (otros["Documento"].astype(int) + 7_000_000_000).astype(str)
    sap = pd.concat([sap, otros], ignore_index=True)

    sap = _ensuciar_sap(sap, rng)
    tms = _ensuciar_tms(tms, sap, rng)
    return sap, tms, maestro_mat


def _ensuciar_sap(df, rng):
    n = len(df)
    df = df.copy()
    df["Importe_Flete_COP"] = df["Importe_Flete_COP"].astype(object)
    df["Peso_Kg"] = df["Peso_Kg"].astype(object)
    df["Fecha_Contabilizacion"] = df["Fecha_Contabilizacion"].dt.strftime("%Y-%m-%d").astype(object)

    def idx(p):
        return rng.choice(n, int(n * p), replace=False)

    df.loc[idx(0.07), "Importe_Flete_COP"] = rng.choice([0, 1], int(n * 0.07))          # 0 o 1 peso
    i = idx(0.01); df.loc[i, "Importe_Flete_COP"] = [float(v) * 10 for v in df.loc[i, "Importe_Flete_COP"]]   # dígito de más
    i = idx(0.005); df.loc[i, "Importe_Flete_COP"] = [float(v) / 10 for v in df.loc[i, "Importe_Flete_COP"]]
    i = idx(0.10)                                                                       # formato texto colombiano
    df.loc[i, "Importe_Flete_COP"] = [f"{float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") for v in df.loc[i, "Importe_Flete_COP"]]
    df.loc[idx(0.03), "Peso_Kg"] = None
    i = idx(0.08); df.loc[i, "Tipo_Movimiento"] = rng.choice([" xi", "Xi ", "XI "], len(i))
    i = idx(0.05); df.loc[i, "Centro_Destino"] = [f" {str(v).lower()} " for v in df.loc[i, "Centro_Destino"]]
    i = idx(0.004); df.loc[i, "Centro_Destino"] = df.loc[i, "Centro_Origen"]
    i = idx(0.003); df.loc[i, "Centro_Destino"] = "C99"
    i = idx(0.006); df.loc[i, "Cantidad"] = -df.loc[i, "Cantidad"]
    df.loc[idx(0.003), "Fecha_Contabilizacion"] = "1900-01-01"
    df.loc[idx(0.03), "Viaje"] = None
    i = idx(0.05); df.loc[i, "Viaje"] = [str(v).lstrip("0") if v else v for v in df.loc[i, "Viaje"]]
    dup = df.sample(frac=0.02, random_state=7)
    return pd.concat([df, dup], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)


def _ensuciar_tms(tms, sap, rng):
    tms = tms.copy()
    # 15 % de los viajes de SAP no llegan al TMS
    tms = tms.drop(tms.sample(frac=0.15, random_state=11).index)
    i = tms.sample(frac=0.02, random_state=12).index
    tms.loc[i, "Km_Recorridos"] = 0
    i = tms.sample(frac=0.01, random_state=14).index
    tms.loc[i, "Km_Recorridos"] = tms.loc[i, "Km_Recorridos"] * 10      # error de digitación
    # viajes del TMS que no son de traslados (otras operaciones)
    extra = tms.sample(frac=0.05, random_state=13).copy()
    extra["ID_Viaje"] = [str(9_000_000 + k) for k in range(len(extra))]
    return pd.concat([tms, extra], ignore_index=True)
