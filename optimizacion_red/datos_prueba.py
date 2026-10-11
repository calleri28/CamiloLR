"""
Generador de DATOS DE PRUEBA con la estructura del documento
«Estructura Información» (hojas SAP, 03_Referencias, 04_Clientes, 05_Nodos,
Info_CEDI), más un TMS supuesto y las tablas de parámetros del modelo.

Escribe los CSV en `datos_entrada/` con separador «;» y coma decimal, como
los exporta normalmente un SAP o un Excel configurado para Colombia.

Inyecta a propósito los problemas de calidad típicos:
  * IMPORTE SAP en 0 o 1 peso, con dígito de más, o en texto «1.234.567,00»
  * Volumen vacío, en notación científica o con error de unidad
  * Viaje sin ceros a la izquierda o vacío; viajes sin match en el TMS
  * Tipo « xi», centros en minúscula, centros sin homologación, origen = destino
  * Poblaciones con y sin tilde (BOGOTÁ / BOGOTA), registros duplicados
  * Referencias que no existen en el maestro, Unidad Negocio vacía
  * Maestros con longitudes positivas, coordenadas vacías, volúmenes de caja
    en cm³, situaciones post-cambio inconsistentes
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import parametros as P

# --------------------------------------------------------------------------
# Mundo de prueba
# --------------------------------------------------------------------------
PLANTAS = [  # ID_Nodo, código SAP, nombre, ciudad, departamento, lat, lon
    ("PL01", "CO45", "Planta Tocancipá", "Tocancipá", "Cundinamarca", 4.965, -73.913),
    ("PL02", "CO46", "Planta Yumbo", "Yumbo", "Valle del Cauca", 3.585, -76.495),
    ("PL03", "CO47", "Planta Rionegro", "Rionegro", "Antioquia", 6.155, -75.374),
    ("PL04", "CO48", "Planta Malambo", "Malambo", "Atlántico", 10.859, -74.773),
]
PLANTA_QUE_CIERRA = "PL02"

CEDIS = [  # ID_Nodo, código SAP, nombre, ciudad, departamento, zona, lat, lon, peso demanda
    ("CD01", "CO60", "CEDI Funza", "Bogotá", "Bogotá D.C.", "Centro", 4.711, -74.072, 0.27),
    ("CD02", "CO61", "CEDI Itagüí", "Medellín", "Antioquia", "Antioquia", 6.244, -75.581, 0.16),
    ("CD03", "CO62", "CEDI Acopi", "Cali", "Valle del Cauca", "Suroccidente", 3.452, -76.532, 0.12),
    ("CD04", "CO63", "CEDI Barranquilla", "Barranquilla", "Atlántico", "Caribe", 10.964, -74.796, 0.08),
    ("CD05", "CO64", "CEDI Girón", "Bucaramanga", "Santander", "Santanderes", 7.119, -73.122, 0.07),
    ("CD06", "CO65", "CEDI Dosquebradas", "Pereira", "Risaralda", "Eje Cafetero", 4.813, -75.696, 0.05),
    ("CD07", "CO66", "CEDI Mamonal", "Cartagena", "Bolívar", "Caribe", 10.391, -75.479, 0.05),
    ("CD08", "CO67", "CEDI Villavicencio", "Villavicencio", "Meta", "Centro", 4.142, -73.626, 0.04),
    ("CD09", "CO68", "CEDI Pasto", "Pasto", "Nariño", "Suroccidente", 1.213, -77.281, 0.04),
    ("CD10", "CO69", "CEDI Neiva", "Neiva", "Huila", "Centro", 2.927, -75.281, 0.04),
    ("CD11", "CO70", "CEDI Ibagué", "Ibagué", "Tolima", "Centro", 4.438, -75.232, 0.04),
    ("CD12", "CO71", "CEDI Montería", "Montería", "Córdoba", "Caribe", 8.748, -75.881, 0.04),
]
# Lanzaderas cortas que solo hace la flota propia (sin viajes de terceros para comparar)
RUTAS_SOLO_FLOTA_PROPIA = {("PL01", "CD01"), ("PL03", "CD02"), ("PL04", "CD04")}
REEXPEDICION = {"CD08": ("CD01", 0.6), "CD09": ("CD03", 0.5), "CD12": ("CD07", 0.4)}

FAMILIAS = {  # nombre: demanda m3/mes, caja (L, A, H cm), peso caja kg, costo var COP/m3, CAPEX traslado
    "Family Care": (30_000, (60, 40, 35), 4.5, 180_000, 9_000_000_000),
    "Baby Care": (9_000, (50, 38, 30), 7.0, 520_000, 6_500_000_000),
    "Feminine Care": (5_000, (40, 30, 25), 4.0, 610_000, 5_500_000_000),
    "Professional Hygiene": (6_000, (55, 45, 40), 9.0, 240_000, 4_000_000_000),
}
CAPACIDAD = {  # (planta, familia): m3/mes (líneas actuales)
    ("PL01", "Family Care"): 13_000, ("PL01", "Baby Care"): 5_500, ("PL01", "Professional Hygiene"): 3_500,
    ("PL02", "Family Care"): 9_000, ("PL02", "Feminine Care"): 2_800, ("PL02", "Baby Care"): 4_500,
    ("PL03", "Feminine Care"): 3_500, ("PL03", "Professional Hygiene"): 3_500,
    ("PL04", "Family Care"): 10_000,
}
PARAM_PLANTAS = {"PL01": (2_400_000_000, 1.00, 2), "PL02": (1_900_000_000, 0.97, 2),
                 "PL03": (1_600_000_000, 1.03, 2), "PL04": (1_300_000_000, 1.01, 2)}
PLAN_TRASLADO = {"Family Care": {"PL04": 0.7, "PL01": 0.3}, "Baby Care": {"PL01": 1.0}, "Feminine Care": {"PL03": 1.0}}

VEHICULOS = {  # capacidad m3, capacidad kg, costo base por viaje, costo por km
    "TRACTOMULA": (90, 34_000, 900_000, 7_200),
    "DOBLETROQUE": (48, 17_000, 600_000, 4_800),
    "SENCILLO": (32, 9_000, 400_000, 3_300),
}
CADENAS = ["ÉXITO", "OLÍMPICA", "D1", "ARA", "CENCOSUD", "DROGUERÍAS CAFAM", "MAKRO", "DISTRIBUIDORA EL SOL",
           "SUPERTIENDAS LA 14", "COOPERATIVA CENTRAL", "ALKOSTO", "FARMATODO"]


def km_vial(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return max(15.0, 2 * r * math.asin(math.sqrt(a)) * P.FACTOR_SINUOSIDAD_VIAL)


def _num_co(v, dec=2):
    """Número con formato colombiano: 1.234.567,89"""
    return f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _sin_tilde(s):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# --------------------------------------------------------------------------
def generar(carpeta: Path | None = None, rng: np.random.Generator | None = None) -> Path:
    carpeta = Path(carpeta or P.CARPETA_ENTRADA)
    carpeta.mkdir(parents=True, exist_ok=True)
    rng = rng or np.random.default_rng(P.SEMILLA)
    nodos = {p[0]: dict(sap=p[1], nombre=p[2], ciudad=p[3], depto=p[4], lat=p[5], lon=p[6]) for p in PLANTAS}
    nodos.update({c[0]: dict(sap=c[1], nombre=c[2], ciudad=c[3], depto=c[4], zona=c[5], lat=c[6], lon=c[7], peso=c[8]) for c in CEDIS})

    refs = _referencias(rng)
    clientes = _clientes(rng)
    sap, tms = _movimientos(rng, nodos, refs, clientes)

    escribir = lambda df, clave: df.to_csv(carpeta / P.ARCHIVOS[clave], sep=";", decimal=",", index=False, encoding="utf-8-sig")
    escribir(sap, "sap")
    escribir(tms, "tms")
    escribir(_ensuciar_referencias(refs, rng), "referencias")
    escribir(clientes, "clientes")
    escribir(_nodos(nodos), "nodos")
    escribir(_info_cedi(rng), "info_cedi")
    escribir(pd.DataFrame([dict(Codigo_SAP=v["sap"], ID_Nodo=k) for k, v in nodos.items()]), "homologacion")
    escribir(pd.DataFrame([dict(ID_Nodo=i, Familia=f, Capacidad_M3_Mes=c) for (i, f), c in CAPACIDAD.items()]), "capacidad")
    escribir(pd.DataFrame([dict(ID_Nodo=i, Costo_Fijo_COP_Mes=v[0], Factor_Costo_Variable=_num_co(v[1]), Max_Lineas_Recibir=v[2],
                                Costo_Posicion_COP_Mes=30_000, Dias_Inventario_Planta=5)
                           for i, v in PARAM_PLANTAS.items()]), "param_plantas")
    escribir(pd.DataFrame([dict(ID_Nodo=c[0], Costo_Manejo_COP_Estiba=int(rng.integers(15, 22)) * 1000,
                                Costo_Posicion_COP_Mes=(62_000 if c[0] in ("CD07", "CD10") else int(rng.integers(40, 50)) * 1000),
                                Costo_Posicion_Externa_COP_Mes=78_000, Dias_Cobertura_Base=9)
                           for c in CEDIS]), "param_cedis")
    escribir(pd.DataFrame([dict(Familia=f, Costo_Var_Base_COP_M3=v[3], CAPEX_Traslado_Linea_COP=v[4], Pct_Capacidad_Trasladable="0,9")
                           for f, v in FAMILIAS.items()]), "param_familias")
    return carpeta


def _referencias(rng):
    filas, n = [], 10_045_000
    capaces = {f: [i for (i, ff) in CAPACIDAD if ff == f] for f in FAMILIAS}
    for f, (dem, (l, a, h), peso, *_r) in FAMILIAS.items():
        for k in range(24):
            n += int(rng.integers(3, 40))
            L, A, H = (round(x * rng.uniform(0.88, 1.12), 1) for x in (l, a, h))
            vol = L * A * H / 1e6
            planta = capaces[f][k % len(capaces[f])]
            sit, nueva, sust, fecha = "Sin cambio", planta, "", ""
            if planta == PLANTA_QUE_CIERRA:
                u = rng.random()
                fecha = "2027-01-01"
                if u < 0.78:
                    plan = PLAN_TRASLADO[f]
                    sit, nueva = "Trasladada", str(rng.choice(list(plan), p=list(plan.values())))
                elif u < 0.93:
                    sit, nueva = "Descontinuada", ""
            elif rng.random() < 0.04:
                sit, nueva, fecha = "Descontinuada", "", "2027-01-01"
            filas.append(dict(
                Cod_Ref=str(n), Descripcion_Ref=f"{f.upper()} {['PREMIUM', 'CLASICO', 'ECONOMICO', 'XL', 'PACK'][k % 5]} {int(rng.integers(4, 48))} UND",
                Familia=f, Subfamilia="", Categoria=f"CAT{['10', '20', '30', '40'][list(FAMILIAS).index(f)]}",
                Planta_Produccion_Antes=planta, Situacion_Post_Cambio=sit, Planta_Produccion_Nueva=nueva,
                Fecha_Efectiva_Cambio=fecha, Cod_Ref_Sustituta=sust, Control_Consistencia="OK",
                Unidad_Medida_Base="Caja", Unidades_x_Caja=int(rng.choice([6, 12, 24, 48])),
                Cajas_x_Estiba=int(1.7 / vol), Largo_Caja_cm=L, Ancho_Caja_cm=A, Alto_Caja_cm=H,
                Volumen_Caja_m3=round(vol, 4), Peso_Caja_kg=round(peso * rng.uniform(0.85, 1.15), 2),
                Altura_Estiba_m=1.75, Niveles_Apilamiento=int(rng.integers(2, 4)), Manejo_Especial="",
                Vida_Util_Dias=int(rng.choice([720, 1095])), Clasificacion_ABC=str(rng.choice(["A", "B", "C"], p=[0.2, 0.3, 0.5]))))
    refs = pd.DataFrame(filas)
    # La mitad de las descontinuadas de la planta que cierra tienen sustituta de la misma familia
    for idx in refs.index[(refs.Situacion_Post_Cambio == "Descontinuada")][::2]:
        f = refs.at[idx, "Familia"]
        cands = refs[(refs.Familia == f) & (refs.Situacion_Post_Cambio != "Descontinuada")]
        refs.at[idx, "Cod_Ref_Sustituta"] = cands.Cod_Ref.iloc[0]
    return refs


def _ensuciar_referencias(refs, rng):
    r = refs.copy()
    tras = r.index[r.Situacion_Post_Cambio == "Trasladada"]
    r.loc[tras[-2:], "Planta_Produccion_Nueva"] = ""                    # trasladada sin planta nueva
    desc = r.index[r.Situacion_Post_Cambio == "Descontinuada"]
    r.loc[desc[:1], "Planta_Produccion_Nueva"] = "PL01"                   # descontinuada con planta nueva
    r.loc[[3, 17], "Volumen_Caja_m3"] = r.loc[[3, 17], "Volumen_Caja_m3"] * 1e6   # quedó en cm3
    r["Volumen_Caja_m3"] = r["Volumen_Caja_m3"].map(lambda v: _num_co(v, 4))
    r = pd.concat([r, r.iloc[[8]]], ignore_index=True)                     # referencia duplicada
    return r


def _clientes(rng):
    filas, k = [], 0
    for c in CEDIS:
        for _ in range(5):
            k += 1
            ciudad = c[3]
            filas.append(dict(Cod_Cliente=f"CL{k:04d}", Nombre_Cliente=str(rng.choice(CADENAS)), Tipo_Cliente="",
                              Ciudad=ciudad if rng.random() > 0.3 else _sin_tilde(ciudad).upper(), Departamento=c[4],
                              Zona_Comercial=c[5], Latitud=_num_co(c[6] + rng.normal(0, 0.03), 5),
                              Longitud=_num_co(c[7] + rng.normal(0, 0.03), 5), Frecuencia_Entrega_Semana=int(rng.integers(1, 6)),
                              Ventana_Horaria=str(rng.choice(["06:00-12:00", "08:00-17:00", "22:00-05:00"])),
                              Vehiculo_Max_Permitido=str(rng.choice(["TRACTOMULA", "DOBLETROQUE", "SENCILLO"])),
                              Lead_Time_Comprometido_Dias=int(rng.integers(1, 5)), Prioridad_Cliente=str(rng.choice(["Alta", "Media", "Baja"]))))
    cl = pd.DataFrame(filas)
    cl.loc[[4, 22], "Longitud"] = cl.loc[[4, 22], "Longitud"].str.lstrip("-")   # signo perdido
    cl.loc[[9, 31], ["Latitud", "Longitud"]] = ""                                # sin coordenadas
    cl.loc[len(cl)] = dict(Cod_Cliente="CL9001", Nombre_Cliente="IMPORTADORA ANDINA", Tipo_Cliente="", Ciudad="Quito",
                           Departamento="EC-PICHINCHA", Zona_Comercial="Exportación", Latitud="-0,18065", Longitud="-78,46784",
                           Frecuencia_Entrega_Semana=1, Ventana_Horaria="08:00-17:00", Vehiculo_Max_Permitido="TRACTOMULA",
                           Lead_Time_Comprometido_Dias=8, Prioridad_Cliente="Media")
    return cl


def _nodos(nodos):
    filas = []
    for k, v in nodos.items():
        es_planta = k.startswith("PL")
        filas.append(dict(ID_Nodo=k, Nombre_Nodo=v["nombre"],
                          Tipo_Nodo="Planta" if es_planta else ("Centro tercerizado" if k in ("CD07", "CD10") else "Centro de distribución propio"),
                          Estado="Activo", Fecha_Inicio_Operacion="2012-01-01",
                          Fecha_Cierre="2026-12-31" if k == PLANTA_QUE_CIERRA else "",
                          Modalidad="Propio" if es_planta else ("3PL" if k in ("CD07", "CD10") else "Arrendado"),
                          Direccion="", Ciudad=v["ciudad"], Departamento=v["depto"],
                          Latitud=_num_co(v["lat"], 5), Longitud=_num_co(v["lon"], 5),
                          Observaciones="Cierre programado; producción se traslada según maestro de referencias" if k == PLANTA_QUE_CIERRA else ""))
    filas.append(dict(ID_Nodo="MQ01", Nombre_Nodo="Maquilador Andino", Tipo_Nodo="Maquilador/Proveedor", Estado="Activo",
                      Fecha_Inicio_Operacion="2020-01-01", Fecha_Cierre="", Modalidad="Tercero", Direccion="", Ciudad="Bogotá",
                      Departamento="Bogotá D.C.", Latitud="4,65000", Longitud="-74,10000", Observaciones="Capacidad de maquila"))
    df = pd.DataFrame(filas)
    df.loc[df.ID_Nodo == "CD07", "Longitud"] = df.loc[df.ID_Nodo == "CD07", "Longitud"].str.lstrip("-")   # signo perdido
    df.loc[df.ID_Nodo == "CD11", ["Latitud", "Longitud"]] = ""                                               # sin coordenadas
    return df


def _info_cedi(rng):
    filas = []
    total = sum(v[0] for v in FAMILIAS.values())
    for c in CEDIS:
        estibas_dia = total * c[8] / 1.7 / P.DIAS_OPERACION_MES
        holgura = 1.05 if c[0] == "CD05" else rng.uniform(1.2, 1.6)
        pos_factor = 0.8 if c[0] in ("CD05", "CD09") else 1.0
        for zona, frac in (("Almacenamiento general", 0.8), ("Picking", 0.2)):
            filas.append(dict(ID_Nodo=c[0], Zona_Almacenamiento=zona, Tipo_Almacenamiento="Rack selectivo" if zona != "Picking" else "Estantería",
                              Posiciones_Estiba=int(estibas_dia * 12 * frac * pos_factor), Dias_Operacion_Semana=6, Turnos_Dia=2, Horas_x_Turno=8,
                              Muelles_Recibo=max(2, int(estibas_dia / 40)), Muelles_Despacho=max(2, int(estibas_dia / 30)),
                              Cap_Recibo_Estibas_Dia=int(estibas_dia * holgura), Cap_Despacho_Estibas_Dia=int(estibas_dia * holgura * 1.1),
                              Peso_Max_Posicion_kg=1000, Posiciones_Ampliables=int(estibas_dia * 2), Personal_Operativo_Turno=int(estibas_dia / 6) + 4,
                              Equipos_Montacarga=max(2, int(estibas_dia / 25)), Restricciones_Horario="", Observaciones="",
                              Area_m2=int(estibas_dia * 25), Niveles=4, Altura_Max_Posicion_m="1,8", Volumen_Util_m3=int(estibas_dia * 12 * frac * 1.7),
                              Ocupacion_Promedio_Pct=int(rng.integers(60, 80)), Ocupacion_Pico_Pct=int(rng.integers(82, 98)),
                              Utilizacion_Max_Operable_Pct=88, Vigencia_Desde="2026-01-01", Muelles_Mixtos=1,
                              Cap_Recibo_Vehiculos_Dia=max(4, int(estibas_dia * holgura / 30)), Cap_Despacho_Vehiculos_Dia=max(6, int(estibas_dia / 15)),
                              Tiempo_Descargue_Min=60, Tiempo_Cargue_Min=75, Cap_Picking_Cajas_Dia=int(estibas_dia * 40)))
    return pd.DataFrame(filas)


def _asignacion_actual(nodos):
    """Abastecimiento AS-IS (no óptimo): planta capaz más cercana 75 %, segunda 25 %."""
    asign = {}
    for f in FAMILIAS:
        capaces = [i for (i, ff) in CAPACIDAD if ff == f]
        for c in [c[0] for c in CEDIS]:
            orden = sorted(capaces, key=lambda p: km_vial(nodos[p]["lat"], nodos[p]["lon"], nodos[c]["lat"], nodos[c]["lon"]))
            asign[(c, f)] = [(orden[0], 0.75), (orden[1], 0.25)] if len(orden) > 1 else [(orden[0], 1.0)]
    return asign


def _movimientos(rng, nodos, refs, clientes):
    sap, tms = [], []
    viaje_id = 4_500_000
    refs_pf = {k: g for k, g in refs.groupby(["Planta_Produccion_Antes", "Familia"])}
    refs_f = {k: g for k, g in refs.groupby("Familia")}
    meses = pd.date_range("2025-10-01", periods=12, freq="MS")
    asign = _asignacion_actual(nodos)

    def poblacion(ciudad):
        u = rng.random()
        return ciudad.upper() if u < 0.5 else (_sin_tilde(ciudad).upper() if u < 0.85 else ciudad)

    def viaje(origen, destino, vol_m3, mes, fam, tipo_mov="XI", cliente=None, ciudad_dest=None, ref_pool=None):
        nonlocal viaje_id
        o, d = nodos[origen], nodos[destino] if destino in nodos else None
        lat2, lon2 = (d["lat"], d["lon"]) if d else cliente[1]
        km = km_vial(o["lat"], o["lon"], lat2, lon2)
        if vol_m3 < 2:
            return
        tipo = "TRACTOMULA" if km > 120 and rng.random() < 0.85 else str(rng.choice(list(VEHICULOS)))
        if vol_m3 < VEHICULOS[tipo][0] * 0.6:
            tipo = next((t for t in ("SENCILLO", "DOBLETROQUE", "TRACTOMULA") if VEHICULOS[t][0] * 0.95 >= vol_m3), "TRACTOMULA")
        cap_m3, cap_kg, base, por_km = VEHICULOS[tipo]
        n = math.ceil(vol_m3 / (cap_m3 * 0.92))
        pool = ref_pool if ref_pool is not None else refs_pf.get((origen, fam), refs_f[fam])
        for _ in range(n):
            carga = min(cap_m3 * 0.99, vol_m3 / n * rng.uniform(0.95, 1.05))
            km_real = km * rng.uniform(0.97, 1.08)
            costo = (base + por_km * km_real) * rng.lognormal(0, 0.07)
            # Flota propia: SAP no liquida flete (queda en 0 o 1 peso) y el TMS no trae costo
            propia = tipo_mov == "XI" and ((origen, destino) in RUTAS_SOLO_FLOTA_PROPIA
                                           or (km < 600 and tipo != "SENCILLO" and rng.random() < 0.25))
            importe_propia = int(rng.choice([0, 1]))
            viaje_id += 1
            fecha = mes + pd.Timedelta(days=int(rng.integers(0, 28)))
            partes = rng.dirichlet(np.ones(int(rng.integers(1, 6))))
            for frac in partes:
                ref = pool.iloc[int(rng.integers(len(pool)))]
                cajas = max(1, round(carga * frac / ref.Volumen_Caja_m3))
                vol = cajas * ref.Volumen_Caja_m3
                productor = ref.Planta_Produccion_Antes if tipo_mov != "XI" else origen
                sap.append({"Viaje": f"{viaje_id:010d}",
                            "Cliente": f"{cliente[0]} - {ciudad_dest.upper()}" if cliente else f"{d['nombre'].upper()} - {d['ciudad'].upper()}",
                            "Referencia": ref.Cod_Ref, "Unidad Negocio": fam,
                            "Centro productor": nodos[productor]["sap"], "CEDI": nodos[destino]["sap"] if d else o["sap"],
                            "CorredorTransporte": f"{_sin_tilde(o['ciudad']).upper()} - {_sin_tilde(ciudad_dest or d['ciudad']).upper()}",
                            "Poblacion": poblacion(ciudad_dest or d["ciudad"]), "Periodo": fecha.strftime("%Y-%m-%d 00:00:00"),
                            "Mercado": "N", "Tipo": tipo_mov, "Cantidad": cajas,
                            "IMPORTE SAP": importe_propia if propia else round(costo * frac), "Volumen": round(vol, 3)})
            tms.append({"ID_Viaje": str(viaje_id), "Fecha_Despacho": fecha.strftime("%Y-%m-%d"),
                        "Placa": f"{''.join(rng.choice(list('ABCDEFGHJKLMNPRSTUVWXYZ'), 3))}{rng.integers(100, 999)}",
                        "Tipo_Vehiculo": tipo, "Capacidad_M3": cap_m3, "Capacidad_Kg": cap_kg,
                        "Km_Recorridos": round(km_real, 1), "Costo_Viaje_COP": "" if propia else round(costo * rng.uniform(0.98, 1.03)),
                        "Tipo_Flota": "PROPIA" if propia else "TERCERO"})

    cl_por_ciudad = {}
    for c in CEDIS:
        sub = clientes[clientes.Departamento == c[4]].head(5)
        cl_por_ciudad[c[0]] = [(r.Nombre_Cliente, (c[6] + rng.normal(0, 0.05), c[7] + rng.normal(0, 0.05)), c[3]) for r in sub.itertuples()]

    for mes in meses:
        estac = 1 + 0.10 * math.sin((mes.month - 3) / 12 * 2 * math.pi) + (0.12 if mes.month == 12 else 0)
        for (c, f), fuentes in asign.items():
            dem = FAMILIAS[f][0] * nodos[c]["peso"] * estac * rng.uniform(0.93, 1.07)
            hub, frac_hub = REEXPEDICION.get(c, (None, 0))
            for planta, w in fuentes:
                if hub:
                    viaje(planta, hub, dem * w * frac_hub, mes, f)
                    viaje(planta, c, dem * w * (1 - frac_hub), mes, f)
                else:
                    viaje(planta, c, dem * w, mes, f)
            if hub:
                viaje(hub, c, dem * frac_hub, mes, f, ref_pool=refs_f[f])
            # Distribución secundaria (AU): una muestra de entregas a clientes del CEDI
            for nombre, (la, lo), ciudad in cl_por_ciudad[c][:2]:
                viaje(c, None, dem * 0.03, mes, f, tipo_mov="AU", cliente=(nombre, (la, lo)), ciudad_dest=ciudad, ref_pool=refs_f[f])

    sap = pd.DataFrame(sap)
    tms = pd.DataFrame(tms)
    sap.loc[sap.sample(n=3, random_state=5).index, ["Mercado", "Cliente", "Poblacion"]] = ["E", "IMPORTADORA ANDINA - QUITO", "QUITO"]
    sap.loc[sap.sample(frac=0.01, random_state=6).index, "Tipo"] = "Z5"
    return _ensuciar_sap(sap, rng), _ensuciar_tms(tms)


def _ensuciar_sap(df, rng):
    df = df.copy()
    n = len(df)
    for c in ["IMPORTE SAP", "Volumen", "Viaje", "Periodo", "Referencia"]:
        df[c] = df[c].astype(object)

    def idx(p):
        return rng.choice(n, int(n * p), replace=False)

    df.loc[idx(0.02), "IMPORTE SAP"] = rng.choice([0, 1], int(n * 0.02))                    # 0 o 1 peso sin liquidar
    i = idx(0.01); df.loc[i, "IMPORTE SAP"] = [float(v) * 10 for v in df.loc[i, "IMPORTE SAP"]]   # dígito de más
    i = idx(0.005); df.loc[i, "IMPORTE SAP"] = [float(v) / 10 for v in df.loc[i, "IMPORTE SAP"]]
    i = idx(0.10); df.loc[i, "IMPORTE SAP"] = [_num_co(float(v)) for v in df.loc[i, "IMPORTE SAP"]]   # texto 1.234.567,00
    df["Volumen"] = [_num_co(float(v), 3) for v in df["Volumen"]]
    i = idx(0.02); df.loc[i, "Volumen"] = [f"{float(str(v).replace('.', '').replace(',', '.')):.1E}".replace(".", ",") for v in df.loc[i, "Volumen"]]
    i = idx(0.01); df.loc[i, "Volumen"] = [_num_co(float(str(v).replace(".", "").replace(",", ".")) * 1000, 1) for v in df.loc[i, "Volumen"]]   # en litros
    df.loc[idx(0.03), "Volumen"] = ""
    i = rng.choice(df.index[df["Tipo"] == "XI"], int(n * 0.05), replace=False)
    df.loc[i, "Tipo"] = rng.choice([" xi", "Xi ", "XI "], len(i))
    i = idx(0.05); df.loc[i, "CEDI"] = [f" {str(v).lower()} " for v in df.loc[i, "CEDI"]]
    i = idx(0.003); df.loc[i, "CEDI"] = "CO99"
    xi = df.index[df.Tipo == "XI"]
    j = rng.choice(xi, 40, replace=False)
    df.loc[j, "CEDI"] = df.loc[j, "Centro productor"]                                         # origen = destino
    i = idx(0.006); df.loc[i, "Cantidad"] = -df.loc[i, "Cantidad"]
    df.loc[idx(0.003), "Periodo"] = "1900-01-01 00:00:00"
    df.loc[idx(0.03), "Viaje"] = ""
    i = idx(0.06); df.loc[i, "Viaje"] = [str(v).lstrip("0") for v in df.loc[i, "Viaje"]]
    df.loc[idx(0.02), "Unidad Negocio"] = ""
    df.loc[idx(0.005), "Referencia"] = "99999999"
    i = idx(0.05); df.loc[i, "Referencia"] = ["00000000" + str(v) for v in df.loc[i, "Referencia"]]   # ceros SAP
    dup = df.sample(frac=0.02, random_state=7)
    return pd.concat([df, dup], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)


def _ensuciar_tms(tms):
    tms = tms.copy()
    tms = tms.drop(tms.sample(frac=0.15, random_state=11).index)          # viajes que no llegan al TMS
    tms.loc[tms.sample(frac=0.02, random_state=12).index, "Km_Recorridos"] = 0
    i = tms.sample(frac=0.01, random_state=14).index
    tms.loc[i, "Km_Recorridos"] = tms.loc[i, "Km_Recorridos"] * 10
    extra = tms.sample(frac=0.05, random_state=13).copy()
    extra["ID_Viaje"] = [str(9_000_000 + k) for k in range(len(extra))]
    return pd.concat([tms, extra], ignore_index=True)


if __name__ == "__main__":
    destino = generar()
    print(f"Datos de prueba generados en: {destino}")
