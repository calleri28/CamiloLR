"""
Parámetros de la red logística y del modelo.

Todo lo que es "supuesto de negocio" vive aquí para que se pueda ajustar sin
tocar la lógica. Los valores son DATOS DE PRUEBA (ilustrativos), no reales.
"""

# ---------------------------------------------------------------------------
# 1. Mapeo de columnas: nombre interno -> nombre en el archivo fuente.
#    Si los extractos reales traen otros encabezados, solo se cambia el valor.
# ---------------------------------------------------------------------------
COLUMNAS_SAP = {
    "documento": "Documento",          # Doc. material / documento de transporte
    "posicion": "Posicion",
    "fecha": "Fecha_Contabilizacion",
    "tipo_mov": "Tipo_Movimiento",     # XI = traslado planta/CEDI -> CEDI
    "origen": "Centro_Origen",
    "destino": "Centro_Destino",
    "material": "Material",
    "familia": "Familia",
    "cantidad": "Cantidad",
    "unidad": "Unidad",
    "peso_kg": "Peso_Kg",
    "importe_flete": "Importe_Flete_COP",
    "viaje": "Viaje",
    "transportista": "Transportista",
}

COLUMNAS_TMS = {
    "viaje": "ID_Viaje",
    "fecha_despacho": "Fecha_Despacho",
    "placa": "Placa",
    "tipo_vehiculo": "Tipo_Vehiculo",
    "capacidad_kg": "Capacidad_Kg",
    "km": "Km_Recorridos",
    "costo_tms": "Costo_Viaje_COP",
    "origen": "Origen",
    "destino": "Destino",
}

# ---------------------------------------------------------------------------
# 2. Reglas de calidad de datos
# ---------------------------------------------------------------------------
TIPO_MOV_TRASLADO = "XI"
IMPORTE_MINIMO_VALIDO = 1.0          # importes <= a este valor (0 o 1 peso) se omiten
FECHA_MIN = "2020-01-01"
FECHA_MAX = "2030-12-31"
OUTLIER_IQR_FACTOR = 3.0             # COP/kg fuera de Q1-k*IQR, Q3+k*IQR por ruta = atípico
MIN_REGISTROS_TARIFA = 5             # mínimo de viajes para usar tarifa histórica de la ruta
FACTOR_SINUOSIDAD_VIAL = 1.35        # km carretera ≈ km línea recta × factor (topografía andina)

# ---------------------------------------------------------------------------
# 3. Maestro de nodos (coordenadas aproximadas de ciudades colombianas)
# ---------------------------------------------------------------------------
PLANTAS = {
    # código: nombre, lat, lon, capacidad (t/mes), costo fijo (COP/mes),
    #         familias que produce hoy, costo variable producción (COP/t) por familia
    "P100": dict(nombre="Planta Tocancipá", ciudad="Tocancipá", lat=4.965, lon=-73.913,
                 capacidad_t=9000, costo_fijo=2_400_000_000,
                 familias=["BEBIDAS", "SNACKS", "ASEO_HOGAR"],
                 costo_var={"BEBIDAS": 410_000, "SNACKS": 1_150_000, "ASEO_HOGAR": 760_000, "CUIDADO_PERSONAL": 1_380_000}),
    "P200": dict(nombre="Planta Yumbo (cierre)", ciudad="Yumbo", lat=3.585, lon=-76.495,
                 capacidad_t=6000, costo_fijo=1_900_000_000,
                 familias=["BEBIDAS", "CUIDADO_PERSONAL", "SNACKS"],
                 costo_var={"BEBIDAS": 395_000, "SNACKS": 1_120_000, "ASEO_HOGAR": 740_000, "CUIDADO_PERSONAL": 1_300_000}),
    "P300": dict(nombre="Planta Rionegro", ciudad="Rionegro", lat=6.155, lon=-75.374,
                 capacidad_t=5500, costo_fijo=1_600_000_000,
                 familias=["CUIDADO_PERSONAL", "ASEO_HOGAR"],
                 costo_var={"BEBIDAS": 425_000, "SNACKS": 1_170_000, "ASEO_HOGAR": 745_000, "CUIDADO_PERSONAL": 1_340_000}),
    "P400": dict(nombre="Planta Malambo", ciudad="Malambo", lat=10.859, lon=-74.773,
                 capacidad_t=4500, costo_fijo=1_300_000_000,
                 familias=["BEBIDAS"],
                 costo_var={"BEBIDAS": 400_000, "SNACKS": 1_160_000, "ASEO_HOGAR": 770_000, "CUIDADO_PERSONAL": 1_360_000}),
}

PLANTA_CIERRE = "P200"

CEDIS = {
    "C01": dict(nombre="CEDI Bogotá", ciudad="Bogotá", lat=4.711, lon=-74.072),
    "C02": dict(nombre="CEDI Medellín", ciudad="Medellín", lat=6.244, lon=-75.581),
    "C03": dict(nombre="CEDI Cali", ciudad="Cali", lat=3.452, lon=-76.532),
    "C04": dict(nombre="CEDI Barranquilla", ciudad="Barranquilla", lat=10.964, lon=-74.796),
    "C05": dict(nombre="CEDI Bucaramanga", ciudad="Bucaramanga", lat=7.119, lon=-73.122),
    "C06": dict(nombre="CEDI Pereira", ciudad="Pereira", lat=4.813, lon=-75.696),
    "C07": dict(nombre="CEDI Cartagena", ciudad="Cartagena", lat=10.391, lon=-75.479),
    "C08": dict(nombre="CEDI Villavicencio", ciudad="Villavicencio", lat=4.142, lon=-73.626),
    "C09": dict(nombre="CEDI Pasto", ciudad="Pasto", lat=1.213, lon=-77.281),
    "C10": dict(nombre="CEDI Neiva", ciudad="Neiva", lat=2.927, lon=-75.281),
    "C11": dict(nombre="CEDI Ibagué", ciudad="Ibagué", lat=4.438, lon=-75.232),
    "C12": dict(nombre="CEDI Montería", ciudad="Montería", lat=8.748, lon=-75.881),
}

FAMILIAS = ["BEBIDAS", "SNACKS", "ASEO_HOGAR", "CUIDADO_PERSONAL"]

# ---------------------------------------------------------------------------
# 4. Parámetros de la decisión de traslado de líneas
# ---------------------------------------------------------------------------
# Capacidad que aporta una línea trasladada desde la planta que cierra (t/mes)
CAPACIDAD_LINEA_TRASLADADA_T = {"BEBIDAS": 2600, "SNACKS": 900, "ASEO_HOGAR": 1500, "CUIDADO_PERSONAL": 1700}
# CAPEX de traslado + montaje + curva de aprendizaje, en COP (se amortiza)
CAPEX_TRASLADO_LINEA = {"BEBIDAS": 7_500_000_000, "SNACKS": 4_200_000_000,
                        "ASEO_HOGAR": 5_000_000_000, "CUIDADO_PERSONAL": 5_800_000_000}
MESES_AMORTIZACION = 60
# Una planta puede recibir como máximo este número de líneas trasladadas (espacio físico)
MAX_LINEAS_RECIBIDAS_POR_PLANTA = 2

# Holguras para mantener el modelo factible y medir tensión de capacidad
HORAS_EXTRA_MAX_PCT = 0.12            # % de capacidad adicional con horas extra / turno adicional
SOBRECOSTO_HORAS_EXTRA_PCT = 0.18     # sobrecosto sobre costo variable de producción
COSTO_MAQUILA_PCT = 0.35              # tercerización (maquila): sobrecosto sobre el costo var. de la planta más barata
PENALIDAD_DEMANDA_NO_ATENDIDA = 9_000_000   # COP/t (venta perdida, margen + nivel de servicio)

# Generador de datos de prueba
SEMILLA = 42
MESES_HISTORIA = 12

# Tercerización (maquila): capacidad total disponible en el mercado (t/mes)
MAQUILA_MAX_T_MES = 1200

# Plan corporativo a evaluar (escenario "propuesto"): familia de la planta que
# cierra -> planta receptora de la línea
PLAN_TRASLADO_PROPUESTO = {"BEBIDAS": "P100", "CUIDADO_PERSONAL": "P100"}
