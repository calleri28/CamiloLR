"""
Configuración del modelo.

Aquí está todo lo que se ajusta sin tocar la lógica: rutas de archivos,
nombres de columnas (según el documento «Estructura Información»), reglas
de calidad y supuestos de negocio. Los valores numéricos son de PRUEBA.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Archivos de entrada (CSV con ; o , — o Excel .xlsx con el mismo nombre)
# ---------------------------------------------------------------------------
CARPETA_ENTRADA = Path(__file__).resolve().parent / "datos_entrada"
CARPETA_SALIDA = Path(__file__).resolve().parent / "salida"

ARCHIVOS = {
    # Extractos transaccionales
    "sap": "sap_movimientos.csv",                 # hoja SAP (principal)
    "tms": "tms_viajes.csv",                      # TMS (complementario, opcional)
    # Maestros del documento de estructura
    "referencias": "maestro_referencias.csv",     # hoja 03_Referencias
    "clientes": "maestro_clientes.csv",           # hoja 04_Clientes
    "nodos": "maestro_nodos.csv",                 # hoja 05_Nodos
    "info_cedi": "info_cedi.csv",                 # hoja Info_CEDI
    # Tablas de apoyo que el documento no trae y el modelo necesita
    "homologacion": "homologacion_centros.csv",   # código SAP (CO45) -> ID_Nodo (PL01)
    "capacidad": "capacidad_plantas.csv",         # capacidad por planta y familia (líneas)
    "param_plantas": "parametros_plantas.csv",    # costo fijo, factor de costo, máx. líneas
    "param_familias": "parametros_familias.csv",  # costo variable, CAPEX de traslado
}

# ---------------------------------------------------------------------------
# 2. Nombres de columnas: nombre interno -> encabezado en el archivo.
#    La comparación ignora mayúsculas, tildes, espacios y guiones bajos,
#    así que "IMPORTE SAP", "Importe_SAP" o "importe sap" funcionan igual.
# ---------------------------------------------------------------------------
COLUMNAS_SAP = {
    "viaje": "Viaje",
    "cliente": "Cliente",
    "referencia": "Referencia",
    "unidad_negocio": "Unidad Negocio",
    "centro_productor": "Centro productor",
    "cedi": "CEDI",
    "corredor": "CorredorTransporte",
    "poblacion": "Poblacion",
    "periodo": "Periodo",
    "mercado": "Mercado",
    "tipo": "Tipo",
    "cantidad": "Cantidad",
    "importe": "IMPORTE SAP",
    "volumen": "Volumen",
}

# El documento no describe el TMS: esta es la estructura mínima supuesta.
COLUMNAS_TMS = {
    "viaje": "ID_Viaje",
    "fecha_despacho": "Fecha_Despacho",
    "placa": "Placa",
    "tipo_vehiculo": "Tipo_Vehiculo",
    "capacidad_m3": "Capacidad_M3",
    "capacidad_kg": "Capacidad_Kg",
    "km": "Km_Recorridos",
    "costo_tms": "Costo_Viaje_COP",
}

COLUMNAS_REFERENCIAS = {
    "cod_ref": "Cod_Ref", "descripcion": "Descripcion_Ref", "familia": "Familia",
    "subfamilia": "Subfamilia", "categoria": "Categoria",
    "planta_antes": "Planta_Produccion_Antes", "situacion": "Situacion_Post_Cambio",
    "planta_nueva": "Planta_Produccion_Nueva", "fecha_cambio": "Fecha_Efectiva_Cambio",
    "ref_sustituta": "Cod_Ref_Sustituta", "control": "Control_Consistencia",
    "unidad_base": "Unidad_Medida_Base", "unid_caja": "Unidades_x_Caja",
    "cajas_estiba": "Cajas_x_Estiba", "largo_cm": "Largo_Caja_cm", "ancho_cm": "Ancho_Caja_cm",
    "alto_cm": "Alto_Caja_cm", "vol_caja_m3": "Volumen_Caja_m3", "peso_caja_kg": "Peso_Caja_kg",
    "altura_estiba_m": "Altura_Estiba_m", "niveles": "Niveles_Apilamiento",
    "manejo_especial": "Manejo_Especial", "vida_util": "Vida_Util_Dias", "abc": "Clasificacion_ABC",
}

COLUMNAS_CLIENTES = {
    "cod_cliente": "Cod_Cliente", "nombre": "Nombre_Cliente", "tipo": "Tipo_Cliente",
    "ciudad": "Ciudad", "departamento": "Departamento", "zona": "Zona_Comercial",
    "lat": "Latitud", "lon": "Longitud", "frecuencia": "Frecuencia_Entrega_Semana",
    "ventana": "Ventana_Horaria", "vehiculo_max": "Vehiculo_Max_Permitido",
    "lead_time": "Lead_Time_Comprometido_Dias", "prioridad": "Prioridad_Cliente",
}

COLUMNAS_NODOS = {
    "id_nodo": "ID_Nodo", "nombre": "Nombre_Nodo", "tipo": "Tipo_Nodo", "estado": "Estado",
    "fecha_inicio": "Fecha_Inicio_Operacion", "fecha_cierre": "Fecha_Cierre",
    "modalidad": "Modalidad", "direccion": "Direccion", "ciudad": "Ciudad",
    "departamento": "Departamento", "lat": "Latitud", "lon": "Longitud",
    "observaciones": "Observaciones",
}

# La hoja Info_CEDI no trae el código del CEDI: se agregó ID_Nodo como primera columna.
COLUMNAS_INFO_CEDI = {
    "id_nodo": "ID_Nodo", "zona": "Zona_Almacenamiento", "tipo_alm": "Tipo_Almacenamiento",
    "posiciones": "Posiciones_Estiba", "dias_semana": "Dias_Operacion_Semana",
    "turnos": "Turnos_Dia", "horas_turno": "Horas_x_Turno",
    "muelles_recibo": "Muelles_Recibo", "muelles_despacho": "Muelles_Despacho",
    "cap_recibo_estibas_dia": "Cap_Recibo_Estibas_Dia", "cap_despacho_estibas_dia": "Cap_Despacho_Estibas_Dia",
    "peso_max_posicion": "Peso_Max_Posicion_kg", "posiciones_ampliables": "Posiciones_Ampliables",
    "personal_turno": "Personal_Operativo_Turno", "montacargas": "Equipos_Montacarga",
    "restricciones": "Restricciones_Horario", "observaciones": "Observaciones", "area_m2": "Area_m2",
    "niveles": "Niveles", "altura_max": "Altura_Max_Posicion_m", "volumen_util_m3": "Volumen_Util_m3",
    "ocup_prom": "Ocupacion_Promedio_Pct", "ocup_pico": "Ocupacion_Pico_Pct",
    "util_max_operable": "Utilizacion_Max_Operable_Pct", "vigencia": "Vigencia_Desde",
    "muelles_mixtos": "Muelles_Mixtos", "cap_recibo_veh_dia": "Cap_Recibo_Vehiculos_Dia",
    "cap_despacho_veh_dia": "Cap_Despacho_Vehiculos_Dia", "t_descargue": "Tiempo_Descargue_Min",
    "t_cargue": "Tiempo_Cargue_Min", "cap_picking": "Cap_Picking_Cajas_Dia",
}

COLUMNAS_HOMOLOGACION = {"codigo_sap": "Codigo_SAP", "id_nodo": "ID_Nodo"}
COLUMNAS_CAPACIDAD = {"id_nodo": "ID_Nodo", "familia": "Familia", "capacidad_m3_mes": "Capacidad_M3_Mes"}
COLUMNAS_PARAM_PLANTAS = {"id_nodo": "ID_Nodo", "costo_fijo_mes": "Costo_Fijo_COP_Mes",
                          "factor_costo_var": "Factor_Costo_Variable", "max_lineas": "Max_Lineas_Recibir"}
COLUMNAS_PARAM_FAMILIAS = {"familia": "Familia", "costo_var_m3": "Costo_Var_Base_COP_M3",
                           "capex_traslado": "CAPEX_Traslado_Linea_COP", "pct_trasladable": "Pct_Capacidad_Trasladable"}

# ---------------------------------------------------------------------------
# 3. Reglas de calidad de datos
# ---------------------------------------------------------------------------
TIPO_MOV_TRASLADO = "XI"            # traslado planta/CEDI -> CEDI
IMPORTE_MINIMO_VALIDO = 1.0         # importes <= 1 peso (0 o 1) se omiten del cálculo de tarifa
FECHA_MIN, FECHA_MAX = "2020-01-01", "2030-12-31"
OUTLIER_IQR_FACTOR = 3.0            # atípico de COP/m3 por ruta: fuera de Q1-k·IQR, Q3+k·IQR (en log)
TOLERANCIA_VOLUMEN = (0.5, 2.0)     # Volumen SAP / (Cantidad × Volumen_Caja_m3) fuera de este rango = inconsistente
MIN_VIAJES_TARIFA = 5               # mínimo de viajes válidos para usar la tarifa histórica de la ruta
FACTOR_SINUOSIDAD_VIAL = 1.35       # km carretera ≈ km en línea recta × factor (topografía andina)
LIMITES_COLOMBIA = dict(lat=(-4.3, 13.5), lon=(-82.0, -66.8))
# Si es True, las líneas SAP con importe 0/1 toman el costo del TMS prorrateado por volumen
# cuando el viaje cruza. Por defecto se omiten, como pide el negocio.
COMPLETAR_IMPORTE_CON_TMS = False

# Tipos de nodo (se comparan sin tildes y en mayúsculas, por contenido)
PALABRA_PLANTA = "PLANTA"
PALABRAS_CEDI = ("CENTRO DE DISTRIBUCION", "CEDI", "BODEGA", "TERCERIZADO", "3PL")

# ---------------------------------------------------------------------------
# 4. Supuestos del modelo de optimización
# ---------------------------------------------------------------------------
PLANTA_CIERRE = None                 # None = se detecta en maestro de nodos (Fecha_Cierre o Estado)
HORAS_EXTRA_MAX_PCT = 0.12           # capacidad adicional con horas extra / turno adicional
SOBRECOSTO_HORAS_EXTRA_PCT = 0.18    # sobrecosto sobre el costo variable
COSTO_MAQUILA_PCT = 0.35             # maquila: sobrecosto sobre el costo variable base de la familia
MAQUILA_MAX_M3_MES = 3000            # capacidad de maquila disponible en el mercado
PENALIDAD_NO_ATENDIDA_COP_M3 = 900_000   # venta perdida por m3 no atendido
MESES_AMORTIZACION = 60
DIAS_OPERACION_MES = 26              # para convertir capacidad diaria de recibo de CEDI a mensual
SEMILLA = 42                         # datos de prueba reproducibles
