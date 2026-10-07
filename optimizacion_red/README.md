# Modelo de optimización de red: cierre de planta y traslado de líneas

Modelo en Python para evaluar escenarios de la red **planta → CEDI** cuando se cierra una planta y parte de su producción se traslada a otra. Hace tres cosas:

1. **Valida y corrige** los extractos de SAP (fuente principal) y TMS (fuente complementaria).
2. **Optimiza** la red con un modelo de programación entera mixta (MILP en PuLP).
3. **Genera un HTML autocontenido** con los resultados y un simulador donde se pueden mover las variables y ver el efecto al instante.

> Los valores de plantas, costos, capacidades y CAPEX en `parametros.py` son **datos de prueba**. Reemplácelos por los reales.

## Ejecución

```bash
pip install -r requirements.txt
python modelo_red.py                       # genera datos de prueba y corre todo
python modelo_red.py --sap sap.xlsx --tms tms.xlsx --materiales materiales.xlsx
```

Archivos generados en `salida/`:

| Archivo | Contenido |
|---|---|
| `reporte_optimizacion.html` | Reporte interactivo (se abre en cualquier navegador y funciona sin conexión) |
| `bitacora_calidad.csv` | Cada regla de calidad con los registros, toneladas e importe que afecta |
| `sap_xi_depurado.csv` | Traslados XI ya limpios y enriquecidos con TMS |
| `demanda_cedi_familia.csv` | Demanda neta mensual estimada por CEDI y familia |
| `tarifas_planta_cedi.csv` | Tarifa COP/t por ruta y su fuente (histórica o estimada) |
| `resultados_escenarios.csv`, `flujos_optimos.csv` | Resultados del MILP |
| `datos_prueba/` | Los CSV sintéticos con la estructura esperada (solo en modo prueba) |

## Estructura esperada de los datos

Si los encabezados reales son distintos, solo cambie `COLUMNAS_SAP` y `COLUMNAS_TMS` en `parametros.py`.

**SAP (principal)**: Documento, Posicion, Fecha_Contabilizacion, Tipo_Movimiento (`XI` = traslado planta/CEDI → CEDI), Centro_Origen, Centro_Destino, Material, Familia, Cantidad, Unidad, Peso_Kg, Importe_Flete_COP, Viaje, Transportista.

**TMS (complementario, se une por viaje)**: ID_Viaje, Fecha_Despacho, Placa, Tipo_Vehiculo, Capacidad_Kg, Km_Recorridos, Costo_Viaje_COP, Origen, Destino.

**Maestro de materiales (opcional)**: Material, Familia, Kg_Unidad. Se usa para imputar pesos faltantes.

## Validaciones (bitácora R01–R15)

| Regla | Qué detecta | Acción |
|---|---|---|
| R01 | Importes en texto con formato colombiano (`1.234.567,00`) | Se convierten a número |
| R02 | Tipo de movimiento o centro con espacios o en minúsculas (`" xi"`) | Se normalizan |
| R03 | Movimientos distintos a XI | Se excluyen (fuera de alcance) |
| R04 | Duplicados por documento + posición | Se excluyen |
| R05 | Fechas nulas o fuera de rango | Se excluyen |
| R06 | Cantidades negativas (reversiones) | Se excluyen |
| R07–R08 | Centros que no están en el maestro, u origen = destino | Se excluyen |
| R09 | Peso faltante | Se imputa con cantidad × kg/unidad |
| **R10** | **Flete en 0 o 1 peso** | **Se excluye del cálculo de tarifa; el volumen sí cuenta como demanda** |
| R11 | COP/kg atípico dentro de la ruta (IQR sobre el logaritmo) | Se excluye del cálculo de tarifa |
| R12 | Línea sin número de viaje | Se conserva, pero no se puede enriquecer con TMS |
| R13–R14 | Km del TMS en cero o incoherentes con la ruta | Se usa el km teórico |
| R15 | Viaje SAP sin match en TMS | Se conserva el SAP y se usa el km teórico |

El cruce SAP ↔ TMS **normaliza el número de viaje** (quita ceros a la izquierda y caracteres no numéricos) antes de unir. SAP siempre manda: el TMS solo aporta km, vehículo y capacidad cuando hay match.

## Modelo de optimización

Minimiza el costo mensual = producción variable + flete primario + horas extra + maquila + penalidad por demanda no atendida + costo fijo de plantas abiertas + CAPEX de traslado de líneas amortizado.

- **Variables**: toneladas planta → CEDI por familia; horas extra; maquila; demanda no atendida; binaria `z[i,f]` = trasladar la línea de la familia *f* a la planta *i*.
- **Restricciones**: atender la demanda; capacidad de planta más la capacidad de las líneas recibidas; tope de horas extra; una planta solo produce las familias que ya tiene o que recibe; cada línea va a máximo una planta; máximo de líneas por planta; tope de maquila.
- **Tarifas de rutas nuevas**: las rutas sin historia se estiman con una regresión COP/t = a + b·km ajustada sobre las rutas con datos.

### Escenarios precalculados

| Id | Escenario |
|---|---|
| E0 | Red actual optimizada (referencia) |
| E1 | Cierre sin traslado de líneas |
| E2 | Cierre con traslado óptimo (el modelo decide) |
| E3 | Cierre con el plan propuesto (`PLAN_TRASLADO_PROPUESTO`) |
| E4 / E5 | Sensibilidad: demanda +10 %, flete +15 % |

## Simulador en el HTML

Permite abrir o cerrar plantas, ajustar su capacidad, elegir a qué planta va cada línea (o pulsar **Optimizar traslados**), y mover el crecimiento de la demanda, el índice de flete (ACPM), el costo de producción, el tope de horas extra, la capacidad de maquila y los meses de amortización.

Con los traslados fijos, el problema es un flujo de costo mínimo, y el navegador lo resuelve de forma exacta en cada cambio. «Optimizar traslados» enumera todas las combinaciones, lo que equivale a las binarias del MILP. Para los seis escenarios se verificó que el simulador reproduce el costo del MILP de Python.

## Archivos

- `parametros.py`: mapeo de columnas, reglas de calidad, maestro de plantas y CEDIs, costos y capacidades
- `datos_prueba.py`: generador de datos sintéticos con problemas de calidad inyectados
- `calidad_datos.py`: limpieza, bitácora y cruce SAP–TMS
- `optimizador.py`: estimación de demanda y tarifas, y el MILP
- `modelo_red.py`: punto de entrada y generación del reporte
- `plantilla_reporte.html`: plantilla del reporte y del simulador (JavaScript sin dependencias externas)
