# Modelo de optimización de red: cierre de planta y traslado de líneas

Modelo en Python que:

1. **Valida y corrige** la información de SAP (fuente principal), TMS (complementaria, se une por *Viaje*) y los maestros del documento «Estructura Información»: Referencias, Clientes, Nodos e Info_CEDI.
2. **Optimiza** la red planta → CEDI con un modelo MILP (PuLP + CBC). Decide qué planta abastece a cada CEDI y qué líneas de la planta que cierra conviene trasladar, y a dónde.
3. **Genera un HTML interactivo** con los resultados y un simulador para mover variables. También genera un Excel con todas las tablas.

> Los archivos de `datos_entrada/` son **datos de prueba** con la misma estructura que su documento y con errores de calidad inyectados a propósito.

---

## Paso a paso para ejecutarlo en su PC con Visual Studio Code (Windows)

### Paso 1. Instalar lo necesario (una sola vez)

1. **Python 3.11 o 3.12**: descárguelo de <https://www.python.org/downloads/>.
   En el instalador marque **«Add python.exe to PATH»** antes de dar *Install Now*.
2. **Visual Studio Code**: <https://code.visualstudio.com/>.
3. En VS Code abra Extensiones (`Ctrl+Shift+X`), busque **Python** (de Microsoft) e instálela.

Para comprobar, abra una terminal de Windows y escriba `py --version`. Debe aparecer `Python 3.11.x` o `3.12.x`.

### Paso 2. Descargar el proyecto

Use una de estas dos opciones:

- **ZIP**: en GitHub, abra la rama `claude/supply-chain-optimization-model-qn8nut`, pulse **Code → Download ZIP** y descomprímalo, por ejemplo en `C:\Proyectos\CamiloLR`.
- **Git**:
  ```
  git clone -b claude/supply-chain-optimization-model-qn8nut https://github.com/calleri28/CamiloLR.git C:\Proyectos\CamiloLR
  ```

### Paso 3. Abrir la carpeta en VS Code

*File → Open Folder…* y seleccione **la carpeta `optimizacion_red`** (no la carpeta raíz del repositorio).
Si VS Code pregunta si confía en los autores, responda que sí.

### Paso 4. Crear el entorno virtual e instalar librerías (una sola vez)

Abra la terminal de VS Code (*Terminal → New Terminal*, o `` Ctrl+` ``) y ejecute:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Si PowerShell bloquea `Activate.ps1` («la ejecución de scripts está deshabilitada en este sistema»), tiene tres salidas. Ninguna requiere permisos de administrador:
  1. **Habilitar solo para esa terminal**: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y repetir `.venv\Scripts\Activate.ps1`. Hay que repetirlo cada vez que abra una terminal nueva.
  2. **Habilitar para su usuario**: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (responda `S`). Puede estar bloqueado por políticas de la empresa.
  3. **No activar el entorno**: use su Python directamente.
     `.venv\Scripts\python.exe -m pip install -r requirements.txt` y luego `.venv\Scripts\python.exe modelo_red.py --abrir`.
     El archivo `ejecutar_modelo.bat` y el botón F5 funcionan así, sin activar nada.
- Al quedar activo, la terminal muestra `(.venv)` al inicio de la línea.
- Luego seleccione el intérprete: `Ctrl+Shift+P` → **Python: Select Interpreter** → elija el que dice `.venv`.

### Paso 5. Ejecutar el modelo

Tiene tres formas de hacerlo:

| Forma | Cómo |
|---|---|
| **Botón Ejecutar (F5)** | Panel *Run and Debug* (`Ctrl+Shift+D`) → elija **«1. Ejecutar modelo (datos_entrada)»** → `F5` |
| **Terminal** | `python modelo_red.py --abrir` |
| **Doble clic** | `ejecutar_modelo.bat` (crea el entorno la primera vez y abre el reporte) |

Tarda unos segundos. En la terminal se ven el resumen de la limpieza y el costo de cada escenario, y al final el reporte se abre en el navegador.

### Paso 6. Revisar los resultados (carpeta `salida/`)

| Archivo | Contenido |
|---|---|
| `reporte_optimizacion.html` | Reporte interactivo: escenarios, simulador, calidad de datos, tarifas y metodología. Funciona sin internet y se puede enviar por correo. |
| `resultados_modelo.xlsx` | Escenarios, flujos óptimos, bitácora de calidad, demanda, tarifas y plan de traslado |
| `*.csv` | Las mismas tablas en CSV (separador `;`, coma decimal) |
| `sap_xi_depurado.csv` | Los traslados XI ya limpios y enriquecidos, para auditar |

### Paso 7. Cambiar a datos reales

1. Reemplace los archivos de `datos_entrada/` por los reales. Debe **conservar el nombre del archivo**, pero puede ser `.csv` o `.xlsx` (por ejemplo `sap_movimientos.xlsx`).
2. Si un encabezado se llama distinto, ajuste el diccionario `COLUMNAS_…` en `parametros.py`. Al comparar, se ignoran mayúsculas, tildes, espacios y guiones bajos.
3. Complete las tablas de apoyo (ver abajo): homologación de centros, capacidad y costos.
4. Ejecute de nuevo (`F5`). Si falta un archivo o una columna obligatoria, el mensaje dice exactamente cuál.

Para volver a crear los datos de prueba, use la configuración **«2. Regenerar datos de prueba y ejecutar»** o ejecute `python modelo_red.py --generar-prueba`.
**Ojo:** esta opción sobrescribe `datos_entrada/`.

---

## Archivos de entrada (`datos_entrada/`)

| Archivo | Origen | Obligatorio | Uso en el modelo |
|---|---|---|---|
| `sap_movimientos.csv` | Hoja **SAP** | Sí | Traslados XI: demanda, tarifas e historial |
| `tms_viajes.csv` | TMS (estructura supuesta: ID_Viaje, Fecha_Despacho, Placa, Tipo_Vehiculo, Capacidad_M3, Capacidad_Kg, Km_Recorridos, Costo_Viaje_COP) | No | Km reales, vehículo y ocupación cuando el viaje cruza |
| `maestro_referencias.csv` | Hoja **03_Referencias** | Sí | Familia, volumen por caja, plan de traslado (Situacion_Post_Cambio / Planta_Produccion_Nueva) y descontinuadas |
| `maestro_clientes.csv` | Hoja **04_Clientes** | No | Validación del campo *Cliente* de SAP y coordenadas de respaldo |
| `maestro_nodos.csv` | Hoja **05_Nodos** | Sí | Plantas y CEDI, coordenadas y planta que cierra (`Fecha_Cierre`) |
| `info_cedi.csv` | Hoja **Info_CEDI** + columna **ID_Nodo** | No | Capacidad de recibo (estibas/día) y ocupación máxima operable |
| `homologacion_centros.csv` | *Nueva*: Codigo_SAP → ID_Nodo | Sí | Une «Centro productor» y «CEDI» de SAP (CO45) con el maestro de nodos (PL01) |
| `capacidad_plantas.csv` | *Nueva*: ID_Nodo, Familia, Capacidad_M3_Mes | Sí | Capacidad de las líneas actuales |
| `parametros_plantas.csv` | *Nueva*: ID_Nodo, Costo_Fijo_COP_Mes, Factor_Costo_Variable, Max_Lineas_Recibir | Sí | Costos fijos y diferencias de costo entre plantas |
| `parametros_familias.csv` | *Nueva*: Familia, Costo_Var_Base_COP_M3, CAPEX_Traslado_Linea_COP, Pct_Capacidad_Trasladable | Sí | Costo de producción y de trasladar cada línea |

**Vacíos del documento de estructura que el modelo resuelve con supuestos** (conviene confirmarlos):

- **Info_CEDI no trae código de CEDI.** Se agregó `ID_Nodo` como primera columna. Si viene por zona, se suman las posiciones y se toma la capacidad de recibo del CEDI.
- **SAP no tiene origen y destino explícitos.** En los movimientos XI se toma *Centro productor* como origen y *CEDI* como destino.
- **La unidad de *Cantidad* no está definida.** Se valida contra `Volumen_Caja_m3` del maestro de referencias; en los datos de prueba corresponde a cajas.
- **SAP no trae peso.** El modelo trabaja en **m³**, que es lo que llena el vehículo en productos voluminosos.
- **El TMS no está descrito.** Se supuso la estructura mínima indicada arriba.
- ***Cliente* viene como «Nombre - Ciudad».** Se separa por « - ». Se recomienda incluir `Cod_Cliente` en el extracto.

## Validaciones (ver pestaña «Calidad de datos» del reporte)

| Tabla | Reglas principales |
|---|---|
| SAP | Tipo ≠ XI fuera de alcance · duplicados · periodos inválidos · reversiones · centros sin homologación · origen = destino · referencia fuera del maestro · Unidad Negocio vacía o distinta · Volumen vacío, en notación científica o inconsistente (corregido con el maestro) · **IMPORTE SAP en 0 o 1 peso: se omite del cálculo de tarifa** (el volumen sí cuenta como demanda) · COP/m³ atípico por ruta · sin viaje |
| TMS | Viaje normalizado (ceros a la izquierda) antes del cruce · km en cero o incoherentes → km teórico · viajes sin match: se conserva SAP |
| Referencias | Duplicados · Volumen_Caja_m3 vs largo × ancho × alto · «Trasladada» sin planta nueva · «Descontinuada» con planta nueva · sustituta inexistente · Control_Consistencia = OK con inconsistencias · «Sin cambio» en la planta que cierra |
| Nodos / Clientes | Longitud positiva (signo) · coordenadas faltantes o fuera de Colombia · nodos que no son planta ni CEDI |

Si quiere que las líneas con importe 0/1 tomen el costo del TMS cuando el viaje cruza, cambie `COMPLETAR_IMPORTE_CON_TMS = True` en `parametros.py`. Por defecto se omiten.

## Escenarios

| Id | Escenario |
|---|---|
| E0 | Red actual optimizada (referencia) |
| E1 | Cierre sin trasladar líneas: capacidad existente, horas extra y maquila |
| E2 | Cierre con el traslado que elige el modelo |
| E3 | Cierre con el **plan del maestro de referencias**, ponderado por volumen histórico |
| E4 / E5 | Sensibilidad: demanda +10 % y flete +15 % |

En el **simulador** puede:

- abrir o cerrar plantas y ajustar su capacidad;
- elegir el destino de cada línea («No trasladar», «Según maestro de referencias» o una planta), o pulsar **Optimizar traslados**;
- mover la demanda, el flete, el costo de producción, las horas extra, la maquila y la amortización;
- incluir o no el descuento de las referencias descontinuadas.

Todo se recalcula al instante, incluida la alerta de capacidad de recibo de los CEDI.

## Estructura del código

| Archivo | Qué hace |
|---|---|
| `parametros.py` | Rutas, nombres de columnas, reglas de calidad y supuestos (lo que normalmente se ajusta) |
| `calidad_datos.py` | Lectura robusta (CSV `;`/`,`, UTF-8/Latin-1, Excel), validaciones, bitácora y cruce SAP–TMS |
| `optimizador.py` | Demanda, tarifas, plan de traslado y modelo MILP |
| `modelo_red.py` | Punto de entrada: orquesta todo y genera el reporte y el Excel |
| `datos_prueba.py` | Generador de los CSV de prueba |
| `plantilla_reporte.html` | Reporte y simulador (JavaScript sin dependencias externas) |

## Problemas frecuentes

| Mensaje | Solución |
|---|---|
| `'py' no se reconoce…` | Reinstale Python marcando «Add python.exe to PATH», o use `python` en lugar de `py` |
| `ModuleNotFoundError: No module named 'pandas'` | No está activo el entorno: ejecute `.venv\Scripts\Activate.ps1` y elija el intérprete `.venv` en VS Code |
| `faltan columnas obligatorias: [...]` | El encabezado real es distinto: ajuste `COLUMNAS_…` en `parametros.py` |
| `Centro productor o CEDI sin homologación (COxx)` en la bitácora | Agregue ese código a `homologacion_centros.csv` |
| `Familias sin parámetros…` | Agregue la familia a `parametros_familias.csv` con el mismo nombre que en SAP o en el maestro |
| Tildes raras en un CSV abierto en Excel | Los CSV se guardan en UTF-8 con BOM; ábralos con doble clic o con *Datos → Desde texto/CSV* |
