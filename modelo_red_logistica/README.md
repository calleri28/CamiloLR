# Modelo de optimización de red logística: cierre de planta y reubicación de producción

Este modelo responde, con los datos de SAP, TMS y los maestros, a la pregunta:

> Con el cierre de la planta **CO10 (Medellín)** y el traslado de su producción a otras plantas,
> **¿es mejor usar una bodega externa o trasladar a un CEDI de otra ciudad que tenga capacidad?**

Una sola ejecución hace todo esto:

1. **Valida la calidad de los datos**: 52 validaciones sobre SAP, TMS y los 7 maestros.
2. **Limpia y homologa la información.** Omite los fletes en 0 o 1 peso, normaliza el número de viaje para cruzar SAP con TMS, homologa las plantas del maestro de referencias y los clientes que en realidad son nodos.
3. **Construye los parámetros**: demanda por ciudad, tarifas por ruta, capacidades y costos.
4. **Optimiza** un modelo de programación lineal entera mixta (MILP) para 5 escenarios y corre una sensibilidad.
5. **Genera los resultados**:
   - `salidas/reporte_red_logistica.html`: tablero interactivo. Trae un **simulador** que vuelve a optimizar en el navegador cuando usted mueve los parámetros. Funciona sin internet.
   - `salidas/resultados_red_logistica.xlsx`: resultados consolidados. Incluye la hoja **Simulador** con fórmulas vivas y tablas con filtros.

---

## Paso a paso para ejecutarlo en su PC con Visual Studio Code (Windows)

### 1. Instale los requisitos (solo la primera vez)
1. **Python 3.11, 3.12 o 3.13**: descárguelo de https://www.python.org/downloads/.
   En el instalador marque **"Add python.exe to PATH"**.
2. **Visual Studio Code**: https://code.visualstudio.com/.
3. En VS Code abra Extensiones (`Ctrl+Shift+X`) e instale **Python** (de Microsoft).

### 2. Arme la carpeta del proyecto
Copie la carpeta `modelo_red_logistica` a su equipo, por ejemplo en `C:\Proyectos\modelo_red_logistica`.
Descomprima `Datos.rar` dentro de la carpeta `datos`. Debe quedar así:

```
modelo_red_logistica/
├── ejecutar_modelo.py          <- EJECUTABLE PRINCIPAL
├── ejecutar_windows.bat        <- alternativa: doble clic
├── parametros.yaml             <- TODOS los supuestos editables
├── requirements.txt
├── README.md
├── .vscode/                    <- configuración para ejecutar con F5
├── datos/
│   ├── SAP.csv
│   ├── TMS.csv
│   ├── maestro_nodos.csv, maestro_referencias.csv, maestro_clientes.csv
│   ├── info_cedi.csv, capacidad_plantas.csv, parametros_plantas.csv
│   └── homologacion_centros.csv     (también sirve si quedan dentro de datos/Datos/)
├── src/
│   ├── carga_datos.py          lectura de los CSV
│   ├── preparacion.py          limpieza, homologación, tarifas, demanda y capacidades
│   ├── calidad_datos.py        validaciones de calidad
│   ├── modelo_optimizacion.py  modelo MILP (HiGHS vía SciPy)
│   ├── escenarios.py           escenarios, sensibilidad y modelo del simulador
│   ├── resumen.py              recomendación y hallazgos automáticos
│   ├── reporte_excel.py        Excel con simulador de fórmulas
│   ├── reporte_html.py         tablero HTML
│   └── recursos/               plantilla HTML, simulador JS y solver JS (sin internet)
└── salidas/                    (se crea al ejecutar)
```

### 3. Abra la carpeta en VS Code
`Archivo > Abrir carpeta...` y seleccione `modelo_red_logistica`.

### 4. Cree el entorno virtual e instale las librerías (solo la primera vez)
Abra la terminal de VS Code (`Ctrl+ñ` o `Terminal > Nueva terminal`) y ejecute:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Si PowerShell bloquea `Activate.ps1`, ejecute una sola vez:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelva a activar.
> En Mac o Linux use `python3 -m venv .venv` y `source .venv/bin/activate`.

Luego, en VS Code: `Ctrl+Shift+P` > **Python: Select Interpreter** > elija `.venv`.

### 5. Ejecute el modelo
Puede hacerlo de cualquiera de estas formas:
- **F5** (o `Ejecutar > Iniciar depuración`) y elija **"Ejecutar modelo de red"**.
- En la terminal: `python ejecutar_modelo.py`
- Doble clic en `ejecutar_windows.bat`. La primera vez crea el entorno solo.

Opciones disponibles:
```powershell
python ejecutar_modelo.py --datos "D:\otra\ruta\Datos"   # leer los CSV de otra carpeta
python ejecutar_modelo.py --sin-sensibilidad             # corrida más rápida
python ejecutar_modelo.py --no-abrir                     # no abrir el HTML al terminar
```
La corrida toma alrededor de 1 minuto: leer 1,9 millones de líneas de SAP es lo que más tarda. Desde la segunda vez usa caché y es más rápida.
Al terminar se abre el HTML. Los resultados quedan en `salidas/`.

### 6. Modifique escenarios
Edite `parametros.yaml` y vuelva a ejecutar. Puede cambiar, entre otros:
- `dias_inventario`: `auto` lo calibra con los datos; también puede poner un número, por ejemplo `15`.
- `utilizacion_maxima`, `costos_por_modalidad` (manejo por m3 y costo por posición) y `nodos_opcionales_tobe`.
- `bodegas_candidatas`: posiciones, costo fijo, tarifa por posición y manejo de cada bodega externa cotizada. Puede agregar más.
- `escenarios`: agregue o quite escenarios. Cada uno define cómo se produce, si se cierra la planta, si hay bodegas candidatas y si se limitan los traslados.
- `umbral_importe_invalido`, `periodo_base`, `percentil_outlier_tarifa` y `homologacion`.

---

## Qué hace el modelo

### Reglas de negocio con los datos
| Regla | Implementación |
|---|---|
| SAP es la fuente principal | La demanda, las tarifas y los traslados salen de SAP. TMS solo complementa. |
| Tipo **XI** = traslado entre nodos | El campo CEDI es el origen; el Cliente (C716 Rionegro, C718 Cajicá, C720 Tocancipá...) es el nodo destino. |
| Fletes en 0 o 1 peso | Se omiten del cálculo de tarifas (numerador y denominador). Su volumen sí cuenta como demanda. |
| Cruce SAP–TMS por viaje | SAP trae ceros a la izquierda (`0984938`). Al normalizarlos, el cruce pasa de 77,6 % a **96,2 %**. |
| Complemento TMS | Origen real del traslado, tiempos de tránsito, m3 por viaje (para estimar viajes por mes) y población faltante en SAP. |
| Planta TO-BE | Sale del maestro de referencias. Sus códigos (CO05, CO06, CO12...) se homologan a nodos SAP por volumen. |
| Tarifas | COP/m3 por ruta observada. Las rutas sin historia se estiman con una regresión por distancia. |
| Capacidad | Posiciones × m3 por posición (cajas por estiba × m3 por caja) × utilización máxima. |

### Escenarios incluidos
| Id | Escenario |
|---|---|
| E0 | AS-IS optimizado (CO10 operando) |
| E1 | **Opción B**: traslado a CEDI de otra ciudad, sin bodega externa |
| E2 | **Opción A**: bodega externa. Se mantiene CO99 y puede abrir candidatas; los traslados a otros CEDI se limitan al nivel AS-IS. |
| E3 | Óptimo TO-BE: el modelo combina libremente |
| E4 | Óptimo TO-BE reasignando producción entre plantas compatibles |

### Formulación
Ver la pestaña **Metodología** del HTML o el encabezado de `src/modelo_optimizacion.py`.
- **Objetivo**: minimizar fletes de traslado + fletes de distribución + manejo + posiciones + costos fijos + penalización por falta de capacidad.
- **Restricciones**: demanda, balance en almacén, capacidad de almacenamiento (con abrir o cerrar bodegas), capacidad de planta y límite de traslados en la Opción A.

### Interacción
- **HTML > Simulador interactivo**: mueva los días de inventario, la utilización, los factores de flete y el crecimiento. También puede editar las posiciones y los costos de cada almacén. El navegador resuelve las tres opciones (A, B y Óptimo) en milisegundos y muestra la recomendación, el costo fijo de equilibrio de la bodega externa y la ocupación.
- **Excel > Simulador**: las celdas amarillas son editables (días, volumen, planta origen, reparto por zona, costos y posiciones). Las fórmulas recalculan el costo de cada opción, revisan si cabe, arman el ranking y la decisión (bodega externa o traslado a CEDI) y actualizan la gráfica.
- **Excel, resto de hojas**: tablas con filtros para escenarios, flujos, almacenes, calidad de datos, tarifas, tiempos TMS, homologaciones, demanda y parámetros.

## Supuestos que debe validar
Los valores marcados `[SUPUESTO]` en `parametros.yaml` no vienen en los datos:
- Costos variables de manejo por modalidad.
- Costos de las bodegas externas candidatas.
- Utilización máxima: `info_cedi` trae vacío `Utilizacion_Max_Operable_Pct`.
- Días de inventario: se calibran asumiendo que Cajicá está lleno, porque hoy desborda a CO99.

El modelo no incluye costos de producción por planta ni restricciones de nivel de servicio: los campos de lead time del maestro de clientes están vacíos.
