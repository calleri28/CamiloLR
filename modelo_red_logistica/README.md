# Modelo diario de optimización de red logística: cierre de planta y reubicación de producción

Este modelo responde, con los datos de SAP, TMS y los maestros, a la pregunta:

> Con el cierre de la planta **CO10 (Medellín)** y el traslado de su producción a otras plantas, ¿cómo debe atenderse cada cliente
> y cada referencia, día a día, al menor costo logístico sin perder servicio? En particular, **¿conviene una bodega externa o un traslado XI
> a un CEDI de otra ciudad con capacidad?**

Una sola ejecución hace todo esto:

1. **Valida la calidad de los datos**: 53 validaciones sobre SAP, TMS y los 7 maestros.
2. **Limpia, homologa e imputa.** Omite los fletes en 0/1 COP de las tarifas y cruza SAP con TMS por viaje (SAP manda). A los XI de flota propia les imputa el flete: por viajes similares o por costo fijo + COP/km.
3. **Construye la base diaria**: demanda por día, ciudad y producto (familia × planta), tarifas por tramo, tiempos de tránsito y capacidades en posiciones.
4. **Reconstruye la operación actual** con los despachos y traslados observados.
5. **Optimiza** un modelo MILP multiperiodo diario (HiGHS) para 7 escenarios y todas las combinaciones de bodegas externas.
6. **Corre las pruebas de consistencia A–H**: balance, traslado ficticio, inventario acumulado, capacidad, bodegas, rutas, servicio y coherencia.
7. **Genera los resultados**:
   - `salidas/reporte_red_logistica.html`: tablero de 7 vistas. La **Simulación** vuelve a optimizar en el navegador con el mismo modelo y solver (HiGHS WebAssembly). Funciona sin internet.
   - `salidas/resultados_red_logistica.xlsx`: resultados consolidados y la hoja **Validador_rutas** con fórmulas vivas.

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
├── datos/                      <- SAP.csv, TMS.csv y maestros (también sirve datos/Datos/)
├── src/
│   ├── carga_datos.py          lectura de los CSV
│   ├── preparacion.py          limpieza, homologación, imputación de flota propia, tarifas
│   ├── calidad_datos.py        validaciones de calidad
│   ├── modelo_optimizacion.py  tarifario (costos por tramo) y envoltura del solver HiGHS
│   ├── base_modelo.py          base diaria común a Python y navegador
│   ├── modelo_diario.py        modelo MILP multiperiodo y reconstrucción de la operación actual
│   ├── escenarios_diarios.py   escenarios y combinaciones de bodegas externas
│   ├── pruebas.py              pruebas de consistencia A–H
│   ├── resumen.py, metodologia.py   textos del resumen y de la metodología
│   ├── reporte_excel.py, reporte_html.py
│   └── recursos/               plantilla HTML, motor.js (espejo del modelo), HiGHS WebAssembly, mapa base
├── pruebas/                    paridad Python vs navegador (opcional, requiere Node.js)
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
python ejecutar_modelo.py --sin-pruebas                  # omite las pruebas A–H (~2 min menos)
python ejecutar_modelo.py --sin-combinaciones            # omite las 9 combinaciones de bodegas
python ejecutar_modelo.py --no-abrir                     # no abrir el HTML al terminar
```
La corrida completa toma de 4 a 6 minutos. Leer 1,9 millones de líneas de SAP solo pesa la primera vez, porque después usa caché.

### 6. Modifique escenarios
Edite `parametros.yaml` y vuelva a ejecutar. También puede simular directamente en el HTML (pestaña Simulación).
- `horizonte`: ventana de demanda y número de días a optimizar.
- `modelo`: proyección y crecimiento, producción (`plan` o `flexible`), inventario inicial y final, utilización, servicio (tiempo máximo, meta), penalizaciones, factores de tarifa y costos por modalidad.
- `escenarios_diarios`: cada escenario define el modo de cada bodega externa (`fijo`, `opcional`, `forzado`, `cerrado`) y sus ajustes.
- `bodegas_candidatas`: posiciones, costo fijo y costos variables de cada bodega cotizada.

---

## Qué hace el modelo (versión 3, diaria)

| Elemento | Implementación |
|---|---|
| Tiempo | Horizonte diario configurable (por defecto 28 días desde el 4-may-2026) |
| Demanda | Despachos SAP no XI por día, ciudad y producto; réplica o perfil semanal + crecimientos por mercado, zona y familia |
| Producción | Plan estable en días hábiles = demanda del horizonte × factor (supuesto), o flexible dentro de la capacidad de planta |
| Inventario | Balance diario por nodo y producto: inicial + producción + llegadas (con tránsito) − salidas − despachos |
| Inventario final | Cada nodo termina con la cobertura de la demanda que atiende (evita el efecto fin de horizonte) |
| Capacidad | Posiciones × m3 por posición × utilización; exceso explícito y penalizado |
| Bodegas externas | Binarias con costo fijo; combinaciones ninguna / A / B / ambas / el optimizador decide |
| Traslado ficticio | Planta y CEDI del mismo código son el mismo nodo: costo 0, tránsito 0 |
| Servicio | Tiempo máximo (excluye nodos), meta con penalización por hora, atraso y no atendido penalizados |
| Solver | HiGHS (SciPy) en Python y HiGHS WebAssembly en el navegador; mismo modelo, paridad verificada |

Formulación completa, supuestos y limitaciones: pestaña **Metodología** del HTML.

## Escenarios incluidos
| Id | Escenario |
|---|---|
| S0 | Actual: operación observada (reconstruida, no optimizada) |
| S1 | Actual optimizado (con la planta operando) |
| S2 | Sin planta, con los CEDI históricos de cada cliente |
| S3 | Optimizado: cierre, el modelo decide nodos y bodegas |
| S4 | Uso obligatorio de bodegas externas |
| S5 | Sin bodegas externas |
| S6 | Optimizado con demanda +20 % |
| S7 | Parametrizado (base editable para la simulación) |

## Supuestos que debe validar
Los valores marcados `[SUPUESTO]` en `parametros.yaml` no vienen en los datos:
- **Plan de producción diario** e **inventario inicial** por nodo.
- Costos de las bodegas externas candidatas y costos de manejo por modalidad.
- Penalizaciones: atraso, no atendido, exceso de capacidad y stock de seguridad.
- Tiempos de servicio: el maestro de clientes no trae lead times.
- Tarifas de distribución en rutas sin historial: se estiman por distancia.

## Prueba de paridad (opcional)
```powershell
python pruebas/exportar_paridad.py
node pruebas/paridad.js
```
