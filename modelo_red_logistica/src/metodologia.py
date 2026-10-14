"""Texto didáctico de la metodología (pestaña 7 del HTML y hoja del Excel)."""
from __future__ import annotations

from html import escape


def _tabla(cols, filas):
    h = "<div class='tabla-wrap' style='max-height:none'><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>"
    for f in filas:
        h += "<tr>" + "".join(f"<td>{v}</td>" for v in f) + "</tr>"
    return h + "</tbody></table></div>"


CONJUNTOS = [
    ("t ∈ T", "Días del horizonte", "día", "—", "Parámetro (horizonte_dias)", "Índice", "t = 0 … 27 (28 días desde el 4-may-2026)"),
    ("n ∈ N", "Nodos activos: plantas (cada una con su CEDI del mismo código), CEDI propios, CEDI 3PL, bodegas externas", "—", "—",
     "Maestro de nodos + parámetros", "Índice", "CO01 Cajicá, CO02, CO99, BE_ANTIOQUIA"),
    ("k ∈ K", "Producto = familia × planta productora (la referencia se agrega en su familia)", "—", "—", "Maestro de referencias", "Índice", "FaCa|CO01"),
    ("z ∈ Z", "Ciudad destino del cliente (el detalle por cliente se recupera con su participación histórica)", "—", "—", "SAP + maestro clientes", "Índice", "MEDELLIN"),
    ("a=(u,v) ∈ A", "Arcos de traslado XI permitidos entre nodos según la política de traslados", "—", "—", "Rutas SAP/TMS + parámetro", "Índice", "CO01→CO02"),
    ("(z,k) ∈ P", "Pares ciudad-producto con demanda en el horizonte", "—", "—", "Derivado", "Índice", "(MEDELLIN, FaCa|CO01)"),
]
DATOS = [
    ("D[z,k,t]", "Demanda diaria proyectada", "m3/día", "z,k,t", "SAP (despachos no XI) × proyección", "Dato / supuesto", "MEDELLIN, FaCa|CO01, día 3 = 12,4"),
    ("c_out[n,z]", "Tarifa de distribución desde n a la ciudad z", "COP/m3", "n,z", "SAP fletes > 1 COP; regresión por km si no hay ruta", "Dato / estimado", "CO02→MEDELLIN 8.145"),
    ("h_out[n,z]", "Tiempo de entrega n→z", "horas", "n,z", "TMS; estimado por km si no hay registro", "Dato / estimado", "11,9 h"),
    ("tt[u,v], tp[u,v]", "Tarifa XI de terceros y costo imputado de flota propia", "COP/m3", "u,v", "SAP; flota propia imputada (viajes similares o fijo+km)", "Dato / estimado", "CO01→CO02 ≈ 52.000"),
    ("s[u,v]", "Participación histórica de flota propia en la ruta", "fracción", "u,v", "SAP (viajes en 0/1 COP)", "Dato", "0,62"),
    ("L[u,v]", "Tiempo de tránsito en días = ⌊horas/24⌋", "días", "u,v", "TMS", "Dato", "CO01→CO02: 0 (llega el mismo día)"),
    ("pos[n]", "Posiciones de estiba del nodo", "posiciones", "n", "Maestro de nodos / parámetros", "Dato", "CO10: 450"),
    ("Fijo[n]", "Costo fijo mensual del nodo", "COP/mes", "n", "Parámetros de plantas / bodegas", "Dato / supuesto", "BE_ANTIOQUIA 56 M"),
]
PARAMS = [
    ("m3p", "m3 por posición de estiba", "m3/posición", "—", "Maestro de referencias (cajas × estiba × m3)", "Parámetro", "2,26 (ponderado por volumen)"),
    ("util", "Utilización máxima de las posiciones", "fracción", "—", "parametros.yaml", "Supuesto", "0,85"),
    ("Cap[n] = pos·m3p·util", "Capacidad física útil", "m3", "n", "Derivado", "Parámetro derivado", "CO10: 450 × 2,26 × 0,85 ≈ 864 m3"),
    ("Plan[k,t]", "Producción estable: total demanda × factor repartido en días hábiles", "m3/día", "k,t", "Supuesto (no hay datos de producción)", "Supuesto", "100/semana → 16,7/día lun-sáb"),
    ("I0[n,k]", "Inventario inicial = días × demanda diaria (tope ocupación inicial)", "m3", "n,k", "Supuesto", "Supuesto", "8 días planta, 10 días CEDI"),
    ("SS", "Días de stock de seguridad", "días", "—", "parametros.yaml", "Supuesto", "7"),
    ("ms[n], me[n], al[n]", "Manejo salida, entrada y almacenaje", "COP/m3, COP/m3·día", "n", "Tarifas por modalidad", "Dato / supuesto", "Propio 4.500 / 0 / 0"),
    ("π_atr, π_na, π_ex, π_ss", "Penalizaciones: atraso, no atendido, exceso, falta de stock de seguridad", "COP/m3(·día)", "—", "parametros.yaml", "Supuesto calibrable", "60.000 / 1.500.000 / 100.000 / 2.000"),
    ("Lmax, meta, π_h", "Tiempo máximo, meta de entrega y penalización por hora sobre la meta", "h, COP/m3·h", "—", "parametros.yaml", "Supuesto", "— / 72 / 0"),
    ("f_terc, f_prop, f_dist, f_zona", "Factores de sensibilidad de tarifas", "×", "zona", "Simulación", "Supuesto", "1,0"),
]
VARIABLES = [
    ("s[n,z,k] ∈ [0,1]", "Fracción de la demanda (z,k) que atiende el nodo n (política fija en el horizonte)", "fracción", "n,z,k", "Optimizado", "CO02 atiende 100 % de Medellín FaCa"),
    ("x[a,k,t] ≥ 0", "m3 trasladados por el arco a del producto k que salen el día t", "m3", "a,k,t", "Optimizado", "CO01→CO02 FaCa día 2 = 30"),
    ("I[n,k,t] ≥ 0", "Inventario físico al cierre del día", "m3", "n,k,t", "Optimizado", "CO02 FaCa = 210"),
    ("B[n,k,t] ≥ 0", "Pedido atrasado (backlog) acumulado", "m3", "n,k,t", "Holgura", "0 si hay producto"),
    ("E[n,t] ≥ 0", "Inventario por encima de la capacidad (exceso explícito)", "m3", "n,t", "Holgura", "CO10 = 0"),
    ("u[z,k] ∈ [0,1]", "Fracción de la demanda no atendida", "fracción", "z,k", "Holgura", "0"),
    ("q[n,k], F[n,k,t]", "Objetivo de stock de seguridad y su faltante", "m3", "n,k,t", "Auxiliar / holgura", "q = 7 × demanda diaria"),
    ("y[n] ∈ {0,1}", "Abrir la bodega externa opcional n", "binaria", "n", "Optimizado", "BE_SABANA = 0"),
    ("P[k,t] ≥ 0", "Producción diaria (solo en modo flexible)", "m3", "k,t", "Optimizado", "—"),
]
RESTR = [
    ("R1 Asignación", "Σₙ s[n,z,k] + u[z,k] = 1", "Que cualquier nodo con ruta y servicio permitido atienda la demanda, o que una parte quede sin atender (penalizada).",
     "Que se pierda demanda sin registrarlo.", "Demanda sin dueño o duplicada."),
    ("R2 Balance diario", "I[n,k,t] − B[n,k,t] = I[n,k,t−1] − B[n,k,t−1] + Plan[k,t]·1(n=planta k) + Σ x[(u,n),k,t−L] − Σ x[(n,v),k,t] − Σ D[z,k,t]·s[n,z,k]",
     "Mover inventario entre días y nodos, con el tiempo de tránsito.", "Crear o perder producto; despachar lo que no ha llegado.", "Soluciones físicamente imposibles."),
    ("R3 Capacidad", "Σₖ I[n,k,t] − E[n,t] ≤ Cap[n]·(y[n] si es opcional)", "Usar el nodo hasta su capacidad; el exceso queda explícito y penalizado.",
     "Ocupación por encima de las posiciones disponibles sin verlo.", "Ocultar la necesidad de bodega externa."),
    ("R4 Stock de seguridad", "q[n,k] = SS·Σ s·D/H ;  I[n,k,t] + F[n,k,t] ≥ q[n,k]", "Bajar del stock de seguridad pagando una penalización.",
     "Operar con inventario cero sin costo.", "Planes frágiles ante variaciones."),
    ("R5 Apertura", "s[n,·] ≤ y[n] ;  Σ x[(u,n),k,t] ≤ Cap[n]·y[n]", "Usar una bodega opcional solo si se paga su costo fijo.",
     "Usar gratis una bodega cerrada.", "Subestimar el costo de la bodega externa."),
    ("R6 Inventario final por nodo", "I[n,k,H−1] + tránsito hacia n ≥ φₖ·(días_CEDI·Σ s[n,z,k]·D̄[z,k] + 1(n=planta k)·días_planta·D̄ₖ),  φₖ = final_min·Σₙ I0[n,k] / ((días_planta+días_CEDI)·D̄ₖ)",
     "Terminar el horizonte con el mismo inventario total de la red, ubicado donde el nuevo esquema lo necesita (cobertura de la demanda que atiende cada nodo).",
     "Vaciar los CEDI consumiendo su inventario inicial y dejar el producto en la planta al final, ahorrando traslados que en la operación real sí ocurren.",
     "Efecto fin de horizonte: subestimar traslados XI y sobrestimar el ahorro. Con final_por_nodo = false se exige solo el total de la red."),
    ("R7 Capacidad de producción", "Σₖ∈planta P[k,t] ≤ CapProd[p,t]", "Producir más algunos días si hay capacidad (modo flexible).",
     "Producir sin límite.", "Planes de producción irreales."),
    ("Traslado ficticio", "Arco u=v no existe; el CEDI de una planta es el mismo nodo (código igual)", "Despachar desde la planta sin costo ni tiempo de traslado.",
     "Cobrar un traslado XI entre planta y CEDI del mismo código.", "Doble conteo de flete y manejo (prueba B)."),
    ("Servicio", "s[n,z,k] = 0 si h_out[n,z] > Lmax", "Solo nodos que cumplen el tiempo máximo atienden la ciudad.",
     "Atender desde nodos que incumplen la promesa de servicio.", "Ahorro a costa del servicio (prueba G)."),
]


def html(base: dict, cfg: dict, pruebas: list, corrida: dict) -> str:
    S3 = corrida["resultados"].get("S3_OPTIMIZADO") or next(iter(corrida["resultados"].values()))
    hz = cfg.get("horizonte", {})
    tab = lambda c, f: _tabla(c, f)
    out = []
    out.append("<h2>1 · Pregunta de negocio</h2><div class='card'><p>Con el cierre de la planta <b>" + escape(str(base["meta"]["planta_cerrada"])) +
               "</b> y el traslado de su producción a otras plantas, ¿cómo debe atenderse cada cliente y cada referencia, día a día, al menor costo logístico "
               "(transporte + almacenamiento + fijos) sin perder servicio? En particular: ¿conviene una <b>bodega externa</b> o un <b>traslado XI a un CEDI de otra ciudad</b> "
               "con capacidad?</p></div>")
    out.append("<h2>2 · Supuestos</h2><div class='card'><ul class='lista'>" + "".join(f"<li>{x}</li>" for x in [
        f"Horizonte diario de {S3['H']} días tomados de la ventana {hz.get('inicio_ventana')} + {hz.get('dias_ventana')} días (configurable). La demanda se proyecta replicando la ventana o con un perfil por día de semana, más crecimientos por mercado, zona y familia.",
        "<span class='tipo estimado'><i></i>Producción</span>: no existe en los datos. Se supone un plan estable en días hábiles que iguala la demanda del horizonte (factor parametrizable) o, en modo flexible, el modelo decide dentro de la capacidad de planta.",
        "<span class='tipo estimado'><i></i>Inventario inicial</span>: días de cobertura por nodo (planta / CEDI), con un tope de ocupación inicial. El escenario Actual usa además el mínimo físico que la secuencia observada exige.",
        "La política de atención s[n,z,k] es fija en el horizonte (un cliente no cambia de CEDI cada día). Los traslados sí son diarios.",
        "Los traslados XI dentro del mismo código (planta→CEDI propio) no existen como arco: costo 0, tránsito 0. Se excluyen de los datos observados.",
        "Traslados XI de flota propia (fletes de 0/1 COP) valorados por viajes similares de terceros o por costo fijo + COP/km.",
        "Las penalizaciones no son costos contables: son precios de incumplimiento calibrados para que el modelo prefiera siempre cumplir cuando sea posible. Se reportan aparte.",
        "Las tarifas de bodegas externas candidatas son supuestos parametrizables (no hay cotización en los datos)."]) + "</ul></div>")
    out.append("<h2>3 · Conjuntos e índices</h2>" + tab(["Símbolo", "Descripción", "Unidad", "Índices", "Fuente", "Rol", "Ejemplo"], CONJUNTOS))
    out.append("<h2>4 · Datos de entrada</h2>" + tab(["Símbolo", "Descripción", "Unidad", "Índices", "Fuente", "Rol", "Ejemplo"], DATOS))
    out.append("<h2>5 · Parámetros</h2>" + tab(["Símbolo", "Descripción", "Unidad", "Índices", "Fuente", "Rol", "Ejemplo"], PARAMS))
    out.append("<h2>6 · Variables de decisión</h2>" + tab(["Símbolo", "Descripción", "Unidad", "Índices", "Rol", "Ejemplo"], VARIABLES))
    out.append("<h2>7 · Función objetivo</h2><div class='formula'>min  Σ D·s·(c_out·f_dist·f_zona + ms[n] + π_h·max(0, h−meta))            distribución + manejo de salida (+ servicio)\n"
               "   + Σ x[a,k,t]·((1−s)·tt·f_terc + s·tp·f_prop + ms[u] + me[v])                         traslados XI (terceros / flota propia) + manejo\n"
               "   + Σ I[n,k,t]·(al[n] + costo_capital)                                                    almacenaje y costo financiero\n"
               "   + Σ y[n]·Fijo[n]·H/30   + constante Σ Fijo de nodos fijos                               costos fijos\n"
               "   + Σ π_atr·B + π_na·(u·D + B final) + π_ex·E + π_ss·F                                    penalizaciones (incumplimientos)</div>"
               "<p class='nota'>Los costos se reportan en el horizonte y en mes equivalente (× 30 / H). El objetivo se escala por 10⁻⁶ para estabilidad numérica.</p>")
    out.append("<h2>8 · Restricciones</h2>" + tab(["Restricción", "Fórmula", "Qué permite", "Qué impide", "Qué problema evita"], RESTR))
    out.append("<h2>9 · Relación entre componentes</h2><div class='card'><p>La demanda D fija cuánto debe salir de la red cada día. La asignación s decide desde qué nodo sale, "
               "lo que define la distribución (segundo tramo) y cuánto inventario necesita ese nodo. El balance R2 obliga a que el inventario llegue antes: por producción en la "
               "planta o por traslados x (primer tramo, con su tiempo de tránsito). La capacidad R3 limita cuánto inventario cabe; si no cabe, el modelo traslada antes de lo "
               "necesario, abre una bodega opcional (y, R5) o muestra el exceso E. R6 impide vaciar la red al final. Así ambos tramos y el almacenamiento se deciden juntos.</p></div>")
    out.append("<h2>10 · Interpretación operativa</h2><div class='card'><ul class='lista'>"
               "<li><b>s</b> se lee como «la ciudad X se atiende desde el CEDI Y» (tabla de clientes que cambian de origen).</li>"
               "<li><b>x</b> es el plan diario de traslados XI por ruta y familia (viajes = m3 / m3 por viaje).</li>"
               "<li><b>I</b> y la ocupación diaria indican cuántas posiciones se necesitan cada día; <b>E</b> &gt; 0 significa que falta espacio físico.</li>"
               "<li><b>y</b> responde la pregunta de bodega externa: se abre solo si su costo fijo + manejo es menor que trasladar a otro CEDI o mover antes.</li>"
               "<li>Penalizaciones &gt; 0 señalan incumplimiento (atraso, no atendido, exceso): revise el diagnóstico antes de usar la cifra.</li></ul></div>")
    out.append("<h2>11 · Limitaciones</h2><div class='card'><ul class='lista'>"
               "<li>La producción diaria y el inventario inicial son supuestos: el modelo no conoce el plan maestro real. Pídalos para afinar el resultado.</li>"
               "<li>La granularidad es familia × planta (no referencia): la tabla de referencias muestra la mezcla histórica dentro de cada familia.</li>"
               "<li>Tarifas lineales por m3 (sin economías de carga completa ni mínimos por viaje) y tiempos de tránsito enteros en días.</li>"
               "<li>Costos de bodegas candidatas y penalizaciones son supuestos parametrizables.</li>"
               "<li>La política de asignación es estática en el horizonte; no modela ventanas horarias ni ruteo de última milla.</li></ul></div>")
    out.append("<h2>Diagnóstico del modelo anterior (v2) y cambios</h2>" + tab(["Aspecto", "Modelo anterior", "Modelo actual"], [
        ("Tiempo", "Estado estable mensual (un periodo)", "Multiperiodo diario con horizonte configurable"),
        ("Inventario", "Despacho × días de inventario / 30 (no se conservaba entre periodos)", "Balance diario con inventario inicial, tránsito y final por nodo (R2, R6)"),
        ("Producción", "Volumen mensual por planta", "Plan diario por producto (o flexible con capacidad)"),
        ("Capacidad", "Déficit penalizado sin detalle diario", "Ocupación diaria por posiciones físicas, exceso explícito E"),
        ("Bodegas externas", "Abrir/cerrar una bodega por escenario", "Combinaciones ninguna/A/B/todas/el optimizador decide, con fijo y variables separados"),
        ("Servicio", "No modelado", "Tiempo máximo, meta y penalización por hora; atraso y no atendido explícitos"),
        ("Traslado ficticio", "Podía cobrar XI dentro del mismo código", "Excluido y verificado (prueba B)"),
        ("Simulador navegador", "Heurística/LP agregado por zona (no era el mismo modelo)", "Mismo MILP que Python con HiGHS WebAssembly; paridad verificada"),
        ("Tipo de resultado", "Mezclaba observado y optimizado", "Cada cifra marcada: histórico, estimado, supuesto, optimizado, simulado"),
    ]))
    out.append("<h2>Pruebas de consistencia (A–H)</h2>" + tab(["Caso", "Prueba", "Qué verifica", "Resultado", "Estado"],
               [(escape(p["caso"]), escape(p["prueba"]), escape(p["que_verifica"]), escape(p["resultado"]),
                 f"<span class='sev {'OK' if p['estado'] == 'OK' else 'Revisar'}'><i></i>{escape(p['estado'])}</span>") for p in pruebas]))
    out.append("<h2>Tecnología de solución</h2><div class='card'><ul class='lista'>"
               f"<li>Programación lineal entera mixta (MILP). Escenario optimizado: {S3.get('variables', 0):,} variables y {S3.get('restricciones', 0):,} restricciones.</li>"
               "<li>Python: <b>HiGHS</b> vía scipy.optimize.milp. Navegador: <b>HiGHS</b> compilado a WebAssembly (highs-js 1.15.3, licencia MIT), embebido en este archivo: funciona sin internet.</li>"
               "<li>Ambos motores construyen exactamente el mismo modelo; la diferencia relativa de costos Python vs navegador es &lt; 10⁻¹² en los casos de prueba.</li>"
               "<li>No hay heurísticas: las cifras «Optimizado» y «Simulado» son óptimos (brecha MIP ≤ 0,1 %). El escenario Actual no se optimiza: se reconstruye con los despachos y traslados observados.</li></ul></div>")
    return "".join(out)
