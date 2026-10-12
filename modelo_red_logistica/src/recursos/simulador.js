/* Simulador de red agregado por zona comercial (se ejecuta en el navegador con jsLPSolver).
 * Réplica exacta de `resolver_reducido` en escenarios.py (Python) — ver prueba de paridad.
 *
 * Opciones (preset):
 *   ACTUAL : planta cerrada operando, producción AS-IS, bodega externa actual abierta, sin candidatas
 *   B      : cierre; traslado a CEDI de otra ciudad (sin bodegas externas)
 *   A      : cierre; bodega externa (la actual se mantiene, candidatas disponibles, traslados limitados al AS-IS)
 *   OPT    : cierre; el modelo decide qué bodegas usar
 *   PERS   : escenario personalizado con la política que defina el usuario
 */
(function (global) {
  "use strict";

  var PRESETS = {
    ACTUAL: { cerrar: false, co99: "abierta", candidatas: false, limitar: false },
    B: { cerrar: true, co99: "cerrada", candidatas: false, limitar: false },
    A: { cerrar: true, co99: "abierta", candidatas: true, limitar: true },
    OPT: { cerrar: true, co99: "decide", candidatas: true, limitar: false }
  };

  function politica(prm, modo) {
    if (modo === "PERS") return prm.pers;
    return PRESETS[modo];
  }

  /* Demanda por zona y planta de origen (k), con su mezcla de mercado N/E */
  function demanda(sim, prm, pol) {
    var D = {}, DM = {}, cerr = sim.planta_cerrada;
    var reparto = prm.reparto || sim.defaults.reparto;
    var tot = 0; Object.keys(reparto).forEach(function (p) { tot += reparto[p] || 0; });
    sim.demanda.forEach(function (r) {
      var zona = r[0], merc = r[1], pa = r[2], pt = r[3], m3 = r[4];
      var g = (1 + (prm.crecimiento || 0)) * (1 + (merc === "E" ? prm.crec_E || 0 : prm.crec_N || 0)) *
              (1 + ((prm.crec_zona && prm.crec_zona[zona]) || 0));
      var v = m3 * g, partes = {};
      if (!pol.cerrar) partes[pa] = v;
      else if (pa === cerr) {
        if (prm.usar_reparto && tot > 0) Object.keys(reparto).forEach(function (p) { if (reparto[p] > 0 && p !== cerr) partes[p] = v * reparto[p] / tot; });
        else partes[pt] = v;
      } else partes[pt === cerr ? pa : pt] = v;
      Object.keys(partes).forEach(function (k) {
        D[zona] = D[zona] || {}; DM[zona] = DM[zona] || {};
        D[zona][k] = (D[zona][k] || 0) + partes[k];
        DM[zona][k] = DM[zona][k] || { N: 0, E: 0 };
        DM[zona][k][merc] += partes[k];
      });
    });
    return { D: D, DM: DM };
  }

  /* Costo de traslado k -> w separado en terceros / flota propia */
  function costoTraslado(sim, prm, k, w) {
    if (k === w) return { terc: 0, prop: 0, s: 0 };
    var L = (sim.lanes[k] || {})[w];
    if (!L) return { terc: 0, prop: 0, s: 0, nd: true };
    var s = (prm.s_propia === null || prm.s_propia === undefined) ? L.s : prm.s_propia;
    var tt = L.tt * prm.factor_flete_traslado;
    var tp;
    if (prm.metodo === "km") {
      tp = L.km > 0 ? (prm.costo_fijo_viaje + prm.costo_km * L.km) / (L.m3v || sim.m3_viaje_def) : 0;
    } else {
      tp = (L.tp === null || L.tp === undefined) ? L.tt : L.tp;
    }
    tp = tp * prm.factor_flota_propia;
    return { terc: (1 - s) * tt, prop: s * tp, s: s };
  }

  function nodosOpcion(sim, prm, pol) {
    return sim.almacenes.map(function (n) {
      return Object.assign({}, n, (prm.ajustes && prm.ajustes[n.id]) || {});
    }).filter(function (n) {
      if (n.activa === false) return false;
      if (n.id === sim.planta_cerrada && pol.cerrar) return false;
      if (n.tipo === "candidata" && !pol.candidatas) return false;
      if (n.tipo === "bodega_externa_actual" && pol.co99 === "cerrada") return false;
      return true;
    });
  }

  function resolverSim(sim, prm, modo) {
    var pol = politica(prm, modo);
    var solver = global.solver;
    var R = sim.regiones, m3p = sim.m3_por_posicion;
    var doi = prm.dias_inventario, util = prm.utilizacion_maxima, fdi = prm.factor_flete_distribucion;
    var pen = prm.penalidad / 1e6;
    var dem = demanda(sim, prm, pol), D = dem.D;
    var K = [];
    R.forEach(function (r) { Object.keys(D[r] || {}).forEach(function (k) { if (K.indexOf(k) < 0) K.push(k); }); });
    K.sort();
    var W = nodosOpcion(sim, prm, pol);
    var model = { optimize: "costo", opType: "min", constraints: {}, variables: {}, ints: {} };
    var C = model.constraints, V = model.variables, binario = {}, cin = {};
    R.forEach(function (r) { K.forEach(function (k) { var dv = (D[r] || {})[k] || 0; if (dv > 1e-9) C["d|" + r + "|" + k] = { equal: dv }; }); });
    W.forEach(function (n) {
      var w = n.id, cv = n.manejo + n.posicion_cop * doi / 30 / m3p, cap = n.posiciones * m3p * util;
      binario[w] = n.tipo === "candidata" || (n.tipo === "bodega_externa_actual" && pol.co99 === "decide");
      C["c|" + w] = { max: binario[w] ? 0 : cap };
      K.forEach(function (k) {
        var ct = costoTraslado(sim, prm, k, w);
        cin[k + "|" + w] = ct;
        C["b|" + w + "|" + k] = { equal: 0 };
        var vx = { costo: (ct.terc + ct.prop) / 1e6 };
        vx["b|" + w + "|" + k] = 1;
        if (pol.limitar && w !== k && (n.tipo === "propio" || n.tipo === "3pl")) {
          C["l|" + k + "|" + w] = { max: ((sim.lim_xi_asis[k] || {})[w]) || 0 };
          vx["l|" + k + "|" + w] = 1;
        }
        V["x|" + k + "|" + w] = vx;
        R.forEach(function (r) {
          if (!(((D[r] || {})[k] || 0) > 1e-9)) return;
          var fz = (prm.factor_zona && prm.factor_zona[r]) || 1;
          var vy = { costo: (sim.c_out[w][r] * fdi * fz + cv) / 1e6 };
          vy["d|" + r + "|" + k] = 1;
          vy["b|" + w + "|" + k] = -1;
          vy["c|" + w] = doi / 30;
          V["y|" + w + "|" + r + "|" + k] = vy;
        });
      });
      if (binario[w]) {
        var va = { costo: n.fijo / 1e6 }; va["c|" + w] = -cap; va["u|" + w] = 1;
        C["u|" + w] = { max: 1 }; V["a|" + w] = va; model.ints["a|" + w] = 1;
      } else {
        var vd = { costo: pen }; vd["c|" + w] = -1; V["f|" + w] = vd;
      }
    });
    var t0 = Date.now(), sol = solver.Solve(model), ms = Date.now() - t0;
    var val = function (name) { return sol[name] || 0; };

    var c = { terc: 0, prop: 0, dist: 0, manejo: 0, pos: 0, fijo_be: 0, fijo_nodos: 0, penal: 0 };
    var nodos = {}, fx = [], fy = [], prodPlanta = {};
    W.forEach(function (n) {
      var w = n.id, thr = 0, entrada = 0;
      K.forEach(function (k) {
        var vx = val("x|" + k + "|" + w), ct = cin[k + "|" + w];
        if (vx > 1e-6) {
          c.terc += vx * ct.terc; c.prop += vx * ct.prop;
          prodPlanta[k] = (prodPlanta[k] || 0) + vx;
          if (k !== w) entrada += vx;
          fx.push({ planta: k, almacen: w, m3: vx, terc: vx * ct.terc, prop: vx * ct.prop, s: ct.s });
        }
        R.forEach(function (r) {
          var vy = val("y|" + w + "|" + r + "|" + k);
          if (vy > 1e-6) {
            var fz = (prm.factor_zona && prm.factor_zona[r]) || 1, mz = dem.DM[r][k], tz = mz.N + mz.E;
            thr += vy; c.dist += vy * sim.c_out[w][r] * fdi * fz;
            fy.push({ almacen: w, zona: r, planta: k, m3: vy, costo: vy * sim.c_out[w][r] * fdi * fz, pE: tz > 0 ? mz.E / tz : 0 });
          }
        });
      });
      var abierto = binario[w] ? val("a|" + w) > 0.5 : true, inv = thr * doi / 30;
      c.manejo += thr * n.manejo; c.pos += inv / m3p * n.posicion_cop;
      var esBE = n.tipo === "candidata" || n.tipo === "bodega_externa_actual";
      if (abierto) { if (esBE) c.fijo_be += n.fijo; else c.fijo_nodos += n.fijo; }
      var def = binario[w] ? 0 : val("f|" + w);
      c.penal += def * prm.penalidad;
      nodos[w] = { id: w, nombre: n.nombre, tipo: n.tipo, abierto: abierto, throughput: thr, inventario: inv,
                   capacidad: n.posiciones * m3p, utilizacion: n.posiciones > 0 ? inv / (n.posiciones * m3p) : 0,
                   deficit: def, entrada_traslado: entrada, lat: n.lat, lon: n.lon };
    });
    // plantas que no tienen almacén abierto (planta cerrada que sigue operando no aplica aquí) pagan su fijo
    c.transporte = c.terc + c.prop + c.dist;
    c.almacenamiento = c.manejo + c.pos + c.fijo_be;
    c.total = c.transporte + c.almacenamiento + c.fijo_nodos + c.penal;
    var vol = 0; fy.forEach(function (f) { vol += f.m3; });
    var plantas = Object.keys(prodPlanta).map(function (k) {
      var cap = sim.cap_planta[k] || null;
      return { planta: k, m3: prodPlanta[k], capacidad: cap, util: cap ? prodPlanta[k] / cap : null };
    });
    return { modo: modo, politica: pol, factible: !!sol.feasible, ms: ms, costos: c, nodos: nodos, fx: fx, fy: fy,
             volumen: vol, plantas: plantas,
             abiertas: W.filter(function (n) { return nodos[n.id].abierto && (n.tipo === "candidata" || n.tipo === "bodega_externa_actual"); })
                        .map(function (n) { return n.id; }) };
  }

  global.SimRed = { resolverSim: resolverSim, PRESETS: PRESETS };
})(typeof window !== "undefined" ? window : this);
