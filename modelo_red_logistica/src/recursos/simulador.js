/* Simulador de red agregado por zona comercial.
 * Replica en el navegador el modelo de Python `resolver_reducido` (escenarios.py) usando jsLPSolver.
 * Modos:  B   = traslado a CEDI de otra ciudad (sin bodega externa)
 *         A   = bodega externa (se mantiene la actual; traslados a otros CEDI limitados al AS-IS)
 *         OPT = óptimo (el modelo decide qué bodegas abrir)
 */
(function (global) {
  "use strict";

  function nodosModo(sim, modo, ajustes) {
    return sim.almacenes
      .map(function (n) { return Object.assign({}, n, (ajustes && ajustes[n.id]) || {}); })
      .filter(function (n) {
        if (n.activa === false) return false;
        if (modo === "B" && (n.tipo === "candidata" || n.tipo === "bodega_externa_actual")) return false;
        return true;
      });
  }

  function resolverSim(sim, prm, modo, ajustes) {
    var solver = global.solver;
    var K = sim.plantas, R = sim.regiones, m3p = sim.m3_por_posicion;
    var doi = prm.dias_inventario, util = prm.utilizacion_maxima;
    var fxi = prm.factor_flete_traslado, fdi = prm.factor_flete_distribucion, crec = prm.crecimiento || 0;
    var pen = prm.penalidad / 1e6;
    var W = nodosModo(sim, modo, ajustes);
    var model = { optimize: "costo", opType: "min", constraints: {}, variables: {}, ints: {} };
    var C = model.constraints, V = model.variables;
    var binario = {};

    R.forEach(function (r) {
      K.forEach(function (k) {
        var dv = sim.demanda[r][k] * (1 + crec);
        if (dv > 0) C["d|" + r + "|" + k] = { equal: dv };
      });
    });
    W.forEach(function (n) {
      var w = n.id;
      var cv = n.manejo + n.posicion_cop * doi / 30 / m3p;
      var cap = n.posiciones * m3p * util;
      binario[w] = n.tipo === "candidata" || (n.tipo === "bodega_externa_actual" && modo === "OPT");
      C["c|" + w] = { max: binario[w] ? 0 : cap };
      K.forEach(function (k) {
        C["b|" + w + "|" + k] = { equal: 0 };
        var vx = { costo: sim.c_in[k][w] * fxi / 1e6 };
        vx["b|" + w + "|" + k] = 1;
        if (modo === "A" && w !== k && (n.tipo === "propio" || n.tipo === "3pl")) {
          C["l|" + k + "|" + w] = { max: sim.lim_xi_asis[k][w] || 0 };
          vx["l|" + k + "|" + w] = 1;
        }
        V["x|" + k + "|" + w] = vx;
        R.forEach(function (r) {
          if (!(sim.demanda[r][k] > 0)) return;
          var vy = { costo: (sim.c_out[w][r] * fdi + cv) / 1e6 };
          vy["d|" + r + "|" + k] = 1;
          vy["b|" + w + "|" + k] = -1;
          vy["c|" + w] = doi / 30;
          V["y|" + w + "|" + r + "|" + k] = vy;
        });
      });
      if (binario[w]) {
        var va = { costo: n.fijo / 1e6 };
        va["c|" + w] = -cap;
        va["u|" + w] = 1;
        C["u|" + w] = { max: 1 };
        V["a|" + w] = va;
        model.ints["a|" + w] = 1;
      } else {
        var vd = { costo: pen };
        vd["c|" + w] = -1;
        V["f|" + w] = vd;
      }
    });

    var t0 = Date.now();
    var sol = solver.Solve(model);
    var ms = Date.now() - t0;
    var val = function (name) { return sol[name] || 0; };

    var cost = { traslado: 0, distribucion: 0, manejo: 0, posiciones: 0, fijo_bodegas: 0, fijo_red: 0, penalizacion: 0 };
    var porNodo = {}, flujosX = [], flujosY = [];
    W.forEach(function (n) {
      var w = n.id, thr = 0, entrada = 0;
      K.forEach(function (k) {
        var vx = val("x|" + k + "|" + w);
        if (vx > 1e-6) {
          cost.traslado += vx * sim.c_in[k][w] * fxi;
          if (k !== w) entrada += vx;
          flujosX.push({ planta: k, almacen: w, m3: vx, tarifa: sim.c_in[k][w] * fxi });
        }
        R.forEach(function (r) {
          var vy = val("y|" + w + "|" + r + "|" + k);
          if (vy > 1e-6) {
            thr += vy;
            cost.distribucion += vy * sim.c_out[w][r] * fdi;
            flujosY.push({ almacen: w, zona: r, planta: k, m3: vy, tarifa: sim.c_out[w][r] * fdi });
          }
        });
      });
      var abierto = binario[w] ? val("a|" + w) > 0.5 : true;
      var inv = thr * doi / 30;
      cost.manejo += thr * n.manejo;
      cost.posiciones += inv / m3p * n.posicion_cop;
      if (binario[w]) { if (abierto) cost.fijo_bodegas += n.fijo; }
      else if (n.tipo === "bodega_externa_actual") cost.fijo_bodegas += n.fijo;
      else cost.fijo_red += n.fijo;
      var def = binario[w] ? 0 : val("f|" + w);
      cost.penalizacion += def * prm.penalidad;
      porNodo[w] = { id: w, nombre: n.nombre, tipo: n.tipo, abierto: abierto, throughput: thr, inventario: inv,
                     capacidad: n.posiciones * m3p, utilizacion: n.posiciones > 0 ? inv / (n.posiciones * m3p) : 0,
                     deficit: def, entrada_traslado: entrada };
    });
    cost.total = cost.traslado + cost.distribucion + cost.manejo + cost.posiciones + cost.fijo_bodegas +
                 cost.fijo_red + cost.penalizacion;
    return { modo: modo, factible: !!sol.feasible, ms: ms, costos: cost, nodos: porNodo,
             flujosX: flujosX, flujosY: flujosY,
             abiertas: W.filter(function (n) { return porNodo[n.id].abierto && (n.tipo === "candidata" || n.tipo === "bodega_externa_actual"); })
                        .map(function (n) { return n.id; }) };
  }

  global.SimRed = { resolverSim: resolverSim };
})(typeof window !== "undefined" ? window : this);
