/* Motor del modelo multiperiodo diario de la red logística (navegador, HiGHS WebAssembly).
 * Réplica exacta de src/modelo_diario.py: mismas reglas, mismas variables, mismas restricciones.
 * pruebas/paridad.js compara ambos con los mismos parámetros.
 */
(function (global) {
  "use strict";
  var ESCALA = 1e-6;
  var COMP = ["Traslado XI terceros", "Traslado XI flota propia", "Distribución", "Manejo", "Almacenaje", "Costo financiero inventario",
    "Fijo bodegas externas", "Fijo nodos (plantas y CEDI)", "Penalización atraso", "Penalización no atendido",
    "Penalización exceso capacidad", "Penalización stock seguridad", "Penalización servicio"];
  var GRUPO = {"Traslado XI terceros": "Transporte", "Traslado XI flota propia": "Transporte", "Distribución": "Transporte",
    "Manejo": "Almacenamiento", "Almacenaje": "Almacenamiento", "Costo financiero inventario": "Almacenamiento",
    "Fijo bodegas externas": "Almacenamiento", "Fijo nodos (plantas y CEDI)": "Fijo nodos", "Penalización atraso": "Penalizaciones",
    "Penalización no atendido": "Penalizaciones", "Penalización exceso capacidad": "Penalizaciones",
    "Penalización stock seguridad": "Penalizaciones", "Penalización servicio": "Penalizaciones"};

  function zeros(n) { var a = new Array(n); for (var i = 0; i < n; i++) a[i] = 0; return a; }
  function sum(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; }
  function g(o, k, d) { return o && o[k] !== undefined && o[k] !== null ? o[k] : d; }

  /* ------------------------------------------------------------ 1. instancia */
  function instancia(base, prm) {
    var nodos = base.nodos, NN = nodos.length, fechas = base.meta.fechas, dow = base.meta.dia_semana, W = fechas.length;
    var off = Math.max(0, Math.min(prm.desplazamiento | 0, W - 1));
    var H = Math.max(1, Math.min(prm.horizonte_dias | 0, W - off)), dias = [];
    for (var d = off; d < off + H; d++) dias.push(d);
    var cerr = prm.planta_cerrada || null, pn = prm.nodos || {};
    var idn = {}; nodos.forEach(function (n, i) { idn[n.id] = i; });
    var modo = nodos.map(function (n) { return g(pn[n.id], "modo", "fijo"); });
    var activo = nodos.map(function (n, i) { return modo[i] !== "cerrado" && n.id !== cerr; });
    var grupos = base.grupos, G = grupos.length, zonas = base.zonas;
    var D0 = []; for (var gi = 0; gi < G; gi++) D0.push(zeros(W));
    base.demanda.forEach(function (r) { D0[r[1]][r[0]] += r[2]; });
    var Dg0 = [];
    if (prm.proyeccion === "perfil") {
      var cnt = zeros(7); for (d = 0; d < W; d++) cnt[dow[d]]++;
      for (gi = 0; gi < G; gi++) { var prof = zeros(7); for (d = 0; d < W; d++) prof[dow[d]] += D0[gi][d];
        Dg0.push(dias.map(function (dd) { return prof[dow[dd]] / (cnt[dow[dd]] || 1); })); }
    } else { for (gi = 0; gi < G; gi++) Dg0.push(dias.map(function (dd) { return D0[gi][dd]; })); }
    var Dg = [];
    for (gi = 0; gi < G; gi++) {
      var gr = grupos[gi];
      var f = (1 + g(prm, "crecimiento", 0)) * (1 + (gr[4] ? g(prm, "crec_E", 0) : g(prm, "crec_N", 0))) *
        (1 + g(prm.crec_zona || {}, zonas[gr[0]].zona || "", 0)) * (1 + g(prm.crec_familia || {}, base.familias[gr[1]], 0));
      Dg.push(Dg0[gi].map(function (v) { return v * f; }));
    }
    var rep = {}, totRep = 0;
    Object.keys(prm.reparto || {}).forEach(function (p) { var v = prm.reparto[p];
      if (v && idn[p] !== undefined && activo[idn[p]] && nodos[idn[p]].es_planta) { rep[p] = v; totRep += v; } });
    var gk = [];
    for (gi = 0; gi < G; gi++) {
      var z = grupos[gi][0], fa = grupos[gi][1], pa = grupos[gi][2], pt = grupos[gi][3], partes;
      if (cerr === null) partes = [[pa, 1]];
      else if (nodos[pa].id === cerr) {
        if (prm.usar_reparto && totRep > 0) partes = Object.keys(rep).sort(function (a, b) { return idn[a] - idn[b]; }).map(function (p) { return [idn[p], rep[p] / totRep]; });
        else partes = [[pt, 1]];
      } else partes = [[nodos[pt].id === cerr ? pa : pt, 1]];
      gk.push(partes.map(function (q) { return [fa * 1000 + q[0], q[1]]; }));
    }
    var Kset = {}; gk.forEach(function (l) { l.forEach(function (q) { Kset[q[0]] = 1; }); });
    var K = Object.keys(Kset).map(Number).sort(function (a, b) { return a - b; });
    var idk = {}; K.forEach(function (k, i) { idk[k] = i; });
    var planta_k = K.map(function (k) { return k % 1000; }), fam_k = K.map(function (k) { return Math.floor(k / 1000); });
    var zk = {};
    for (gi = 0; gi < G; gi++) {
      gk[gi].forEach(function (q) {
        var key = grupos[gi][0] + "|" + idk[q[0]];
        var e = zk[key] || (zk[key] = {z: grupos[gi][0], k: idk[q[0]], D: zeros(H), D0: zeros(H), E: 0, grupos: []});
        for (var t = 0; t < H; t++) { e.D[t] += Dg[gi][t] * q[1]; e.D0[t] += Dg0[gi][t] * q[1]; }
        e.E += sum(Dg[gi]) * q[1] * grupos[gi][4];
        e.grupos.push([gi, q[1]]);
      });
    }
    if (prm.extra && prm.extra.m3 > 0) {          // demanda adicional del validador de rutas
      var ex = prm.extra, kk = idk[ex.familia * 1000 + ex.planta];
      if (kk === undefined) { K.push(ex.familia * 1000 + ex.planta); kk = K.length - 1; idk[K[kk]] = kk; planta_k.push(ex.planta); fam_k.push(ex.familia); }
      var De = zeros(H); De[Math.max(0, Math.min(H - 1, ex.t | 0))] = ex.m3;
      zk["X|" + kk] = {z: ex.z, k: kk, D: De, D0: zeros(H), E: 0, grupos: [], extra: true, forzado: ex.nodo};
    }
    var pares = Object.keys(zk).filter(function (p) { return sum(zk[p].D) > 1e-9; }).sort(function (a, b) {
      var A = zk[a], B = zk[b]; return (A.extra ? 1e9 : A.z) - (B.extra ? 1e9 : B.z) || A.k - B.k; });
    var m3p = base.meta.m3_por_posicion, util = prm.utilizacion_maxima;
    var posi = nodos.map(function (n) { return g(pn[n.id], "posiciones", n.posiciones); });
    var cap = posi.map(function (p) { return p * m3p * util; }), pos_m3 = posi.map(function (p) { return p * m3p; });
    var man_s = nodos.map(function (n) { return g(pn[n.id], "manejo_salida", n.manejo_salida); });
    var man_e = nodos.map(function (n) { return g(pn[n.id], "manejo_entrada", n.manejo_entrada); });
    var alm = nodos.map(function (n) { return g(pn[n.id], "almac_dia", n.almac_dia); });
    var fijo = nodos.map(function (n) { return g(pn[n.id], "costo_fijo_mes", n.costo_fijo_mes); });
    var N = []; for (var i = 0; i < NN; i++) if (activo[i] && (pos_m3[i] > 0 || nodos[i].es_planta)) N.push(i);
    var tr = prm.transporte, pol = prm.traslados || "plantas_y_observados", hist = {};
    base.arcos_hist.forEach(function (a) { hist[a] = 1; });
    var arcos = [];
    N.forEach(function (u) { N.forEach(function (v) {
      if (u === v) return;
      var key = nodos[u].id + "|" + nodos[v].id;
      var ok = pol === "todos" || nodos[u].es_planta || (pol === "plantas_y_observados" && (hist[key] || nodos[u].bodega_externa || nodos[v].bodega_externa));
      if (!ok) return;
      var L = base.lanes[key], s = (tr.s_propia === null || tr.s_propia === undefined) ? L.s : tr.s_propia, tp;
      if (tr.metodo === "km") tp = L.km > 0 ? (tr.fijo_viaje + tr.costo_km * L.km) / (L.m3v || base.meta.m3_viaje_def) : 0;
      else tp = (L.tp === null || L.tp === undefined) ? L.tt : L.tp;
      arcos.push({u: u, v: v, L: Math.floor(L.horas / 24), horas: L.horas, c_terc: (1 - s) * L.tt * tr.f_terc, c_prop: s * tp * tr.f_prop, c_man: man_s[u] + man_e[v]});
    }); });
    var sv = prm.servicio, lmax = (sv.lmax_h === null || sv.lmax_h === undefined || sv.lmax_h === "") ? null : +sv.lmax_h, puede = {};
    pares.forEach(function (p) { var e = zk[p];
      if (e.extra && e.forzado !== null && e.forzado !== undefined) { puede[p] = N.indexOf(e.forzado) >= 0 ? [e.forzado] : []; return; }
      puede[p] = N.filter(function (n) { return lmax === null || base.h_out[n][e.z] <= lmax; }); });
    var inv = prm.inventario, nK = K.length, I0 = [];
    for (i = 0; i < NN; i++) I0.push(zeros(nK));
    var D0k = zeros(nK);
    Object.keys(zk).forEach(function (p) { if (!zk[p].extra) D0k[zk[p].k] += sum(zk[p].D0) / H; });
    for (var k = 0; k < nK; k++) if (activo[planta_k[k]]) I0[planta_k[k]][k] += inv.dias_inicial_planta * D0k[k];
    for (gi = 0; gi < G; gi++) {
      var avg = sum(Dg0[gi]) / H;
      gk[gi].forEach(function (q) { base.asign_hist[gi].forEach(function (h) {
        if (N.indexOf(h[0]) >= 0 && !nodos[h[0]].bodega_externa) I0[h[0]][idk[q[0]]] += inv.dias_inicial_cedi * avg * q[1] * h[1]; }); });
    }
    var oc0 = g(inv, "ocupacion_inicial_max", 0.7);
    for (i = 0; i < NN; i++) { var lim = oc0 * pos_m3[i], t0 = sum(I0[i]);
      if (t0 > lim && lim > 0) I0[i] = I0[i].map(function (v) { return v * lim / t0; }); }
    var pr = prm.produccion, ds = g(pr, "dias_semana", 6);
    var work = dias.map(function (dd) { return dow[dd] < ds ? 1 : 0; });
    if (sum(work) === 0) work = work.map(function () { return 1; });
    var Dk = []; for (k = 0; k < nK; k++) Dk.push(zeros(H));
    Object.keys(zk).forEach(function (p) { if (zk[p].extra) return; var e = zk[p]; for (var t = 0; t < H; t++) Dk[e.k][t] += e.D[t]; });
    var sw = sum(work), plan = Dk.map(function (row) { var tot = g(pr, "factor", 1) * sum(row) / sw; return work.map(function (w) { return tot * w; }); });
    var planExtra = plan.map(function (row) { return row.map(function () { return 0; }); });
    if (prm.extra && prm.extra.m3 > 0) { var kx = idk[prm.extra.familia * 1000 + prm.extra.planta], tx = Math.max(0, Math.min(H - 1, prm.extra.t | 0));
      plan[kx][tx] += prm.extra.m3; planExtra[kx][tx] += prm.extra.m3; }
    var capdia = {};
    planta_k.forEach(function (p) { var cm = nodos[p].cap_produccion_mes;
      if (cm) capdia[p] = work.map(function (w) { return cm / (ds * 52 / 12) * w; }); });
    var fija = {};
    if (prm.asignacion === "historica") {
      pares.forEach(function (p) { var e = zk[p]; if (e.extra) return; var acc = {}, tot = 0;
        e.grupos.forEach(function (q) { var w8 = sum(Dg0[q[0]]) * q[1];
          base.asign_hist[q[0]].forEach(function (h) { if (N.indexOf(h[0]) >= 0) { acc[h[0]] = (acc[h[0]] || 0) + w8 * h[1]; tot += w8 * h[1]; } }); });
        if (tot > 0) { var o = {}; Object.keys(acc).map(Number).sort(function (a, b) { return a - b; }).forEach(function (n) { o[n] = acc[n] / tot; }); fija[p] = o; } });
    }
    return {H: H, dias: dias, fechas: dias.map(function (dd) { return fechas[dd]; }), N: N, K: K, planta_k: planta_k, fam_k: fam_k, zk: zk,
      pares: pares, cap: cap, pos_m3: pos_m3, man_s: man_s, man_e: man_e, alm: alm, fijo: fijo, modo: modo, arcos: arcos, puede: puede,
      I0: I0, plan: plan, plan_extra: planExtra, work: work, capdia: capdia, fija: fija, gk: gk, Dg: Dg, nodos: nodos, prm: prm};
  }

  /* ------------------------------------------------------------ 2. modelo */
  function Lp() { this.c = []; this.lo = []; this.up = []; this.int = []; this.rs = [0]; this.ri = []; this.rv = []; this.rl = []; this.ru = []; }
  Lp.prototype.v = function (c, lo, up, entera) { this.c.push(c); this.lo.push(lo === undefined ? 0 : lo); this.up.push(up === undefined ? Infinity : up); this.int.push(entera ? 1 : 0); return this.c.length - 1; };
  Lp.prototype.r = function (idx, coef, lo, hi) {
    for (var i = 0; i < idx.length; i++) { this.ri.push(idx[i]); this.rv.push(coef[i]); }
    this.rs.push(this.ri.length); this.rl.push(lo === undefined ? -Infinity : lo); this.ru.push(hi === undefined ? Infinity : hi); };

  function construir(base, ins) {
    var prm = ins.prm, H = ins.H, N = ins.N, nK = ins.K.length, pares = ins.pares, hol = prm.holguras, sv = prm.servicio, tr = prm.transporte;
    var m = new Lp(), V = {s: {}, x: {}, I: {}, B: {}, E: {}, F: {}, q: {}, y: {}, P: {}, u: {}};
    var flex = prm.produccion.modo === "flexible", ss = prm.inventario.dias_seguridad || 0, capcost = prm.costo_capital_dia || 0;
    var fz = tr.f_zona || {}, zonas = base.zonas, k, t;
    N.forEach(function (n) { if (ins.modo[n] === "opcional" || ins.modo[n] === "forzado")
      V.y[n] = m.v(ins.fijo[n] * H / 30 * ESCALA, ins.modo[n] === "forzado" ? 1 : 0, 1, true); });
    pares.forEach(function (p) {
      var e = ins.zk[p], tot = sum(e.D), fija = ins.fija[p], z = e.z;
      var lista = fija ? Object.keys(fija).map(Number) : ins.puede[p];
      lista.forEach(function (n) {
        var h = base.h_out[n][z];
        var c = tot * (base.c_out[n][z] * tr.f_dist * g(fz, zonas[z].zona || "", 1) + ins.man_s[n] + g(sv, "pen_hora", 0) * Math.max(0, h - g(sv, "meta_h", 1e9)));
        V.s[n + "|" + p] = fija ? m.v(c * ESCALA, fija[n], fija[n]) : m.v(c * ESCALA, 0, 1);
      });
      if (hol.atraso || !ins.puede[p].length) V.u[p] = m.v(tot * hol.pen_no_atendido * ESCALA, 0, 1);
    });
    ins.arcos.forEach(function (a, ai) { var c = a.c_terc + a.c_prop + a.c_man;
      for (k = 0; k < nK; k++) for (t = 0; t < H; t++) V.x[ai + "|" + k + "|" + t] = m.v(c * ESCALA); });
    N.forEach(function (n) {
      for (k = 0; k < nK; k++) for (t = 0; t < H; t++) {
        V.I[n + "|" + k + "|" + t] = m.v((ins.alm[n] + capcost) * ESCALA);
        if (hol.atraso) V.B[n + "|" + k + "|" + t] = m.v((t === H - 1 ? hol.pen_no_atendido : hol.pen_atraso) * ESCALA);
      }
      if (hol.exceso) for (t = 0; t < H; t++) V.E[n + "|" + t] = m.v(hol.pen_exceso * ESCALA);
    });
    var sirve = {}, sKeys = Object.keys(V.s);
    sKeys.forEach(function (key) { var i = key.indexOf("|"), n = +key.slice(0, i), p = key.slice(i + 1), kk = ins.zk[p].k;
      (sirve[n + "|" + kk] = sirve[n + "|" + kk] || []).push(p); });
    var sirveKeys = Object.keys(sirve).sort(function (a, b) { var A = a.split("|").map(Number), B = b.split("|").map(Number); return A[0] - B[0] || A[1] - B[1]; });
    if (ss > 0) sirveKeys.forEach(function (key) { V.q[key] = m.v(0);
      for (t = 0; t < H; t++) V.F[key + "|" + t] = m.v(g(hol, "pen_stock_seguridad", 0) * ESCALA); });
    if (flex) for (k = 0; k < nK; k++) for (t = 0; t < H; t++) V.P[k + "|" + t] = m.v(0);
    // R1
    pares.forEach(function (p) { var idx = [], coef = [];
      N.forEach(function (n) { var s = V.s[n + "|" + p]; if (s !== undefined) { idx.push(s); coef.push(1); } });
      if (V.u[p] !== undefined) { idx.push(V.u[p]); coef.push(1); }
      m.r(idx, coef, 1, 1); });
    // R2
    var ent = {}, sal = {};
    ins.arcos.forEach(function (a, ai) { (ent[a.v] = ent[a.v] || []).push(ai); (sal[a.u] = sal[a.u] || []).push(ai); });
    N.forEach(function (n) { for (k = 0; k < nK; k++) { var esPl = ins.planta_k[k] === n;
      for (t = 0; t < H; t++) {
        var idx = [V.I[n + "|" + k + "|" + t]], coef = [1];
        if (hol.atraso) { idx.push(V.B[n + "|" + k + "|" + t]); coef.push(-1); }
        if (t > 0) { idx.push(V.I[n + "|" + k + "|" + (t - 1)]); coef.push(-1);
          if (hol.atraso) { idx.push(V.B[n + "|" + k + "|" + (t - 1)]); coef.push(1); } }
        (ent[n] || []).forEach(function (ai) { var t0 = t - ins.arcos[ai].L; if (t0 >= 0) { idx.push(V.x[ai + "|" + k + "|" + t0]); coef.push(-1); } });
        (sal[n] || []).forEach(function (ai) { idx.push(V.x[ai + "|" + k + "|" + t]); coef.push(1); });
        (sirve[n + "|" + k] || []).forEach(function (p) { var dd = ins.zk[p].D[t]; if (dd !== 0) { idx.push(V.s[n + "|" + p]); coef.push(dd); } });
        var rhs = t === 0 ? ins.I0[n][k] : 0;
        if (esPl) { if (flex) { idx.push(V.P[k + "|" + t]); coef.push(-1); rhs += ins.plan_extra[k][t]; } else rhs += ins.plan[k][t]; }
        m.r(idx, coef, rhs, rhs);
      } } });
    // R3
    N.forEach(function (n) { for (t = 0; t < H; t++) {
      var idx = [], coef = []; for (k = 0; k < nK; k++) { idx.push(V.I[n + "|" + k + "|" + t]); coef.push(1); }
      if (hol.exceso) { idx.push(V.E[n + "|" + t]); coef.push(-1); }
      if (V.y[n] !== undefined) { idx.push(V.y[n]); coef.push(-ins.cap[n]); m.r(idx, coef, undefined, 0); }
      else m.r(idx, coef, undefined, ins.cap[n]); } });
    // R4
    if (ss > 0) sirveKeys.forEach(function (key) {
      var n = +key.split("|")[0], kk = +key.split("|")[1], ps = sirve[key];
      m.r([V.q[key]].concat(ps.map(function (p) { return V.s[n + "|" + p]; })), [1].concat(ps.map(function (p) { return -ss * sum(ins.zk[p].D) / H; })), 0, 0);
      for (t = 0; t < H; t++) m.r([V.I[n + "|" + kk + "|" + t], V.F[key + "|" + t], V.q[key]], [1, 1, -1], 0, undefined); });
    // R5
    Object.keys(V.y).map(Number).sort(function (a, b) { return a - b; }).forEach(function (n) {
      var yi = V.y[n];
      sKeys.forEach(function (key) { if (+key.split("|")[0] === n) m.r([V.s[key], yi], [1, -1], undefined, 0); });
      for (t = 0; t < H; t++) { var idx = []; (ent[n] || []).forEach(function (ai) { for (var kk = 0; kk < nK; kk++) idx.push(V.x[ai + "|" + kk + "|" + t]); });
        if (idx.length) m.r(idx.concat([yi]), idx.map(function () { return 1; }).concat([-ins.cap[n]]), undefined, 0); }
    });
    // R6
    //     modo por nodo (defecto): cada nodo termina con la cobertura de la demanda que atiende (proporcional a s)
    var fm = g(prm.inventario, "final_min", 1), porNodo = g(prm.inventario, "final_por_nodo", true);
    var dpl = prm.inventario.dias_inicial_planta, dce = prm.inventario.dias_inicial_cedi, D0k6 = zeros(nK);
    ins.pares.forEach(function (p) { if (!ins.zk[p].extra) D0k6[ins.zk[p].k] += sum(ins.zk[p].D0) / H; });
    for (k = 0; k < nK; k++) { var idx = [], coef = [], s0 = 0;
      for (var n2 = 0; n2 < ins.I0.length; n2++) s0 += ins.I0[n2][k];
      var W6 = (dpl + dce) * D0k6[k];
      if (!porNodo || W6 <= 0) {
        N.forEach(function (n) { idx.push(V.I[n + "|" + k + "|" + (H - 1)]); coef.push(1); });
        ins.arcos.forEach(function (a, ai) { for (var tt = Math.max(0, H - a.L); tt < H; tt++) if (tt + a.L >= H) { idx.push(V.x[ai + "|" + k + "|" + tt]); coef.push(1); } });
        m.r(idx, coef, fm * s0, undefined); continue; }
      var f6 = fm * s0 / W6;
      N.forEach(function (n) { var ix = [V.I[n + "|" + k + "|" + (H - 1)]], cf = [1];
        (ent[n] || []).forEach(function (ai) { var a = ins.arcos[ai]; for (var tt = Math.max(0, H - a.L); tt < H; tt++) { ix.push(V.x[ai + "|" + k + "|" + tt]); cf.push(1); } });
        (sirve[n + "|" + k] || []).forEach(function (p) { var e = ins.zk[p]; if (!e.extra) { ix.push(V.s[n + "|" + p]); cf.push(-f6 * dce * sum(e.D0) / H); } });
        m.r(ix, cf, ins.planta_k[k] === n ? f6 * dpl * D0k6[k] : 0, undefined); });
    }
    // R7
    if (flex) Object.keys(ins.capdia).map(Number).sort(function (a, b) { return a - b; }).forEach(function (p) {
      var ks = []; for (k = 0; k < nK; k++) if (ins.planta_k[k] === p) ks.push(k);
      for (t = 0; t < H; t++) m.r(ks.map(function (kk) { return V.P[kk + "|" + t]; }), ks.map(function () { return 1; }), undefined, ins.capdia[p][t]); });
    var cons = 0; N.forEach(function (n) { if (V.y[n] === undefined) cons += ins.fijo[n] * H / 30; });
    return {m: m, V: V, cons: cons, ent: ent, sal: sal, sirve: sirve};
  }

  function resolverLP(highs, lp, opciones) {
    var INF = highs.infinity !== undefined ? highs.infinity : 1e30;
    var fix = function (a) { return Float64Array.from(a, function (v) { return v === Infinity ? INF : v === -Infinity ? -INF : v; }); };
    var hayInt = lp.int.some(function (v) { return v; });
    var data = {numCols: lp.c.length, numRows: lp.rl.length, colCost: Float64Array.from(lp.c), colLower: fix(lp.lo), colUpper: fix(lp.up),
      rowLower: fix(lp.rl), rowUpper: fix(lp.ru),
      matrix: {format: "csr", numRows: lp.rl.length, numCols: lp.c.length, starts: Int32Array.from(lp.rs), indices: Int32Array.from(lp.ri), values: Float64Array.from(lp.rv)}};
    if (hayInt) data.integrality = Int32Array.from(lp.int);
    var model = highs.createModel(data);
    try {
      model.options.set("output_flag", false);
      if (opciones && opciones.gap) model.options.set("mip_rel_gap", opciones.gap);
      if (opciones && opciones.tiempo) model.options.set("time_limit", opciones.tiempo);
      model.run();
      var st = model.getModelStatus(), MS = highs.constants.modelStatus, nombre = "Otro";
      Object.keys(MS).forEach(function (k) { if (MS[k] === st) nombre = k; });
      var sol = null; try { sol = model.getSolution().colValue; } catch (e) { sol = null; }
      var obj = null; try { obj = model.getObjectiveValue(); } catch (e) { obj = null; }
      return {estado: nombre === "optimal" ? "Optimal" : nombre, valores: sol, objetivo: obj};
    } finally { if (model.dispose) model.dispose(); }
  }

  function resolver(highs, base, prm, opciones) {
    var t0 = Date.now(), ins = instancia(base, prm), mod = construir(base, ins);
    var r = resolverLP(highs, mod.m, opciones || {gap: 1e-3});
    if (!r.valores) return {estado: r.estado, factible: false, H: ins.H, fechas: ins.fechas, costos: totales(cv(), ins.H), kpis: {}, nodos_dia: {}, asignacion: [], traslados: [],
      diagnostico: [{tipo: "Infactible", nodo: "-", detalle: "El modelo no tiene solución factible con estas restricciones (" + r.estado + "). Active las holguras para identificar la causa."}], tipo: "optimizado", tiempo_s: (Date.now() - t0) / 1000};
    var res = extraer(base, ins, r.valores, mod);
    res.estado = r.estado; res.tiempo_s = Math.round((Date.now() - t0) / 100) / 10; res.variables = mod.m.c.length; res.restricciones = mod.m.rl.length;
    res.objetivo = r.objetivo !== null ? r.objetivo / ESCALA + mod.cons : null; res.prm = prm; res.ins = ins;
    return res;
  }

  /* ------------------------------------------------------------ 3. resultados */
  function cv() { var o = {}; COMP.forEach(function (c) { o[c] = 0; }); return o; }
  function totales(c, H) {
    var o = Object.assign({}, c);
    ["Transporte", "Almacenamiento", "Fijo nodos", "Penalizaciones"].forEach(function (gr) { o["Total " + gr.toLowerCase()] = 0;
      Object.keys(c).forEach(function (k) { if (GRUPO[k] === gr) o["Total " + gr.toLowerCase()] += c[k]; }); });
    o["Costo logístico (sin penalizaciones)"] = o["Total transporte"] + o["Total almacenamiento"] + o["Total fijo nodos"];
    o.Total = o["Costo logístico (sin penalizaciones)"] + o["Total penalizaciones"];
    o["Total mensual equivalente"] = o.Total * 30 / H; o["Costo logístico mensual equivalente"] = o["Costo logístico (sin penalizaciones)"] * 30 / H;
    return o;
  }
  function arr3(a, b, c) { var o = []; for (var i = 0; i < a; i++) { var r = []; for (var j = 0; j < b; j++) r.push(zeros(c)); o.push(r); } return o; }

  function extraer(base, ins, X, mod) {
    var V = mod.V, prm = ins.prm, H = ins.H, N = ins.N, nK = ins.K.length, nodos = ins.nodos, NN = nodos.length;
    var sv = prm.servicio, tr = prm.transporte, hol = prm.holguras, zonas = base.zonas, fz = tr.f_zona || {};
    var val = function (i) { return X[i]; }, c = cv(), abiertos = {};
    N.forEach(function (n) { abiertos[n] = V.y[n] !== undefined ? val(V.y[n]) > 0.5 : true; });
    var asign = [], desp = arr3(NN, nK, H), m3meta = 0, m3tot = 0, mh = 0;
    Object.keys(V.s).forEach(function (key) {
      var sh = val(V.s[key]); if (sh < 1e-7) return;
      var i = key.indexOf("|"), n = +key.slice(0, i), p = key.slice(i + 1), e = ins.zk[p], z = e.z, k = e.k, tot = 0;
      for (var t = 0; t < H; t++) { var d = e.D[t] * sh; desp[n][k][t] += d; tot += d; }
      var cd = tot * base.c_out[n][z] * tr.f_dist * g(fz, zonas[z].zona || "", 1), h = base.h_out[n][z];
      c["Distribución"] += cd; c.Manejo += tot * ins.man_s[n];
      c["Penalización servicio"] += tot * g(sv, "pen_hora", 0) * Math.max(0, h - g(sv, "meta_h", 1e9));
      m3tot += tot; mh += tot * h; if (h <= g(sv, "meta_h", 1e9)) m3meta += tot;
      asign.push([n, z, k, sh, tot, cd, h, e.E * sh / Math.max(sum(e.D), 1e-9) * tot, e.extra ? 1 : 0, p]);
    });
    var noAt = 0;
    Object.keys(V.u).forEach(function (p) { var v = val(V.u[p]); if (v > 1e-7) { var q = sum(ins.zk[p].D) * v; noAt += q; c["Penalización no atendido"] += q * hol.pen_no_atendido; } });
    var tras = {}, llega = arr3(NN, nK, H), sale = arr3(NN, nK, H), xdia = {};
    Object.keys(V.x).forEach(function (key) {
      var q = val(V.x[key]); if (q < 1e-7) return;
      var pp = key.split("|"), ai = +pp[0], k = +pp[1], t = +pp[2], a = ins.arcos[ai];
      sale[a.u][k][t] += q; if (t + a.L < H) llega[a.v][k][t + a.L] += q;
      c["Traslado XI terceros"] += q * a.c_terc; c["Traslado XI flota propia"] += q * a.c_prop; c.Manejo += q * a.c_man;
      var tk = a.u + "|" + a.v + "|" + k, r = tras[tk] || (tras[tk] = [a.u, a.v, k, 0, 0, 0, a.L, a.horas]);
      r[3] += q; r[4] += q * a.c_terc; r[5] += q * a.c_prop;
      var dk = nodos[a.u].id + "|" + nodos[a.v].id; (xdia[dk] = xdia[dk] || zeros(H))[t] += q;
    });
    var inv = arr3(NN, nK, H), bkl = arr3(NN, nK, H);
    Object.keys(V.I).forEach(function (key) { var pp = key.split("|"); inv[+pp[0]][+pp[1]][+pp[2]] = val(V.I[key]); });
    Object.keys(V.B).forEach(function (key) { var pp = key.split("|"); bkl[+pp[0]][+pp[1]][+pp[2]] = val(V.B[key]); });
    var prod = []; for (var k = 0; k < nK; k++) { var row = zeros(H);
      for (var t = 0; t < H; t++) row[t] = Object.keys(V.P).length ? val(V.P[k + "|" + t]) + ins.plan_extra[k][t] : ins.plan[k][t]; prod.push(row); }
    var capAlm = prm.costo_capital_dia || 0, nodos_dia = {};
    N.forEach(function (n) {
      var I = zeros(H), at = zeros(H), de = zeros(H), en = zeros(H), sa = zeros(H), pr = zeros(H), E = zeros(H), t;
      for (var kk = 0; kk < nK; kk++) for (t = 0; t < H; t++) { I[t] += inv[n][kk][t]; at[t] += bkl[n][kk][t]; de[t] += desp[n][kk][t]; en[t] += llega[n][kk][t]; sa[t] += sale[n][kk][t];
        if (ins.planta_k[kk] === n) pr[t] += prod[kk][t]; }
      if (hol.exceso) for (t = 0; t < H; t++) E[t] = val(V.E[n + "|" + t]);
      var Ii = [sum(ins.I0[n])].concat(I.slice(0, H - 1));
      nodos_dia[nodos[n].id] = {inv_ini: Ii, produccion: pr, entradas: en, salidas_tr: sa, despachos: de, inv_fin: I, atraso: at, cap: ins.cap[n], pos_m3: ins.pos_m3[n],
        ocupacion: I.map(function (v) { return ins.pos_m3[n] > 0 ? v / ins.pos_m3[n] : 0; }), exceso: E, abierto: abiertos[n],
        costo_almac: I.map(function (v) { return v * (ins.alm[n] + capAlm); }), costo_manejo: de.map(function (v) { return v * ins.man_s[n]; })};
      c.Almacenaje += sum(I) * ins.alm[n]; c["Costo financiero inventario"] += sum(I) * capAlm;
      c["Penalización exceso capacidad"] += sum(E) * g(hol, "pen_exceso", 0);
      for (kk = 0; kk < nK; kk++) { for (t = 0; t < H - 1; t++) c["Penalización atraso"] += bkl[n][kk][t] * g(hol, "pen_atraso", 0);
        c["Penalización no atendido"] += bkl[n][kk][H - 1] * g(hol, "pen_no_atendido", 0); }
      var fc = abiertos[n] ? ins.fijo[n] * H / 30 : 0;
      if (nodos[n].bodega_externa) c["Fijo bodegas externas"] += fc; else c["Fijo nodos (plantas y CEDI)"] += fc;
    });
    Object.keys(V.F).forEach(function (key) { c["Penalización stock seguridad"] += val(V.F[key]) * g(hol, "pen_stock_seguridad", 0); });
    var resBal = 0;
    N.forEach(function (n) { for (var kk = 0; kk < nK; kk++) { var prev = ins.I0[n][kk], prevb = 0;
      for (var t = 0; t < H; t++) { var entra = llega[n][kk][t] + (ins.planta_k[kk] === n ? prod[kk][t] : 0);
        var r = (inv[n][kk][t] - bkl[n][kk][t]) - (prev - prevb + entra - sale[n][kk][t] - desp[n][kk][t]);
        resBal = Math.max(resBal, Math.abs(r)); prev = inv[n][kk][t]; prevb = bkl[n][kk][t]; } } });
    var produccion = {};
    for (k = 0; k < nK; k++) { var p = nodos[ins.planta_k[k]].id; produccion[p] = produccion[p] || zeros(H);
      for (t = 0; t < H; t++) produccion[p][t] += prod[k][t]; }
    var capProd = {}; Object.keys(ins.capdia).forEach(function (p) { capProd[nodos[+p].id] = ins.capdia[p]; });
    var atr = 0, fin = 0;
    N.forEach(function (n) { for (var kk = 0; kk < nK; kk++) { for (var t = 0; t < H - 1; t++) atr += bkl[n][kk][t]; fin += bkl[n][kk][H - 1]; } });
    var totDem = 0; Object.keys(ins.zk).forEach(function (p) { totDem += sum(ins.zk[p].D); });
    var traslM3 = 0; Object.keys(tras).forEach(function (kk) { traslM3 += tras[kk][3]; });
    var exMax = 0; Object.keys(nodos_dia).forEach(function (nid) { exMax = Math.max(exMax, Math.max.apply(null, nodos_dia[nid].exceso)); });
    var kpis = {demanda_m3: totDem, atendido_m3: m3tot, no_atendido_m3: noAt + fin, atraso_m3_dia: atr,
      fill_rate: totDem ? 1 - (noAt + fin) / totDem : 1, pct_en_meta: m3tot ? m3meta / m3tot : 0, horas_promedio: m3tot ? mh / m3tot : 0,
      traslados_m3: traslM3, exceso_max_m3: exMax, residuo_balance: resBal,
      bodegas_abiertas: Object.keys(V.y).map(Number).sort(function (a, b) { return a - b; }).filter(function (n) { return abiertos[n]; }).map(function (n) { return nodos[n].id; })};
    var demDia = zeros(H); Object.keys(ins.zk).forEach(function (p) { if (!ins.zk[p].extra) for (var t = 0; t < H; t++) demDia[t] += ins.zk[p].D[t]; });
    return {H: H, fechas: ins.fechas, K: ins.K.map(function (kk, i) { return [ins.fam_k[i], ins.planta_k[i]]; }), costos: totales(c, H), kpis: kpis,
      nodos_dia: nodos_dia, asignacion: asign, traslados: Object.keys(tras).map(function (kk) { return tras[kk]; }), traslados_dia: xdia,
      produccion: produccion, cap_produccion: capProd, abiertos: (function () { var o = {}; N.forEach(function (n) { o[nodos[n].id] = abiertos[n]; }); return o; })(),
      diagnostico: diagnostico(ins, nodos_dia, kpis, produccion, capProd), demanda_dia: demDia, tipo: "optimizado"};
  }

  function nf(v) { return Math.round(v).toLocaleString("es-CO"); }
  function diagnostico(ins, nodos_dia, kpis, produccion, capProd) {
    var out = [];
    Object.keys(nodos_dia).forEach(function (nid) { var d = nodos_dia[nid], ex = d.exceso, mx = Math.max.apply(null, ex);
      if (mx > 0.5) out.push({tipo: "Capacidad", nodo: nid, detalle: nid + " supera su capacidad " + ex.filter(function (v) { return v > 0.5; }).length + " días (máximo " + nf(mx) + " m3). Falta capacidad de almacenamiento: amplíe posiciones o habilite una bodega externa."});
      var dias = d.cap > 0 ? d.inv_fin.filter(function (v) { return v / d.cap > 0.98; }).length : 0;
      if (dias && mx <= 0.5) out.push({tipo: "Restricción activa", nodo: nid, detalle: nid + " opera al límite de capacidad " + dias + " días: la capacidad está condicionando la solución."});
      var am = Math.max.apply(null, d.atraso);
      if (am > 0.5) out.push({tipo: "Servicio", nodo: nid, detalle: nid + " acumula atrasos (máx. " + nf(am) + " m3): falta producto disponible a tiempo (producción, inventario inicial o tiempos de tránsito)."});
    });
    Object.keys(produccion).forEach(function (p) { if (!capProd[p]) return; var ex = produccion[p].map(function (v, i) { return v - capProd[p][i]; }), mx = Math.max.apply(null, ex);
      if (mx > 0.5) out.push({tipo: "Producción", nodo: p, detalle: "El plan de " + p + " supera su capacidad diaria " + ex.filter(function (v) { return v > 0.5; }).length + " días (máx. +" + nf(mx) + " m3/día). Revise el reparto de producción o use el modo flexible."}); });
    var sin = ins.pares.filter(function (p) { return !ins.puede[p].length; }).length;
    if (sin) out.push({tipo: "Servicio", nodo: "-", detalle: sin + " combinaciones ciudad-producto no tienen ningún nodo dentro del tiempo máximo de entrega: la restricción de servicio las hace infactibles (quedan como no atendidas)."});
    if (kpis.no_atendido_m3 > 0.5) out.push({tipo: "Servicio", nodo: "-", detalle: nf(kpis.no_atendido_m3) + " m3 quedan sin atender al final del horizonte."});
    return out;
  }

  /* ------------------------------------------------------------ 4. Actual (reconstrucción observada) */
  function reconstruirActual(base, prm) {
    var p = Object.assign({}, prm, {planta_cerrada: null, asignacion: "libre"}), ins = instancia(base, p);
    var nodos = base.nodos, NN = nodos.length, H = ins.H, d0 = ins.dias[0];
    var ks = {}; base.grupos.forEach(function (gr) { ks[gr[1] * 1000 + gr[2]] = 1; }); base.obs_xi.forEach(function (r) { ks[r[3] * 1000 + r[4]] = 1; });
    var keys = Object.keys(ks).map(Number).sort(function (a, b) { return a - b; }), idk = {}; keys.forEach(function (k, i) { idk[k] = i; });
    var nk = keys.length, desp = arr3(NN, nk, H), ent = arr3(NN, nk, H), sal = arr3(NN, nk, H), tr = prm.transporte, c = cv();
    var sv = prm.servicio, zonas = base.zonas, fz = tr.f_zona || {}, m3meta = 0, m3tot = 0, mh = 0, flu = [];
    base.obs_desp.forEach(function (r) { var t = r[0] - d0; if (t < 0 || t >= H) return;
      var n = r[1], gi = r[2], m3 = r[3], gr = base.grupos[gi], z = gr[0], k = idk[gr[1] * 1000 + gr[2]];
      desp[n][k][t] += m3; var cd = m3 * base.c_out[n][z] * tr.f_dist * g(fz, zonas[z].zona || "", 1);
      c["Distribución"] += cd; c.Manejo += m3 * ins.man_s[n]; var h = base.h_out[n][z];
      m3tot += m3; mh += m3 * h; if (h <= g(sv, "meta_h", 1e9)) m3meta += m3; flu.push([n, gi, m3, cd]); });
    var fic = prm.traslado_ficticio_mismo_codigo !== false, tras = {};
    base.obs_xi.forEach(function (r) { var t = r[0] - d0, u = r[1], w = r[2]; if (t < 0 || t >= H || (fic && u === w)) return;
      var k = idk[r[3] * 1000 + r[4]], m3 = r[5], costo = r[6], propia = r[7];
      sal[u][k][t] += m3; var lane = w >= 0 ? base.lanes[nodos[u].id + "|" + nodos[w].id] : null, L = lane ? Math.floor(lane.horas / 24) : 0;
      if (w >= 0 && t + L < H) ent[w][k][t + L] += m3;
      if (propia) c["Traslado XI flota propia"] += costo; else c["Traslado XI terceros"] += costo;
      c.Manejo += m3 * (ins.man_s[u] + (w >= 0 ? ins.man_e[w] : 0));
      var tk = u + "|" + w + "|" + k, rr = tras[tk] || (tras[tk] = [u, w, k, 0, 0, 0, L, lane ? lane.horas : 0]);
      rr[3] += m3; rr[propia ? 5 : 4] += costo; });
    var work = ins.work, sw = sum(work), prod = [];
    keys.forEach(function (k, kk) { var pl = k % 1000, neto = sum(desp[pl][kk]) + sum(sal[pl][kk]) - sum(ent[pl][kk]);
      prod.push(work.map(function (w) { return Math.max(0, neto) / sw * w; })); });
    var inv = prm.inventario, avgTot = keys.map(function (k, kk) { var s = 0; for (var n = 0; n < NN; n++) s += sum(desp[n][kk]); return s / H; });
    var REG = [], I0 = [], I0min = {};
    for (var n = 0; n < NN; n++) { var rr2 = keys.map(function (k, kk) { return inv.dias_inicial_cedi * sum(desp[n][kk]) / H + (k % 1000 === n ? inv.dias_inicial_planta * avgTot[kk] : 0); });
      var lim = g(inv, "ocupacion_inicial_max", 0.7) * ins.pos_m3[n], s = sum(rr2); if (s > lim && lim > 0) rr2 = rr2.map(function (v) { return v * lim / s; }); REG.push(rr2); }
    for (n = 0; n < NN; n++) { var row = zeros(nk);
      for (var kk = 0; kk < nk; kk++) { var cum = 0, mn = 0;
        for (var t = 0; t < H; t++) { cum += (keys[kk] % 1000 === n ? prod[kk][t] : 0) + ent[n][kk][t] - sal[n][kk][t] - desp[n][kk][t]; mn = Math.min(mn, cum); }
        var minimo = Math.max(0, -mn); row[kk] = Math.max(REG[n][kk], minimo);
        if (minimo > REG[n][kk] + 1e-6) I0min[nodos[n].id] = (I0min[nodos[n].id] || 0) + minimo - REG[n][kk]; }
      I0.push(row); }
    var nodos_dia = {}, capCap = prm.costo_capital_dia || 0;
    for (n = 0; n < NN; n++) { if (nodos[n].tipo === "candidata") continue;
      var pr = zeros(H), en = zeros(H), sa = zeros(H), de = zeros(H), I = zeros(H), acc = sum(I0[n]);
      for (kk = 0; kk < nk; kk++) for (t = 0; t < H; t++) { if (keys[kk] % 1000 === n) pr[t] += prod[kk][t]; en[t] += ent[n][kk][t]; sa[t] += sal[n][kk][t]; de[t] += desp[n][kk][t]; }
      for (t = 0; t < H; t++) { acc += pr[t] + en[t] - sa[t] - de[t]; I[t] = acc; }
      var capn = ins.cap[n], pos = ins.pos_m3[n];
      nodos_dia[nodos[n].id] = {inv_ini: [sum(I0[n])].concat(I.slice(0, H - 1)), produccion: pr, entradas: en, salidas_tr: sa, despachos: de, inv_fin: I, atraso: zeros(H),
        cap: capn, pos_m3: pos, ocupacion: I.map(function (v) { return pos > 0 ? v / pos : 0; }), exceso: I.map(function (v) { return Math.max(0, v - capn); }), abierto: true,
        costo_almac: I.map(function (v) { return v * (ins.alm[n] + capCap); }), costo_manejo: de.map(function (v) { return v * ins.man_s[n]; })};
      c.Almacenaje += sum(I) * ins.alm[n]; c["Costo financiero inventario"] += sum(I) * capCap;
      var fc = ins.fijo[n] * H / 30; if (nodos[n].bodega_externa) c["Fijo bodegas externas"] += fc; else c["Fijo nodos (plantas y CEDI)"] += fc; }
    var produccion = {}; keys.forEach(function (k, kk) { var p2 = nodos[k % 1000].id; produccion[p2] = produccion[p2] || zeros(H);
      for (var t = 0; t < H; t++) produccion[p2][t] += prod[kk][t]; });
    var exMax = 0; Object.keys(nodos_dia).forEach(function (nid) { exMax = Math.max(exMax, Math.max.apply(null, nodos_dia[nid].exceso)); });
    var tot = 0, tsal = 0; for (n = 0; n < NN; n++) for (kk = 0; kk < nk; kk++) { tot += sum(desp[n][kk]); tsal += sum(sal[n][kk]); }
    var kpis = {demanda_m3: tot, atendido_m3: tot, no_atendido_m3: 0, atraso_m3_dia: 0, fill_rate: 1, pct_en_meta: m3tot ? m3meta / m3tot : 0,
      horas_promedio: m3tot ? mh / m3tot : 0, traslados_m3: tsal, exceso_max_m3: exMax, residuo_balance: 0,
      bodegas_abiertas: nodos.filter(function (x) { return x.tipo === "bodega_externa"; }).map(function (x) { return x.id; }), inventario_inicial_ajustado: I0min};
    var diag = [{tipo: "Dato estimado", nodo: "-", detalle: "La producción diaria no existe en los datos: se estima como una tasa estable en días hábiles que equilibra las salidas observadas de cada planta."}];
    Object.keys(I0min).forEach(function (nid) { diag.push({tipo: "Dato estimado", nodo: nid, detalle: "La secuencia observada en " + nid + " requiere " + nf(I0min[nid]) + " m3 más de inventario inicial que la regla de días para no ser negativa; se usó el mínimo físico."}); });
    Object.keys(nodos_dia).forEach(function (nid) { var ex = nodos_dia[nid].exceso, mx = Math.max.apply(null, ex);
      if (mx > 0.5) diag.push({tipo: "Capacidad", nodo: nid, detalle: "Con la operación observada " + nid + " supera su capacidad estimada " + ex.filter(function (v) { return v > 0.5; }).length + " días (máx. " + nf(mx) + " m3)."}); });
    var ag = {}; flu.forEach(function (f) { var z = base.grupos[f[1]][0], kk2 = f[0] + "|" + z + "|" + f[1];
      var r = ag[kk2] || (ag[kk2] = [f[0], z, f[1], 1, 0, 0, base.h_out[f[0]][z], 0]); r[4] += f[2]; r[5] += f[3]; r[7] += f[2] * base.grupos[f[1]][4]; });
    var demDia = zeros(H); for (n = 0; n < NN; n++) for (kk = 0; kk < nk; kk++) for (t = 0; t < H; t++) demDia[t] += desp[n][kk][t];
    return {H: H, fechas: ins.fechas, K: keys.map(function (k) { return [Math.floor(k / 1000), k % 1000]; }), costos: totales(c, H), kpis: kpis, nodos_dia: nodos_dia,
      asignacion_grupos: Object.keys(ag).map(function (k) { return ag[k]; }), traslados: Object.keys(tras).map(function (k) { return tras[k]; }),
      produccion: produccion, cap_produccion: {}, abiertos: (function () { var o = {}; Object.keys(nodos_dia).forEach(function (k) { o[k] = true; }); return o; })(),
      diagnostico: diag, estado: "Observado (reconstruido)", demanda_dia: demDia, tipo: "historico", objetivo: null, tiempo_s: 0, variables: 0, restricciones: 0};
  }

  /* ------------------------------------------------------------ 5. flujos a nivel cliente / grupo (mapa y análisis) */
  /* devuelve filas [planta, cedi, grupo, m3, costo_dist] — el detalle por cliente se obtiene con base.cli_shares[grupo] */
  function flujosGrupo(base, res) {
    var out = [];
    if (res.tipo === "historico") { (res.asignacion_grupos || []).forEach(function (r) { out.push([base.grupos[r[2]][2], r[0], r[2], r[4], r[5]]); }); return out; }
    var ins = res.ins || instancia(base, res.prm);
    (res.asignacion || []).forEach(function (r) {
      if (r[8]) return;
      var n = r[0], k = r[2], sh = r[3], p = r[9] !== undefined ? r[9] : (r[1] + "|" + k), e = ins.zk[p], cu = r[4] > 0 ? r[5] / r[4] : 0;
      if (!e) return;
      e.grupos.forEach(function (q) { var m3 = sum(ins.Dg[q[0]]) * q[1] * sh; if (m3 > 1e-6) out.push([ins.planta_k[k], n, q[0], m3, m3 * cu]); });
    });
    return out;
  }

  global.MotorRed = {instancia: instancia, construir: construir, resolver: resolver, resolverLP: resolverLP, reconstruirActual: reconstruirActual,
    flujosGrupo: flujosGrupo, totales: totales, COMP: COMP, GRUPO: GRUPO};
})(typeof window !== "undefined" ? window : this);
