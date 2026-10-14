/* Prueba de paridad: resuelve en Node.js (HiGHS WebAssembly, mismo motor del HTML) los casos exportados por
 * `python pruebas/exportar_paridad.py` y compara con el resultado de Python (HiGHS nativo vía SciPy).
 * Uso:  node pruebas/paridad.js            (requiere Node.js 18+; no es necesario para usar el modelo)
 */
const fs = require("fs"), path = require("path"), vm = require("vm");
(async () => {
  const R = path.join(__dirname, "..", "src", "recursos"), S = path.join(__dirname, "salida");
  const loadHighs = require(path.join(R, "highs.js"));
  const highs = await loadHighs({wasmBinary: fs.readFileSync(path.join(R, "highs.wasm"))});
  const ctx = {console, Math, Object, Array, Date, JSON, Float64Array, Int32Array, Infinity}; ctx.window = ctx; vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(path.join(R, "motor.js"), "utf8"), ctx);
  const base = JSON.parse(fs.readFileSync(path.join(S, "base.json"))), casos = JSON.parse(fs.readFileSync(path.join(S, "casos.json")));
  const py = JSON.parse(fs.readFileSync(path.join(S, "resultados_python.json")));
  let peor = 0;
  for (const [k, prm] of Object.entries(casos)) {
    const r = ctx.MotorRed.resolver(highs, base, prm, {gap: 1e-7, tiempo: 600});
    const rel = Math.abs(r.costos.Total - py[k]) / Math.abs(py[k]); peor = Math.max(peor, rel);
    console.log(k.padEnd(22), r.estado.padEnd(9), "JS", (r.costos.Total / 1e6).toFixed(3), "Python", (py[k] / 1e6).toFixed(3), "dif. relativa", rel.toExponential(2));
  }
  console.log(peor < 1e-6 ? "PARIDAD OK" : "DIFERENCIAS", "(máx. diferencia relativa " + peor.toExponential(2) + ")");
})().catch(e => { console.error(e); process.exit(1); });
