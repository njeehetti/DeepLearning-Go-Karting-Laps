// Runs the untouched v1 code (legacy/kart-lap-lab.html) and records reference outputs that the TS port
// must reproduce exactly. Usage: node scripts/gen-v1-fixture.cjs
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'legacy/kart-lap-lab.html'), 'utf8');
const grab = (id) => html.split(`<script id="${id}"`)[1].split('>').slice(1).join('>').split('</script>')[0];
const ctx = { module: undefined };
vm.createContext(ctx);
vm.runInContext(grab('core-src') + '\n' + grab('ga-src') + '\nthis.KartCore = KartCore; this.KartGA = KartGA;', ctx);
const { KartCore, KartGA } = ctx;

const def = JSON.parse(fs.readFileSync(path.join(root, 'data/tracks/lyl.json'), 'utf8'));
// exactly what the v1 page passes to buildTrack
const v1def = { coords: def.coords, width: 8, officialLength: def.officialLength, bridge: { height: def.bridge.height, lowerIsBridge: false } };
const tr = KartCore.buildTrack(v1def);
const ev = new KartCore.Evaluator(tr, 16);
const sum = (a) => Array.from(a).reduce((p, q) => p + q, 0);

function genes(seed) {
  let a = seed >>> 0;
  const r = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const lim = ev.limit();
  return Array.from({ length: ev.M }, () => (r() * 2 - 1) * lim);
}

const optsList = [{}, { mu: 1.0, margin: 0.9, driver: 90 }];
const laps = [];
for (const seed of [1, 2, 3, 4]) for (const kart of ['k390', 'k270']) for (const opts of optsList) {
  laps.push({ seed, kart, opts, T: ev.lapTime(genes(seed), KartCore.kartParams(kart, opts)) });
}
laps.push({ seed: 0, kart: 'k390', opts: {}, T: ev.lapTime(new Array(ev.M).fill(0), KartCore.kartParams('k390', {})) });

const tel = ev.telemetry(genes(1), KartCore.kartParams('k390', {}));
const rep = KartCore.cornerReport(tr, tel);

const ga = {};
for (const kart of ['k390', 'k270']) {
  const g = new KartGA.GA(ev, KartCore.kartParams(kart, {}), { popSize: 80, seed: 12345 });
  let r;
  for (let i = 0; i < 200; i++) r = g.step();
  ga[kart] = { gens: 200, best: r.best, mean: r.mean, sigma: r.sigma };
}

const out = {
  generatedFrom: 'legacy/kart-lap-lab.html',
  track: { N: tr.N, L: tr.L, M: ev.M, corners: tr.corners.map((c) => [c.name, c.kind, c.dir, c.iMid]), crossings: tr.crossings,
    sumX: sum(tr.x), sumY: sum(tr.y), sumZ: sum(tr.z), sumKc: sum(tr.kc), sumGrade: sum(tr.grade) },
  laps,
  telemetry: { T: tel.T, sumV: sum(tel.v), sumThr: sum(tel.thr), sumBrk: sum(tel.brk), sumTime: sum(tel.time) },
  cornerReport: rep.map((c) => ({ name: c.name, brakeStart: c.brakeStart, brakeDist: c.brakeDist, vMin: c.vMin, vExit: c.vExit, latG: c.latG })),
  ga200: ga,
};
fs.writeFileSync(path.join(root, 'tests/fixtures/v1-reference.json'), JSON.stringify(out, null, 1));
console.log('N', tr.N, 'L', tr.L.toFixed(1), 'GA200', ga.k390.best.toFixed(3), ga.k270.best.toFixed(3));
