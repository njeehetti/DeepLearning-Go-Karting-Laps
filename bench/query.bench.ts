// Micro-benchmark of track queries (npx tsx bench/query.bench.ts).
import lyl from '../data/tracks/lyl.json';
import { buildTrack } from '../src/track/build';
import { Proj, TrackQuery } from '../src/track/query';
import type { TrackDef } from '../src/track/types';

const t = buildTrack(lyl as TrackDef, { scale: 'official' });
const q = new TrackQuery(t);
const p = new Proj();
const n = 400_000;
let acc = 0;
let t0 = performance.now();
for (let k = 0; k < n; k++) {
  const i = k % t.N;
  q.project(t.x[i] + t.nx[i], t.y[i] + t.ny[i], i, p, 3);
  acc += p.d;
}
const proj = ((performance.now() - t0) / n) * 1000;
t0 = performance.now();
for (let k = 0; k < n; k++) {
  const i = k % t.N;
  const a = (k % 9) / 8 * Math.PI - Math.PI / 2;
  const c = Math.cos(a);
  const s = Math.sin(a);
  acc += q.rayEdges(t.x[i], t.y[i], c * t.tx[i] - s * t.ty[i], s * t.tx[i] + c * t.ty[i], i);
}
const ray = ((performance.now() - t0) / n) * 1000;
console.log(`projection ${proj.toFixed(3)} µs · rayon ${ray.toFixed(3)} µs · 9 rayons ${(9 * ray).toFixed(2)} µs par décision (${acc.toFixed(0)})`);
