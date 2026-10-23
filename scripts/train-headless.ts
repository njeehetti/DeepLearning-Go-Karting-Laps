// Headless training benchmark (criteria 1 and 2): npx tsx scripts/train-headless.ts [--kart k270]
// [--track lyl] [--minutes 10] [--lambda 256] [--workers N] [--seed 1] [--out out/brains]
import { availableParallelism } from 'node:os';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { Worker } from 'node:worker_threads';
import type { TrackDef } from '../src/track/types';
import { DEFAULT_CALIBRATION } from '../src/sim/defaults';
import { World, type FromWorker, type WorldSpec } from '../src/ai/evalCore';
import { WorkerPool, type WorkerLike } from '../src/ai/pool';
import { Trainer } from '../src/ai/trainer';

const arg = (k: string, d: string) => {
  const i = process.argv.indexOf('--' + k);
  return i > 0 ? process.argv[i + 1] : d;
};
const kartId = arg('kart', 'k270') as 'k270' | 'k390';
const minutes = parseFloat(arg('minutes', '10'));
const lambda = parseInt(arg('lambda', '256'), 10);
const nWorkers = parseInt(arg('workers', String(Math.max(1, availableParallelism() - 1))), 10);
const seed = parseInt(arg('seed', '1'), 10);
const outDir = arg('out', 'out/brains');
const trackId = arg('track', 'lyl');
const def = JSON.parse(readFileSync(new URL(`../data/tracks/${trackId}.json`, import.meta.url), 'utf8')) as TrackDef;

const c = DEFAULT_CALIBRATION[kartId];
const world: WorldSpec = {
  track: def, scale: 'official',
  kart: { kartId, mu: c.mu, powerScale: c.powerScale, vTopKmh: c.vTopKmh, transmission: c.transmission, driver: c.driver },
};
const local = new World(world);

const workers: WorkerLike[] = Array.from({ length: nWorkers }, () => {
  const w = new Worker(new URL('../src/ai/node.worker.ts', import.meta.url), { execArgv: ['--import', 'tsx'] });
  return {
    post: (m) => w.postMessage(m),
    onMessage: (cb) => w.on('message', (m: FromWorker) => cb(m)),
    terminate: () => void w.terminate(),
  };
});
const pool = new WorkerPool(workers);
await pool.init(world);
const trainer = new Trainer(pool, { world, lambda, seed, L: local.q.L, startSpeed: (s) => local.startSpeed(s) });

const fmt = (T: number) => (Number.isFinite(T) ? `${Math.floor(T / 60)}:${(T % 60).toFixed(2).padStart(5, '0')}` : '–');
console.log(`Entraînement ${kartId}, λ=${lambda}, ${nWorkers} workers, ${minutes} min, tour théorique ${fmt(c.lap)}`);
const t0 = performance.now();
let firstLap = -1;
let firstClean = -1;
while ((performance.now() - t0) / 60000 < minutes) {
  const s = await trainer.step();
  const el = (performance.now() - t0) / 1000;
  if (firstLap < 0 && s.completion > 0) firstLap = el;
  if (firstClean < 0 && Number.isFinite(s.bestLap)) firstClean = el;
  if (s.gen % 5 === 0 || Number.isFinite(s.bestLap)) {
    console.log(
      `g${String(s.gen).padStart(4)} t=${el.toFixed(0).padStart(4)}s  best ${s.best.toFixed(0).padStart(5)}  moy ${s.mean.toFixed(0).padStart(5)}  ` +
        `tours ${(s.completion * 100).toFixed(0).padStart(3)}%  propres ${(s.cleanRate * 100).toFixed(0).padStart(3)}%  ` +
        `tour ${fmt(s.bestLap)}  volant ${Number.isFinite(s.bestSteer) ? s.bestSteer.toFixed(1) : "–"}/s  record ${fmt(trainer.best?.lap ?? NaN)}  σ ${s.sigma.toFixed(3)}  ${s.wallMs.toFixed(0)} ms`,
    );
  }
}
console.log(`Premier tour bouclé : ${firstLap < 0 ? 'jamais' : firstLap.toFixed(0) + ' s'} ; premier tour propre : ${firstClean < 0 ? 'jamais' : firstClean.toFixed(0) + ' s'} ; meilleur ${fmt(trainer.best?.lap ?? NaN)}`);
const brain = trainer.brain();
if (brain) {
  mkdirSync(outDir, { recursive: true });
  const file = `${outDir}/${brain.track}-${kartId}-${Date.now()}.json`;
  writeFileSync(file, JSON.stringify({ brain, history: trainer.history }));
  console.log('Cerveau sauvegardé :', file);
}
pool.close();
