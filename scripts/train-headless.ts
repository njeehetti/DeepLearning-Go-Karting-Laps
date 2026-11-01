// Headless training benchmark (criteria 1 and 2): npx tsx scripts/train-headless.ts [--kart k270]
// [--track lyl] [--minutes 10] [--lambda 256] [--workers N] [--seed 1] [--out out/brains]
// Generalist: --others all (or genk,pfi,…) also drives the other circuits. --x0 <brain.json> resumes a brain.
// --eval <brain.json> only measures a brain on every circuit. --weather sec|gras|humide|pluie sets the conditions.
// --hidden 32 (or 32,16) sizes the network; an --x0 brain with a smaller single hidden layer is widened.
// --driver 80 sets the driver's mass in kit (kg, default from the calibration).
import { availableParallelism } from 'node:os';
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { Worker } from 'node:worker_threads';
import type { TrackDef } from '../src/track/types';
import type { Weather } from '../src/sim/weather';
import { DEFAULT_CALIBRATION } from '../src/sim/defaults';
import { World, type FromWorker, type WorldSpec } from '../src/ai/evalCore';
import { WorkerPool, type WorkerLike } from '../src/ai/pool';
import { Trainer } from '../src/ai/trainer';
import { Pilot, runEpisode } from '../src/ai/episode';
import { driverShape, widen, type MlpShape } from '../src/ai/mlp';
import { mulberry32 } from '../src/sim/rng';

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
const loadTrack = (id: string) => JSON.parse(readFileSync(new URL(`../data/tracks/${id}.json`, import.meta.url), 'utf8')) as TrackDef;
const allIds = readdirSync(new URL('../data/tracks/', import.meta.url)).filter((f) => f.endsWith('.json')).map((f) => f.slice(0, -5));
const def = loadTrack(trackId);
const othersArg = arg('others', '');
const otherIds = othersArg === 'all' ? allIds.filter((id) => id !== trackId) : othersArg ? othersArg.split(',') : [];
const readBrain = (f: string): { params: number[]; shape: MlpShape } => {
  const b = JSON.parse(readFileSync(f, 'utf8'));
  const o = b.brain ?? b;
  return { params: o.params, shape: { sizes: o.shape ?? driverShape().sizes } };
};
const hiddenArg = arg('hidden', '');
const hidden = hiddenArg ? hiddenArg.split(',').map((n) => parseInt(n, 10)) : undefined;
const fmt = (T: number) => (Number.isFinite(T) ? `${Math.floor(T / 60)}:${(T % 60).toFixed(2).padStart(5, '0')}` : '–');

const c = DEFAULT_CALIBRATION[kartId];
const world: WorldSpec = {
  track: def, scale: 'official',
  kart: { kartId, mu: parseFloat(arg('mu', String(c.mu))), powerScale: parseFloat(arg('power', String(c.powerScale))), vTopKmh: c.vTopKmh, transmission: c.transmission, driver: parseFloat(arg('driver', String(c.driver))), weather: arg('weather', 'sec') as Weather },
  ...(otherIds.length ? { others: otherIds.map(loadTrack) } : {}),
  ...(hidden ? { hidden } : {}),
};
const local = new World(world);

/** A flying lap from the start line of every circuit: lap time and whether it stayed on the track. */
function evalAll(params: ArrayLike<number>, shape: MlpShape): void {
  for (const id of allIds) {
    const w = new World({ ...world, track: loadTrack(id), others: undefined });
    const pilot = new Pilot(shape);
    const r = runEpisode(w.sim, pilot, params, { s0: 0, d0: 0, v0: w.startSpeed(0), maxTime: 200, timeWeight: 1 });
    console.log(`  ${id.padEnd(13)} ${r.completed ? fmt(r.lapTime) : 'non bouclé (' + r.end + ', ' + r.progress.toFixed(0) + ' m)'}${r.completed && !r.clean ? ' (herbe)' : ''}`);
  }
}
const evalFile = arg('eval', '');
if (evalFile) {
  console.log('Tour lancé sur chaque circuit :');
  const b = readBrain(evalFile);
  evalAll(b.params, b.shape);
  process.exit(0);
}
const x0File = arg('x0', '');
const shape = driverShape(hidden);
let x0: ArrayLike<number> | undefined;
if (x0File) {
  const b = readBrain(x0File);
  x0 = b.shape.sizes.join() === shape.sizes.join() ? b.params : widen(b.params, b.shape, shape, mulberry32(seed));
}

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
const otherWorlds = (world.others ?? []).map((track) => new World({ ...world, track, others: undefined }));
const trainer = new Trainer(pool, {
  world, lambda, seed, L: local.q.L, startSpeed: (s) => local.startSpeed(s),
  others: otherWorlds.map((w) => ({ L: w.q.L, startSpeed: (s: number) => w.startSpeed(s) })),
  ...(x0 ? { x0 } : {}),
});

console.log(`Entraînement ${kartId}, réseau ${shape.sizes.join('-')}, λ=${lambda}, ${nWorkers} workers, ${minutes} min, tour théorique ${fmt(c.lap)}`);
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
// generalist: the best start-line lap only speaks for the main circuit; the population mean is the driver
// that learnt all of them
if (brain && otherIds.length) {
  brain.params = Array.from(trainer.es.m);
  brain.track = 'generaliste';
}
if (brain) {
  mkdirSync(outDir, { recursive: true });
  const file = `${outDir}/${brain.track}-${kartId}-${Date.now()}.json`;
  writeFileSync(file, JSON.stringify({ brain, history: trainer.history }));
  console.log('Cerveau sauvegardé :', file);
  if (otherIds.length) {
    console.log('Tour lancé sur chaque circuit :');
    evalAll(brain.params, shape);
  }
}
pool.close();
