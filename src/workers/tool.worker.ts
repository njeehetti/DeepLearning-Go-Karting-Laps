// Heavy one-off jobs off the UI thread: record a lap for replay, theoretical-lap ghost, calibration.
import { World, type KartSetup, type WorldSpec } from '../ai/evalCore';
import { recordRun, type Run } from '../ai/record';
import { N_OBS } from '../ai/obs';
import { buildTrack } from '../track/build';
import { TrackQuery } from '../track/query';
import { calibrate, paramsOf, type CalibrationInput } from '../sim/calibrate';
import { kartParams } from '../sim/karts';
import { LineGA } from '../sim/lineGA';
import { LineEvaluator, lineLimits } from '../sim/qss';
import { makeQss2Ideal, solveQss2 } from '../sim/qss2';

export type ToolRequest =
  | { id: number; type: 'record'; world: WorldSpec; params: number[] }
  | { id: number; type: 'theory'; world: WorldSpec; gens?: number }
  | { id: number; type: 'calibrate'; world: WorldSpec; input: CalibrationInput };

export type ToolResponse =
  | { id: number; type: 'record'; run: Run }
  | { id: number; type: 'theory'; run: Run; lap: number }
  | { id: number; type: 'calibrate'; result: ReturnType<typeof calibrate> }
  | { id: number; type: 'error'; message: string };

const worlds = new Map<string, World>();
function worldOf(spec: WorldSpec): World {
  const key = JSON.stringify([spec.track.id, spec.scale, spec.width, spec.kart]);
  let w = worlds.get(key);
  if (!w) {
    w = new World(spec);
    worlds.set(key, w);
  }
  return w;
}

/** Theoretical lap (ideal bound) as a Run, so it can be replayed as a ghost. */
function theoryRun(spec: WorldSpec, kart: KartSetup, gens: number): { run: Run; lap: number } {
  const track = buildTrack(spec.track, { scale: spec.scale, width: spec.width });
  const q = new TrackQuery(track);
  const ev = new LineEvaluator(track, 16, lineLimits(track, q.kerb));
  const m = makeQss2Ideal(paramsOf(kart));
  const ga = new LineGA(ev, kartParams(kart.kartId), {
    popSize: 60, seed: 3,
    cost: (g) => {
      ev.lineFromGenes(g);
      return solveQss2(ev, m);
    },
  });
  for (let i = 0; i < gens; i++) ga.step();
  ev.lineFromGenes(ga.pop[0]);
  const lap = solveQss2(ev, m);
  const N = ev.N;
  const f = () => new Float32Array(N + 1);
  const run: Run = { hz: 0, n: N + 1, t: f(), x: f(), y: f(), z: f(), psi: f(), v: f(), ax: f(), ay: f(), thr: f(), brk: f(), steer: f(), s: f(), dist: f(), d: f(), obs: new Float32Array((N + 1) * N_OBS),
    result: { fitness: 0, progress: track.L, completed: true, clean: true, lapTime: lap, offTime: 0, end: 'tour' } };
  let t = 0;
  let dist = 0;
  for (let k = 0; k <= N; k++) {
    const i = k % N;
    const j = (i + 1) % N;
    run.t[k] = t;
    run.x[k] = ev.px[i];
    run.y[k] = ev.py[i];
    run.z[k] = track.z[i];
    run.psi[k] = Math.atan2(ev.py[j] - ev.py[i], ev.px[j] - ev.px[i]);
    run.v[k] = ev.v[i];
    const a = (ev.v[j] ** 2 - ev.v[i] ** 2) / (2 * ev.ds[i]);
    run.ax[k] = a;
    run.ay[k] = ev.v[i] ** 2 * ev.k[i];
    run.thr[k] = a > 0.05 ? 1 : 0;
    run.brk[k] = a < -0.8 ? 1 : 0;
    run.s[k] = track.s[i];
    run.dist[k] = dist;
    run.d[k] = ev.d[i];
    t += (2 * ev.ds[i]) / (ev.v[i] + ev.v[j]);
    dist += track.s[i + 1] - track.s[i];
  }
  return { run, lap };
}

const post = (m: ToolResponse) => (self as unknown as Worker).postMessage(m);

self.onmessage = (e: MessageEvent<ToolRequest>) => {
  const r = e.data;
  try {
    if (r.type === 'record') {
      const w = worldOf(r.world);
      const run = recordRun(w, r.params, { s0: 0, d0: 0, v0: w.startSpeed(0), maxTime: 200, timeWeight: 1 });
      post({ id: r.id, type: 'record', run });
    } else if (r.type === 'theory') {
      post({ id: r.id, type: 'theory', ...theoryRun(r.world, r.world.kart, r.gens ?? 400) });
    } else if (r.type === 'calibrate') {
      const track = buildTrack(r.world.track, { scale: r.world.scale, width: r.world.width });
      post({ id: r.id, type: 'calibrate', result: calibrate(track, r.input) });
    }
  } catch (err) {
    post({ id: r.id, type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
