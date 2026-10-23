// Evaluation core shared by every pool (in-thread, Node worker_threads, Web Workers).
import { buildTrack } from '../track/build';
import { TrackQuery } from '../track/query';
import type { TrackDef, TrackScale } from '../track/types';
import { paramsOf, type Calibration } from '../sim/calibrate';
import { KartSim, KartState } from '../sim/kart';
import { LineEvaluator } from '../sim/qss';
import { makeQss2Ideal, solveQss2 } from '../sim/qss2';
import { Pilot, runEpisode, type EpisodeResult, type EpisodeSpec } from './episode';
import { sampleCandidate, type SampleSpec } from './es';
import { DRIVER_SHAPE, paramCount } from './mlp';

export type KartSetup = Pick<Calibration, 'kartId' | 'mu' | 'powerScale' | 'vTopKmh' | 'transmission' | 'driver'>;

export interface WorldSpec {
  track: TrackDef;
  scale: TrackScale;
  width?: number;
  kart: KartSetup;
}

export interface CandidateResult {
  fitness: number;
  /** Episodes completed / clean. */
  completed: number;
  clean: number;
  /** Lap time of the episode from the start line (index 0), NaN if not completed. */
  lap: number;
  /** The start-line lap was completed without any wheel on the grass. */
  lapClean: boolean;
  steerActivity: number;
  progress: number;
  offTime: number;
  /** Start-line episode trajectory at 10 Hz (x, y, psi, throttle, brake per sample), only when the job asks for it. */
  traj?: Float32Array;
}

/** Floats per sample in a live trajectory: x, y, psi, throttle, brake. */
export const LIVE_STRIDE = 5;

export interface GenJob {
  sample: SampleSpec;
  episodes: EpisodeSpec[];
  /** Record the start-line trajectory of candidates 0 .. recordN − 1 (live view). */
  recordN?: number;
}

export class World {
  readonly q: TrackQuery;
  readonly sim: KartSim;
  /** Theoretical (ideal) speed profile on the centreline, used for rolling starts. */
  readonly profile: Float64Array;
  readonly dim = paramCount(DRIVER_SHAPE);

  constructor(spec: WorldSpec) {
    const track = buildTrack(spec.track, { scale: spec.scale, width: spec.width });
    this.q = new TrackQuery(track);
    const p = paramsOf(spec.kart);
    this.sim = new KartSim(this.q, p);
    const ev = new LineEvaluator(track, 16);
    ev.lineFromGenes(new Float64Array(ev.M));
    solveQss2(ev, makeQss2Ideal(p));
    this.profile = Float64Array.from(ev.v);
  }

  /** Rolling start: a bit below the theoretical centreline speed at s. */
  startSpeed(s: number): number {
    return 0.8 * this.q.at(this.profile, s);
  }
}

export class Evaluator {
  readonly world: World;
  private readonly pilot = new Pilot();
  private readonly st = new KartState();
  private readonly x: Float64Array;
  private readonly z: Float64Array;

  constructor(world: World) {
    this.world = world;
    this.pilot.profile = world.profile;
    this.x = new Float64Array(world.dim);
    this.z = new Float64Array(world.dim);
  }

  episode(params: ArrayLike<number>, spec: EpisodeSpec): EpisodeResult {
    return runEpisode(this.world.sim, this.pilot, params, spec, this.st);
  }

  candidate(job: GenJob, i: number): CandidateResult {
    sampleCandidate(job.sample, i, this.x, this.z);
    let fitness = 0;
    let completed = 0;
    let clean = 0;
    let progress = 0;
    let offTime = 0;
    let lap = NaN;
    let lapClean = false;
    let steerActivity = 0;
    let traj: Float32Array | undefined;
    job.episodes.forEach((spec, e) => {
      let r: EpisodeResult;
      if (e === 0 && i < (job.recordN ?? 0)) {
        const buf: number[] = [];
        r = runEpisode(this.world.sim, this.pilot, this.x, spec, this.st, (st, _inp, decided) => {
          if (decided && Math.round(st.t * 30) % 3 === 0) buf.push(st.x, st.y, st.psi, st.thr, st.brk);
        });
        traj = Float32Array.from(buf);
      } else r = this.episode(this.x, spec);
      fitness += r.fitness;
      progress += r.progress;
      offTime += r.offTime;
      if (r.completed) completed++;
      if (r.clean) clean++;
      if (e === 0 && r.completed) lap = r.lapTime;
      if (e === 0) {
        lapClean = r.clean;
        steerActivity = r.steerActivity;
      }
    });
    const n = job.episodes.length;
    return { fitness: fitness / n, completed, clean, lap, lapClean, steerActivity, progress: progress / n, offTime, traj };
  }
}

/** Worker protocol (Node and browser). */
export type ToWorker =
  | { type: 'init'; world: WorldSpec }
  | { type: 'job'; id: number; job: GenJob; from: number; to: number };
export type FromWorker =
  | { type: 'ready' }
  | { type: 'result'; id: number; from: number; results: CandidateResult[] }
  | { type: 'error'; message: string };

/** Message handler used by both worker entry points. */
export function makeHandler(post: (m: FromWorker) => void): (m: ToWorker) => void {
  let ev: Evaluator | null = null;
  return (m) => {
    try {
      if (m.type === 'init') {
        ev = new Evaluator(new World(m.world));
        post({ type: 'ready' });
      } else if (m.type === 'job') {
        if (!ev) throw new Error('worker non initialisé');
        const results: CandidateResult[] = [];
        for (let i = m.from; i < m.to; i++) results.push(ev.candidate(m.job, i));
        post({ type: 'result', id: m.id, from: m.from, results });
      }
    } catch (e) {
      post({ type: 'error', message: e instanceof Error ? e.stack ?? e.message : String(e) });
    }
  };
}
