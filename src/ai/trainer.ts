// Training loop (UI-agnostic): sep-CMA-ES over the driver MLP, curriculum "finish the lap, then go
// fast", best brain bookkeeping. Used by the headless CLI and by the browser coordinator worker.
import { mulberry32 } from '../sim/rng';
import type { EpisodeSpec } from './episode';
import type { CandidateResult, WorldSpec } from './evalCore';
import { SepCmaEs, sampleCandidate } from './es';
import { driverShape, paramCount } from './mlp';
import type { EvalPool } from './pool';

export interface TrainConfig {
  world: WorldSpec;
  lambda?: number;
  sigma0?: number;
  seed?: number;
  /** Extra random common starts per generation, on top of the start line. */
  randomStarts?: number;
  maxTime?: number;
  /** Starting brain (resume from a saved one). */
  x0?: ArrayLike<number>;
  /** Lap length (m) and start-speed function, provided by the caller's World. */
  L: number;
  startSpeed: (s: number) => number;
  /** Same for the extra circuits of a generalist training (world.others, in order). */
  others?: { L: number; startSpeed: (s: number) => number }[];
}

export interface GenStats {
  gen: number;
  best: number;
  mean: number;
  /** Share of episodes that completed a lap / a clean lap. */
  completion: number;
  cleanRate: number;
  /** Fastest clean lap from the start line among this generation's candidates (s), NaN if none. */
  bestLap: number;
  /** Steering activity of that driver (|Δ command| per second). */
  bestSteer: number;
  sigma: number;
  timeWeight: number;
  wallMs: number;
}

export interface Brain {
  shape: number[];
  params: number[];
  lap: number;
  gen: number;
  kart: WorldSpec['kart'];
  track: string;
  scale: string;
  createdAt: string;
}

export class Trainer {
  readonly cfg: Required<Omit<TrainConfig, 'world' | 'L' | 'startSpeed' | 'x0' | 'others'>> & TrainConfig;
  readonly es: SepCmaEs;
  readonly pool: EvalPool;
  readonly history: GenStats[] = [];
  /** Best clean lap found so far. */
  best: { params: Float64Array; lap: number; gen: number } | null = null;
  /** Best brain of every generation (for "génération N vs N−50" ghosts). */
  readonly perGen: { gen: number; lap: number; params: Float64Array }[] = [];
  private timeWeight = 0;

  constructor(pool: EvalPool, cfg: TrainConfig) {
    this.cfg = { lambda: 256, sigma0: 0.5, seed: 1, randomStarts: cfg.others?.length ? 3 : 1, maxTime: 150, ...cfg };
    this.pool = pool;
    this.es = new SepCmaEs(paramCount(driverShape(cfg.world.hidden)), { lambda: this.cfg.lambda, sigma0: this.cfg.sigma0, seed: this.cfg.seed, x0: this.cfg.x0 });
  }

  /** Common starts of a generation: the start line first (its lap time is the one reported). */
  episodes(gen: number): EpisodeSpec[] {
    const r = mulberry32((this.cfg.seed * 7919 + gen) >>> 0);
    const eps: EpisodeSpec[] = [{ s0: 0, d0: 0, v0: this.cfg.startSpeed(0), maxTime: this.cfg.maxTime, timeWeight: this.timeWeight }];
    const circuits = [{ L: this.cfg.L, startSpeed: this.cfg.startSpeed }, ...(this.cfg.others ?? [])];
    for (let k = 0; k < this.cfg.randomStarts; k++) {
      // generalist: each extra start is on a circuit drawn at random (the same for every candidate)
      const track = circuits.length > 1 ? Math.floor(r() * circuits.length) : 0;
      const c = circuits[track];
      const s0 = r() * c.L;
      eps.push({ s0, d0: 0, v0: c.startSpeed(s0), maxTime: this.cfg.maxTime, timeWeight: this.timeWeight, track });
    }
    return eps;
  }

  /** Candidates whose start-line trajectory is recorded for the live view (0 = none). */
  recordN = 0;
  /** Trajectories of the last generation (index = candidate) and their best clean index. */
  lastTraj: (Float32Array | undefined)[] = [];
  lastBestIndex = -1;

  async step(): Promise<GenStats> {
    const t0 = performance.now();
    const es = this.es;
    const sample = { ...es.spec(), m: Float64Array.from(es.m), D: Float64Array.from(es.D) };
    const episodes = this.episodes(es.gen);
    const res: CandidateResult[] = await this.pool.run({ sample, episodes, recordN: this.recordN }, es.lambda);
    this.lastTraj = res.map((r) => r.traj);
    const fit = res.map((r) => r.fitness);
    let completedEp = 0;
    let cleanEp = 0;
    let bestLap = NaN;
    let bestI = -1;
    let bestSteer = NaN;
    res.forEach((r, i) => {
      completedEp += r.completed;
      cleanEp += r.clean;
      if (r.lapClean && Number.isFinite(r.lap) && !(r.lap >= bestLap)) {
        bestLap = r.lap;
        bestI = i;
        bestSteer = r.steerActivity;
      }
    });
    const nEp = res.length * episodes.length;
    const stats: GenStats = {
      gen: es.gen,
      best: Math.max(...fit),
      mean: fit.reduce((a, b) => a + b, 0) / fit.length,
      completion: completedEp / nEp,
      cleanRate: cleanEp / nEp,
      bestLap,
      bestSteer,
      sigma: es.sigma,
      timeWeight: this.timeWeight,
      wallMs: 0,
    };
    this.lastBestIndex = bestI;
    if (bestI >= 0) {
      const params = sampleCandidate(sample, bestI, new Float64Array(es.dim));
      this.perGen.push({ gen: es.gen, lap: bestLap, params });
      if (!this.best || bestLap < this.best.lap) this.best = { params, lap: bestLap, gen: es.gen };
    }
    // elitism: the best brain so far drives again next generation, so the generation best never regresses
    if (this.best) es.elite = this.best.params;
    // the elite (candidate 1) is only re-evaluated, it must not bias the step-size / covariance update
    if (sample.elite) fit[1] = -Infinity;
    es.tell(fit);
    // curriculum: lap time only counts once a fifth of the episodes finish
    this.timeWeight = stats.completion >= 0.2 ? 1 : 0;
    stats.wallMs = performance.now() - t0;
    this.history.push(stats);
    return stats;
  }

  brain(): Brain | null {
    if (!this.best) return null;
    return {
      shape: driverShape(this.cfg.world.hidden).sizes, params: Array.from(this.best.params), lap: this.best.lap, gen: this.best.gen,
      kart: this.cfg.world.kart, track: this.cfg.world.track.id, scale: this.cfg.world.scale, createdAt: new Date().toISOString(),
    };
  }
}
