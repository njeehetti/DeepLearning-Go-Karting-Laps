// Training loop (UI-agnostic): sep-CMA-ES over the driver MLP, curriculum "finish the lap, then go
// fast", best brain bookkeeping. Used by the headless CLI and by the browser coordinator worker.
import { mulberry32 } from '../sim/rng';
import type { EpisodeSpec } from './episode';
import type { CandidateResult, WorldSpec } from './evalCore';
import { SepCmaEs, sampleCandidate } from './es';
import { DRIVER_SHAPE, paramCount } from './mlp';
import type { EvalPool } from './pool';

export interface TrainConfig {
  world: WorldSpec;
  lambda?: number;
  sigma0?: number;
  seed?: number;
  /** Extra random common starts per generation, on top of the start line. */
  randomStarts?: number;
  maxTime?: number;
  /** Lap length (m) and start-speed function, provided by the caller's World. */
  L: number;
  startSpeed: (s: number) => number;
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
  readonly cfg: Required<Omit<TrainConfig, 'world' | 'L' | 'startSpeed'>> & TrainConfig;
  readonly es: SepCmaEs;
  readonly pool: EvalPool;
  readonly history: GenStats[] = [];
  /** Best clean lap found so far. */
  best: { params: Float64Array; lap: number; gen: number } | null = null;
  /** Best brain of every generation (for "génération N vs N−50" ghosts). */
  readonly perGen: { gen: number; lap: number; params: Float64Array }[] = [];
  private timeWeight = 0;

  constructor(pool: EvalPool, cfg: TrainConfig) {
    this.cfg = { lambda: 256, sigma0: 0.5, seed: 1, randomStarts: 1, maxTime: 150, ...cfg };
    this.pool = pool;
    this.es = new SepCmaEs(paramCount(DRIVER_SHAPE), { lambda: this.cfg.lambda, sigma0: this.cfg.sigma0, seed: this.cfg.seed });
  }

  /** Common starts of a generation: the start line first (its lap time is the one reported). */
  episodes(gen: number): EpisodeSpec[] {
    const r = mulberry32((this.cfg.seed * 7919 + gen) >>> 0);
    const eps: EpisodeSpec[] = [{ s0: 0, d0: 0, v0: this.cfg.startSpeed(0), maxTime: this.cfg.maxTime, timeWeight: this.timeWeight }];
    for (let k = 0; k < this.cfg.randomStarts; k++) {
      const s0 = r() * this.cfg.L;
      eps.push({ s0, d0: 0, v0: this.cfg.startSpeed(s0), maxTime: this.cfg.maxTime, timeWeight: this.timeWeight });
    }
    return eps;
  }

  async step(): Promise<GenStats> {
    const t0 = performance.now();
    const es = this.es;
    const sample = { ...es.spec(), m: Float64Array.from(es.m), D: Float64Array.from(es.D) };
    const episodes = this.episodes(es.gen);
    const res: CandidateResult[] = await this.pool.run({ sample, episodes }, es.lambda);
    const fit = res.map((r) => r.fitness);
    let completedEp = 0;
    let cleanEp = 0;
    let bestLap = NaN;
    let bestI = -1;
    res.forEach((r, i) => {
      completedEp += r.completed;
      cleanEp += r.clean;
      if (r.clean === episodes.length && Number.isFinite(r.lap) && !(r.lap >= bestLap)) {
        bestLap = r.lap;
        bestI = i;
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
      sigma: es.sigma,
      timeWeight: this.timeWeight,
      wallMs: 0,
    };
    if (bestI >= 0) {
      const params = sampleCandidate(sample, bestI, new Float64Array(es.dim));
      this.perGen.push({ gen: es.gen, lap: bestLap, params });
      if (!this.best || bestLap < this.best.lap) this.best = { params, lap: bestLap, gen: es.gen };
    }
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
      shape: DRIVER_SHAPE.sizes, params: Array.from(this.best.params), lap: this.best.lap, gen: this.best.gen,
      kart: this.cfg.world.kart, track: this.cfg.world.track.id, scale: this.cfg.world.scale, createdAt: new Date().toISOString(),
    };
  }
}
