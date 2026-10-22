// Genetic algorithm over racing lines (v1 KartGA, verbatim port). Used to compute the theoretical lap.
import type { QssParams } from './karts';
import type { LineEvaluator } from './qss';
import { mulberry32 } from './rng';

export interface LineGAOptions {
  popSize?: number;
  elite?: number;
  sigma?: number;
  seed?: number;
  seedGenes?: ArrayLike<number> | null;
}

export interface LineGAStep {
  gen: number;
  best: number;
  mean: number;
  sigma: number;
}

export class LineGA {
  readonly ev: LineEvaluator;
  readonly kp: QssParams;
  readonly M: number;
  readonly lim: number;
  readonly popSize: number;
  readonly elite: number;
  readonly sigma0: number;
  sigma: number;
  readonly minSigma = 0.04;
  private readonly rand: () => number;
  gen = 0;
  evals = 0;
  bestEver = Infinity;
  stall = 0;
  readonly history: [number, number, number][] = [];
  pop: Float64Array[] = [];
  fit: Float64Array;

  constructor(evaluator: LineEvaluator, kp: QssParams, opts: LineGAOptions = {}) {
    this.ev = evaluator;
    this.kp = kp;
    this.M = evaluator.M;
    this.lim = evaluator.limit();
    this.popSize = opts.popSize || 80;
    this.elite = opts.elite || 4;
    this.sigma0 = opts.sigma || 1.4;
    this.sigma = this.sigma0;
    this.rand = mulberry32(opts.seed || 12345);
    this.fit = new Float64Array(this.popSize);
    for (let p = 0; p < this.popSize; p++) this.pop.push(this.randomIndividual(p === 0 ? 0 : 0.85));
    if (opts.seedGenes && opts.seedGenes.length === this.M) this.pop[1] = Float64Array.from(opts.seedGenes);
    this.evaluateAll();
  }

  private gauss(): number {
    const u = 1 - this.rand();
    const v = this.rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  randomIndividual(amp: number): Float64Array {
    const M = this.M;
    const g = new Float64Array(M);
    if (amp > 0) {
      const nb = 3 + Math.floor(this.rand() * 6);
      for (let b = 0; b < nb; b++) {
        const f = 1 + Math.floor(this.rand() * 12);
        const ph = this.rand() * 6.283;
        const a = (((this.rand() * 2 - 1) * this.lim * amp) / Math.sqrt(nb)) * 1.6;
        for (let j = 0; j < M; j++) g[j] += a * Math.sin(ph + (f * 6.283 * j) / M);
      }
    }
    this.clamp(g);
    return g;
  }

  private clamp(g: Float64Array): void {
    const l = this.lim;
    for (let j = 0; j < g.length; j++) {
      if (g[j] > l) g[j] = l;
      else if (g[j] < -l) g[j] = -l;
    }
  }

  private evaluateAll(): void {
    for (let p = 0; p < this.popSize; p++) this.fit[p] = this.ev.lapTime(this.pop[p], this.kp);
    this.evals += this.popSize;
    this.sort();
  }

  private sort(): void {
    const idx: number[] = [];
    for (let p = 0; p < this.popSize; p++) idx.push(p);
    const fit = this.fit;
    idx.sort((a, b) => fit[a] - fit[b]);
    const np: Float64Array[] = [];
    const nf = new Float64Array(this.popSize);
    for (let p = 0; p < this.popSize; p++) {
      np.push(this.pop[idx[p]]);
      nf[p] = fit[idx[p]];
    }
    this.pop = np;
    this.fit = nf;
  }

  private tournament(): Float64Array {
    let best = Math.floor(this.rand() * this.popSize);
    for (let k = 0; k < 2; k++) {
      const c = Math.floor(this.rand() * this.popSize);
      if (c < best) best = c;
    }
    return this.pop[best];
  }

  private child(): Float64Array {
    const M = this.M;
    const A = this.tournament();
    const B = this.tournament();
    const g = new Float64Array(M);
    // circular two-point crossover with a soft blend at the cuts
    const c1 = Math.floor(this.rand() * M);
    const len = 2 + Math.floor(this.rand() * M * 0.5);
    const blend = 3;
    for (let j = 0; j < M; j++) {
      const dpos = (j - c1 + M) % M;
      const w = dpos < len ? Math.min(1, Math.min(dpos + 1, len - dpos) / blend) : 0;
      g[j] = A[j] * (1 - w) + B[j] * w;
    }
    // mutation: smooth gaussian bumps
    const nb = this.rand() < 0.9 ? 1 + Math.floor(-Math.log(1 - this.rand()) * 1.2) : 0;
    for (let b = 0; b < nb; b++) {
      const c = this.rand() * M;
      const wd = 0.8 + this.rand() * 3.5;
      const amp = this.gauss() * this.sigma;
      const r = Math.ceil(wd * 3);
      for (let q = -r; q <= r; q++) {
        const jj = (((Math.floor(c) + q) % M) + M) % M;
        const x = (Math.floor(c) + q - c) / wd;
        g[jj] += amp * Math.exp(-x * x);
      }
    }
    this.clamp(g);
    return g;
  }

  step(): LineGAStep {
    const next: Float64Array[] = [];
    const nf = new Float64Array(this.popSize);
    for (let e = 0; e < this.elite; e++) {
      next.push(this.pop[e]);
      nf[e] = this.fit[e];
    }
    for (let p = this.elite; p < this.popSize; p++) {
      const g = this.child();
      next.push(g);
      nf[p] = this.ev.lapTime(g, this.kp);
    }
    this.evals += this.popSize - this.elite;
    this.pop = next;
    this.fit = nf;
    this.sort();
    this.gen++;
    const best = this.fit[0];
    let mean = 0;
    for (let p = 0; p < this.popSize; p++) mean += this.fit[p];
    mean /= this.popSize;
    if (best < this.bestEver - 1e-4) {
      this.bestEver = best;
      this.stall = 0;
    } else this.stall++;
    // adaptive mutation strength
    this.sigma = Math.max(this.minSigma, this.sigma * 0.994);
    if (this.stall > 80) {
      this.sigma = Math.min(this.sigma0 * 0.6, this.sigma * 3);
      this.stall = 0;
    }
    this.history.push([this.gen, best, mean]);
    return { gen: this.gen, best, mean, sigma: this.sigma };
  }
}
