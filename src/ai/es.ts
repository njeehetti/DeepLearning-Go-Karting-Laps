// Separable CMA-ES (Ros & Hansen 2008): diagonal covariance, O(n) per sample, robust defaults.
// Candidates are regenerated from (seed, generation, index), so workers only need m, σ and the
// diagonal to rebuild them: tiny messages and bit-identical results wherever they run.
import { mulberry32 } from '../sim/rng';

/** Generic trainer interface, so a gradient method (PPO) can be plugged in later. */
export interface PolicyTrainer {
  readonly dim: number;
  readonly lambda: number;
  readonly gen: number;
  /** Writes candidate i of the current generation into out. */
  candidate(i: number, out: Float64Array): Float64Array;
  /** Reports fitness (higher is better) for all candidates of the current generation. */
  tell(fitness: ArrayLike<number>): void;
  mean(): Float64Array;
  toJSON(): unknown;
}

export interface SampleSpec {
  seed: number;
  gen: number;
  m: Float64Array;
  sigma: number;
  D: Float64Array;
  /** Best-so-far brain re-entered as candidate 1 (elitism: a generation never does worse). */
  elite?: Float64Array | null;
}

function hash3(a: number, b: number, c: number): number {
  let h = (a ^ 0x9e3779b9) >>> 0;
  h = Math.imul(h ^ (b + 0x85ebca6b), 0xc2b2ae35) >>> 0;
  h = Math.imul(h ^ (c + 0x27d4eb2f), 0x165667b1) >>> 0;
  return (h ^ (h >>> 15)) >>> 0;
}

/** Standard normal vector for (seed, gen, i), Box-Muller on mulberry32. */
export function gaussVec(seed: number, gen: number, i: number, out: Float64Array): Float64Array {
  const r = mulberry32(hash3(seed, gen, i));
  for (let k = 0; k < out.length; k += 2) {
    const u = 1 - r();
    const v = r();
    const rad = Math.sqrt(-2 * Math.log(u));
    out[k] = rad * Math.cos(2 * Math.PI * v);
    if (k + 1 < out.length) out[k + 1] = rad * Math.sin(2 * Math.PI * v);
  }
  return out;
}

/** x_i = m + σ·D·z_i. Candidate 0 is the mean itself, candidate 1 the elite when there is one. */
export function sampleCandidate(s: SampleSpec, i: number, out: Float64Array, z: Float64Array = new Float64Array(out.length)): Float64Array {
  if (i === 0) {
    out.set(s.m);
    return out;
  }
  if (i === 1 && s.elite) {
    out.set(s.elite);
    return out;
  }
  gaussVec(s.seed, s.gen, i, z);
  for (let k = 0; k < out.length; k++) out[k] = s.m[k] + s.sigma * s.D[k] * z[k];
  return out;
}

export interface SepCmaOptions {
  lambda?: number;
  sigma0?: number;
  seed?: number;
  x0?: ArrayLike<number>;
}

export class SepCmaEs implements PolicyTrainer {
  readonly dim: number;
  readonly lambda: number;
  readonly mu: number;
  readonly seed: number;
  gen = 0;
  sigma: number;
  readonly m: Float64Array;
  readonly C: Float64Array;
  readonly D: Float64Array;
  elite: Float64Array | null = null;
  private readonly pc: Float64Array;
  private readonly ps: Float64Array;
  private readonly w: Float64Array;
  private readonly mueff: number;
  private readonly cs: number;
  private readonly ds: number;
  private readonly cc: number;
  private readonly c1: number;
  private readonly cmu: number;
  private readonly chiN: number;

  constructor(dim: number, opts: SepCmaOptions = {}) {
    const n = (this.dim = dim);
    this.lambda = opts.lambda ?? 4 + Math.floor(3 * Math.log(n));
    this.mu = Math.floor(this.lambda / 2);
    this.seed = opts.seed ?? 1;
    this.sigma = opts.sigma0 ?? 0.5;
    this.m = opts.x0 ? Float64Array.from(opts.x0) : new Float64Array(n);
    this.C = new Float64Array(n).fill(1);
    this.D = new Float64Array(n).fill(1);
    this.pc = new Float64Array(n);
    this.ps = new Float64Array(n);
    const w = new Float64Array(this.mu);
    let sw = 0;
    for (let i = 0; i < this.mu; i++) sw += w[i] = Math.log(this.mu + 0.5) - Math.log(i + 1);
    let sw2 = 0;
    for (let i = 0; i < this.mu; i++) {
      w[i] /= sw;
      sw2 += w[i] * w[i];
    }
    this.w = w;
    const mueff = (this.mueff = 1 / sw2);
    this.cs = (mueff + 2) / (n + mueff + 5);
    this.ds = 1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (n + 1)) - 1) + this.cs;
    this.cc = (4 + mueff / n) / (n + 4 + (2 * mueff) / n);
    const c1 = 2 / ((n + 1.3) ** 2 + mueff);
    const cmu = Math.min(1 - c1, (2 * (mueff - 2 + 1 / mueff)) / ((n + 2) ** 2 + mueff));
    // separable variant learns (n + 2) / 3 times faster
    const f = (n + 2) / 3;
    this.c1 = Math.min(0.5, c1 * f);
    this.cmu = Math.min(1 - this.c1, cmu * f);
    this.chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n));
  }

  spec(): SampleSpec {
    return { seed: this.seed, gen: this.gen, m: this.m, sigma: this.sigma, D: this.D, elite: this.elite };
  }

  candidate(i: number, out: Float64Array): Float64Array {
    return sampleCandidate(this.spec(), i, out);
  }

  mean(): Float64Array {
    return Float64Array.from(this.m);
  }

  tell(fitness: ArrayLike<number>): void {
    const n = this.dim;
    const lam = this.lambda;
    const idx = Array.from({ length: lam }, (_, i) => i).sort((a, b) => fitness[b] - fitness[a] || a - b);
    const yw = new Float64Array(n);
    const zw = new Float64Array(n);
    const z = new Float64Array(n);
    const sumY2 = new Float64Array(n);
    for (let r = 0; r < this.mu; r++) {
      const i = idx[r];
      if (i === 0) z.fill(0);
      else if (i === 1 && this.elite) {
        // injected solution: its step expressed in the sampling frame, clipped like a normal draw
        for (let k = 0; k < n; k++) z[k] = Math.max(-4, Math.min(4, (this.elite[k] - this.m[k]) / (this.sigma * this.D[k])));
      } else gaussVec(this.seed, this.gen, i, z);
      const wr = this.w[r];
      for (let k = 0; k < n; k++) {
        const y = this.D[k] * z[k];
        yw[k] += wr * y;
        zw[k] += wr * z[k];
        sumY2[k] += wr * y * y;
      }
    }
    for (let k = 0; k < n; k++) this.m[k] += this.sigma * yw[k];
    const csn = Math.sqrt(this.cs * (2 - this.cs) * this.mueff);
    let psNorm = 0;
    for (let k = 0; k < n; k++) {
      this.ps[k] = (1 - this.cs) * this.ps[k] + csn * zw[k];
      psNorm += this.ps[k] * this.ps[k];
    }
    psNorm = Math.sqrt(psNorm);
    const hs = psNorm / Math.sqrt(1 - (1 - this.cs) ** (2 * (this.gen + 1))) < (1.4 + 2 / (n + 1)) * this.chiN ? 1 : 0;
    const ccn = Math.sqrt(this.cc * (2 - this.cc) * this.mueff);
    for (let k = 0; k < n; k++) {
      this.pc[k] = (1 - this.cc) * this.pc[k] + hs * ccn * yw[k];
      const c = this.C[k];
      this.C[k] = (1 - this.c1 - this.cmu) * c + this.c1 * (this.pc[k] ** 2 + (1 - hs) * this.cc * (2 - this.cc) * c) + this.cmu * sumY2[k];
      this.D[k] = Math.sqrt(this.C[k]);
    }
    this.sigma *= Math.exp((this.cs / this.ds) * (psNorm / this.chiN - 1));
    this.sigma = Math.min(this.sigma, 5);
    this.gen++;
  }

  toJSON() {
    return { kind: 'sep-cma-es', dim: this.dim, lambda: this.lambda, seed: this.seed, gen: this.gen, sigma: this.sigma,
      m: Array.from(this.m), C: Array.from(this.C), pc: Array.from(this.pc), ps: Array.from(this.ps) };
  }

  static fromJSON(o: { dim: number; lambda: number; seed: number; gen: number; sigma: number; m: number[]; C: number[]; pc: number[]; ps: number[] }): SepCmaEs {
    const es = new SepCmaEs(o.dim, { lambda: o.lambda, seed: o.seed, sigma0: o.sigma, x0: o.m });
    es.gen = o.gen;
    es.C.set(o.C);
    for (let k = 0; k < o.dim; k++) es.D[k] = Math.sqrt(o.C[k]);
    es.pc.set(o.pc);
    es.ps.set(o.ps);
    return es;
  }
}
