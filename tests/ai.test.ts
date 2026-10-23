import { describe, expect, it } from 'vitest';
import lyl from '../data/tracks/lyl.json';
import type { TrackDef } from '../src/track/types';
import { DEFAULT_CALIBRATION } from '../src/sim/defaults';
import { DRIVER_SHAPE, LEGACY_SHAPE, Mlp, driverShape, paramCount, toDriverShape, widen } from '../src/ai/mlp';
import { N_OBS } from '../src/ai/obs';
import { SepCmaEs, gaussVec, sampleCandidate } from '../src/ai/es';
import { Evaluator, World, type WorldSpec } from '../src/ai/evalCore';
import { SyncPool } from '../src/ai/pool';
import { Trainer } from '../src/ai/trainer';
import { mulberry32 } from '../src/sim/rng';

const c = DEFAULT_CALIBRATION.k270;
const world: WorldSpec = {
  track: lyl as TrackDef, scale: 'official',
  kart: { kartId: 'k270', mu: c.mu, powerScale: c.powerScale, vTopKmh: c.vTopKmh, transmission: c.transmission, driver: c.driver },
};

describe('réseau', () => {
  it('a 22 entrées, 32 neurones cachés et 3 sorties', () => {
    expect(DRIVER_SHAPE.sizes).toEqual([22, 32, 3]);
    expect(N_OBS).toBe(22);
    expect(paramCount(DRIVER_SHAPE)).toBe(32 * 23 + 3 * 33);
    const out = new Mlp().forward(new Float64Array(paramCount(DRIVER_SHAPE)).fill(0.1), new Float64Array(22).fill(0.5));
    expect(out).toHaveLength(3);
    expect(Number.isFinite(out[0])).toBe(true);
  });
  it('s’élargit sans changer ce que fait le pilote', () => {
    const r = mulberry32(3);
    const p = Float64Array.from({ length: paramCount(LEGACY_SHAPE) }, () => r() * 2 - 1);
    const big = driverShape([32]);
    const q = widen(p, LEGACY_SHAPE, big, mulberry32(4));
    expect(toDriverShape(p)).toHaveLength(paramCount(DRIVER_SHAPE));
    expect(toDriverShape([1, 2, 3])).toBeNull();
    expect(q).toHaveLength(paramCount(big));
    const x = Float64Array.from({ length: 22 }, () => r() * 2 - 1);
    const a = Array.from(new Mlp(LEGACY_SHAPE).forward(p, x));
    const b = Array.from(new Mlp(big).forward(q, x));
    b.forEach((v, k) => expect(v).toBeCloseTo(a[k], 12));
  });
});

describe('sep-CMA-ES', () => {
  it('régénère les mêmes candidats à partir de la graine', () => {
    const a = gaussVec(7, 3, 11, new Float64Array(50));
    const b = gaussVec(7, 3, 11, new Float64Array(50));
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it('réinjecte l’élite en candidat 1 et la moyenne en candidat 0', () => {
    const es = new SepCmaEs(5, { lambda: 8, seed: 1, x0: [1, 2, 3, 4, 5] });
    es.elite = Float64Array.of(9, 9, 9, 9, 9);
    expect(Array.from(sampleCandidate(es.spec(), 0, new Float64Array(5)))).toEqual([1, 2, 3, 4, 5]);
    expect(Array.from(sampleCandidate(es.spec(), 1, new Float64Array(5)))).toEqual([9, 9, 9, 9, 9]);
  });

  it('minimise une sphère en dimension 30', () => {
    const n = 30;
    const es = new SepCmaEs(n, { lambda: 24, seed: 2, sigma0: 1, x0: new Float64Array(n).fill(3) });
    const x = new Float64Array(n);
    for (let g = 0; g < 300; g++) {
      const fit = Array.from({ length: es.lambda }, (_, i) => -es.candidate(i, x).reduce((s, v) => s + v * v, 0));
      es.tell(fit);
    }
    expect(es.mean().reduce((s, v) => s + v * v, 0)).toBeLessThan(1e-3);
  });
});

describe('épisode', () => {
  const ev = new Evaluator(new World(world));
  const spec = { s0: 0, d0: 0, v0: 10, maxTime: 20, timeWeight: 0 };

  it('est déterministe', () => {
    const r = mulberry32(5);
    const p = Float64Array.from({ length: paramCount(DRIVER_SHAPE) }, () => (r() - 0.5) * 0.6);
    expect(ev.episode(p, spec)).toEqual(ev.episode(p, spec));
  });

  it('pénalise un volant agité', () => {
    const n = paramCount(DRIVER_SHAPE);
    // steering output driven by the "Braquage" input with a large negative gain: a self-sustained saw
    const saw = new Float64Array(n);
    const calm = new Float64Array(n);
    const hid = DRIVER_SHAPE.sizes[1] * (N_OBS + 1);
    saw[14] = -8; // hidden 0 ← input 14 (wheel angle)
    saw[N_OBS] = 0.3; // hidden 0 bias
    saw[hid + 0] = 6; // steer ← hidden 0
    const a = ev.episode(saw, spec);
    const b = ev.episode(calm, spec);
    expect(a.steerActivity).toBeGreaterThan(b.steerActivity);
  });
});

describe('entraînement', () => {
  it('ne fait jamais régresser le meilleur tour d’une génération à l’autre', async () => {
    const pool = new SyncPool();
    await pool.init(world);
    const w = new World(world);
    const tr = new Trainer(pool, { world, lambda: 24, seed: 3, maxTime: 60, L: w.q.L, startSpeed: (s) => w.startSpeed(s) });
    let prev = Infinity;
    for (let g = 0; g < 12; g++) {
      const s = await tr.step();
      if (Number.isFinite(s.bestLap)) {
        expect(s.bestLap).toBeLessThanOrEqual(prev + 1e-9);
        prev = s.bestLap;
      } else expect(prev).toBe(Infinity);
    }
  }, 120_000);
});
