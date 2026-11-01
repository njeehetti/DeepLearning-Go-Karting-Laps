import { describe, expect, it } from 'vitest';
import { Adam, Backprop } from '../src/ai/backprop';
import { DRIVER_SHAPE, Mlp, controlsFromOutput, paramCount } from '../src/ai/mlp';
import { mulberry32 } from '../src/sim/rng';
import { World } from '../src/ai/evalCore';
import { Imitation, bestLap, type Attempt } from '../src/ai/imitation';
import { DEFAULT_CALIBRATION } from '../src/sim/defaults';
import type { TrackDef } from '../src/track/types';
import genk from '../data/tracks/genk.json';
import generalist from '../data/brains/generaliste-k270.json';

const shape = DRIVER_SHAPE;
const rnd = (seed: number, n: number, a = 1) => {
  const r = mulberry32(seed);
  return Float64Array.from({ length: n }, () => (r() * 2 - 1) * a);
};

describe('rétropropagation', () => {
  it('donne les mêmes commandes que le pilote', () => {
    const p = rnd(1, paramCount(shape), 0.4);
    const x = rnd(2, shape.sizes[0]);
    const bp = new Backprop(shape);
    const ctl = bp.forward(p, x);
    const ref = { steer: 0, throttle: 0, brake: 0 };
    controlsFromOutput(new Mlp(shape).forward(p, x), ref);
    expect(ctl[0]).toBeCloseTo(ref.steer, 12);
    expect(ctl[1]).toBeCloseTo(ref.throttle, 12);
    expect(ctl[2]).toBeCloseTo(ref.brake, 12);
  });

  it('calcule le gradient exact (comparé à la dérivée numérique)', () => {
    const p = rnd(3, paramCount(shape), 0.4);
    // brake output bias pushed up so the brake is outside its dead zone (where it has a true derivative)
    p[p.length - 1] = 3;
    const x = rnd(4, shape.sizes[0]);
    const target = Float64Array.of(0.3, 0.8, 0.6);
    const bp = new Backprop(shape);
    const grad = new Float64Array(p.length);
    bp.forward(p, x);
    bp.backward(p, target, grad);
    const loss = (q: Float64Array) => {
      bp.forward(q, x);
      return bp.backward(q, target, new Float64Array(q.length));
    };
    const r = mulberry32(5);
    for (let n = 0; n < 40; n++) {
      const k = Math.floor(r() * p.length);
      const h = 1e-6;
      const a = Float64Array.from(p);
      const b = Float64Array.from(p);
      a[k] += h;
      b[k] -= h;
      expect(grad[k]).toBeCloseTo((loss(a) - loss(b)) / (2 * h), 6);
    }
  });

  it('fait baisser l’erreur avec Adam', () => {
    const bp = new Backprop(shape);
    const p = rnd(6, paramCount(shape), 0.2);
    const xs = Array.from({ length: 64 }, (_, i) => rnd(100 + i, shape.sizes[0]));
    // a simple driving rule to learn: steer towards the first input, throttle when the second is positive
    const ys = xs.map((x) => Float64Array.of(Math.tanh(x[0]), x[1] > 0 ? 0.9 : 0.2, 0));
    const adam = new Adam(p.length, 1e-2);
    const epoch = () => {
      const g = new Float64Array(p.length);
      let l = 0;
      xs.forEach((x, i) => {
        bp.forward(p, x);
        l += bp.backward(p, ys[i], g, 1 / xs.length);
      });
      adam.apply(p, g);
      return l / xs.length;
    };
    const first = epoch();
    let last = first;
    for (let e = 0; e < 300; e++) last = epoch();
    expect(last).toBeLessThan(first * 0.2);
  });
});

describe('imitation du champion', () => {
  it('apprend à boucler un tour propre en quelques leçons', () => {
    const c = DEFAULT_CALIBRATION.k270;
    const w = new World({ track: genk as TrackDef, scale: 'official', kart: { kartId: 'k270', mu: c.mu, powerScale: c.powerScale, vTopKmh: c.vTopKmh, transmission: c.transmission, driver: c.driver } });
    const im = new Imitation(w, generalist.params, 1);
    const first = im.step().attempt.result;
    expect(first.completed).toBe(false);
    // at µ 1.15 the teacher drives close to the limit: the student needs about 15 lessons, not always clean yet
    let clean = false;
    for (let e = 0; e < 24 && !clean; e++) {
      im.step();
      const r = im.attempt(false).result;
      clean = r.completed && r.clean;
    }
    expect(clean).toBe(true);
  }, 120_000);
});

describe('choix du professeur', () => {
  const lap = (completed: boolean, clean: boolean, lapTime: number, progress: number) =>
    ({ result: { completed, clean, lapTime, progress }, path: new Float32Array(), steerGap: 0 }) as unknown as Attempt;
  it('prend le tour propre le plus rapide, puis le plus long parcours', () => {
    expect(bestLap([lap(true, true, 141.7, 1200), lap(true, true, 95.7, 1200)])).toBe(1);
    expect(bestLap([lap(true, false, 90, 1200), lap(true, true, 99, 1200)])).toBe(1);
    expect(bestLap([lap(false, false, 0, 800), lap(true, false, 120, 1200)])).toBe(1);
    expect(bestLap([lap(false, false, 0, 800), lap(false, false, 0, 510)])).toBe(0);
  });
});
