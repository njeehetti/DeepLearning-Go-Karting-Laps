// Physical invariants of the dynamic model. Absolute grip levels are calibrated on the driver's lap
// time, so these tests check behaviour and orders of magnitude, not tuned constants.
import { describe, expect, it } from 'vitest';
import lyl from '../data/tracks/lyl.json';
import { buildTrack } from '../src/track/build';
import { TrackQuery } from '../src/track/query';
import type { TrackDef } from '../src/track/types';
import { DT, KartSim, KartState, chassisParams, type KartInput } from '../src/sim/kart';
import { driveForce, ENGINES, makePowertrain } from '../src/sim/engine';
import { PursuitDriver, constantLine } from '../src/ai/pursuit';
import { mulberry32 } from '../src/sim/rng';
import { circle, stadium } from './helpers/synth';

const G = 9.81;
const qStadium = new TrackQuery(buildTrack(stadium()));
const qWide = new TrackQuery(buildTrack(stadium(900, 80, 40)));
const qLyl = new TrackQuery(buildTrack(lyl as TrackDef, { scale: 'official' }));

function run(sim: KartSim, st: KartState, seconds: number, ctrl: (st: KartState) => KartInput) {
  for (let k = 0; k < Math.round(seconds / DT); k++) sim.step(st, ctrl(st));
}

describe('moteur', () => {
  it('donne un couple de GX plausible et coupe au régulateur', () => {
    const pt = makePowertrain(ENGINES.k270, 62 / 3.6);
    // flat-ish torque: force at 40 km/h within 15 % of force at 60 km/h
    const f40 = driveForce(pt, 40 / 3.6, 1);
    const f60 = driveForce(pt, 60 / 3.6, 1);
    expect(f40 / f60).toBeGreaterThan(1);
    expect(f40 / f60).toBeLessThan(1.15);
    expect(driveForce(pt, 64.5 / 3.6, 1)).toBe(0);
    expect(driveForce(pt, 30 / 3.6, 0)).toBe(0);
    // 390cc pulls harder than 270cc at the same road speed
    expect(driveForce(makePowertrain(ENGINES.k390, 76 / 3.6), 40 / 3.6, 1)).toBeGreaterThan(f40);
  });
});

describe('ligne droite', () => {
  it.each([
    ['k270', 62],
    ['k390', 76],
  ] as const)('%s atteint sa vitesse de pointe (%i km/h) sans la dépasser', (id, vTop) => {
    const sim = new KartSim(qStadium, chassisParams(id));
    const st = sim.reset(new KartState(), 1, 0, 0);
    const drv = new PursuitDriver(qStadium, constantLine(qStadium.N, 0, 40));
    let vmax = 0;
    run(sim, st, 30, (s) => {
      const i = drv.drive(sim, s);
      i.throttle = 1;
      i.brake = 0;
      if (s.s > 10 && s.s < 880) vmax = Math.max(vmax, s.vx);
      return i;
    });
    expect(vmax * 3.6).toBeGreaterThan(vTop - 0.5);
    expect(vmax * 3.6).toBeLessThan(vTop + 1.5);
  });

  it('démarre arrêté sans NaN et accélère', () => {
    const sim = new KartSim(qStadium, chassisParams('k270'));
    const st = sim.reset(new KartState(), 1, 0, 0);
    run(sim, st, 5, () => ({ steer: 0, throttle: 1, brake: 0 }));
    expect(Number.isFinite(st.x + st.y + st.vx + st.vy + st.r)).toBe(true);
    expect(st.vx).toBeGreaterThan(5);
  });

  it('lâche les gaz dès qu’on freine : gaz et frein ensemble = frein seul', () => {
    const sim = new KartSim(qStadium, chassisParams('k270'));
    const brakeWith = (throttle: number) => {
      const st = sim.reset(new KartState(), 100, 0, 50 / 3.6);
      run(sim, st, 1, () => ({ steer: 0, throttle, brake: 0.3 }));
      return st;
    };
    const both = brakeWith(1);
    expect(both.thr).toBe(0);
    expect(both.vx).toBe(brakeWith(0).vx);
  });

  it('freine à environ 0,5 g au seuil, roues arrière seules, et moins une fois bloqué', () => {
    const sim = new KartSim(qStadium, chassisParams('k270'));
    const decel = (brake: number) => {
      const st = sim.reset(new KartState(), 100, 0, 60 / 3.6);
      const drv = new PursuitDriver(qStadium, constantLine(qStadium.N, 0, 0));
      let t = 0;
      let locked = false;
      while (st.vx > 20 / 3.6 && t < 10) {
        const i = drv.drive(sim, st);
        i.throttle = 0;
        i.brake = brake;
        sim.step(st, i);
        t += DT;
        locked ||= st.locked;
      }
      return { g: (40 / 3.6 / t) / G, locked };
    };
    let best = { g: 0, locked: false };
    for (let b = 0.2; b <= 1; b += 0.05) {
      const r = decel(b);
      if (!r.locked && r.g > best.g) best = r;
    }
    expect(best.g).toBeGreaterThan(0.45);
    expect(best.g).toBeLessThan(0.6);
    const full = decel(1);
    expect(full.locked).toBe(true);
    expect(full.g).toBeLessThan(best.g);
  });
});

describe('virage', () => {
  function skidpad(mu: number, R: number): number {
    const q = new TrackQuery(buildTrack(circle(R, 12)));
    const sim = new KartSim(q, chassisParams('k270', { mu }));
    const line = { d: new Float64Array(q.N), v: new Float64Array(q.N).fill(5) };
    const st = sim.reset(new KartState(), 1, 0, 5);
    const drv = new PursuitDriver(q, line, { l0: 3, kv: 0.3, kSpeed: 1 });
    let best = 0;
    for (let k = 0; k < 120 * 120; k++) {
      line.v.fill(5 + st.t * 0.12);
      sim.step(st, drv.drive(sim, st));
      if (Math.abs(st.d) > 1.5 || st.wheelsOff) break;
      if (Math.abs(st.d) < 0.6) best = Math.max(best, (st.vx * st.vx) / R);
    }
    return best / G;
  }

  it('tient en régime permanent entre 60 % et 100 % de µg, et plus avec plus d’adhérence', () => {
    const lo = skidpad(1.15, 12);
    const hi = skidpad(1.45, 12);
    expect(lo / 1.15).toBeGreaterThan(0.6);
    expect(lo / 1.15).toBeLessThan(1.0);
    expect(hi).toBeGreaterThan(lo * 1.15);
  });

  it('fait sous-virer l’essieu rigide tant que la roue intérieure porte', () => {
    const sim = new KartSim(new TrackQuery(buildTrack(circle(25, 12))), chassisParams('k270'));
    const st = sim.reset(new KartState(), 1, 0, 9);
    run(sim, st, 5, () => ({ steer: 0.05 / 0.4, throttle: 0.3, brake: 0 }));
    // geometric (Ackermann) radius at 0.05 rad would be L/δ = 21 m; the rigid axle pushes it wider
    expect(st.vx / st.r).toBeGreaterThan((sim.p.a + sim.p.b) / 0.05);
    expect(st.lift).toBeGreaterThan(0);
    expect(st.lift).toBeLessThan(1);
  });

  it('permet le tête-à-queue : frein bloqué en braquant à 60 km/h', () => {
    const sim = new KartSim(qWide, chassisParams('k270'));
    const st = sim.reset(new KartState(), 100, 0, 60 / 3.6);
    let maxBeta = 0;
    run(sim, st, 3, (s) => {
      if (Math.hypot(s.vx, s.vy) > 1) maxBeta = Math.max(maxBeta, Math.abs(Math.atan2(s.vy, s.vx)));
      return { steer: 0.6, throttle: 0, brake: 1 };
    });
    expect(maxBeta).toBeGreaterThan(Math.PI / 2);
  });
});

describe('circuit', () => {
  it('est déterministe au bit près', () => {
    const go = () => {
      const sim = new KartSim(qLyl, chassisParams('k390'));
      const st = sim.reset(new KartState(), 20, 0.5, 8);
      const rnd = mulberry32(42);
      const drv = new PursuitDriver(qLyl, constantLine(qLyl.N, 0, 11));
      run(sim, st, 20, (s) => {
        const i = drv.drive(sim, s);
        return { steer: i.steer + (rnd() - 0.5) * 0.2, throttle: i.throttle, brake: i.brake };
      });
      return [st.x, st.y, st.psi, st.vx, st.vy, st.r, st.dist];
    };
    expect(go()).toEqual(go());
  });

  it('reste fini sous commandes aléatoires et détecte le mur de pneus', () => {
    const sim = new KartSim(qLyl, chassisParams('k390'));
    const st = sim.reset(new KartState(), 20, 0, 10);
    const rnd = mulberry32(7);
    run(sim, st, 30, () => ({ steer: rnd() * 2 - 1, throttle: rnd(), brake: rnd() < 0.2 ? 1 : 0 }));
    expect(Number.isFinite(st.x + st.y + st.vx + st.vy + st.r + st.alphaF + st.alphaR)).toBe(true);
    const st2 = sim.reset(new KartState(), 20, 0, 15);
    run(sim, st2, 6, () => ({ steer: 1, throttle: 1, brake: 0 }));
    expect(st2.crashed || st2.wheelsOff > 0).toBe(true);
  });

  it('ralentit dans la montée du pont', () => {
    const cr = qLyl.track.crossings[0];
    const coast = (s0: number) => {
      const sim = new KartSim(qLyl, chassisParams('k270'));
      const st = sim.reset(new KartState(), s0, 0, 12);
      const drv = new PursuitDriver(qLyl, constantLine(qLyl.N, 0, 12));
      run(sim, st, 2, (s) => {
        const i = drv.drive(sim, s);
        i.throttle = 0;
        i.brake = 0;
        return i;
      });
      return st.vx;
    };
    // the ramp starts ~48 m before the top of the bridge; compare with a flat stretch just before it
    expect(coast(cr.sHigh - 45)).toBeLessThan(coast(cr.sHigh - 110) - 0.3);
  });
});
