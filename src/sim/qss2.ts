// "Tour théorique v2": quasi-steady lap solver that shares its limits with the dynamic model, so the AI
// is measured against what this kart can actually do:
//  - lateral: the steady-state envelope ayMax(v) of the dynamic model, found open-loop (fixed steering)
//    so it includes rigid-axle effects and does not depend on any driver model;
//  - acceleration: the same engine torque curve and governor;
//  - braking: rear brakes only, limited by longitudinal load transfer.
import { G } from '../track/geometry';
import { buildTrack } from '../track/build';
import { TrackQuery } from '../track/query';
import type { TrackDef } from '../track/types';
import { driveForce, makePowertrain, scaleEngine } from './engine';
import { KartSim, KartState, type ChassisParams } from './kart';
import type { LineEvaluator } from './qss';

/** Lateral acceleration capacity sampled every `dv` m/s from 0 to vMax. */
export interface Envelope {
  dv: number;
  ay: Float64Array;
}

function circleDef(R: number, width: number): TrackDef {
  const M = (6371000 * Math.PI) / 180;
  const n = Math.round(2 * Math.PI * R);
  const coords: [number, number][] = [];
  for (let k = 0; k < n; k++) {
    const a = (k / n) * 2 * Math.PI;
    coords.push([(R * Math.sin(a)) / M, (R * Math.cos(a)) / M]);
  }
  return { id: 'skidpad', name: 'skidpad', width, coords, startFrac: 0 };
}

/** A paddock so wide the kart never leaves the asphalt during the open-loop tests. */
let paddock: TrackQuery | null = null;

/**
 * Largest stable steady-state lateral acceleration at speed v: fixed steering angle, speed held by a
 * PI on the throttle, no path controller. The result is deterministic and independent of any driver
 * model. Beyond the limit the kart either spins (oversteer) or saturates its front (understeer).
 */
function steadyAy(p: ChassisParams, v: number): number {
  paddock ??= new TrackQuery(buildTrack(circleDef(200, 390)));
  const sim = new KartSim(paddock, p);
  const st = new KartState();
  const L = p.a + p.b;
  const tryDelta = (delta: number): number => {
    sim.reset(st, 1, 0, v);
    let I = 0;
    let sum = 0;
    let sum2 = 0;
    let n = 0;
    const steer = delta / p.deltaMax;
    for (let k = 0; k < 120 * 3.5; k++) {
      const e = v - st.vx;
      I += e / 120;
      const u = e + 0.5 * I;
      sim.step(st, { steer, throttle: Math.max(0, Math.min(1, u)), brake: u < -0.5 ? Math.min(1, -u * 0.2) : 0 });
      if (k >= 120 * 2.5) {
        const ay = st.vx * st.r;
        sum += ay;
        sum2 += ay * ay;
        n++;
      }
    }
    const mean = sum / n;
    const sd = Math.sqrt(Math.max(0, sum2 / n - mean * mean));
    const stable = Math.abs(st.vx - v) < 0.4 && sd < 0.05 * Math.abs(mean) + 0.05 && Math.abs(Math.atan2(st.vy, st.vx)) < 0.35;
    return stable ? mean : -1;
  };
  // coarse log sweep, then refine around the best
  const lo = Math.max(0.002, (L * 0.2 * G) / (v * v));
  let best = 0;
  let bestD = lo;
  const nC = 40;
  for (let k = 0; k <= nC; k++) {
    const d = lo * Math.pow(p.deltaMax / lo, k / nC);
    const ay = tryDelta(d);
    if (ay > best) {
      best = ay;
      bestD = d;
    }
  }
  const f = Math.pow(p.deltaMax / lo, 1 / nC);
  for (let k = 1; k <= 8; k++) {
    const d = bestD * Math.pow(f, (k - 4.5) / 4);
    const ay = tryDelta(d);
    if (ay > best) best = ay;
  }
  return best;
}

/**
 * Steady-state lateral envelope ayMax(v) of the dynamic model (≈ 0.1 s). Run with an oversized,
 * ungoverned engine so high speeds are reachable: holding a steady speed only needs the force that
 * beats the resistances, so cornering behaviour is unchanged; real power is handled by the solver.
 */
export function measureEnvelope(p: ChassisParams): Envelope {
  const strong: ChassisParams = { ...p, pt: makePowertrain(scaleEngine(p.pt.spec, 4), 45, p.pt.rWheel, 'cvt') };
  const dv = 0.5;
  const n = Math.ceil(p.pt.vTop / dv) + 4;
  const speeds = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 24];
  const pts = speeds.map((v) => steadyAy(strong, v));
  // fill dips: a driver can trade throttle for stability, so a speed band that is unstable at the
  // regulated throttle is not a hard limit; never exceed what both neighbours prove achievable
  const raw = pts.slice();
  for (let i = 1; i < raw.length - 1; i++) {
    const left = Math.max(...raw.slice(0, i));
    const right = Math.max(...raw.slice(i + 1));
    pts[i] = Math.max(raw[i], Math.min(left, right));
  }
  const ay = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const v = k * dv;
    if (v <= speeds[0]) ay[k] = pts[0];
    else if (v >= speeds[speeds.length - 1]) ay[k] = pts[pts.length - 1];
    else {
      let j = 0;
      while (speeds[j + 1] < v) j++;
      const u = (v - speeds[j]) / (speeds[j + 1] - speeds[j]);
      ay[k] = pts[j] + (pts[j + 1] - pts[j]) * u;
    }
  }
  return { dv, ay };
}

export function envelopeAt(env: Envelope, v: number): number {
  const f = Math.max(0, v) / env.dv;
  const k = Math.min(env.ay.length - 2, Math.floor(f));
  const u = Math.min(1, f - k);
  return env.ay[k] + (env.ay[k + 1] - env.ay[k]) * u;
}

export interface Qss2 {
  p: ChassisParams;
  env: Envelope;
  /** Rear-only braking deceleration limit on a straight (m/s²). */
  aBrake: number;
}

export function makeQss2(p: ChassisParams, env = measureEnvelope(p)): Qss2 {
  const L = p.a + p.b;
  const rear = p.a / L;
  return { p, env, aBrake: (p.mu * G * rear) / (1 + (p.mu * p.h) / L) };
}

/** Lap time of the line currently held by `ev` (call ev.lineFromGenes first); fills ev.v. */
export function solveQss2(ev: LineEvaluator, m: Qss2): number {
  const { N, k, ds, v } = ev;
  const t = ev.track;
  const p = m.p;
  const vTop = p.pt.vTop;
  let i0 = 0;
  let vmin = 1e9;
  for (let i = 0; i < N; i++) {
    const ak = Math.abs(k[i]);
    let vl = vTop;
    if (ak > 1e-6) {
      // fixed point: v = sqrt(ayMax(v) / k)
      vl = Math.sqrt(envelopeAt(m.env, vTop) / ak);
      for (let it = 0; it < 6; it++) vl = Math.sqrt(envelopeAt(m.env, Math.min(vl, vTop)) / ak);
    }
    if (vl > vTop) vl = vTop;
    v[i] = vl;
    if (vl < vmin) {
      vmin = vl;
      i0 = i;
    }
  }
  const cd = (0.5 * p.rho * p.CdA) / p.m;
  const rr = p.Crr * G;
  for (let q = 0; q < N; q++) {
    const a0 = (i0 + q) % N;
    const a1 = (a0 + 1) % N;
    const vv = v[a0];
    let r = (vv * vv * Math.abs(k[a0])) / envelopeAt(m.env, vv);
    if (r > 1) r = 1;
    const acc = (driveForce(p.pt, vv, 1) / p.m) * Math.sqrt(1 - r * r) - cd * vv * vv - rr - G * t.grade[a0];
    const vn2 = vv * vv + 2 * acc * ds[a0];
    const vn = vn2 > 0 ? Math.sqrt(vn2) : 0.5;
    if (vn < v[a1]) v[a1] = vn;
  }
  for (let q = 0; q < N; q++) {
    const b1 = (i0 - q + N) % N;
    const b0 = (b1 - 1 + N) % N;
    const vb = v[b1];
    let r = (vb * vb * Math.abs(k[b0])) / envelopeAt(m.env, vb);
    if (r > 1) r = 1;
    let dec = m.aBrake * Math.sqrt(1 - r * r) + cd * vb * vb + rr + G * t.grade[b0];
    if (dec < 0.2) dec = 0.2;
    const vp = Math.sqrt(vb * vb + 2 * dec * ds[b0]);
    if (vp < v[b0]) v[b0] = vp;
  }
  let T = 0;
  for (let i = 0; i < N; i++) T += (2 * ds[i]) / (v[i] + v[(i + 1) % N]);
  return T;
}

