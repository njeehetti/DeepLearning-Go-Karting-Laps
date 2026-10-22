// "Tour théorique v2": quasi-steady lap solver that shares its limits with the dynamic model, so the AI
// is measured against what this kart can actually do:
//  - lateral: the steady-state envelope ayMax(v) measured by simulated skidpads on the dynamic model
//    (includes rigid-axle and power understeer);
//  - acceleration: the same engine torque curve and governor;
//  - braking: rear brakes only, limited by longitudinal load transfer.
import { G } from '../track/geometry';
import { buildTrack } from '../track/build';
import { TrackQuery } from '../track/query';
import type { TrackDef } from '../track/types';
import { driveForce } from './engine';
import { KartSim, KartState, type ChassisParams } from './kart';
import type { LineEvaluator } from './qss';

/** Lateral acceleration capacity sampled every `dv` m/s from 0 to vMax. */
export interface Envelope {
  dv: number;
  ay: Float64Array;
}

function circleDef(R: number): TrackDef {
  const M = (6371000 * Math.PI) / 180;
  const n = Math.round(2 * Math.PI * R);
  const coords: [number, number][] = [];
  for (let k = 0; k < n; k++) {
    const a = (k / n) * 2 * Math.PI;
    coords.push([(R * Math.sin(a)) / M, (R * Math.cos(a)) / M]);
  }
  return { id: 'skidpad', name: 'skidpad', width: 12, coords, startFrac: 0 };
}

/** Highest steady speed held on a circle of radius R (ramped, scripted driver on the centreline). */
function skidpad(p: ChassisParams, R: number): number {
  const q = new TrackQuery(buildTrack(circleDef(R)));
  const sim = new KartSim(q, p);
  const st = sim.reset(new KartState(), 1, 0, 4);
  const L = p.a + p.b;
  let best = 0;
  for (let k = 0; k < 120 * 150; k++) {
    const vT = 4 + st.t * 0.12;
    // pure pursuit on the circle + speed hold
    const ld = 3 + 0.3 * st.vx;
    const sA = st.s + ld;
    const tx = q.at(q.track.x, sA) - st.x;
    const ty = q.at(q.track.y, sA) - st.y;
    const alpha = Math.atan2(ty, tx) - st.psi;
    const steer = Math.atan2(2 * L * Math.sin(alpha), Math.hypot(tx, ty)) / p.deltaMax;
    const e = vT - st.vx;
    sim.step(st, { steer, throttle: e > 0 ? Math.min(1, e + 0.2) : 0, brake: 0 });
    if (Math.abs(st.d) > 1.5 || st.wheelsOff || st.crashed) break;
    if (Math.abs(st.d) < 0.6) best = Math.max(best, st.vx);
    if (st.vx < vT - 3) break; // power-limited: cannot go faster on this radius
  }
  return best;
}

/** Steady-state lateral envelope of the dynamic model (≈ 0.3 s to compute). */
export function measureEnvelope(p: ChassisParams): Envelope {
  const radii = [6, 8, 10, 13, 17, 22, 30, 40, 60, 90];
  const pts: [number, number][] = radii.map((R) => {
    const v = skidpad(p, R);
    return [v, (v * v) / R];
  });
  pts.sort((a, b) => a[0] - b[0]);
  const dv = 0.5;
  const n = Math.ceil(p.pt.vTop / dv) + 4;
  const ay = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const v = k * dv;
    if (v <= pts[0][0]) ay[k] = pts[0][1];
    else if (v >= pts[pts.length - 1][0]) ay[k] = pts[pts.length - 1][1];
    else {
      let j = 0;
      while (pts[j + 1][0] < v) j++;
      const u = (v - pts[j][0]) / (pts[j + 1][0] - pts[j][0] || 1);
      ay[k] = pts[j][1] + (pts[j + 1][1] - pts[j][1]) * u;
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

