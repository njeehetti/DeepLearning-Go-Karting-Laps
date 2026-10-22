// Recording and analysis of a driven lap (replay, telemetry, corner sheet, network view). DOM-free.
import type { Track } from '../track/types';
import { KartState } from '../sim/kart';
import { Pilot, runEpisode, type EpisodeResult, type EpisodeSpec } from './episode';
import type { World } from './evalCore';
import { N_OBS } from './obs';

/** Samples at 30 Hz (every decision). */
export interface Run {
  hz: number;
  n: number;
  t: Float32Array;
  x: Float32Array;
  y: Float32Array;
  z: Float32Array;
  psi: Float32Array;
  v: Float32Array;
  ax: Float32Array;
  ay: Float32Array;
  thr: Float32Array;
  brk: Float32Array;
  steer: Float32Array;
  s: Float32Array;
  dist: Float32Array;
  d: Float32Array;
  /** Observations fed to the network, N_OBS per sample. */
  obs: Float32Array;
  result: EpisodeResult;
}

export function recordRun(world: World, params: ArrayLike<number>, spec: EpisodeSpec): Run {
  const cap = Math.ceil(spec.maxTime * 30) + 2;
  const f = () => new Float32Array(cap);
  const run = { hz: 30, n: 0, t: f(), x: f(), y: f(), z: f(), psi: f(), v: f(), ax: f(), ay: f(), thr: f(), brk: f(), steer: f(), s: f(), dist: f(), d: f(), obs: new Float32Array(cap * N_OBS) };
  const pilot = new Pilot();
  const result = runEpisode(world.sim, pilot, params, spec, new KartState(), (st, inp, decided) => {
    if (!decided || run.n >= cap) return;
    const k = run.n++;
    run.t[k] = st.t;
    run.x[k] = st.x;
    run.y[k] = st.y;
    run.z[k] = st.z;
    run.psi[k] = st.psi;
    run.v[k] = Math.hypot(st.vx, st.vy);
    run.ax[k] = st.ax;
    run.ay[k] = st.ay;
    run.thr[k] = st.thr;
    run.brk[k] = st.brk;
    run.steer[k] = inp.steer;
    run.s[k] = st.s;
    run.dist[k] = st.dist;
    run.d[k] = st.d;
    run.obs.set(pilot.obs, k * N_OBS);
  });
  return { ...run, result };
}

/** Interpolated pose at time t (wraps over the run). */
export function poseAt(run: Run, t: number) {
  const T = run.t[run.n - 1];
  const tt = ((t % T) + T) % T;
  let lo = 0;
  let hi = run.n - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (run.t[m] <= tt) lo = m;
    else hi = m;
  }
  const u = Math.min(1, Math.max(0, (tt - run.t[lo]) / Math.max(1e-6, run.t[hi] - run.t[lo])));
  const L = (a: Float32Array) => a[lo] + (a[hi] - a[lo]) * u;
  const p0 = run.psi[lo];
  let p1 = run.psi[hi];
  while (p1 - p0 > Math.PI) p1 -= 2 * Math.PI;
  while (p1 - p0 < -Math.PI) p1 += 2 * Math.PI;
  return { k: lo, t: tt, x: L(run.x), y: L(run.y), z: L(run.z), psi: p0 + (p1 - p0) * u, v: L(run.v), ax: L(run.ax), ay: L(run.ay), thr: run.thr[lo], brk: run.brk[lo], steer: run.steer[lo], dist: L(run.dist), d: L(run.d) };
}
export type Pose = ReturnType<typeof poseAt>;

export interface CornerSheetRow {
  name: string;
  dir: 'G' | 'D';
  kind: string;
  angle: number;
  /** Distance from braking point to apex (m), 0 if no braking. */
  brakeDist: number;
  vEntry: number;
  vApex: number;
  vExit: number;
  latG: number;
  /** Distance of the kart centre from the inner edge at the apex (m). */
  apexFromInside: number;
  /** Run times of the apex and of the braking point (s), for "replay here". */
  tApex: number;
  tBrake: number;
  /** Time from 30 m before to 30 m after the apex (s), to compare runs. */
  sectorTime: number;
  /** Unwrapped run distance of the apex and of the braking point (m). */
  distApex: number;
  distBrake: number;
}

/** Index of the last sample with dist ≤ value. */
export function indexAtDist(run: Run, dist: number): number {
  let lo = 0;
  let hi = run.n - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (run.dist[m] <= dist) lo = m;
    else hi = m;
  }
  return lo;
}

/** Corner sheet of a recorded lap (uses the unwrapped distance, so it works from any start). */
export function cornerSheet(track: Track, run: Run): CornerSheetRow[] {
  const L = track.L;
  const at = (dist: number) => indexAtDist(run, dist);
  const sStart = run.s[0];
  return track.corners.map((c) => {
    const sMid = track.s[c.iMid];
    const dMid = (((sMid - sStart) % L) + L) % L;
    const k0 = at(dMid - 25);
    const k1 = at(dMid + 15);
    let kA = k0;
    for (let k = k0; k <= k1; k++) if (run.v[k] < run.v[kA]) kA = k;
    let kB = -1;
    for (let k = kA; k >= Math.max(0, at(dMid - 90)); k--) {
      if (run.brk[k] > 0.08) kB = k;
      else if (kB >= 0) break;
    }
    const kExit = at(run.dist[kA] + 25);
    let latG = 0;
    for (let k = k0; k <= k1; k++) latG = Math.max(latG, Math.abs(run.ay[k]) / 9.81);
    const inside = c.dir === 'G' ? run.d[kA] : -run.d[kA];
    const kS0 = at(run.dist[kA] - 30);
    const kS1 = at(run.dist[kA] + 30);
    return {
      name: c.name, dir: c.dir, kind: c.kind, angle: c.angle,
      brakeDist: kB >= 0 ? run.dist[kA] - run.dist[kB] : 0,
      vEntry: run.v[kB >= 0 ? kB : kA], vApex: run.v[kA], vExit: run.v[kExit], latG,
      apexFromInside: track.half - inside, tApex: run.t[kA], tBrake: kB >= 0 ? run.t[kB] : run.t[kA],
      sectorTime: run.t[kS1] - run.t[kS0], distApex: run.dist[kA], distBrake: kB >= 0 ? run.dist[kB] : -1,
    };
  });
}
