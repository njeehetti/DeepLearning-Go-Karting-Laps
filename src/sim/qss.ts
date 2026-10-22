// Quasi-steady-state lap solver over a racing line ("tour théorique"). Verbatim port of v1 KartCore.Evaluator.
import { G } from '../track/geometry';
import type { Track } from '../track/types';
import type { QssParams } from './karts';

export interface Telemetry {
  T: number;
  N: number;
  x: Float64Array;
  y: Float64Array;
  z: Float64Array;
  /** Lateral offset from the centreline (m, + = left). */
  d: Float64Array;
  v: Float64Array;
  k: Float64Array;
  ds: Float64Array;
  s: Float64Array;
  time: Float64Array;
  ax: Float64Array;
  ay: Float64Array;
  thr: Float64Array;
  brk: Float64Array;
  heading: Float64Array;
}

/** Racing line = lateral offsets at M control points (every geneSpacing m), periodic Catmull-Rom. */
export class LineEvaluator {
  readonly N: number;
  readonly M: number;
  readonly track: Track;
  private readonly j0: Int32Array;
  private readonly w: Float64Array;
  readonly d: Float64Array;
  readonly px: Float64Array;
  readonly py: Float64Array;
  readonly k: Float64Array;
  readonly ds: Float64Array;
  readonly vlim: Float64Array;
  readonly v: Float64Array;
  /** Optional per-point limit on |d| (v2: tighter on the bridge); v1 uses limit() everywhere. */
  readonly lims: Float64Array | null;

  constructor(track: Track, geneSpacing = 16, lims: Float64Array | null = null) {
    this.lims = lims;
    const N = track.N;
    const step = Math.max(2, Math.round(geneSpacing / 2));
    const M = Math.floor(N / step);
    this.N = N;
    this.M = M;
    this.track = track;
    this.j0 = new Int32Array(N);
    this.w = new Float64Array(N * 4);
    for (let i = 0; i < N; i++) {
      const u = (i * M) / N;
      const j = Math.floor(u);
      const t = u - j;
      const t2 = t * t;
      const t3 = t2 * t;
      this.j0[i] = j;
      this.w[i * 4] = -0.5 * t3 + t2 - 0.5 * t;
      this.w[i * 4 + 1] = 1.5 * t3 - 2.5 * t2 + 1;
      this.w[i * 4 + 2] = -1.5 * t3 + 2 * t2 + 0.5 * t;
      this.w[i * 4 + 3] = 0.5 * t3 - 0.5 * t2;
    }
    this.d = new Float64Array(N);
    this.px = new Float64Array(N);
    this.py = new Float64Array(N);
    this.k = new Float64Array(N);
    this.ds = new Float64Array(N);
    this.vlim = new Float64Array(N);
    this.v = new Float64Array(N);
  }

  limit(): number {
    return this.track.half - 0.65;
  }

  lineFromGenes(g: ArrayLike<number>): void {
    const { N, M, track: t, w } = this;
    const lim0 = this.limit();
    const lims = this.lims;
    for (let i = 0; i < N; i++) {
      const lim = lims ? lims[i] : lim0;
      const j = this.j0[i];
      const b = i * 4;
      let dd = g[(j - 1 + M) % M] * w[b] + g[j % M] * w[b + 1] + g[(j + 1) % M] * w[b + 2] + g[(j + 2) % M] * w[b + 3];
      if (dd > lim) dd = lim;
      else if (dd < -lim) dd = -lim;
      this.d[i] = dd;
      this.px[i] = t.x[i] + t.nx[i] * dd;
      this.py[i] = t.y[i] + t.ny[i] * dd;
    }
    for (let i = 0; i < N; i++) {
      const i2 = (i + 1) % N;
      this.ds[i] = Math.hypot(this.px[i2] - this.px[i], this.py[i2] - this.py[i], t.z[i2] - t.z[i]);
      const a = (i - 2 + N) % N;
      const c = (i + 2) % N;
      const abx = this.px[i] - this.px[a];
      const aby = this.py[i] - this.py[a];
      const bcx = this.px[c] - this.px[i];
      const bcy = this.py[c] - this.py[i];
      const acx = this.px[c] - this.px[a];
      const acy = this.py[c] - this.py[a];
      const den = Math.sqrt((abx * abx + aby * aby) * (bcx * bcx + bcy * bcy) * (acx * acx + acy * acy));
      this.k[i] = den < 1e-12 ? 0 : (2 * (abx * bcy - aby * bcx)) / den;
    }
  }

  /** Lap time of the current line; fills this.v. */
  solve(kp: QssParams): number {
    const { N, track: t, k, ds, v, vlim } = this;
    const mu = kp.muLat * G;
    let i0 = 0;
    let vmin = 1e9;
    for (let i = 0; i < N; i++) {
      const ak = Math.abs(k[i]);
      let vl = ak > 1e-6 ? Math.sqrt(mu / ak) : 1e9;
      if (vl > kp.vTop) vl = kp.vTop;
      vlim[i] = vl;
      v[i] = vl;
      if (vl < vmin) {
        vmin = vl;
        i0 = i;
      }
    }
    const cd = (0.5 * kp.rho * kp.CdA) / kp.m;
    const rr = kp.Crr * G;
    // forward (acceleration)
    for (let q = 0; q < N; q++) {
      const a0 = (i0 + q) % N;
      const a1 = (a0 + 1) % N;
      const vv = v[a0];
      const ay = vv * vv * Math.abs(k[a0]);
      let r = ay / mu;
      if (r > 1) r = 1;
      const sc = Math.sqrt(1 - r * r);
      let drive = kp.P / (kp.m * Math.max(vv, 1.5));
      if (drive > kp.aTrac) drive = kp.aTrac;
      const acc = drive * sc - cd * vv * vv - rr - G * t.grade[a0];
      const vn2 = vv * vv + 2 * acc * ds[a0];
      const vn = vn2 > 0 ? Math.sqrt(vn2) : 0.5;
      if (vn < v[a1]) v[a1] = vn;
    }
    // backward (braking)
    for (let q = 0; q < N; q++) {
      const b1 = (i0 - q + N) % N;
      const b0 = (b1 - 1 + N) % N;
      const vb = v[b1];
      const ay2 = vb * vb * Math.abs(k[b0]);
      let r2 = ay2 / mu;
      if (r2 > 1) r2 = 1;
      let dec = kp.aBrk * Math.sqrt(1 - r2 * r2) + cd * vb * vb + rr + G * t.grade[b0];
      if (dec < 0.2) dec = 0.2;
      const vp = Math.sqrt(vb * vb + 2 * dec * ds[b0]);
      if (vp < v[b0]) v[b0] = vp;
    }
    let T = 0;
    for (let i = 0; i < N; i++) T += (2 * ds[i]) / (v[i] + v[(i + 1) % N]);
    return T;
  }

  lapTime(g: ArrayLike<number>, kp: QssParams): number {
    this.lineFromGenes(g);
    return this.solve(kp);
  }

  telemetry(g: ArrayLike<number>, kp: QssParams): Telemetry {
    const T = this.lapTime(g, kp);
    const { N, track: t } = this;
    const out: Telemetry = {
      T, N, x: Float64Array.from(this.px), y: Float64Array.from(this.py), z: Float64Array.from(t.z),
      d: Float64Array.from(this.d), v: Float64Array.from(this.v), k: Float64Array.from(this.k), ds: Float64Array.from(this.ds),
      s: new Float64Array(N + 1), time: new Float64Array(N + 1), ax: new Float64Array(N), ay: new Float64Array(N),
      thr: new Float64Array(N), brk: new Float64Array(N), heading: new Float64Array(N),
    };
    const cd = (0.5 * kp.rho * kp.CdA) / kp.m;
    const rr = kp.Crr * G;
    for (let i = 0; i < N; i++) {
      const i2 = (i + 1) % N;
      const v0 = out.v[i];
      const v1 = out.v[i2];
      const ds = out.ds[i];
      out.s[i + 1] = out.s[i] + ds;
      out.time[i + 1] = out.time[i] + (2 * ds) / (v0 + v1);
      const ax = (v1 * v1 - v0 * v0) / (2 * ds);
      out.ax[i] = ax;
      out.ay[i] = v0 * v0 * out.k[i];
      const need = ax + cd * v0 * v0 + rr + G * t.grade[i];
      const drive = Math.min(kp.P / (kp.m * Math.max(v0, 1.5)), kp.aTrac);
      if (need > 0.05) out.thr[i] = Math.min(1, need / drive);
      else if (need < -0.4) out.brk[i] = Math.min(1, -need / kp.aBrk);
      else out.thr[i] = Math.max(0, Math.min(1, need / drive));
      out.heading[i] = Math.atan2(out.y[i2] - out.y[i], out.x[i2] - out.x[i]);
    }
    return out;
  }
}

export interface CornerStats {
  name: string;
  dir: 'G' | 'D';
  kind: string;
  angle: number;
  radius: number;
  iMid: number;
  iApex: number;
  brakeStart: number;
  brakeDist: number;
  vEntry: number;
  vMin: number;
  vExit: number;
  latG: number;
  apexOffset: number;
  sApex: number;
  tApex: number;
}

/** Corner-by-corner summary from telemetry (v1 logic). */
export function cornerReport(track: Track, tel: Telemetry): CornerStats[] {
  const N = tel.N;
  const cs = track.corners;
  return cs.map((c, idx) => {
    const prev = cs[(idx - 1 + cs.length) % cs.length];
    const nxt = cs[(idx + 1) % cs.length];
    const a = Math.floor(prev.iMid + (((c.iMid - prev.iMid) + N) % N) * 0.5) % N;
    const b = Math.floor(c.iMid + (((nxt.iMid - c.iMid) + N) % N) * 0.5) % N;
    const span = ((b - a) + N) % N || N;
    let iMin = c.iMid;
    let vMin = 1e9;
    const r0 = (c.i0 - 5 + N) % N;
    const len = ((c.i1 - c.i0 + N) % N) + 10;
    for (let q = 0; q <= len; q++) {
      const i = (r0 + q) % N;
      if (tel.v[i] < vMin) {
        vMin = tel.v[i];
        iMin = i;
      }
    }
    let brakeStart = -1;
    for (let q = 0; q < span; q++) {
      const i = (iMin - q + N) % N;
      if (((iMin - i + N) % N) > ((iMin - a + N) % N)) break;
      if (tel.brk[i] > 0.05) brakeStart = i;
      else if (brakeStart >= 0) break;
    }
    const iEntry = brakeStart >= 0 ? brakeStart : iMin;
    const vEntry = tel.v[iEntry];
    const iExit = (Math.max(((c.i1 - iMin + N) % N), 1) + iMin + 8) % N;
    let aymax = 0;
    for (let q = 0; q <= len; q++) aymax = Math.max(aymax, Math.abs(tel.ay[(r0 + q) % N]));
    let brakeDist = brakeStart >= 0 ? ((tel.s[iMin] - tel.s[brakeStart]) + tel.s[N]) % tel.s[N] : 0;
    if (brakeDist > tel.s[N] / 2) {
      brakeDist = 0;
      brakeStart = -1;
    }
    return {
      name: c.name, dir: c.dir, kind: c.kind, angle: c.angle, radius: c.radius, iMid: c.iMid, iApex: iMin,
      brakeStart, brakeDist, vEntry, vMin, vExit: tel.v[iExit], latG: aymax / G, apexOffset: tel.d[iMin],
      sApex: tel.s[iMin], tApex: tel.time[iMin],
    };
  });
}

/**
 * v2 lateral limits on the kart centre: wheels on asphalt (|d| ≤ half − 0.65), or on the kerb where
 * there is one (kerb widths from TrackQuery.kerb), with 0.3 m more on the bridge (parapet).
 */
export function lineLimits(track: Track, kerb?: Float64Array): Float64Array {
  const out = new Float64Array(track.N);
  for (let i = 0; i < track.N; i++) {
    const k = kerb ? kerb[i] : 0;
    out[i] = k > 0 ? track.half + k - 0.65 : track.half - 0.65 - (track.z[i] > 0.1 ? 0.3 : 0);
  }
  return out;
}
