// Spatial queries on a built track, written for the simulation hot loop: no allocation per call.
// Every query takes a centreline index hint and only searches a local window around it, which is what
// keeps the upper and lower pass of the flyover apart (they share x, y but are far apart in s).
import type { Track } from './types';

/** Reusable projection result. */
export class Proj {
  /** Segment index (from point i to i + 1). */
  i = 0;
  /** Position along the segment, 0..1. */
  t = 0;
  /** Distance along the centreline, 0..L. */
  s = 0;
  /** Signed lateral offset, + = left of the centreline. */
  d = 0;
  /** Unit tangent of the segment. */
  tx = 1;
  ty = 0;
}

export class TrackQuery {
  readonly track: Track;
  readonly N: number;
  readonly L: number;
  /** Mean spacing of centreline points (m). */
  readonly ds: number;
  private readonly segX: Float64Array;
  private readonly segY: Float64Array;
  private readonly segInvLen2: Float64Array;
  private readonly segLen: Float64Array;
  /** Left / right edge polylines (centreline ± half width). */
  readonly lx: Float64Array;
  readonly ly: Float64Array;
  readonly rx: Float64Array;
  readonly ry: Float64Array;
  private readonly tmp = new Proj();

  constructor(track: Track) {
    this.track = track;
    const N = (this.N = track.N);
    this.L = track.L;
    this.ds = track.L / N;
    this.segX = new Float64Array(N);
    this.segY = new Float64Array(N);
    this.segInvLen2 = new Float64Array(N);
    this.segLen = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const j = (i + 1) % N;
      const dx = track.x[j] - track.x[i];
      const dy = track.y[j] - track.y[i];
      this.segX[i] = dx;
      this.segY[i] = dy;
      const l2 = dx * dx + dy * dy || 1e-12;
      this.segInvLen2[i] = 1 / l2;
      this.segLen[i] = Math.sqrt(l2);
    }
    this.lx = new Float64Array(N);
    this.ly = new Float64Array(N);
    this.rx = new Float64Array(N);
    this.ry = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      this.lx[i] = track.x[i] + track.nx[i] * track.half;
      this.ly[i] = track.y[i] + track.ny[i] * track.half;
      this.rx[i] = track.x[i] - track.nx[i] * track.half;
      this.ry[i] = track.y[i] - track.ny[i] * track.half;
    }
  }

  /**
   * Distance from (x, y) along unit direction (dx, dy) to the first track edge, capped at maxDist.
   * Exact ray / polyline intersection over the edge segments from hint − back to hint + ahead, so the
   * other branch of the loop (e.g. across the flyover) is never considered.
   */
  rayEdges(x: number, y: number, dx: number, dy: number, hint: number, maxDist = 40): number {
    const { N, lx, ly, rx, ry } = this;
    const ahead = Math.ceil(maxDist / this.ds) + 2;
    let best = maxDist;
    for (let k = -4; k <= ahead; k++) {
      const i = (((hint + k) % N) + N) % N;
      const j = i + 1 === N ? 0 : i + 1;
      // left edge segment
      let ax = lx[i] - x;
      let ay = ly[i] - y;
      let ex = lx[j] - lx[i];
      let ey = ly[j] - ly[i];
      let den = dx * ey - dy * ex;
      if (den !== 0) {
        const r = (ax * ey - ay * ex) / den;
        const u = (ax * dy - ay * dx) / den;
        if (r >= 0 && r < best && u >= -1e-9 && u <= 1 + 1e-9) best = r;
      }
      // right edge segment
      ax = rx[i] - x;
      ay = ry[i] - y;
      ex = rx[j] - rx[i];
      ey = ry[j] - ry[i];
      den = dx * ey - dy * ex;
      if (den !== 0) {
        const r = (ax * ey - ay * ex) / den;
        const u = (ax * dy - ay * dx) / den;
        if (r >= 0 && r < best && u >= -1e-9 && u <= 1 + 1e-9) best = r;
      }
    }
    return best;
  }

  /** Projects (x, y) onto the centreline, searching segments hint − w .. hint + w. */
  project(x: number, y: number, hint: number, out: Proj, w = 6): Proj {
    const { N, track: t, segX, segY, segInvLen2 } = this;
    let best = Infinity;
    let bi = 0;
    let bt = 0;
    for (let k = -w; k <= w; k++) {
      const i = (((hint + k) % N) + N) % N;
      const px = x - t.x[i];
      const py = y - t.y[i];
      let u = (px * segX[i] + py * segY[i]) * segInvLen2[i];
      if (u < 0) u = 0;
      else if (u > 1) u = 1;
      const ex = px - u * segX[i];
      const ey = py - u * segY[i];
      const d2 = ex * ex + ey * ey;
      if (d2 < best) {
        best = d2;
        bi = i;
        bt = u;
      }
    }
    const inv = 1 / this.segLen[bi];
    const tx = segX[bi] * inv;
    const ty = segY[bi] * inv;
    out.i = bi;
    out.t = bt;
    out.s = t.s[bi] + bt * this.segLen[bi];
    out.d = (x - t.x[bi]) * -ty + (y - t.y[bi]) * tx;
    out.tx = tx;
    out.ty = ty;
    return out;
  }

  /** Projection without a hint: nearest segment overall, preferring the level closest to zHint. */
  projectGlobal(x: number, y: number, out: Proj, zHint = 0): Proj {
    const { N, track: t } = this;
    let best = Infinity;
    let bi = 0;
    for (let i = 0; i < N; i++) {
      const dx = x - t.x[i];
      const dy = y - t.y[i];
      const dz = (zHint - t.z[i]) * 4;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best) {
        best = d2;
        bi = i;
      }
    }
    return this.project(x, y, bi, out, 2);
  }

  /** Segment index containing distance s (wraps); binary search, points are only nearly equidistant. */
  indexAt(s: number): number {
    const S = this.track.s;
    s -= Math.floor(s / this.L) * this.L;
    let lo = 0;
    let hi = this.N;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (S[mid] <= s) lo = mid;
      else hi = mid;
    }
    return lo;
  }

  /** Linear interpolation of a per-point array at distance s (wraps). */
  at(arr: Float64Array, s: number): number {
    s -= Math.floor(s / this.L) * this.L;
    const i = this.indexAt(s);
    const S = this.track.s;
    const u = (s - S[i]) / (S[i + 1] - S[i]);
    return arr[i] + (arr[(i + 1) % this.N] - arr[i]) * u;
  }

  /** Centreline curvature (1/m, + = left) at distance s. */
  curvature(s: number): number {
    return this.at(this.track.kc, s);
  }

  /**
   * Free distance from (x, y) along direction (dx, dy) (unit) until the ray leaves the track
   * (|d| > half + margin), capped at maxDist. Sphere tracing on the lateral clearance, with the
   * projection hint following the ray so it never jumps to another branch of the loop.
   */
  ray(x: number, y: number, dx: number, dy: number, hint: number, maxDist = 40, margin = 0): number {
    const half = this.track.half + margin;
    const p = this.tmp;
    let dist = 0;
    let h = hint;
    for (let it = 0; it < 48; it++) {
      this.project(x + dx * dist, y + dy * dist, h, p, 4);
      h = p.i;
      const clear = half - Math.abs(p.d);
      if (clear <= 0) {
        if (dist === 0) return 0;
        // bisection back to the edge
        let lo = Math.max(0, dist - 2);
        let hi = dist;
        for (let b = 0; b < 8; b++) {
          const mid = (lo + hi) / 2;
          this.project(x + dx * mid, y + dy * mid, h, p, 4);
          if (half - Math.abs(p.d) > 0) lo = mid;
          else hi = mid;
        }
        return (lo + hi) / 2;
      }
      if (dist >= maxDist) return maxDist;
      dist = Math.min(maxDist, dist + Math.min(2, Math.max(0.15, clear * 0.8)));
    }
    return dist;
  }
}
