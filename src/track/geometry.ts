// Planar geometry helpers, ported verbatim from v1 KartCore so lap times stay bit-identical.

export type Pt = [number, number];

export const G = 9.81;
const R_EARTH = 6371000;
export const DEG = Math.PI / 180;

/** Equirectangular projection around the centroid; returns metres (x east, y north). */
export function project(coords: readonly Pt[]): { pts: Pt[]; lat0: number; lon0: number } {
  let lat0 = 0;
  let lon0 = 0;
  const n = coords.length;
  for (let i = 0; i < n; i++) {
    lat0 += coords[i][0];
    lon0 += coords[i][1];
  }
  lat0 /= n;
  lon0 /= n;
  const kx = Math.cos(lat0 * DEG) * R_EARTH * DEG;
  const ky = R_EARTH * DEG;
  return { pts: coords.map((c): Pt => [(c[1] - lon0) * kx, (c[0] - lat0) * ky]), lat0, lon0 };
}

export function closedLength(p: readonly Pt[]): number {
  let L = 0;
  for (let i = 0; i < p.length; i++) {
    const a = p[i];
    const b = p[(i + 1) % p.length];
    L += Math.hypot(b[0] - a[0], b[1] - a[1]);
  }
  return L;
}

export function resampleClosed(p: readonly Pt[], ds: number): Pt[] {
  const n = p.length;
  const cum = [0];
  for (let i = 0; i < n; i++) {
    const a = p[i];
    const b = p[(i + 1) % n];
    cum.push(cum[i] + Math.hypot(b[0] - a[0], b[1] - a[1]));
  }
  const L = cum[n];
  const m = Math.max(8, Math.round(L / ds));
  const step = L / m;
  const out: Pt[] = [];
  let seg = 0;
  for (let k = 0; k < m; k++) {
    const s = k * step;
    while (cum[seg + 1] < s) seg++;
    const t = (s - cum[seg]) / Math.max(1e-9, cum[seg + 1] - cum[seg]);
    const A = p[seg];
    const B = p[(seg + 1) % n];
    out.push([A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t]);
  }
  return out;
}

/** Repeated triangular-kernel moving average on a closed polyline. */
export function smoothClosed(p: readonly Pt[], radius: number, passes: number): Pt[] {
  const n = p.length;
  let cur = p as Pt[];
  for (let q = 0; q < passes; q++) {
    const nxt: Pt[] = new Array(n);
    for (let i = 0; i < n; i++) {
      let sx = 0;
      let sy = 0;
      let w = 0;
      for (let k = -radius; k <= radius; k++) {
        const c = cur[(i + k + n) % n];
        const wk = radius + 1 - Math.abs(k);
        sx += c[0] * wk;
        sy += c[1] * wk;
        w += wk;
      }
      nxt[i] = [sx / w, sy / w];
    }
    cur = nxt;
  }
  return cur;
}

/** Proper intersection of segments ab and cd; returns [t on ab, u on cd] or null. */
export function segIntersect(a: Pt, b: Pt, c: Pt, d: Pt): [number, number] | null {
  const r0 = b[0] - a[0];
  const r1 = b[1] - a[1];
  const s0 = d[0] - c[0];
  const s1 = d[1] - c[1];
  const den = r0 * s1 - r1 * s0;
  if (Math.abs(den) < 1e-12) return null;
  const t = ((c[0] - a[0]) * s1 - (c[1] - a[1]) * s0) / den;
  const u = ((c[0] - a[0]) * r1 - (c[1] - a[1]) * r0) / den;
  if (t > 0 && t < 1 && u > 0 && u < 1) return [t, u];
  return null;
}

/** Signed curvature of the circle through p[i-h], p[i], p[i+h] (+ = left turn). */
export function signedCurv(p: readonly Pt[], i: number, h: number): number {
  const n = p.length;
  const a = p[(i - h + n) % n];
  const b = p[i];
  const c = p[(i + h) % n];
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const bcx = c[0] - b[0];
  const bcy = c[1] - b[1];
  const acx = c[0] - a[0];
  const acy = c[1] - a[1];
  const cross = abx * bcy - aby * bcx;
  const den = Math.hypot(abx, aby) * Math.hypot(bcx, bcy) * Math.hypot(acx, acy);
  return den < 1e-12 ? 0 : (2 * cross) / den;
}

/** Shoelace signed area: < 0 means the loop is driven clockwise. */
export function signedArea(p: readonly Pt[]): number {
  let A = 0;
  for (let i = 0; i < p.length; i++) {
    const a = p[i];
    const b = p[(i + 1) % p.length];
    A += a[0] * b[1] - b[0] * a[1];
  }
  return A / 2;
}
