import { DEG, closedLength, project, resampleClosed, segIntersect, signedCurv, smoothClosed, type Pt } from './geometry';
import type { BuildOptions, Corner, Crossing, Track, TrackDef } from './types';

const DS = 2;

/**
 * Builds the simulation track from a generic definition. With scale 'osm' this is a verbatim port of
 * v1 KartCore.buildTrack (lap-time regression depends on it). Scale 'official' rescales the smoothed
 * centreline, so the final loop length really equals officialLength (v1 scaled before smoothing and
 * ended up ~18 m short).
 */
export function buildTrack(def: TrackDef, opts: BuildOptions = {}): Track {
  const width = opts.width ?? def.width;
  const pr = project(def.coords);
  let dense = smoothClosed(resampleClosed(pr.pts, 1), 3, 4);
  if (opts.scale === 'official' && def.officialLength) {
    const f = def.officialLength / closedLength(dense);
    dense = dense.map((q): Pt => [q[0] * f, q[1] * f]);
  }
  let c = resampleClosed(dense, DS);
  const N = c.length;

  // centreline curvature (for start-line choice and corner detection)
  const kc = new Float64Array(N);
  for (let i = 0; i < N; i++) kc[i] = signedCurv(c, i, 3);
  let ks = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let acc = 0;
    for (let k = -3; k <= 3; k++) acc += kc[(i + k + N) % N];
    ks[i] = acc / 7;
  }

  // start line: middle of the longest straight unless given
  let start = 0;
  if (typeof def.startFrac === 'number') start = Math.round(def.startFrac * N) % N;
  else {
    let best = 0;
    let bestEnd = 0;
    let run = 0;
    for (let q = 0; q < 2 * N; q++) {
      if (Math.abs(ks[q % N]) < 1 / 120) {
        run++;
        if (run > best && run <= N) {
          best = run;
          bestEnd = q;
        }
      } else run = 0;
    }
    start = (bestEnd - Math.floor(best * 0.5) + N) % N;
  }
  const rotated = new Float64Array(N);
  for (let j = 0; j < N; j++) rotated[j] = ks[(j + start) % N];
  ks = rotated;
  c = c.slice(start).concat(c.slice(0, start));

  const x = new Float64Array(N);
  const y = new Float64Array(N);
  const tx = new Float64Array(N);
  const ty = new Float64Array(N);
  const nx = new Float64Array(N);
  const ny = new Float64Array(N);
  const s = new Float64Array(N + 1);
  for (let i = 0; i < N; i++) {
    x[i] = c[i][0];
    y[i] = c[i][1];
  }
  for (let i = 0; i < N; i++) {
    const a = c[(i - 1 + N) % N];
    const b = c[(i + 1) % N];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const l = Math.hypot(dx, dy) || 1;
    tx[i] = dx / l;
    ty[i] = dy / l;
    nx[i] = -ty[i];
    ny[i] = tx[i];
    s[i + 1] = s[i] + Math.hypot(c[(i + 1) % N][0] - c[i][0], c[(i + 1) % N][1] - c[i][1]);
  }
  const L = s[N];

  // self-crossings -> flyover: the later pass is the bridge
  const crossings: Crossing[] = [];
  for (let i = 0; i < N; i++) {
    for (let j = i + 2; j < N; j++) {
      if (i === 0 && j === N - 1) continue;
      const hit = segIntersect(c[i], c[(i + 1) % N], c[j], c[(j + 1) % N]);
      if (hit) {
        crossings.push({
          sLow: s[i] + hit[0] * (s[i + 1] - s[i]),
          sHigh: s[j] + hit[1] * (s[j + 1] - s[j]),
          x: c[i][0] + (c[(i + 1) % N][0] - c[i][0]) * hit[0],
          y: c[i][1] + (c[(i + 1) % N][1] - c[i][1]) * hit[0],
        });
      }
    }
  }
  const z = new Float64Array(N);
  const bridge = def.bridge;
  if (bridge && crossings.length) {
    const H = bridge.height || 4.5;
    const plateau = bridge.plateau || 10;
    const ramp = bridge.ramp || 38;
    for (const cr of crossings) {
      if (bridge.lowerIsBridge) {
        const tmp = cr.sLow;
        cr.sLow = cr.sHigh;
        cr.sHigh = tmp;
      }
      for (let i = 0; i < N; i++) {
        let d = Math.abs(s[i] - cr.sHigh);
        d = Math.min(d, L - d);
        const h = d <= plateau ? H : d >= plateau + ramp ? 0 : H * 0.5 * (1 + Math.cos((Math.PI * (d - plateau)) / ramp));
        if (h > z[i]) z[i] = h;
      }
    }
  }
  const grade = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    grade[i] = (z[(i + 1) % N] - z[(i - 1 + N) % N]) / (s[i + 1] - s[i] + (s[i] - (i ? s[i - 1] : s[N - 1] - L)));
  }

  const track: Track = {
    N, L, x, y, z, grade, tx, ty, nx, ny, s, kc: ks,
    width, half: width / 2, crossings, origin: { lat0: pr.lat0, lon0: pr.lon0 }, corners: [],
  };
  track.corners = detectCorners(track);
  return track;
}

export function detectCorners(t: Pick<Track, 'N' | 'kc'>): Corner[] {
  const N = t.N;
  const TH = 1 / 45;
  const regions: [number, number][] = [];
  let inR = false;
  let st = 0;
  // start scanning at a straight so regions don't wrap
  let off = 0;
  for (let i = 0; i < N; i++) {
    if (Math.abs(t.kc[i]) < TH * 0.5) {
      off = i;
      break;
    }
  }
  for (let q = 0; q <= N; q++) {
    const i2 = (q + off) % N;
    const on = Math.abs(t.kc[i2]) > TH && q < N;
    if (on && !inR) {
      inR = true;
      st = q;
    }
    if (!on && inR) {
      inR = false;
      regions.push([st, q - 1]);
    }
  }
  // merge close regions turning the same way
  const merged: { a: number; b: number; sign: number }[] = [];
  for (const r of regions) {
    const sign = Math.sign(t.kc[(r[0] + off + Math.floor((r[1] - r[0]) / 2)) % N]);
    const last = merged[merged.length - 1];
    if (last && r[0] - last.b <= 4 && last.sign === sign) last.b = r[1];
    else merged.push({ a: r[0], b: r[1], sign });
  }
  const out: Corner[] = [];
  for (const m of merged) {
    let ang = 0;
    let kmax = 0;
    let imax = m.a;
    for (let q = m.a; q <= m.b; q++) {
      const kk = t.kc[(q + off) % N];
      ang += kk * 2;
      if (Math.abs(kk) > kmax) {
        kmax = Math.abs(kk);
        imax = q;
      }
    }
    if (Math.abs(ang) < 20 * DEG) continue;
    out.push({
      i0: (m.a + off) % N, i1: (m.b + off) % N, iMid: (imax + off) % N,
      dir: ang < 0 ? 'D' : 'G', angle: Math.abs(ang) / DEG, radius: 1 / kmax, name: '', kind: 'Rapide',
    });
  }
  out.sort((a, b) => a.iMid - b.iMid);
  out.forEach((c, k) => {
    c.name = 'V' + (k + 1);
    c.kind = c.angle > 140 ? 'Épingle' : c.radius < 14 ? 'Serré' : c.radius < 28 ? 'Moyen' : 'Rapide';
  });
  return out;
}
