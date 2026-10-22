import { describe, expect, it } from 'vitest';
import lyl from '../data/tracks/lyl.json';
import { buildTrack } from '../src/track/build';
import { Proj, TrackQuery } from '../src/track/query';
import type { TrackDef } from '../src/track/types';

const t = buildTrack(lyl as TrackDef, { scale: 'official' });
const q = new TrackQuery(t);
const p = new Proj();

/** Index of the longest straight's middle (the start line is placed there). */
const STRAIGHT = 0;

describe('projection sur l’axe', () => {
  it('retrouve s et d pour des points décalés de l’axe', () => {
    for (const i of [0, 37, 200, 411, t.N - 3]) {
      for (const off of [-3.5, 0, 2]) {
        const x = t.x[i] + t.nx[i] * off;
        const y = t.y[i] + t.ny[i] * off;
        q.project(x, y, i + 2, p);
        expect(Math.abs(p.d - off)).toBeLessThan(0.12);
        const ds = Math.abs(p.s - t.s[i]);
        expect(Math.min(ds, t.L - ds)).toBeLessThan(0.6);
      }
    }
  });

  it('lève l’ambiguïté au croisement du pont grâce à l’indice suivi', () => {
    const cr = t.crossings[0];
    const iLow = Math.round((cr.sLow / t.L) * t.N);
    const iHigh = Math.round((cr.sHigh / t.L) * t.N);
    q.project(cr.x, cr.y, iLow, p);
    expect(Math.abs(p.s - cr.sLow)).toBeLessThan(0.5);
    q.project(cr.x, cr.y, iHigh, p);
    expect(Math.abs(p.s - cr.sHigh)).toBeLessThan(0.5);
    // global projection picks the level by height
    q.projectGlobal(cr.x, cr.y, p, 4.5);
    expect(Math.abs(p.s - cr.sHigh)).toBeLessThan(0.5);
    q.projectGlobal(cr.x, cr.y, p, 0);
    expect(Math.abs(p.s - cr.sLow)).toBeLessThan(0.5);
  });

  it('suit un tour complet sans saut de s, pont compris', () => {
    let hint = 0;
    let prev = 0;
    for (let k = 1; k < t.N * 4; k++) {
      const s = (k / (t.N * 4)) * t.L;
      const i = Math.floor((s / t.L) * t.N);
      const u = (s / t.L) * t.N - i;
      const j = (i + 1) % t.N;
      const x = t.x[i] + (t.x[j] - t.x[i]) * u + t.nx[i] * 1.5;
      const y = t.y[i] + (t.y[j] - t.y[i]) * u + t.ny[i] * 1.5;
      q.project(x, y, hint, p, 3);
      hint = p.i;
      expect(p.s - prev).toBeGreaterThan(-0.05);
      expect(p.s - prev).toBeLessThan(1.5);
      prev = p.s;
    }
  });
});

describe('rayons', () => {
  const i = STRAIGHT;
  const x = t.x[i];
  const y = t.y[i];
  const tx = t.tx[i];
  const ty = t.ty[i];

  it('mesurent la demi-largeur perpendiculairement à la ligne droite', () => {
    expect(q.ray(x, y, -ty, tx, i)).toBeCloseTo(t.half, 1);
    expect(q.ray(x, y, ty, -tx, i)).toBeCloseTo(t.half, 1);
  });

  it('mesurent demi-largeur × √2 à 45°', () => {
    const c = Math.SQRT1_2;
    expect(q.ray(x, y, c * tx - c * ty, c * ty + c * tx, i)).toBeCloseTo(t.half * Math.SQRT2, 0);
  });

  it('voient loin dans l’axe de la ligne droite et plafonnent à la portée', () => {
    const r = q.ray(x, y, tx, ty, i, 40);
    expect(r).toBeGreaterThan(25);
    expect(r).toBeLessThanOrEqual(40);
  });

  it('valent 0 hors piste', () => {
    expect(q.ray(x + t.nx[i] * (t.half + 1), y + t.ny[i] * (t.half + 1), tx, ty, i)).toBe(0);
  });

  it('ne voient pas l’autre branche sous le pont', () => {
    const cr = t.crossings[0];
    const iLow = Math.round((cr.sLow / t.L) * t.N);
    // looking sideways on the lower pass: the other branch runs across, the ray must stop at our edge
    const r = q.ray(cr.x, cr.y, -t.ty[iLow], t.tx[iLow], iLow);
    expect(r).toBeLessThan(t.half + 0.6);
  });
});

/** Independent reference: is (x, y) inside one of the track quads near index i? */
function inStrip(x: number, y: number, i0: number): boolean {
  const inTri = (ax: number, ay: number, bx: number, by: number, cx: number, cy: number) => {
    const d1 = (x - bx) * (ay - by) - (ax - bx) * (y - by);
    const d2 = (x - cx) * (by - cy) - (bx - cx) * (y - cy);
    const d3 = (x - ax) * (cy - ay) - (cx - ax) * (y - ay);
    return !((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0));
  };
  for (let k = -6; k <= 26; k++) {
    const i = (((i0 + k) % t.N) + t.N) % t.N;
    const j = (i + 1) % t.N;
    if (inTri(q.lx[i], q.ly[i], q.lx[j], q.ly[j], q.rx[j], q.ry[j]) || inTri(q.lx[i], q.ly[i], q.rx[j], q.ry[j], q.rx[i], q.ry[i])) return true;
  }
  return false;
}
function marchRef(x: number, y: number, dx: number, dy: number, i: number, max = 40): number {
  for (let r = 0; r <= max; r += 0.02) if (!inStrip(x + dx * r, y + dy * r, i)) return r;
  return max;
}

describe('rayons analytiques (rayEdges)', () => {
  it('concordent avec une marche fine dans la bande de piste, 9 directions, tout le tour', () => {
    let worst = 0;
    for (let i = 0; i < t.N; i += 5) {
      for (let k = 0; k < 9; k++) {
        const a = (k / 8) * Math.PI - Math.PI / 2;
        const c = Math.cos(a);
        const s = Math.sin(a);
        const dx = c * t.tx[i] - s * t.ty[i];
        const dy = s * t.tx[i] + c * t.ty[i];
        // start mid-segment: rays from a vertex run exactly along a quad boundary of the reference
        const j = (i + 1) % t.N;
        const x = t.x[i] + (t.x[j] - t.x[i]) * 0.37 + t.nx[i] * 1.2;
        const y = t.y[i] + (t.y[j] - t.y[i]) * 0.37 + t.ny[i] * 1.2;
        const ref = marchRef(x, y, dx, dy, i, 40);
        const fast = q.rayEdges(x, y, dx, dy, i, 40);
        worst = Math.max(worst, Math.abs(ref - fast));
      }
    }
    expect(worst).toBeLessThan(0.05);
  });

  it('s’arrêtent au bord sous le pont', () => {
    const cr = t.crossings[0];
    const iLow = Math.round((cr.sLow / t.L) * t.N);
    expect(q.rayEdges(cr.x, cr.y, -t.ty[iLow], t.tx[iLow], iLow)).toBeLessThan(t.half + 0.6);
  });
});

describe('interpolation', () => {
  it('donne la courbure et la hauteur le long de s', () => {
    expect(q.curvature(t.s[50])).toBeCloseTo(t.kc[50], 9);
    const cr = t.crossings[0];
    expect(q.at(t.z, cr.sHigh)).toBeCloseTo(4.5, 1);
    expect(q.at(t.z, cr.sLow + t.L)).toBeCloseTo(0, 6);
  });
});
