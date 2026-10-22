// The TS port must reproduce the untouched v1 code bit for bit (fixture: scripts/gen-v1-fixture.cjs).
import { describe, expect, it } from 'vitest';
import ref from './fixtures/v1-reference.json';
import lyl from '../data/tracks/lyl.json';
import { buildTrack } from '../src/track/build';
import type { TrackDef } from '../src/track/types';
import { kartParams, type KartId } from '../src/sim/karts';
import { LineEvaluator, cornerReport } from '../src/sim/qss';
import { LineGA } from '../src/sim/lineGA';
import { mulberry32 } from '../src/sim/rng';

const track = buildTrack(lyl as TrackDef);
const ev = new LineEvaluator(track, 16);
const sum = (a: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s;
};
const EPS = 1e-9;

function genes(seed: number): number[] {
  const r = mulberry32(seed);
  const lim = ev.limit();
  return Array.from({ length: ev.M }, () => (r() * 2 - 1) * lim);
}

describe('portage v1 : géométrie', () => {
  it('produit le même circuit', () => {
    expect(track.N).toBe(ref.track.N);
    expect(ev.M).toBe(ref.track.M);
    expect(Math.abs(track.L - ref.track.L)).toBeLessThan(EPS);
    expect(Math.abs(sum(track.x) - ref.track.sumX)).toBeLessThan(1e-6);
    expect(Math.abs(sum(track.y) - ref.track.sumY)).toBeLessThan(1e-6);
    expect(Math.abs(sum(track.z) - ref.track.sumZ)).toBeLessThan(1e-6);
    expect(Math.abs(sum(track.kc) - ref.track.sumKc)).toBeLessThan(EPS);
    expect(Math.abs(sum(track.grade) - ref.track.sumGrade)).toBeLessThan(EPS);
    expect(track.corners.map((c) => [c.name, c.kind, c.dir, c.iMid])).toEqual(ref.track.corners);
    expect(track.crossings).toEqual(ref.track.crossings);
  });
});

describe('portage v1 : solveur quasi-statique', () => {
  it.each(ref.laps.map((l) => [l.seed, l.kart, JSON.stringify(l.opts), l] as const))(
    'gènes #%i, %s, %s : même temps au tour',
    (seed, kart, _o, lap) => {
      const g = seed === 0 ? new Array(ev.M).fill(0) : genes(seed);
      const T = ev.lapTime(g, kartParams(kart as KartId, lap.opts));
      expect(Math.abs(T - lap.T)).toBeLessThan(EPS);
    },
  );

  it('même télémétrie et même fiche virages', () => {
    const tel = ev.telemetry(genes(1), kartParams('k390'));
    expect(Math.abs(tel.T - ref.telemetry.T)).toBeLessThan(EPS);
    expect(Math.abs(sum(tel.v) - ref.telemetry.sumV)).toBeLessThan(1e-7);
    expect(Math.abs(sum(tel.thr) - ref.telemetry.sumThr)).toBeLessThan(1e-7);
    expect(Math.abs(sum(tel.brk) - ref.telemetry.sumBrk)).toBeLessThan(1e-7);
    expect(Math.abs(sum(tel.time) - ref.telemetry.sumTime)).toBeLessThan(1e-6);
    const rep = cornerReport(track, tel);
    expect(rep.length).toBe(ref.cornerReport.length);
    rep.forEach((c, i) => {
      const r = ref.cornerReport[i];
      expect(c.name).toBe(r.name);
      expect(c.brakeStart).toBe(r.brakeStart);
      expect(Math.abs(c.brakeDist - r.brakeDist)).toBeLessThan(1e-7);
      expect(Math.abs(c.vMin - r.vMin)).toBeLessThan(EPS);
      expect(Math.abs(c.latG - r.latG)).toBeLessThan(EPS);
    });
  });
});

describe('portage v1 : algorithme génétique', () => {
  it.each(['k390', 'k270'] as const)('%s : 200 générations seedées identiques', (kart) => {
    const ga = new LineGA(ev, kartParams(kart), { popSize: 80, seed: 12345 });
    let r = ga.step();
    for (let i = 1; i < 200; i++) r = ga.step();
    const want = ref.ga200[kart];
    expect(Math.abs(r.best - want.best)).toBeLessThan(EPS);
    expect(Math.abs(r.mean - want.mean)).toBeLessThan(EPS);
    expect(Math.abs(r.sigma - want.sigma)).toBeLessThan(EPS);
  }, 30_000);
});
