import { describe, expect, it } from 'vitest';
import lyl from '../data/tracks/lyl.json';
import { buildTrack } from '../src/track/build';
import { project, signedArea } from '../src/track/geometry';
import type { TrackDef } from '../src/track/types';
import { TRACKS } from '../src/track/registry';

const def = lyl as TrackDef;

describe('chargement du LYL', () => {
  const t = buildTrack(def);

  it('a la bonne longueur OSM, 19 virages et un croisement', () => {
    expect(t.L).toBeGreaterThan(1565);
    expect(t.L).toBeLessThan(1595);
    expect(t.corners).toHaveLength(19);
    expect(t.crossings).toHaveLength(1);
    expect(t.width).toBe(8);
  });

  it('se roule dans le sens horaire', () => {
    expect(signedArea(project(def.coords).pts)).toBeLessThan(0);
  });

  it('met le passage le plus tardif sur le pont à 4,5 m', () => {
    const cr = t.crossings[0];
    expect(cr.sHigh).toBeGreaterThan(cr.sLow);
    const at = (s: number) => t.z[Math.round(s / (t.L / t.N)) % t.N];
    expect(at(cr.sHigh)).toBeCloseTo(4.5, 1);
    expect(at(cr.sLow)).toBe(0);
    expect(Math.max(...t.z)).toBeCloseTo(4.5, 6);
  });

  it('a des rampes en cosinus : pente max = H·π/(2·rampe) ≈ 18,6 %, moyenne ≈ 12 %', () => {
    const peak = (4.5 * Math.PI) / (2 * 38);
    expect(Math.max(...t.grade.map(Math.abs))).toBeCloseTo(peak, 2);
  });

  it("respecte la largeur demandée", () => {
    const w = buildTrack(def, { width: 9.5 });
    expect(w.half).toBe(4.75);
  });
});

describe('option de mise à l’échelle', () => {
  it('« officielle » donne 1 514 m après lissage', () => {
    const t = buildTrack(def, { scale: 'official' });
    expect(Math.abs(t.L - 1514)).toBeLessThan(1);
    expect(t.corners).toHaveLength(19);
  });

  it('« osm » ignore la longueur officielle', () => {
    expect(buildTrack(def, { scale: 'osm' }).L).toBeCloseTo(buildTrack(def).L, 9);
  });
});

describe('circuits du monde', () => {
  it.each(TRACKS.map((d) => [d.id, d] as const))('%s : boucle cohérente avec sa longueur officielle et son sens', (_id, d) => {
    expect(d.coords.length).toBeGreaterThan(40);
    expect(d.attribution).toMatch(/OpenStreetMap/);
    const osm = buildTrack(d, { scale: 'osm' });
    // the OSM centreline is within 5 % of the published length, so the rescale stays a small correction
    expect(Math.abs(osm.L / d.officialLength! - 1)).toBeLessThan(0.05);
    const t = buildTrack(d, { scale: 'official' });
    expect(Math.abs(t.L - d.officialLength!)).toBeLessThan(2);
    expect(t.corners.length).toBeGreaterThanOrEqual(10);
    expect(t.crossings).toHaveLength(d.bridge ? 1 : 0);
    if (d.direction) expect(signedArea(project(d.coords).pts) < 0 ? 'horaire' : 'anti-horaire').toBe(d.direction);
  });
});
