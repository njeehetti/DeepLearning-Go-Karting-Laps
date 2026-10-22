import { describe, expect, it } from 'vitest';
import lyl from '../data/tracks/lyl.json';
import { parseGeoJson, parseGpx, parseTrackFile, stitchOverpass } from '../src/track/import';
import { buildTrack } from '../src/track/build';
import type { TrackDef } from '../src/track/types';

const coords = (lyl as TrackDef).coords;

describe('import GPX', () => {
  const gpx = `<?xml version="1.0"?><gpx version="1.1" creator="RaceChrono"><trk><trkseg>
${coords.map(([la, lo]) => `<trkpt lat="${la}" lon='${lo}'><ele>40</ele><time>2026-10-01T10:00:00Z</time></trkpt>`).join('\n')}
<trkpt lat="${coords[0][0]}" lon="${coords[0][1]}"></trkpt></trkseg></trk></gpx>`;

  it('lit les points dans l’ordre et retire le point de fermeture', () => {
    expect(parseGpx(gpx)).toEqual(coords);
    expect(parseTrackFile(gpx, 'tour.gpx')).toEqual(coords);
  });

  it('donne un circuit identique à la source', () => {
    const def = { ...(lyl as TrackDef), coords: parseGpx(gpx) };
    expect(buildTrack(def).L).toBeCloseTo(buildTrack(lyl as TrackDef).L, 6);
  });
});

describe('import GeoJSON', () => {
  it('prend la LineString la plus longue', () => {
    const gj = JSON.stringify({
      type: 'FeatureCollection',
      features: [
        { type: 'Feature', geometry: { type: 'LineString', coordinates: [[101.84, 3.05], [101.85, 3.05]] } },
        { type: 'Feature', geometry: { type: 'LineString', coordinates: coords.map(([la, lo]) => [lo, la]) } },
      ],
    });
    expect(parseGeoJson(gj)).toEqual(coords);
    expect(parseTrackFile(gj, 'piste.geojson')).toEqual(coords);
  });
});

describe('assemblage Overpass', () => {
  const ids = (n: number, from: number) => Array.from({ length: n }, (_, k) => from + k);
  const n = coords.length;
  const cut = 100;
  const nodes = ids(n, 1);
  const geom = (a: number, b: number) => coords.slice(a, b).map(([lat, lon]) => ({ lat, lon }));
  const main = { type: 'way', id: 1, nodes: nodes.slice(0, cut + 1), geometry: geom(0, cut + 1), tags: {} };
  const bridge = { type: 'way', id: 2, nodes: [...nodes.slice(cut), 1], geometry: [...geom(cut, n), { lat: coords[0][0], lon: coords[0][1] }], tags: { oneway: 'yes', bridge: 'yes' } };

  it('referme la boucle dans l’ordre du way principal', () => {
    expect(stitchOverpass({ elements: [bridge, main] }, undefined, 1)).toEqual(coords);
  });

  it('retourne un way à double sens si besoin', () => {
    const rev = { ...bridge, tags: {}, nodes: [...bridge.nodes].reverse(), geometry: [...bridge.geometry].reverse() };
    expect(stitchOverpass({ elements: [main, rev] }, undefined, 1)).toEqual(coords);
  });

  it('refuse un sens unique pris à contresens', () => {
    const rev = { ...bridge, nodes: [...bridge.nodes].reverse(), geometry: [...bridge.geometry].reverse() };
    expect(() => stitchOverpass({ elements: [main, rev] }, undefined, 1)).toThrow(/sens unique/);
  });

  it('refuse une boucle ouverte', () => {
    expect(() => stitchOverpass({ elements: [main] })).toThrow(/boucle/);
  });
});
