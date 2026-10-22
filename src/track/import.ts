// Track importers. DOM-free (string parsing only) so they also run in workers and under Node.
import type { Pt } from './geometry';

/** Drops non-finite points and a duplicated closing point. */
function tidy(out: Pt[]): Pt[] {
  const pts = out.filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
  if (pts.length > 2) {
    const a = pts[0];
    const b = pts[pts.length - 1];
    if (Math.abs(a[0] - b[0]) < 1e-7 && Math.abs(a[1] - b[1]) < 1e-7) pts.pop();
  }
  return pts;
}

/** GPX track or route points, in file order. Returns [lat, lon]. */
export function parseGpx(text: string): Pt[] {
  const out: Pt[] = [];
  const tag = /<(?:\w+:)?(?:trkpt|rtept)\b([^>]*)>/g;
  let m: RegExpExecArray | null;
  while ((m = tag.exec(text))) {
    const lat = /\blat\s*=\s*["']([^"']+)["']/.exec(m[1]);
    const lon = /\blon\s*=\s*["']([^"']+)["']/.exec(m[1]);
    if (lat && lon) out.push([parseFloat(lat[1]), parseFloat(lon[1])]);
  }
  return tidy(out);
}

/** Longest LineString found anywhere in a GeoJSON document. Returns [lat, lon]. */
export function parseGeoJson(text: string): Pt[] {
  const lines: number[][][] = [];
  const visit = (g: unknown): void => {
    if (!g || typeof g !== 'object') return;
    const o = g as { type?: string; features?: unknown[]; geometry?: unknown; geometries?: unknown[]; coordinates?: unknown };
    if (o.type === 'FeatureCollection') o.features?.forEach(visit);
    else if (o.type === 'Feature') visit(o.geometry);
    else if (o.type === 'GeometryCollection') o.geometries?.forEach(visit);
    else if (o.type === 'LineString') lines.push(o.coordinates as number[][]);
    else if (o.type === 'MultiLineString' || o.type === 'Polygon') (o.coordinates as number[][][]).forEach((l) => lines.push(l));
  };
  visit(JSON.parse(text));
  lines.sort((a, b) => b.length - a.length);
  return lines[0] ? tidy(lines[0].map((c): Pt => [c[1], c[0]])) : [];
}

interface OverpassWay {
  type: 'way';
  id: number;
  nodes: number[];
  geometry: { lat: number; lon: number }[];
  tags?: Record<string, string>;
}

/**
 * Chains OSM ways that share end nodes into one closed loop. The driving direction is the node order
 * of `mainWayId` (or of the first way); other ways are reversed as needed unless tagged oneway=yes,
 * in which case they must already point the right way.
 */
export function stitchOverpass(json: { elements: unknown[] }, wayIds?: number[], mainWayId?: number): Pt[] {
  let ways = json.elements.filter((e): e is OverpassWay => (e as OverpassWay).type === 'way' && !!(e as OverpassWay).geometry);
  if (wayIds) ways = ways.filter((w) => wayIds.includes(w.id));
  if (!ways.length) throw new Error('Aucun way avec géométrie dans la réponse Overpass.');
  const first = ways.find((w) => w.id === mainWayId) ?? ways[0];
  const left = ways.filter((w) => w !== first);
  const nodes = [...first.nodes];
  const geo = [...first.geometry];
  while (left.length) {
    const tail = nodes[nodes.length - 1];
    const k = left.findIndex((w) => w.nodes[0] === tail || w.nodes[w.nodes.length - 1] === tail);
    if (k < 0) throw new Error(`Les ways ne se raccordent pas après le nœud ${tail}.`);
    const w = left.splice(k, 1)[0];
    const forward = w.nodes[0] === tail;
    if (!forward && w.tags?.oneway === 'yes') throw new Error(`Le way ${w.id} est à sens unique dans le mauvais sens.`);
    const wn = forward ? w.nodes : [...w.nodes].reverse();
    const wg = forward ? w.geometry : [...w.geometry].reverse();
    nodes.push(...wn.slice(1));
    geo.push(...wg.slice(1));
  }
  if (nodes[nodes.length - 1] !== nodes[0]) throw new Error('Les ways ne forment pas une boucle fermée.');
  return tidy(geo.map((g): Pt => [g.lat, g.lon]));
}

/** Picks a parser from the file name / content. */
export function parseTrackFile(text: string, name: string): Pt[] {
  if (/\.gpx$/i.test(name) || /<gpx[\s>]/i.test(text)) return parseGpx(text);
  const json = JSON.parse(text);
  if (Array.isArray(json?.elements)) return stitchOverpass(json);
  return parseGeoJson(text);
}
