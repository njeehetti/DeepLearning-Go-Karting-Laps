// Fetches a circuit's centreline from OpenStreetMap (Overpass) and compares it with data/tracks/<id>.json.
// The file names its ways in `osm.ways` (stitched in driving order from `osm.main`).
// Usage: npx tsx scripts/fetch-osm.ts [--track lyl] [--write] [--reverse]
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { stitchOverpass } from '../src/track/import';
import type { TrackDef } from '../src/track/types';

const ENDPOINTS = ['https://overpass-api.de/api/interpreter', 'https://overpass.private.coffee/api/interpreter', 'https://maps.mail.ru/osm/tools/overpass/api/interpreter'];
const arg = (k: string, d: string) => {
  const i = process.argv.indexOf('--' + k);
  return i > 0 ? process.argv[i + 1] : d;
};
const id = arg('track', 'lyl');
const file = fileURLToPath(new URL(`../data/tracks/${id}.json`, import.meta.url));
const def = JSON.parse(readFileSync(file, 'utf8')) as TrackDef;
if (!def.osm?.ways.length) throw new Error(`${id}.json n'a pas de champ osm.ways.`);
const { ways, main } = def.osm;
const QUERY = `[out:json][timeout:30];way(id:${ways.join(',')});out body geom;`;

async function fetchOverpass(): Promise<{ elements: unknown[] }> {
  let last: unknown;
  for (let attempt = 0; attempt < 3; attempt++) {
    for (const url of ENDPOINTS) {
      try {
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded', Accept: 'application/json', 'User-Agent': 'kart-lap-lab/2.0' },
          body: 'data=' + encodeURIComponent(QUERY),
          signal: AbortSignal.timeout(45_000),
        });
        if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
        return (await res.json()) as { elements: unknown[] };
      } catch (e) {
        last = e;
        console.warn(String(e));
      }
    }
    await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)));
  }
  throw last;
}

const json = await fetchOverpass();
for (const e of json.elements as { id: number; tags?: Record<string, string> }[]) console.log(`way ${e.id} :`, JSON.stringify(e.tags ?? {}));
let coords = stitchOverpass(json, ways, main);
if (process.argv.includes('--reverse')) coords = coords.reverse();
let maxDiff = 0;
const old = def.coords ?? [];
const same = coords.length === old.length;
if (same) coords.forEach((c, i) => (maxDiff = Math.max(maxDiff, Math.abs(c[0] - old[i][0]), Math.abs(c[1] - old[i][1]))));
console.log(`OSM : ${coords.length} points ; fichier : ${old.length} points ; écart max ${same ? maxDiff.toExponential(2) + '°' : 'n/a (nombre différent)'}`);
if (process.argv.includes('--write')) {
  def.coords = coords;
  writeFileSync(file, JSON.stringify(def, null, 1).replace(/\[\n\s+([-\d.]+),\n\s+([-\d.]+)\n\s+\]/g, '[$1,$2]'));
  console.log(`data/tracks/${id}.json mis à jour.`);
}
