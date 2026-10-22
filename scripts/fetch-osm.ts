// Fetches the LYL loop from OpenStreetMap (Overpass) and compares it with data/tracks/lyl.json.
// Usage: npx tsx scripts/fetch-osm.ts [--write]
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { stitchOverpass } from '../src/track/import';
import type { TrackDef } from '../src/track/types';

const MAIN_WAY = 1388683501;
const BRIDGE_WAY = 1388688194;
const ENDPOINTS = ['https://overpass-api.de/api/interpreter', 'https://overpass.private.coffee/api/interpreter', 'https://maps.mail.ru/osm/tools/overpass/api/interpreter'];
const QUERY = `[out:json][timeout:30];(way(${MAIN_WAY});way(${BRIDGE_WAY}););out body geom;`;
const file = fileURLToPath(new URL('../data/tracks/lyl.json', import.meta.url));

async function fetchOverpass(): Promise<{ elements: unknown[] }> {
  let last: unknown;
  for (let attempt = 0; attempt < 3; attempt++) {
    for (const url of ENDPOINTS) {
      try {
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded', Accept: 'application/json', 'User-Agent': 'kart-lap-lab/2.0' },
          body: 'data=' + encodeURIComponent(QUERY),
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
const bridge = json.elements.find((e) => (e as { id: number }).id === BRIDGE_WAY) as { tags?: Record<string, string> } | undefined;
console.log('Pont :', JSON.stringify(bridge?.tags ?? {}));
const coords = stitchOverpass(json, [MAIN_WAY, BRIDGE_WAY], MAIN_WAY);
const def = JSON.parse(readFileSync(file, 'utf8')) as TrackDef;
let maxDiff = 0;
const same = coords.length === def.coords.length;
if (same) coords.forEach((c, i) => (maxDiff = Math.max(maxDiff, Math.abs(c[0] - def.coords[i][0]), Math.abs(c[1] - def.coords[i][1]))));
console.log(`OSM : ${coords.length} points ; fichier : ${def.coords.length} points ; écart max ${same ? maxDiff.toExponential(2) + '°' : 'n/a (nombre différent)'}`);
if (process.argv.includes('--write')) {
  def.coords = coords;
  writeFileSync(file, JSON.stringify(def, null, 1).replace(/\[\n\s+([-\d.]+),\n\s+([-\d.]+)\n\s+\]/g, '[$1,$2]'));
  console.log('data/tracks/lyl.json mis à jour.');
}
