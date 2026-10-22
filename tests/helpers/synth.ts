// Synthetic test circuits (equator, so 1° ≈ 111.195 km in both axes).
import type { TrackDef } from '../../src/track/types';

const M_PER_DEG = (6371000 * Math.PI) / 180;

function toDef(id: string, pts: [number, number][], width: number): TrackDef {
  return { id, name: id, width, coords: pts.map(([x, y]) => [y / M_PER_DEG, x / M_PER_DEG]), startFrac: 0 };
}

/** Stadium driven anticlockwise: two straights of `straight` m joined by half-circles of radius R. */
export function stadium(straight = 900, R = 80, width = 12): TrackDef {
  const pts: [number, number][] = [];
  const step = 2;
  for (let x = 0; x < straight; x += step) pts.push([x, -R]);
  for (let a = -Math.PI / 2; a < Math.PI / 2; a += step / R) pts.push([straight + R * Math.cos(a), R * Math.sin(a)]);
  for (let x = straight; x > 0; x -= step) pts.push([x, R]);
  for (let a = Math.PI / 2; a < (3 * Math.PI) / 2; a += step / R) pts.push([R * Math.cos(a), R * Math.sin(a)]);
  return toDef('stadium', pts, width);
}

/** Circle of radius R driven anticlockwise (left turn). */
export function circle(R = 25, width = 10): TrackDef {
  const pts: [number, number][] = [];
  const n = Math.round((2 * Math.PI * R) / 1);
  for (let k = 0; k < n; k++) {
    const a = (k / n) * 2 * Math.PI;
    pts.push([R * Math.cos(a), R * Math.sin(a)]);
  }
  return toDef('circle', pts, width);
}
