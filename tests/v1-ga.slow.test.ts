// Full v1 regression: seeded GA over 3000 generations must land on ≈1:37 (390cc) and ≈1:44 (270cc).
import { expect, it } from 'vitest';
import lyl from '../data/tracks/lyl.json';
import { buildTrack } from '../src/track/build';
import type { TrackDef } from '../src/track/types';
import { kartParams } from '../src/sim/karts';
import { LineEvaluator } from '../src/sim/qss';
import { LineGA } from '../src/sim/lineGA';

const ev = new LineEvaluator(buildTrack(lyl as TrackDef), 16);

it.each([
  ['k390', 96.85],
  ['k270', 103.86],
] as const)('%s : tour théorique v1 ≈ %f s', (kart, want) => {
  const ga = new LineGA(ev, kartParams(kart), { popSize: 80, seed: 12345 });
  let best = Infinity;
  for (let i = 0; i < 3000; i++) best = ga.step().best;
  expect(Math.abs(best - want)).toBeLessThan(0.5);
});
