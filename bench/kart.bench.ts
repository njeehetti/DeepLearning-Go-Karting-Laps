// Steps per second of the dynamic model on the LYL (npx tsx bench/kart.bench.ts).
import lyl from '../data/tracks/lyl.json';
import { buildTrack } from '../src/track/build';
import { TrackQuery } from '../src/track/query';
import type { TrackDef } from '../src/track/types';
import { KartSim, KartState, chassisParams } from '../src/sim/kart';
import { PursuitDriver, constantLine } from '../src/ai/pursuit';

const q = new TrackQuery(buildTrack(lyl as TrackDef, { scale: 'official' }));
const sim = new KartSim(q, chassisParams('k270'));
const st = sim.reset(new KartState(), 20, 0, 8);
const drv = new PursuitDriver(q, constantLine(q.N, 0, 9));
const inp = { steer: 0, throttle: 0, brake: 0 };
const n = 1_000_000;
const t0 = performance.now();
for (let k = 0; k < n; k++) {
  if (k % 4 === 0) Object.assign(inp, drv.drive(sim, st));
  sim.step(st, inp);
  if (st.crashed) sim.reset(st, st.s, 0, 8);
}
const us = ((performance.now() - t0) / n) * 1000;
console.log(`pas physique + pilote à 30 Hz : ${us.toFixed(3)} µs → ${(1 / us).toFixed(2)} M pas/s par cœur ; tours ${st.lap}`);
