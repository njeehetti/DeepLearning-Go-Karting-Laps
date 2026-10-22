// Validation gate: the scripted driver following the "tour théorique v2" line and speed profile in the
// dynamic model must be slower than the theoretical lap (it is an upper bound) but by no more than 10 %.
import { expect, it } from 'vitest';
import lyl from '../data/tracks/lyl.json';
import { buildTrack } from '../src/track/build';
import { TrackQuery } from '../src/track/query';
import type { TrackDef } from '../src/track/types';
import { KartSim, KartState, chassisParams } from '../src/sim/kart';
import { LineEvaluator, lineLimits } from '../src/sim/qss';
import { LineGA } from '../src/sim/lineGA';
import { kartParams } from '../src/sim/karts';
import { makeQss2, solveQss2 } from '../src/sim/qss2';
import { PursuitDriver } from '../src/ai/pursuit';

it('le pilote scripté boucle de +3 % à +10 % au-dessus du tour théorique v2', () => {
  const track = buildTrack(lyl as TrackDef, { scale: 'official' });
  const q = new TrackQuery(track);
  const p = chassisParams('k270', { mu: 1.45 });
  const m = makeQss2(p);
  const ev = new LineEvaluator(track, 16, lineLimits(track));
  const ga = new LineGA(ev, kartParams('k270'), {
    popSize: 60,
    seed: 3,
    cost: (g) => {
      ev.lineFromGenes(g);
      return solveQss2(ev, m);
    },
  });
  for (let i = 0; i < 400; i++) ga.step();
  ev.lineFromGenes(ga.pop[0]);
  const T = solveQss2(ev, m);

  const sim = new KartSim(q, p);
  const st = sim.reset(new KartState(), 0, ev.d[0], ev.v[0]);
  const drv = new PursuitDriver(q, { d: Float64Array.from(ev.d), v: Float64Array.from(ev.v) });
  const inp = { steer: 0, throttle: 0, brake: 0 };
  let t1 = -1;
  let t2 = -1;
  for (let k = 0; k < 120 * 300 && !st.crashed; k++) {
    if (k % 4 === 0) Object.assign(inp, drv.drive(sim, st));
    sim.step(st, inp);
    if (st.lap === 1 && t1 < 0) t1 = st.t;
    if (st.lap >= 2) {
      t2 = st.t;
      break;
    }
  }
  expect(st.crashed).toBe(false);
  const ratio = (t2 - t1) / T;
  expect(ratio).toBeGreaterThan(1.03);
  expect(ratio).toBeLessThan(1.1);
}, 120_000);
