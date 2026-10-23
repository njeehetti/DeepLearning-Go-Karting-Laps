// Calibration of the kart model on a known lap time. The brief's rental-kart figures (9 hp, 62 km/h,
// µ 1.15) cannot produce the lap times actually driven at LYL, so the model is fitted with "effective"
// values that absorb what it does not capture. One lap time fixes one degree of freedom: the mode says
// which parameter moves.
import type { Track } from '../track/types';
import type { Transmission } from './engine';
import { chassisParams, type ChassisParams } from './kart';
import { kartParams, type KartId } from './karts';
import { LineGA } from './lineGA';
import { LineEvaluator, lineLimits } from './qss';
import { makeQss2Ideal, solveQss2 } from './qss2';
import { TrackQuery } from '../track/query';
import { withWeather, type Weather } from './weather';

export type CalibrationMode =
  /** keep µ, fit engine power */
  | 'power'
  /** keep power, fit µ (the brief's original idea; may be unreachable) */
  | 'grip'
  /** move both by the same factor */
  | 'balanced';

export interface Calibration {
  kartId: KartId;
  mode: CalibrationMode;
  /** Lap time the theoretical (perfect) lap is fitted to, s. */
  targetLap: number;
  mu: number;
  powerScale: number;
  vTopKmh: number;
  transmission: Transmission;
  driver: number;
  /** Theoretical lap reached with these values, s (≈ targetLap unless unreachable). */
  lap: number;
  reachable: boolean;
}

export interface CalibrationInput {
  kartId: KartId;
  targetLap: number;
  mode?: CalibrationMode;
  /** Starting / fixed values. */
  mu?: number;
  powerScale?: number;
  vTopKmh: number;
  transmission?: Transmission;
  driver?: number;
  /** GA effort for the racing line (generations). */
  lineGens?: number;
}

/** Chassis parameters of a calibrated kart, in the given track conditions (dry by default). */
export function paramsOf(c: Pick<Calibration, 'kartId' | 'mu' | 'powerScale' | 'vTopKmh' | 'transmission' | 'driver'> & { weather?: Weather }): ChassisParams {
  return withWeather(chassisParams(c.kartId, { mu: c.mu, powerScale: c.powerScale, vTopKmh: c.vTopKmh, transmission: c.transmission, driver: c.driver }), c.weather);
}

/** Optimises a racing line for params p; returns the genes and the theoretical lap. */
export function theoreticalLap(track: Track, p: ChassisParams, gens = 400, seedGenes?: Float64Array | null): { genes: Float64Array; T: number } {
  const ev = new LineEvaluator(track, 16, lineLimits(track, new TrackQuery(track).kerb));
  const m = makeQss2Ideal(p);
  const ga = new LineGA(ev, kartParams(p.kartId), {
    popSize: 60,
    seed: 3,
    seedGenes: seedGenes ?? null,
    sigma: seedGenes ? 0.5 : 1.4,
    cost: (g) => {
      ev.lineFromGenes(g);
      return solveQss2(ev, m);
    },
  });
  for (let i = 0; i < gens; i++) ga.step();
  return { genes: Float64Array.from(ga.pop[0]), T: ga.fit[0] };
}

export function calibrate(track: Track, inp: CalibrationInput): Calibration {
  const mode = inp.mode ?? 'power';
  const base = {
    kartId: inp.kartId,
    mu: inp.mu ?? 1.3,
    powerScale: inp.powerScale ?? 1,
    vTopKmh: inp.vTopKmh,
    transmission: inp.transmission ?? ('cvt' as Transmission),
    driver: inp.driver ?? 75,
  };
  // x scales the free parameter(s)
  const at = (x: number) => ({
    ...base,
    mu: mode === 'power' ? base.mu : base.mu * (mode === 'balanced' ? Math.sqrt(x) : x),
    powerScale: mode === 'grip' ? base.powerScale : base.powerScale * (mode === 'balanced' ? Math.sqrt(x) : x),
  });
  const ev = new LineEvaluator(track, 16, lineLimits(track, new TrackQuery(track).kerb));
  const lapOn = (genes: Float64Array, x: number) => {
    ev.lineFromGenes(genes);
    return solveQss2(ev, makeQss2Ideal(paramsOf(at(x))));
  };
  const xMax = mode === 'grip' ? 3 : 4;
  let x = mode === 'grip' ? 1.5 : 2;
  let genes = theoreticalLap(track, paramsOf(at(x)), inp.lineGens ?? 400).genes;
  for (let pass = 0; pass < 2; pass++) {
    let lo = 0.6;
    let hi = xMax;
    if (lapOn(genes, hi) > inp.targetLap) {
      x = hi;
      break;
    }
    for (let i = 0; i < 16; i++) {
      const mid = (lo + hi) / 2;
      if (lapOn(genes, mid) > inp.targetLap) lo = mid;
      else hi = mid;
    }
    x = (lo + hi) / 2;
    // the best line depends a little on the kart: re-optimise it and refine once
    genes = theoreticalLap(track, paramsOf(at(x)), Math.round((inp.lineGens ?? 400) / 2), genes).genes;
  }
  const c = at(x);
  const lap = lapOn(genes, x);
  return { ...c, mode, targetLap: inp.targetLap, lap, reachable: Math.abs(lap - inp.targetLap) < 0.5 };
}
