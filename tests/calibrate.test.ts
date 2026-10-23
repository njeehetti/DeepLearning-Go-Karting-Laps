import { describe, expect, it } from 'vitest';
import lyl from '../data/tracks/lyl.json';
import { buildTrack } from '../src/track/build';
import type { TrackDef } from '../src/track/types';
import { calibrate, paramsOf, theoreticalLap } from '../src/sim/calibrate';
import { DEFAULT_CALIBRATION } from '../src/sim/defaults';
import { WEATHER_IDS } from '../src/sim/weather';

const track = buildTrack(lyl as TrackDef, { scale: 'official' });

describe('calage', () => {
  it('atteint la cible en ajustant la puissance', () => {
    const c = calibrate(track, { kartId: 'k270', targetLap: 90, mode: 'power', mu: 1.3, vTopKmh: 80, lineGens: 120 });
    expect(c.reachable).toBe(true);
    expect(Math.abs(c.lap - 90)).toBeLessThan(0.5);
    expect(c.mu).toBe(1.3);
    expect(c.powerScale).toBeGreaterThan(1);
  }, 60_000);

  it('demande plus d’adhérence pour un tour plus rapide (mode adhérence)', () => {
    const slow = calibrate(track, { kartId: 'k270', targetLap: 98, mode: 'grip', vTopKmh: 80, lineGens: 120 });
    const fast = calibrate(track, { kartId: 'k270', targetLap: 94, mode: 'grip', vTopKmh: 80, lineGens: 120 });
    expect(fast.mu).toBeGreaterThan(slow.mu);
    expect(fast.powerScale).toBe(1);
  }, 120_000);

  it('signale une cible inatteignable', () => {
    const c = calibrate(track, { kartId: 'k270', targetLap: 60, mode: 'grip', vTopKmh: 62, lineGens: 60 });
    expect(c.reachable).toBe(false);
  }, 60_000);

  it('la calibration par défaut du 270cc redonne ≈ 1:31.5', () => {
    const { T } = theoreticalLap(track, paramsOf(DEFAULT_CALIBRATION.k270), 400);
    expect(Math.abs(T - 91.5)).toBeLessThan(1);
  }, 60_000);
});

describe('météo', () => {
  it('ralentit le tour parfait dans l’ordre sec < piste grasse < humide < pluie, et coupe l’adhérence des vibreurs', () => {
    const c = DEFAULT_CALIBRATION.k270;
    const laps = WEATHER_IDS.map((weather) => theoreticalLap(track, paramsOf({ ...c, weather }), 120).T);
    for (let k = 1; k < laps.length; k++) expect(laps[k]).toBeGreaterThan(laps[k - 1] + 1);
    // rain on slicks: a lap time in the range seen in rental karts in the wet (+15 % to +35 %)
    expect(laps[3] / laps[0]).toBeGreaterThan(1.15);
    expect(laps[3] / laps[0]).toBeLessThan(1.35);
    const dry = paramsOf(c);
    const wet = paramsOf({ ...c, weather: 'pluie' });
    expect(wet.kerbGrip / dry.kerbGrip).toBeLessThan(wet.mu / dry.mu);
    expect(wet.brakeMax).toBe(dry.brakeMax);
  }, 120_000);
});
