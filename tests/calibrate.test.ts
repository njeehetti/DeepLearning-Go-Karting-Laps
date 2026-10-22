import { describe, expect, it } from 'vitest';
import lyl from '../data/tracks/lyl.json';
import { buildTrack } from '../src/track/build';
import type { TrackDef } from '../src/track/types';
import { calibrate, paramsOf, theoreticalLap } from '../src/sim/calibrate';
import { DEFAULT_CALIBRATION } from '../src/sim/defaults';

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
