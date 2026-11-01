// Default calibration (2026-10-30), computed with src/sim/calibrate.ts on the LYL at official scale.
// Evidence: driver's best 270cc 1:36.27; a faster driver 1:33.33 (38.278 / 25.507 / 29.547) at 57.5 km/h
// average, i.e. a timed lap of ≈ 1 491 m; public videos down to 1:32.2. The perfect lap is therefore set
// to 1:31.5. Mode 'power': µ kept at 1.15, engine power fitted (×1.35, ≈ 12.2 hp effective) against the
// ideal theoretical bound (full grip, 0.6 µg braking, engine-limited acceleration, kerbs usable), which a
// trained AI approaches from above. Transmission and top
// speed are unknown: 'cvt' and 80 km/h are assumptions shown in the UI.
// Why µ 1.15 and not 1.3: at 1.3 g of lateral grip the trained drivers barely brake (≈ 4 % of the lap) and
// shed speed by scrubbing sideways through the hairpins; at 1.15 they brake before every slow corner
// (≈ 15 %), as rental drivers do, for a similar lap time. The extra effective power also absorbs what the
// model simplifies (drag, converter losses).
import type { Calibration } from './calibrate';

export const DEFAULT_CALIBRATION: Record<'k270' | 'k390', Calibration> = {
  k270: {
    kartId: 'k270', mode: 'power', targetLap: 91.5, mu: 1.15, powerScale: 1.352, vTopKmh: 80,
    transmission: 'cvt', driver: 75, lap: 91.28, reachable: true,
  },
  // no 390cc data yet: same tyres and power factor, top speed scaled from the 270cc assumption
  k390: {
    kartId: 'k390', mode: 'power', targetLap: NaN, mu: 1.15, powerScale: 1.352, vTopKmh: 90,
    transmission: 'cvt', driver: 75, lap: NaN, reachable: false,
  },
};

/** Evidence the calibration is based on, for display. */
export const LAP_EVIDENCE = {
  driverBest270: 96.27,
  fastLap: { T: 93.332, sectors: [38.278, 25.507, 29.547], avgKmh: 57.5 },
  publicBest270: 92.195,
};
