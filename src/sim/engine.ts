// Rental kart powertrain: single-cylinder Honda GX, centrifugal clutch, fixed chain ratio, rev governor.
// Unlike v1's constant-power P/v, the wheel force follows the (fairly flat) engine torque, so a slow
// corner exit really is slow.
import type { KartId } from './karts';

export interface EngineSpec {
  /** Peak torque (N·m) and its rpm. */
  tPeak: number;
  rpmPeak: number;
  /** Torque at the governed rpm (rated power point). */
  tGov: number;
  /** Governor cut-in rpm (rated). */
  rpmGov: number;
  /** Clutch engagement rpm: below it the clutch slips and transmits about tClutch. */
  rpmClutch: number;
  /** Driveline efficiency (chain, bearings). */
  eta: number;
}

/** Honda data sheets: GX270 9 hp (6.7 kW) @ 3600, 19.1 N·m @ 2500; GX390 13 hp (9.6 kW) @ 3600, 26.5 N·m @ 2500. */
export const ENGINES: Record<KartId, EngineSpec> = {
  k270: { tPeak: 19.1, rpmPeak: 2500, tGov: 17.7, rpmGov: 3600, rpmClutch: 1900, eta: 0.85 },
  k390: { tPeak: 26.5, rpmPeak: 2500, tGov: 25.4, rpmGov: 3600, rpmClutch: 1900, eta: 0.85 },
};

/**
 * 'direct': centrifugal clutch + fixed chain ratio (force follows engine torque).
 * 'cvt': torque converter / variator keeping the engine near rated power (force ≈ P/v).
 */
export type Transmission = 'direct' | 'cvt';

export interface Powertrain {
  spec: EngineSpec;
  transmission: Transmission;
  /** Rear tyre rolling radius (m). */
  rWheel: number;
  /** Overall reduction engine → axle, chosen so that rpmGov is reached exactly at vTop. */
  ratio: number;
  vTop: number;
}

export function makePowertrain(spec: EngineSpec, vTop: number, rWheel = 0.14, transmission: Transmission = 'direct'): Powertrain {
  const omegaGov = (spec.rpmGov * 2 * Math.PI) / 60;
  return { spec, transmission, rWheel, vTop, ratio: (omegaGov * rWheel) / vTop };
}

/** Rated power at the wheel (W). */
export function wheelPower(spec: EngineSpec): number {
  return spec.tGov * ((spec.rpmGov * 2 * Math.PI) / 60) * spec.eta;
}

/** Engine torque at rpm with full throttle (parabola through peak and rated points). */
export function engineTorque(spec: EngineSpec, rpm: number): number {
  const span = spec.rpmGov - spec.rpmPeak;
  const k = (spec.tPeak - spec.tGov) / (span * span);
  const d = rpm - spec.rpmPeak;
  return Math.max(0, spec.tPeak - k * d * d);
}

/**
 * Tractive force at the rear contact patch (N) for road speed v (m/s) and throttle 0..1.
 * Throttle scales torque; the governor fades torque to zero over the last 3 % above rpmGov.
 */
export function driveForce(pt: Powertrain, v: number, throttle: number): number {
  if (throttle <= 0) return 0;
  const sp = pt.spec;
  if (pt.transmission === 'cvt') {
    // converter low ratio ≈ 2.5 × the high ratio caps the launch force
    const fMax = (sp.tPeak * pt.ratio * 2.5 * sp.eta) / pt.rWheel;
    let f = Math.min(fMax, wheelPower(sp) / Math.max(v, 0.5)) * throttle;
    const over = (v - pt.vTop) / (0.03 * pt.vTop);
    if (over > 0) f *= Math.max(0, 1 - over);
    return f;
  }
  const rpmWheel = (Math.max(v, 0) / pt.rWheel) * pt.ratio * (60 / (2 * Math.PI));
  // centrifugal clutch: engine sits at engagement rpm and slips until the wheels catch up
  const rpm = Math.max(rpmWheel, sp.rpmClutch);
  let t = engineTorque(sp, rpm) * throttle;
  const over = (rpm - sp.rpmGov) / (0.03 * sp.rpmGov);
  if (over > 0) t *= Math.max(0, 1 - over);
  return (t * pt.ratio * sp.eta) / pt.rWheel;
}
