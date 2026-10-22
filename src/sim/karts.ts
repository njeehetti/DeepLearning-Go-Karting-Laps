import { G } from '../track/geometry';

export type KartId = 'k270' | 'k390';

export interface KartSpec {
  id: KartId;
  label: string;
  engine: string;
  hp: number;
  kartMass: number;
  /** Top speed, m/s. */
  vTop: number;
}

/** Rental Sodikart assumptions, shown in the UI. */
export const KARTS: Record<KartId, KartSpec> = {
  k270: { id: 'k270', label: '270cc', engine: 'Honda GX270', hp: 9, kartMass: 135, vTop: 62 / 3.6 },
  k390: { id: 'k390', label: '390cc', engine: 'Honda GX390', hp: 13, kartMass: 141, vTop: 76 / 3.6 },
};

export interface KartOpts {
  /** Tyre friction coefficient. */
  mu?: number;
  /** Share of the grip the driver exploits (0.8–1). */
  margin?: number;
  /** Driver mass, kg. */
  driver?: number;
}

/** Point-mass parameters of the v1 quasi-steady solver. */
export interface QssParams {
  id: KartId;
  m: number;
  P: number;
  vTop: number;
  muLat: number;
  aTrac: number;
  aBrk: number;
  CdA: number;
  rho: number;
  Crr: number;
}

/** v1 parameters, unchanged (regression target 1:37 / 1:44). Note aBrk is optimistic for rear-only brakes. */
export function kartParams(kartId: KartId, opts: KartOpts = {}): QssParams {
  const k = KARTS[kartId];
  const mu = opts.mu || 1.15;
  const margin = opts.margin == null ? 1 : opts.margin;
  return {
    id: k.id, m: k.kartMass + (opts.driver || 75), P: k.hp * 745.7 * 0.85, vTop: k.vTop,
    muLat: mu * margin, aTrac: 0.62 * mu * margin * G, aBrk: 0.72 * mu * margin * G,
    CdA: 0.75, rho: 1.16, Crr: 0.025,
  };
}
