// Scripted reference driver: pure pursuit on a racing line + speed tracking with brake anticipation.
// Used to validate the physics against the theoretical lap and to bootstrap the AI ("mode assisté").
import type { TrackQuery } from '../track/query';
import type { KartInput, KartSim, KartState } from '../sim/kart';

export interface PursuitLine {
  /** Lateral offset (m) and target speed (m/s) per centreline point. */
  d: Float64Array;
  v: Float64Array;
}

export interface PursuitOpts {
  /** Lookahead = l0 + k·v (m). */
  l0?: number;
  kv?: number;
  /** Speed loop gain (1/(m/s)) and how far ahead the speed target is read (s). */
  kSpeed?: number;
  tAhead?: number;
}

export class PursuitDriver {
  private readonly q: TrackQuery;
  private readonly line: PursuitLine;
  private readonly l0: number;
  private readonly kv: number;
  private readonly kSpeed: number;
  private readonly tAhead: number;
  readonly out: KartInput = { steer: 0, throttle: 0, brake: 0 };

  constructor(q: TrackQuery, line: PursuitLine, opts: PursuitOpts = {}) {
    this.q = q;
    this.line = line;
    this.l0 = opts.l0 ?? 2.5;
    this.kv = opts.kv ?? 0.35;
    this.kSpeed = opts.kSpeed ?? 0.6;
    this.tAhead = opts.tAhead ?? 0.25;
  }

  /** Target point of the line at distance s. */
  private point(s: number): [number, number] {
    const q = this.q;
    const t = q.track;
    const i = q.indexAt(s);
    const d = q.at(this.line.d, s);
    return [q.at(t.x, s) + t.nx[i] * d, q.at(t.y, s) + t.ny[i] * d];
  }

  drive(sim: KartSim, st: KartState): KartInput {
    const v = Math.max(0, st.vx);
    // steering: pure pursuit towards the line ld metres ahead
    const ld = this.l0 + this.kv * v;
    const [tx, ty] = this.point(st.s + ld);
    const dx = tx - st.x;
    const dy = ty - st.y;
    const alpha = Math.atan2(dy, dx) - st.psi;
    const dist = Math.hypot(dx, dy) || 1;
    const L = sim.p.a + sim.p.b;
    const delta = Math.atan2(2 * L * Math.sin(alpha), dist);
    this.out.steer = Math.max(-1, Math.min(1, delta / sim.p.deltaMax));
    // speed: aim at the profile a bit ahead so braking starts on time
    const vT = this.q.at(this.line.v, st.s + v * this.tAhead);
    const e = vT - v;
    const u = e * this.kSpeed;
    this.out.throttle = u > 0 ? Math.min(1, u + 0.2) : 0;
    this.out.brake = u < -0.15 ? Math.min(1, -u * 0.5) : 0;
    return this.out;
  }
}

/** Constant-offset, constant-speed line (tests, skidpad). */
export function constantLine(N: number, d: number, v: number): PursuitLine {
  return { d: new Float64Array(N).fill(d), v: new Float64Array(N).fill(v) };
}
