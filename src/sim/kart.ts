// Dynamic kart model ("bicycle" front axle + two-wheel rigid rear axle), fixed step, deterministic,
// allocation-free in step(). Units SI, body frame x forward / y left, world x east / y north / z up.
//
// What it captures, because it decides lap time on a rental kart:
//  - simplified Pacejka lateral force per axle, relaxed slip angles (stable at low speed);
//  - rear-only drive and brake, per-wheel friction circle, rear lock-up → the rear lets go;
//  - rigid axle: with both rear wheels down, the inner one is forced to spin and the outer one to drag,
//    which costs speed and causes understeer; it stops once the inner rear lifts (lateral g plus caster
//    jacking from steering lock, the reason karts can turn at all);
//  - longitudinal load transfer, slope of the bridge ramps, grass and kerbs per wheel.
import { G } from '../track/geometry';
import { Proj, type TrackQuery } from '../track/query';
import { ENGINES, driveForce, makePowertrain, scaleEngine, type Powertrain, type Transmission } from './engine';
import { KARTS, type KartId } from './karts';

export const DT = 1 / 120;

export interface ChassisParams {
  kartId: KartId;
  m: number;
  /** Yaw inertia (kg·m²). */
  Iz: number;
  /** CG to front / rear axle (m). */
  a: number;
  b: number;
  /** Front / rear track width (m). */
  twF: number;
  twR: number;
  /** CG height (m). */
  h: number;
  /** Peak tyre friction on asphalt. */
  mu: number;
  /** Sliding / peak friction ratio (locked wheel). */
  muSlide: number;
  /** Pacejka shape. */
  B: number;
  C: number;
  E: number;
  /** Tyre relaxation length (m). */
  sigma: number;
  /** Longitudinal slip stiffness, per unit load (1/slip). */
  kSlip: number;
  /** Lateral g that alone lifts the inner rear wheel: (twR/2)/h for a rigid chassis. */
  liftG: number;
  /** Front wheel angle that alone lifts the inner rear wheel (caster jacking), rad. */
  liftSteer: number;
  CdA: number;
  rho: number;
  Crr: number;
  /** Grass: grip factor and rolling resistance coefficient. */
  grassGrip: number;
  grassCrr: number;
  kerbGrip: number;
  /** Max front wheel angle (rad) and actuator rates. */
  deltaMax: number;
  steerRate: number;
  pedalRate: number;
  /** Max rear brake force (N); above the tyre limit the rear locks. */
  brakeMax: number;
  pt: Powertrain;
}

export interface PhysOpts {
  mu?: number;
  driver?: number;
  /** Override top speed (km/h). */
  vTopKmh?: number;
  transmission?: Transmission;
  /** Multiplies engine torque (calibration of an effective power). */
  powerScale?: number;
}

export function chassisParams(kartId: KartId, opts: PhysOpts = {}): ChassisParams {
  const k = KARTS[kartId];
  const m = k.kartMass + (opts.driver ?? 75);
  const L = 1.05;
  const rearShare = 0.58;
  const mu = opts.mu ?? 1.15;
  const vTop = opts.vTopKmh ? opts.vTopKmh / 3.6 : k.vTop;
  return {
    kartId, m, Iz: 0.18 * m, a: L * rearShare, b: L * (1 - rearShare), twF: 1.0, twR: 1.18, h: 0.28,
    mu, muSlide: 0.8, B: 12, C: 1.55, E: 0, sigma: 0.3, kSlip: 12, liftG: 1.18 / 2 / 0.28, liftSteer: 0.2,
    CdA: 0.75, rho: 1.16, Crr: 0.025, grassGrip: 0.5, grassCrr: 0.15, kerbGrip: 0.9,
    deltaMax: 0.4, steerRate: 3, pedalRate: 8,
    brakeMax: 1.6 * mu * m * G * rearShare, pt: makePowertrain(scaleEngine(ENGINES[kartId], opts.powerScale ?? 1), vTop, 0.14, opts.transmission),
  };
}

/** Driver commands. */
export interface KartInput {
  /** −1 (right) .. 1 (left). */
  steer: number;
  throttle: number;
  brake: number;
}

/** Brake demand above which the foot is on the brake pedal and off the throttle. */
export const BRAKE_FOOT = 0.03;

export class KartState {
  x = 0;
  y = 0;
  z = 0;
  /** Heading (rad, world, 0 = east). */
  psi = 0;
  vx = 0;
  vy = 0;
  /** Yaw rate (rad/s). */
  r = 0;
  alphaF = 0;
  alphaR = 0;
  /** Actuator positions. */
  delta = 0;
  thr = 0;
  brk = 0;
  /** Body-frame accelerations of the last step (m/s²). */
  ax = 0;
  ay = 0;
  /** Track progress: centreline index hint, distance s, lateral offset d. */
  idx = 0;
  s = 0;
  d = 0;
  /** Laps completed and unwrapped distance since the start. */
  lap = 0;
  dist = 0;
  /** Rear axle peripheral speed (m/s), sets engine rpm. */
  vAxle = 0;
  /** Number of wheels on grass, rear locked, inner rear unloading (1 = in the air). */
  wheelsOff = 0;
  locked = false;
  lift = 0;
  /** Power lost to rigid-axle scrub (W) at the last step. */
  scrubPower = 0;
  /** Forces of the last step (N): rear longitudinal total, front / rear lateral, resistances. */
  fxRear = 0;
  fyFront = 0;
  fyRear = 0;
  fRes = 0;
  /** Engine force requested before tyre limits (N). */
  fDrive = 0;
  crashed = false;
  t = 0;

  copyFrom(o: KartState): this {
    Object.assign(this, o);
    return this;
  }
}

const WX = new Float64Array(4);
const WY = new Float64Array(4);

/** Pacejka magic formula, normalised (peak = 1). */
function pac(B: number, C: number, E: number, alpha: number): number {
  const ba = B * alpha;
  return Math.sin(C * Math.atan(ba - E * (ba - Math.atan(ba))));
}

export class KartSim {
  readonly p: ChassisParams;
  readonly q: TrackQuery;
  private readonly proj = new Proj();
  /** Surface grip / rolling resistance for each wheel (FL, FR, RL, RR) of the last step. */
  private readonly grip = new Float64Array(4);
  private readonly crr = new Float64Array(4);
  /** Wheel positions in the body frame (FL, FR, RL, RR). */
  private readonly wheelX: Float64Array;
  private readonly wheelY: Float64Array;

  constructor(q: TrackQuery, p: ChassisParams) {
    this.q = q;
    this.p = p;
    this.wheelX = Float64Array.of(p.a, p.a, -p.b, -p.b);
    this.wheelY = Float64Array.of(p.twF / 2, -p.twF / 2, p.twR / 2, -p.twR / 2);
  }

  /** Places the kart on the centreline at distance s, offset d, heading along the track, speed v. */
  reset(st: KartState, s: number, d = 0, v = 0): KartState {
    const q = this.q;
    const t = q.track;
    const i = q.indexAt(s);
    const nx = t.nx[i];
    const ny = t.ny[i];
    Object.assign(st, new KartState());
    st.x = q.at(t.x, s) + nx * d;
    st.y = q.at(t.y, s) + ny * d;
    st.psi = Math.atan2(t.ty[i], t.tx[i]);
    st.vx = v;
    st.idx = i;
    this.locate(st);
    st.dist = 0;
    return st;
  }

  /** Updates idx, s, d, z, lap counting and wheel surfaces. */
  private locate(st: KartState): void {
    const q = this.q;
    const t = q.track;
    const p = this.p;
    const pr = this.proj;
    const prevS = st.s;
    q.project(st.x, st.y, st.idx, pr, 4);
    st.idx = pr.i;
    st.s = pr.s;
    st.d = pr.d;
    st.z = q.at(t.z, pr.s);
    let ds = pr.s - prevS;
    if (ds < -q.L / 2) {
      ds += q.L;
      st.lap++;
    } else if (ds > q.L / 2) {
      ds -= q.L;
      st.lap--;
    }
    st.dist += ds;
    // wheels
    const c = Math.cos(st.psi);
    const sn = Math.sin(st.psi);
    const lx = this.wheelX;
    const ly = this.wheelY;
    let off = 0;
    let crash = false;
    for (let w = 0; w < 4; w++) {
      WX[w] = st.x + c * lx[w] - sn * ly[w];
      WY[w] = st.y + sn * lx[w] + c * ly[w];
      q.project(WX[w], WY[w], st.idx, pr, 3);
      const out = Math.abs(pr.d) - t.half;
      if (out <= 0) {
        this.grip[w] = 1;
        this.crr[w] = p.Crr;
      } else if (out <= q.kerb[pr.i]) {
        this.grip[w] = p.kerbGrip;
        this.crr[w] = p.Crr * 2;
      } else {
        this.grip[w] = p.grassGrip;
        this.crr[w] = p.grassCrr;
        off++;
        if (out >= q.wall[pr.i]) crash = true;
      }
    }
    st.wheelsOff = off;
    st.crashed = st.crashed || crash;
  }

  step(st: KartState, inp: KartInput, dt = DT): void {
    const p = this.p;
    const q = this.q;
    // actuators
    const steerT = Math.max(-1, Math.min(1, inp.steer)) * p.deltaMax;
    const dMax = p.steerRate * dt;
    st.delta += Math.max(-dMax, Math.min(dMax, steerT - st.delta));
    const pMax = p.pedalRate * dt;
    // one right foot: a brake demand lifts it off the throttle (the golden rule of kart driving)
    const brkCmd = Math.max(0, Math.min(1, inp.brake));
    const thrCmd = brkCmd > BRAKE_FOOT ? 0 : Math.max(0, Math.min(1, inp.throttle));
    st.thr += Math.max(-pMax, Math.min(pMax, thrCmd - st.thr));
    st.brk += Math.max(-pMax, Math.min(pMax, brkCmd - st.brk));

    const { m, a, b, h } = p;
    const L = a + b;
    const vx = st.vx;
    const vy = st.vy;
    const r = st.r;
    const speed = Math.hypot(vx, vy);
    const cosD = Math.cos(st.delta);
    const sinD = Math.sin(st.delta);

    // vertical loads (longitudinal transfer from the last step's acceleration)
    const dFzLong = (m * st.ax * h) / L;
    const FzF = Math.max(0, (m * G * b) / L - dFzLong);
    const FzR = Math.max(0, (m * G * a) / L + dFzLong);
    // grip of each axle from its wheels' surfaces
    const gF = (this.grip[0] + this.grip[1]) / 2;
    const muF = p.mu * gF;
    // rear: lateral transfer + jacking, inner wheel lifts
    const liftRatio = Math.abs(st.ay) / (p.liftG * G) + Math.abs(st.delta) / p.liftSteer;
    const transfer = Math.min(1, liftRatio);
    st.lift = transfer;
    const turnLeft = st.ay >= 0; // centripetal acceleration to the left → left wheels are inner
    const FzIn = (FzR / 2) * (1 - transfer);
    const FzOut = (FzR / 2) * (1 + transfer);
    const FzL = turnLeft ? FzIn : FzOut;
    const FzRr = turnLeft ? FzOut : FzIn;

    // relaxed slip angles
    const vxs = Math.max(Math.abs(vx), 0.5);
    const alphaFt = Math.atan2(vy + a * r, vxs) - st.delta;
    const alphaRt = Math.atan2(vy - b * r, vxs);
    const relax = 1 - Math.exp((-Math.max(speed, 0.5) * dt) / p.sigma);
    st.alphaF += (alphaFt - st.alphaF) * relax;
    st.alphaR += (alphaRt - st.alphaR) * relax;

    // front axle: lateral only
    const FyF = -pac(p.B, p.C, p.E, st.alphaF) * muF * FzF;

    // rear axle: one rigid axle, both wheels share its speed vAxle. The axle settles where the two
    // wheel forces add up to the engine/brake demand, so the light inner wheel slips a lot and the
    // loaded outer one barely: the cost of the rigid axle is the yaw moment and the inner wheel's grip.
    const FyRtot = -pac(p.B, p.C, p.E, st.alphaR) * p.mu * FzR;
    const fDrive = driveForce(p.pt, Math.max(st.vAxle, vx), st.thr);
    const fBrake = st.brk * p.brakeMax;
    const dir = vx >= 0 ? 1 : -1;
    const half = p.twR / 2;
    const vL = vx - r * half;
    const vR = vx + r * half;
    const capL = p.mu * this.grip[2] * FzL;
    const capR = p.mu * this.grip[3] * FzRr;
    const shareL = FzR > 0 ? FzL / FzR : 0.5;
    let FyL = FyRtot * shareL * this.grip[2];
    let FyR = FyRtot * (1 - shareL) * this.grip[3];
    let FxL: number;
    let FxR: number;
    let vA: number;
    const locked = fBrake > fDrive && fBrake > capL + capR && Math.abs(vx) > 0.5;
    if (locked) {
      // axle locked: each wheel slides against its own velocity
      vA = 0;
      const wvy = vy - b * r;
      const nL = Math.hypot(vL, wvy) || 1;
      const nR = Math.hypot(vR, wvy) || 1;
      FxL = (-p.muSlide * capL * vL) / nL;
      FyL = (-p.muSlide * capL * wvy) / nL;
      FxR = (-p.muSlide * capR * vR) / nR;
      FyR = (-p.muSlide * capR * wvy) / nR;
    } else {
      const fDem = fDrive - fBrake * dir;
      const vref = Math.max(Math.abs(vx), 2);
      const kL = (p.kSlip * FzL) / vref;
      const kR = (p.kSlip * FzRr) / vref;
      vA = kL + kR > 0 ? (fDem + kL * vL + kR * vR) / (kL + kR) : vx;
      FxL = kL * (vA - vL);
      FxR = kR * (vA - vR);
      if (Math.abs(FxL) > capL) {
        FxL = Math.sign(FxL) * capL;
        FxR = Math.max(-capR, Math.min(capR, fDem - FxL));
        vA = kR > 0 && Math.abs(FxR) < capR ? vR + FxR / kR : vx + Math.sign(fDem) * 0.1 * vref;
      } else if (Math.abs(FxR) > capR) {
        FxR = Math.sign(FxR) * capR;
        FxL = Math.max(-capL, Math.min(capL, fDem - FxR));
        vA = kL > 0 && Math.abs(FxL) < capL ? vL + FxL / kL : vx + Math.sign(fDem) * 0.1 * vref;
      }
      // friction circle per wheel
      let mag = Math.hypot(FxL, FyL);
      if (mag > capL && mag > 0) {
        FxL *= capL / mag;
        FyL *= capL / mag;
      }
      mag = Math.hypot(FxR, FyR);
      if (mag > capR && mag > 0) {
        FxR *= capR / mag;
        FyR *= capR / mag;
      }
    }
    st.vAxle = vA;
    st.locked = locked;
    st.scrubPower = Math.abs(FxL * (vA - vL)) + Math.abs(FxR * (vA - vR));
    st.fxRear = FxL + FxR;
    st.fyFront = FyF;
    st.fyRear = FyL + FyR;
    st.fDrive = fDrive;

    // resistances, against the velocity
    const crrAvg = (this.crr[0] + this.crr[1] + this.crr[2] + this.crr[3]) / 4;
    const fRes = 0.5 * p.rho * p.CdA * speed * speed + crrAvg * m * G * Math.tanh(speed / 0.3);
    const ux = speed > 1e-6 ? vx / speed : 0;
    const uy = speed > 1e-6 ? vy / speed : 0;
    // slope along the kart's heading
    const ti = st.idx;
    const t = q.track;
    const along = Math.cos(st.psi) * t.tx[ti] + Math.sin(st.psi) * t.ty[ti];
    const fGrade = m * G * q.at(t.grade, st.s) * along;
    st.fRes = fRes;

    const Fx = FxL + FxR - FyF * sinD - fRes * ux - fGrade;
    const Fy = FyL + FyR + FyF * cosD - fRes * uy;
    const Mz = a * FyF * cosD - b * (FyL + FyR) + (p.twR / 2) * (FxR - FxL);

    st.ax = Fx / m;
    st.ay = Fy / m;
    st.vx += (st.ax + vy * r) * dt;
    st.vy += (st.ay - vx * r) * dt;
    st.r += (Mz / p.Iz) * dt;
    // standing still: kill creeping instead of oscillating around zero
    if (speed < 0.05 && st.thr < 0.02) {
      st.vx *= 0.5;
      st.vy *= 0.5;
      st.r *= 0.5;
    }
    st.psi += st.r * dt;
    const c = Math.cos(st.psi);
    const sn = Math.sin(st.psi);
    st.x += (st.vx * c - st.vy * sn) * dt;
    st.y += (st.vx * sn + st.vy * c) * dt;
    st.t += dt;
    this.locate(st);
  }
}
