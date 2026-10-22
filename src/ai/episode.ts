// One driving episode of a neural pilot: physics at 120 Hz, decisions at 30 Hz, termination rules and
// fitness. DOM-free, allocation-free per step.
import { DT, KartState, type KartInput, type KartSim } from '../sim/kart';
import type { TrackQuery } from '../track/query';
import { Mlp, controlsFromOutput } from './mlp';
import { N_OBS, observe } from './obs';

export const DECISION_EVERY = 4; // 120 Hz / 4 = 30 Hz
/** Hands-and-wheel inertia: first-order filter on the steering command, τ = 0.1 s at 30 Hz. */
const STEER_ALPHA = 1 - Math.exp(-1 / 30 / 0.1);

export interface EpisodeSpec {
  /** Start position along the track (m), lateral offset and rolling speed (m/s). */
  s0: number;
  d0: number;
  v0: number;
  /** Hard time cap (s). */
  maxTime: number;
  /** Weight of the lap time in the fitness (curriculum: 0 = just finish the lap). */
  timeWeight: number;
}

export type EndReason = 'tour' | 'mur' | 'herbe' | 'immobile' | 'contresens' | 'temps';

export interface EpisodeResult {
  fitness: number;
  /** Distance covered along the centreline (m). */
  progress: number;
  completed: boolean;
  /** Completed without any wheel on the grass. */
  clean: boolean;
  lapTime: number;
  /** Wheel-seconds on the grass. */
  offTime: number;
  /** Mean |change of steering command| per second (smooth driving ≈ 0.5, sawing > 10). */
  steerActivity: number;
  end: EndReason;
}

/** Reference time used to turn lap times into fitness (any value above a realistic lap). */
const T_REF = 150;
const LAP_BONUS = 300;
/** Fitness cost per unit of steering command change: sawing the wheel is slow and unrealistic. */
const STEER_COST = 1;
/** Same for the pedals (dithering the throttle or brake), lighter: modulating them is legitimate. */
const PEDAL_COST = 0.3;

export interface StepHook {
  (st: KartState, inp: KartInput, decided: boolean): void;
}

export class Pilot {
  readonly mlp = new Mlp();
  readonly obs = new Float64Array(N_OBS);
  readonly rays = new Float64Array(9);
  readonly inp: KartInput = { steer: 0, throttle: 0, brake: 0 };
  private steer = 0;

  reset(): void {
    this.steer = 0;
  }

  decide(q: TrackQuery, st: KartState, params: ArrayLike<number>, deltaMax = 0.4): KartInput {
    observe(q, st, this.obs, this.rays, deltaMax);
    controlsFromOutput(this.mlp.forward(params, this.obs), this.inp);
    this.steer += (this.inp.steer - this.steer) * STEER_ALPHA;
    this.inp.steer = this.steer;
    return this.inp;
  }
}

export function runEpisode(sim: KartSim, pilot: Pilot, params: ArrayLike<number>, spec: EpisodeSpec, st = new KartState(), hook?: StepHook): EpisodeResult {
  const q = sim.q;
  const L = q.L;
  sim.reset(st, spec.s0, spec.d0, spec.v0);
  pilot.reset();
  let best = 0;
  let bestAt = 0;
  let allOff = 0;
  let offTime = 0;
  let everOff = false;
  let end: EndReason = 'temps';
  let lapTime = NaN;
  let steerSum = 0;
  let prevSteer = 0;
  let pedalSum = 0;
  let prevThr = 0;
  let prevBrk = 0;
  const maxSteps = Math.round(spec.maxTime / DT);
  for (let k = 0; k < maxSteps; k++) {
    const decided = k % DECISION_EVERY === 0;
    if (decided) {
      pilot.decide(q, st, params, sim.p.deltaMax);
      if (k > 0) {
        steerSum += Math.abs(pilot.inp.steer - prevSteer);
        pedalSum += Math.abs(pilot.inp.throttle - prevThr) + Math.abs(pilot.inp.brake - prevBrk);
      }
      prevSteer = pilot.inp.steer;
      prevThr = pilot.inp.throttle;
      prevBrk = pilot.inp.brake;
    }
    const prevDist = st.dist;
    sim.step(st, pilot.inp);
    hook?.(st, pilot.inp, decided);
    if (st.wheelsOff > 0) {
      offTime += st.wheelsOff * DT;
      everOff = true;
    }
    if (st.dist >= L) {
      // interpolate the crossing of the lap line inside the last step
      const f = (L - prevDist) / Math.max(1e-9, st.dist - prevDist);
      lapTime = st.t - DT + f * DT;
      end = 'tour';
      break;
    }
    if (st.crashed) {
      end = 'mur';
      break;
    }
    allOff = st.wheelsOff === 4 ? allOff + DT : 0;
    if (allOff > 0.5) {
      end = 'herbe';
      break;
    }
    if (st.dist > best + 2) {
      best = st.dist;
      bestAt = st.t;
    } else if (st.t - bestAt > 3) {
      end = 'immobile';
      break;
    }
    if (st.dist < best - 10) {
      end = 'contresens';
      break;
    }
  }
  const completed = end === 'tour';
  const progress = completed ? L : Math.max(0, Math.min(L, Math.max(best, st.dist)));
  let fitness = progress - 3 * offTime - STEER_COST * steerSum - PEDAL_COST * pedalSum;
  if (completed) fitness += LAP_BONUS + spec.timeWeight * 10 * (T_REF - lapTime);
  else if (end === 'mur') fitness -= 20;
  return { fitness, progress, completed, clean: completed && !everOff, lapTime, offTime, end, steerActivity: steerSum / Math.max(st.t, 1e-3) };
}

