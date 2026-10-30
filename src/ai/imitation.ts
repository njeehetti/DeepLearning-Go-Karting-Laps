// Learning to drive by backpropagation: a student network (same 20-32-3 shape as the evolved drivers) copies
// a teacher, the champion found by evolution. The teacher sees exactly the same inputs, so its reflexes can be
// learnt. DAgger: the student drives, the teacher says at every moment what it would have done there, and
// those corrections join the training set. DOM-free.
import { KartState, type KartInput } from '../sim/kart';
import { mulberry32 } from '../sim/rng';
import { Adam, Backprop } from './backprop';
import { Pilot, runEpisode, type EpisodeResult } from './episode';
import type { World } from './evalCore';
import { Mlp, controlsFromOutput } from './mlp';
import { N_OBS, observe } from './obs';

export interface Attempt {
  result: EpisodeResult;
  /** Path at 10 Hz (x, y pairs). */
  path: Float32Array;
  /** Mean |student − teacher| steering command along the attempt (0 = identical, 2 = opposite). */
  steerGap: number;
}

export interface LessonReport {
  lesson: number;
  /** The student's drive at the start of the lesson, before learning from it. */
  attempt: Attempt;
  /** Mean loss of each minibatch step of the lesson. */
  losses: number[];
  samples: number;
  /** Mean |gradient| of every weight over the lesson, its last update, and the weights after it. */
  grad: Float64Array;
  step: Float64Array;
  params: Float64Array;
}

/** Index of the best attempt: a clean lap beats a lap with wheels off, which beats an unfinished one;
 *  then the faster lap, or the longer distance. */
export function bestLap(laps: Attempt[]): number {
  const rank = (a: Attempt) => {
    const r = a.result;
    return r.completed ? (r.clean ? 2e6 : 1e6) - r.lapTime : r.progress - 1e6;
  };
  let best = 0;
  laps.forEach((a, i) => {
    if (rank(a) > rank(laps[best])) best = i;
  });
  return best;
}

const MAX_SAMPLES = 40000;
const START = { s0: 0, d0: 0, maxTime: 150, timeWeight: 1 };

export class Imitation {
  readonly params: Float64Array;
  readonly bp: Backprop;
  readonly adam: Adam;
  private readonly obs = new Float32Array(MAX_SAMPLES * N_OBS);
  private readonly tgt = new Float32Array(MAX_SAMPLES * 3);
  /** Examples stored; once the memory is full the oldest are replaced. */
  n = 0;
  private next = 0;
  lesson = 0;
  private readonly rng: () => number;
  private readonly teacherNet: Mlp;
  private readonly pilot: Pilot;
  private readonly o = new Float64Array(N_OBS);
  private readonly cmd: KartInput = { steer: 0, throttle: 0, brake: 0 };
  private readonly st = new KartState();

  constructor(readonly world: World, readonly teacher: ArrayLike<number>, seed = 1, lr = 3e-3) {
    this.bp = new Backprop(world.shape);
    this.rng = mulberry32(seed);
    // small random weights: the student starts knowing nothing
    this.params = Float64Array.from({ length: this.bp.dim }, () => (this.rng() * 2 - 1) * 0.15);
    this.adam = new Adam(this.bp.dim, lr);
    this.teacherNet = new Mlp(world.shape);
    this.pilot = new Pilot(world.shape);
  }

  /** What the teacher would command in state st (its observation, through its network). */
  private teach(st: KartState): KartInput {
    observe(this.world.q, st, this.o, undefined, this.world.sim.p.deltaMax);
    controlsFromOutput(this.teacherNet.forward(this.teacher, this.o), this.cmd);
    return this.cmd;
  }

  private add(st: KartState): void {
    const t = this.teach(st);
    const k = this.n < MAX_SAMPLES ? this.n++ : this.next++ % MAX_SAMPLES;
    this.obs.set(this.o, k * N_OBS);
    this.tgt[k * 3] = t.steer;
    this.tgt[k * 3 + 1] = t.throttle;
    this.tgt[k * 3 + 2] = t.brake;
  }

  /** The teacher drives one lap and every moment of it becomes an example (the first lesson). */
  demonstrate(): EpisodeResult {
    const w = this.world;
    return runEpisode(w.sim, this.pilot, this.teacher, { ...START, v0: w.startSpeed(0) }, this.st, (st, _inp, decided) => {
      if (decided) this.add(st);
    });
  }

  /** The student (or `driver`) drives from the start line; with `learn`, the teacher labels every state. */
  attempt(learn = true, driver: ArrayLike<number> = this.params): Attempt {
    const w = this.world;
    const path: number[] = [];
    let gap = 0;
    let nGap = 0;
    const result = runEpisode(w.sim, this.pilot, driver, { ...START, v0: w.startSpeed(0) }, this.st, (st, _inp, decided) => {
      if (!decided) return;
      if (Math.round(st.t * 30) % 3 === 0) path.push(st.x, st.y);
      // the driver's own command, before the hands-and-wheel filter, against the teacher's
      observe(w.q, st, this.o, undefined, w.sim.p.deltaMax);
      const own = this.bp.forward(driver, this.o)[0];
      gap += Math.abs(own - this.teach(st).steer);
      nGap++;
      if (learn) this.add(st);
    });
    return { result, path: Float32Array.from(path), steerGap: nGap ? gap / nGap : 0 };
  }

  /** Minibatch gradient steps on the examples gathered so far. */
  train(steps: number, batch = 64): { losses: number[]; grad: Float64Array } {
    const g = new Float64Array(this.bp.dim);
    const gAbs = new Float64Array(this.bp.dim);
    const x = new Float64Array(N_OBS);
    const y = new Float64Array(3);
    const losses: number[] = [];
    for (let s = 0; s < steps; s++) {
      g.fill(0);
      let loss = 0;
      for (let b = 0; b < batch; b++) {
        const k = Math.floor(this.rng() * this.n);
        for (let i = 0; i < N_OBS; i++) x[i] = this.obs[k * N_OBS + i];
        y[0] = this.tgt[k * 3];
        y[1] = this.tgt[k * 3 + 1];
        y[2] = this.tgt[k * 3 + 2];
        this.bp.forward(this.params, x);
        loss += this.bp.backward(this.params, y, g, 1 / batch);
      }
      this.adam.apply(this.params, g);
      losses.push(loss / batch);
      for (let k = 0; k < g.length; k++) gAbs[k] += Math.abs(g[k]) / steps;
    }
    return { losses, grad: gAbs };
  }

  /** One lesson: the student drives (and is corrected), then learns from every correction so far. */
  step(steps = 300): LessonReport {
    if (this.n === 0) this.demonstrate();
    const attempt = this.attempt();
    const { losses, grad } = this.train(steps);
    return { lesson: this.lesson++, attempt, losses, samples: this.n, grad, step: Float64Array.from(this.adam.step), params: Float64Array.from(this.params) };
  }
}
