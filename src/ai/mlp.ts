// Fixed-topology MLP driver brain. Parameters live in one flat Float64Array so that evolution
// strategies (and later PPO) can treat the policy as a vector. Allocation-free forward pass.
import { mulberry32 } from '../sim/rng';

export interface MlpShape {
  /** Layer sizes, input first, e.g. [18, 16, 3]. */
  sizes: number[];
}

// 32 hidden neurons: with 16, one brain could not learn every circuit (generalist training stalled on 5 of 9)
export const DRIVER_SHAPE: MlpShape = { sizes: [20, 32, 3] };
/** Network of the brains saved before the 32-neuron one (they are not convertible since the inputs changed). */
export const LEGACY_SHAPE: MlpShape = { sizes: [20, 16, 3] };

/** Driver network with other hidden layers (same inputs and outputs). */
export function driverShape(hidden?: number[]): MlpShape {
  return hidden?.length ? { sizes: [DRIVER_SHAPE.sizes[0], ...hidden, DRIVER_SHAPE.sizes[DRIVER_SHAPE.sizes.length - 1]] } : DRIVER_SHAPE;
}

/**
 * Grows the hidden layer of a one-hidden-layer brain without changing what it does: the new neurons get
 * small random input weights (`rnd` in [0, 1)) and zero output weights, so evolution can recruit them.
 */
export function widen(params: ArrayLike<number>, from: MlpShape, to: MlpShape, rnd: () => number, scale = 0.1): Float64Array {
  const [nIn, h0, nOut] = from.sizes;
  const h1 = to.sizes[1];
  if (from.sizes.length !== 3 || to.sizes.length !== 3 || to.sizes[0] !== nIn || to.sizes[2] !== nOut || h1 < h0) throw new Error('élargissement impossible');
  const out = new Float64Array(paramCount(to));
  let o = 0;
  for (let j = 0; j < h1; j++) {
    for (let i = 0; i <= nIn; i++) out[o++] = j < h0 ? params[j * (nIn + 1) + i] : (rnd() * 2 - 1) * scale;
  }
  const base = h0 * (nIn + 1);
  for (let k = 0; k < nOut; k++) {
    for (let j = 0; j < h1; j++) out[o + j] = j < h0 ? params[base + k * (h0 + 1) + j] : 0;
    out[o + h1] = params[base + k * (h0 + 1) + h0];
    o += h1 + 1;
  }
  return out;
}

/** A saved brain's parameters on the current driver network (older ones are widened), null if not convertible. */
export function toDriverShape(params: ArrayLike<number>, sizes?: number[]): number[] | null {
  const from = { sizes: sizes ?? (params.length === paramCount(LEGACY_SHAPE) ? LEGACY_SHAPE.sizes : DRIVER_SHAPE.sizes) };
  if (params.length !== paramCount(from)) return null;
  if (from.sizes.join() === DRIVER_SHAPE.sizes.join()) return Array.from(params);
  try {
    return Array.from(widen(params, from, DRIVER_SHAPE, mulberry32(1)));
  } catch {
    return null;
  }
}

export function paramCount(shape: MlpShape): number {
  let n = 0;
  for (let l = 1; l < shape.sizes.length; l++) n += shape.sizes[l] * (shape.sizes[l - 1] + 1);
  return n;
}

export class Mlp {
  readonly shape: MlpShape;
  readonly nParams: number;
  /** Activations of every layer after the last forward pass (index 0 = inputs). */
  readonly act: Float64Array[];

  constructor(shape: MlpShape = DRIVER_SHAPE) {
    this.shape = shape;
    this.nParams = paramCount(shape);
    this.act = shape.sizes.map((n) => new Float64Array(n));
  }

  /**
   * Hidden layers use tanh; the output layer is left linear (the driver head maps it to controls).
   * Weights are row-major per layer: for each output neuron, its inputs then its bias.
   */
  forward(params: ArrayLike<number>, input: ArrayLike<number>): Float64Array {
    const { sizes } = this.shape;
    const a0 = this.act[0];
    for (let i = 0; i < sizes[0]; i++) a0[i] = input[i];
    let o = 0;
    for (let l = 1; l < sizes.length; l++) {
      const prev = this.act[l - 1];
      const cur = this.act[l];
      const nIn = sizes[l - 1];
      const last = l === sizes.length - 1;
      for (let j = 0; j < sizes[l]; j++) {
        let s = 0;
        for (let i = 0; i < nIn; i++) s += params[o + i] * prev[i];
        s += params[o + nIn];
        o += nIn + 1;
        cur[j] = last ? s : Math.tanh(s);
      }
    }
    return this.act[sizes.length - 1];
  }
}

/** Output head: steer in [−1, 1], throttle and brake in [0, 1] with sensible priors at zero weights.
 *  The brake has a dead zone, so the foot can rest fully off it (and on the throttle, see BRAKE_FOOT). */
export function controlsFromOutput(out: ArrayLike<number>, dst: { steer: number; throttle: number; brake: number }): void {
  dst.steer = Math.tanh(out[0]);
  dst.throttle = 1 / (1 + Math.exp(-(out[1] + 1)));
  dst.brake = Math.max(0, (1 / (1 + Math.exp(-(out[2] - 2))) - 0.2) / 0.8);
}
