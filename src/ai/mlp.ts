// Fixed-topology MLP driver brain. Parameters live in one flat Float64Array so that evolution
// strategies (and later PPO) can treat the policy as a vector. Allocation-free forward pass.

export interface MlpShape {
  /** Layer sizes, input first, e.g. [17, 16, 3]. */
  sizes: number[];
}

export const DRIVER_SHAPE: MlpShape = { sizes: [17, 16, 3] };

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

/** Output head: steer in [−1, 1], throttle and brake in [0, 1] with sensible priors at zero weights. */
export function controlsFromOutput(out: ArrayLike<number>, dst: { steer: number; throttle: number; brake: number }): void {
  dst.steer = Math.tanh(out[0]);
  dst.throttle = 1 / (1 + Math.exp(-(out[1] + 1)));
  dst.brake = 1 / (1 + Math.exp(-(out[2] - 2)));
}
