// Gradient learning for the driver network: forward pass with cached activations, backpropagation through
// the tanh hidden layers and the driver head (steer / throttle / brake), and the Adam optimiser.
// Same flat parameter layout as Mlp: per layer, for each output neuron, its input weights then its bias.
import type { MlpShape } from './mlp';
import { paramCount } from './mlp';

const sig = (x: number) => 1 / (1 + Math.exp(-x));

/** Driver head, same as controlsFromOutput, plus its derivative. The brake dead zone has no gradient, so
 *  the gradient goes through as if it were absent (straight-through): an idle foot can still learn. */
export function head(out: ArrayLike<number>, ctl: Float64Array, dctl: Float64Array): void {
  const s = Math.tanh(out[0]);
  ctl[0] = s;
  dctl[0] = 1 - s * s;
  const t = sig(out[1] + 1);
  ctl[1] = t;
  dctl[1] = t * (1 - t);
  const b = sig(out[2] - 2);
  ctl[2] = Math.max(0, (b - 0.2) / 0.8);
  dctl[2] = (b * (1 - b)) / 0.8;
}

/** Relative weight of each control in the loss: steering matters most, the pedals a bit less. */
export const CONTROL_WEIGHTS = Float64Array.of(1, 0.5, 0.5);

export class Backprop {
  readonly shape: MlpShape;
  readonly dim: number;
  /** Activations per layer (index 0 = inputs) and the head outputs of the last forward pass. */
  readonly act: Float64Array[];
  readonly ctl = new Float64Array(3);
  private readonly dctl = new Float64Array(3);
  /** dLoss/d(pre-activation) per layer, reused by backward(). */
  private readonly delta: Float64Array[];
  /** Start of each layer's block in the flat parameter vector. */
  private readonly off: number[];

  constructor(shape: MlpShape) {
    this.shape = shape;
    this.dim = paramCount(shape);
    this.act = shape.sizes.map((n) => new Float64Array(n));
    this.delta = shape.sizes.map((n) => new Float64Array(n));
    this.off = [0];
    for (let l = 1; l < shape.sizes.length; l++) this.off.push(this.off[l - 1] + shape.sizes[l] * (shape.sizes[l - 1] + 1));
  }

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
        let s = params[o + nIn];
        for (let i = 0; i < nIn; i++) s += params[o + i] * prev[i];
        o += nIn + 1;
        cur[j] = last ? s : Math.tanh(s);
      }
    }
    head(this.act[sizes.length - 1], this.ctl, this.dctl);
    return this.ctl;
  }

  /**
   * Weighted squared error between the controls of the last forward() and the target. Adds
   * dLoss/dparams × scale (e.g. 1 / batch size) into `grad` and returns the loss.
   */
  backward(params: ArrayLike<number>, target: ArrayLike<number>, grad: Float64Array, scale = 1): number {
    const { sizes } = this.shape;
    const L = sizes.length - 1;
    let loss = 0;
    const dOut = this.delta[L];
    for (let k = 0; k < sizes[L]; k++) {
      const e = this.ctl[k] - target[k];
      loss += CONTROL_WEIGHTS[k] * e * e;
      dOut[k] = 2 * CONTROL_WEIGHTS[k] * e * this.dctl[k];
    }
    for (let l = L; l >= 1; l--) {
      const nIn = sizes[l - 1];
      const prev = this.act[l - 1];
      const d = this.delta[l];
      const dPrev = this.delta[l - 1];
      dPrev.fill(0);
      let o = this.off[l - 1];
      for (let j = 0; j < sizes[l]; j++) {
        const dj = d[j] * scale;
        for (let i = 0; i < nIn; i++) {
          grad[o + i] += dj * prev[i];
          dPrev[i] += d[j] * params[o + i];
        }
        grad[o + nIn] += dj;
        o += nIn + 1;
      }
      // through the tanh of the layer below (the inputs have no activation)
      if (l > 1) for (let i = 0; i < nIn; i++) dPrev[i] *= 1 - prev[i] * prev[i];
    }
    return loss;
  }
}

/** Adam optimiser on a flat parameter vector. `step` keeps the last update of every weight, for display. */
export class Adam {
  readonly m: Float64Array;
  readonly v: Float64Array;
  readonly step: Float64Array;
  t = 0;
  lr: number;
  private readonly b1 = 0.9;
  private readonly b2 = 0.999;

  constructor(dim: number, lr = 3e-3) {
    this.lr = lr;
    this.m = new Float64Array(dim);
    this.v = new Float64Array(dim);
    this.step = new Float64Array(dim);
  }

  apply(params: Float64Array, grad: ArrayLike<number>): void {
    this.t++;
    const c1 = 1 - this.b1 ** this.t;
    const c2 = 1 - this.b2 ** this.t;
    for (let k = 0; k < params.length; k++) {
      const g = grad[k];
      this.m[k] = this.b1 * this.m[k] + (1 - this.b1) * g;
      this.v[k] = this.b2 * this.v[k] + (1 - this.b2) * g * g;
      const s = (-this.lr * (this.m[k] / c1)) / (Math.sqrt(this.v[k] / c2) + 1e-8);
      this.step[k] = s;
      params[k] += s;
    }
  }
}
