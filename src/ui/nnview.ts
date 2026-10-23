// Network view: neurons as circles coloured by activation, connections by weight (sign and size).
import { DRIVER_SHAPE, Mlp } from '../ai/mlp';
import { OBS_LABELS } from '../ai/obs';

const NS = 'http://www.w3.org/2000/svg';
const OUT_LABELS = ['Direction', 'Gaz', 'Frein'];

export class NnView {
  private readonly svg: SVGSVGElement;
  private readonly nodes: SVGCircleElement[][] = [];
  private readonly edges: { el: SVGLineElement; l: number; i: number; j: number }[] = [];
  private readonly mlp = new Mlp();
  private params: ArrayLike<number> | null = null;

  constructor(host: HTMLElement) {
    this.svg = document.createElementNS(NS, 'svg');
    this.svg.setAttribute('viewBox', '0 0 600 260');
    this.svg.setAttribute('role', 'img');
    this.svg.setAttribute('aria-label', 'Réseau de neurones du meilleur pilote');
    host.appendChild(this.svg);
    const sizes = DRIVER_SHAPE.sizes;
    const xs = [150, 360, 500];
    const pos = sizes.map((n, l) => Array.from({ length: n }, (_, k) => [xs[l], 14 + ((k + 0.5) / n) * 232]));
    const eg = document.createElementNS(NS, 'g');
    this.svg.appendChild(eg);
    for (let l = 1; l < sizes.length; l++) {
      for (let j = 0; j < sizes[l]; j++) {
        for (let i = 0; i < sizes[l - 1]; i++) {
          const el = document.createElementNS(NS, 'line');
          el.setAttribute('x1', String(pos[l - 1][i][0]));
          el.setAttribute('y1', String(pos[l - 1][i][1]));
          el.setAttribute('x2', String(pos[l][j][0]));
          el.setAttribute('y2', String(pos[l][j][1]));
          eg.appendChild(el);
          this.edges.push({ el, l, i, j });
        }
      }
    }
    sizes.forEach((n, l) => {
      this.nodes.push([]);
      for (let k = 0; k < n; k++) {
        const c = document.createElementNS(NS, 'circle');
        c.setAttribute('cx', String(pos[l][k][0]));
        c.setAttribute('cy', String(pos[l][k][1]));
        c.setAttribute('r', l === 1 ? '5.5' : '5');
        c.setAttribute('stroke', '#5a6872');
        this.svg.appendChild(c);
        this.nodes[l].push(c);
        const label = l === 0 ? OBS_LABELS[k] : l === sizes.length - 1 ? OUT_LABELS[k] : '';
        if (label) {
          const t = document.createElementNS(NS, 'text');
          t.textContent = label;
          t.setAttribute('x', String(l === 0 ? pos[l][k][0] - 10 : pos[l][k][0] + 10));
          t.setAttribute('y', String(pos[l][k][1] + 3.5));
          t.setAttribute('text-anchor', l === 0 ? 'end' : 'start');
          t.setAttribute('fill', '#8796a1');
          t.setAttribute('font-size', '10');
          t.setAttribute('font-family', 'Public Sans, system-ui, sans-serif');
          this.svg.appendChild(t);
        }
      }
    });
  }

  setParams(params: ArrayLike<number> | null): void {
    this.params = params;
    if (!params) return;
    const sizes = DRIVER_SHAPE.sizes;
    let maxW = 1e-9;
    for (let k = 0; k < params.length; k++) maxW = Math.max(maxW, Math.abs(params[k]));
    for (const e of this.edges) {
      let o = 0;
      for (let l = 1; l < e.l; l++) o += sizes[l] * (sizes[l - 1] + 1);
      const wgt = params[o + e.j * (sizes[e.l - 1] + 1) + e.i];
      const a = Math.min(1, Math.abs(wgt) / maxW);
      e.el.setAttribute('stroke', wgt > 0 ? 'var(--accent)' : 'var(--ghost)');
      e.el.setAttribute('stroke-opacity', String(0.04 + 0.5 * a * a));
      e.el.setAttribute('stroke-width', String(0.4 + 1.6 * a));
    }
  }

  /** Feeds an observation vector and colours neurons by their activation. */
  show(obs: ArrayLike<number>): void {
    if (!this.params) return;
    this.mlp.forward(this.params, obs);
    this.mlp.act.forEach((layer, l) => {
      const last = l === this.mlp.act.length - 1;
      layer.forEach((v, k) => {
        const a = Math.max(-1, Math.min(1, last ? Math.tanh(v) : v));
        const c = a >= 0 ? `rgba(241,163,59,${0.15 + 0.85 * a})` : `rgba(74,138,214,${0.15 + 0.85 * -a})`;
        this.nodes[l][k].setAttribute('fill', c);
      });
    });
  }
}
