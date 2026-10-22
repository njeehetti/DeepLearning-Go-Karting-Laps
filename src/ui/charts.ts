// Canvas charts: learning curves and lap telemetry.
import type { GenStats } from '../ai/trainer';
import type { Run } from '../ai/record';

const css = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
export const fmtLap = (T: number) => (Number.isFinite(T) ? `${Math.floor(T / 60)}:${(T % 60).toFixed(3).padStart(6, '0')}` : '–:––.–––');

function prep(cv: HTMLCanvasElement) {
  const r = cv.getBoundingClientRect();
  const d = Math.min(window.devicePixelRatio || 1, 2);
  if (cv.width !== Math.round(r.width * d) || cv.height !== Math.round(r.height * d)) {
    cv.width = Math.round(r.width * d);
    cv.height = Math.round(r.height * d);
  }
  const ctx = cv.getContext('2d')!;
  ctx.setTransform(d, 0, 0, d, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  return { ctx, w: r.width, h: r.height };
}

/** Best clean lap per generation (amber), record so far (thin), completion rate (green, right axis). */
export function drawEvo(cv: HTMLCanvasElement, hist: GenStats[], theory: number): string {
  const { ctx, w, h } = prep(cv);
  if (hist.length < 1 || w < 20) return '';
  const pad = { l: 52, r: 40, t: 12, b: 24 };
  const W = w - pad.l - pad.r;
  const H = h - pad.t - pad.b;
  const laps = hist.map((s) => s.bestLap).filter(Number.isFinite);
  const g1 = Math.max(10, hist[hist.length - 1].gen);
  const hi = laps.length ? Math.min(Math.max(...laps), (Number.isFinite(theory) ? theory : Math.min(...laps)) * 1.45) : 150;
  const lo = Math.min(laps.length ? Math.min(...laps) : 90, Number.isFinite(theory) ? theory : 999) - 1;
  const X = (g: number) => pad.l + (g / g1) * W;
  const Y = (v: number) => pad.t + (1 - (v - lo) / Math.max(1, hi - lo)) * H;
  const Yp = (p: number) => pad.t + (1 - p) * H;
  ctx.font = `11px ${css('--font-data')}`;
  ctx.fillStyle = css('--muted');
  ctx.strokeStyle = css('--line');
  ctx.lineWidth = 1;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const span = hi - lo;
  const step = span > 40 ? 10 : span > 15 ? 5 : span > 6 ? 2 : 1;
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    const y = Math.round(Y(v)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + W, y);
    ctx.stroke();
    ctx.fillText(fmtLap(v).replace(/\.\d+$/, ''), pad.l - 6, y);
  }
  ctx.textAlign = 'left';
  for (const p of [0, 0.5, 1]) ctx.fillText(`${p * 100} %`, pad.l + W + 4, Yp(p));
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const gs = g1 > 2000 ? 500 : g1 > 800 ? 200 : g1 > 300 ? 100 : g1 > 100 ? 25 : 10;
  for (let g = 0; g <= g1; g += gs) ctx.fillText(String(g), X(g), pad.t + H + 6);
  ctx.save();
  ctx.beginPath();
  ctx.rect(pad.l, pad.t, W, H);
  ctx.clip();
  if (Number.isFinite(theory)) {
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = css('--ghost');
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(theory));
    ctx.lineTo(pad.l + W, Y(theory));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.strokeStyle = css('--throttle');
  ctx.globalAlpha = 0.7;
  ctx.beginPath();
  hist.forEach((s, k) => (k ? ctx.lineTo(X(s.gen), Yp(s.completion)) : ctx.moveTo(X(s.gen), Yp(s.completion))));
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.fillStyle = css('--accent');
  let rec = Infinity;
  const recPts: [number, number][] = [];
  for (const s of hist) {
    if (!Number.isFinite(s.bestLap)) continue;
    ctx.globalAlpha = 0.45;
    ctx.fillRect(X(s.gen) - 1, Y(s.bestLap) - 1, 2, 2);
    rec = Math.min(rec, s.bestLap);
    recPts.push([s.gen, rec]);
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = css('--accent');
  ctx.lineWidth = 2;
  ctx.beginPath();
  recPts.forEach(([g, v], k) => (k ? ctx.lineTo(X(g), Y(v)) : ctx.moveTo(X(g), Y(v))));
  ctx.stroke();
  ctx.restore();
  const last = hist[hist.length - 1];
  return `record ${fmtLap(rec)} · tours bouclés ${Math.round(last.completion * 100)} % · σ ${last.sigma.toFixed(3)}`;
}

export interface TelGeom {
  pad: { l: number; r: number; t: number; b: number };
  W: number;
  L: number;
}

/** Speed vs distance of the best run (amber) and the ghost (dashed), pedal strips, cursor. */
export function drawTel(cv: HTMLCanvasElement, run: Run | null, ghost: Run | null, L: number, cursorDist: number, hoverDist: number, corners: { dist: number; name: string }[]): TelGeom | null {
  const { ctx, w, h } = prep(cv);
  if (!run || w < 20) return null;
  const pad = { l: 46, r: 12, t: 18, b: 108 };
  const W = w - pad.l - pad.r;
  const H = h - pad.t - pad.b;
  let vmax = 0;
  let vmin = 1e9;
  for (let k = 0; k < run.n; k++) {
    vmax = Math.max(vmax, run.v[k]);
    vmin = Math.min(vmin, run.v[k]);
  }
  const top = Math.ceil((vmax * 3.6 + 4) / 10) * 10;
  const bot = Math.max(0, Math.floor((vmin * 3.6 - 8) / 10) * 10);
  const X = (d: number) => pad.l + (d / L) * W;
  const Y = (v: number) => pad.t + (1 - (v * 3.6 - bot) / (top - bot)) * H;
  ctx.font = `11px ${css('--font-data')}`;
  ctx.fillStyle = css('--muted');
  ctx.strokeStyle = css('--line');
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let v = bot; v <= top; v += 10) {
    const y = Math.round(pad.t + (1 - (v - bot) / (top - bot)) * H) + 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + W, y);
    ctx.stroke();
    ctx.fillText(String(v), pad.l - 6, y);
  }
  ctx.textAlign = 'left';
  ctx.fillText('km/h', 4, 8);
  ctx.textAlign = 'center';
  ctx.font = `600 11px ${css('--font-display')}`;
  for (const c of corners) {
    const x = X(c.dist);
    ctx.strokeStyle = 'rgba(255,255,255,.07)';
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, pad.t + H);
    ctx.stroke();
    ctx.fillText(c.name.slice(1), x, 6);
  }
  // pedal panels: throttle and brake in % along the lap (where and how hard)
  const panel = (top: number, h: number, arr: Float32Array, color: string, ghostArr: Float32Array | null, label: string) => {
    ctx.strokeStyle = css('--line');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, top + h + 0.5);
    ctx.lineTo(pad.l + W, top + h + 0.5);
    ctx.stroke();
    ctx.fillStyle = css('--muted');
    ctx.font = `10px ${css('--font-data')}`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, pad.l - 6, top + h / 2);
    ctx.fillText('100', pad.l - 6, top + 3);
    ctx.beginPath();
    ctx.moveTo(X(run.dist[0]), top + h);
    for (let k = 0; k < run.n && run.dist[k] <= L; k++) ctx.lineTo(X(run.dist[k]), top + h - arr[k] * h);
    ctx.lineTo(X(Math.min(L, run.dist[run.n - 1])), top + h);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.55;
    ctx.fill();
    ctx.globalAlpha = 1;
    if (ghostArr && ghost) {
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = css('--ghost');
      ctx.beginPath();
      for (let k = 0; k < ghost.n && ghost.dist[k] <= L; k++) {
        const px = X(ghost.dist[k]);
        const py = top + h - ghostArr[k] * h;
        if (k) ctx.lineTo(px, py);
        else ctx.moveTo(px, py);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }
  };
  const pt = pad.t + H + 10;
  panel(pt, 36, run.thr, css('--throttle'), ghost ? ghost.thr : null, 'gaz %');
  panel(pt + 44, 36, run.brk, css('--brake'), ghost ? ghost.brk : null, 'frein %');
  ctx.fillStyle = css('--muted');
  ctx.textAlign = 'right';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText('m', pad.l + W, h - 4);
  for (let d = 0; d <= L; d += 200) {
    ctx.textAlign = 'center';
    ctx.fillText(String(d), X(d), h - 4);
  }
  const curve = (r: Run, dash: boolean, color: string, width: number) => {
    ctx.setLineDash(dash ? [4, 3] : []);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    for (let k = 0; k < r.n; k++) {
      const d = r.dist[k];
      if (d > L) break;
      if (k) ctx.lineTo(X(d), Y(r.v[k]));
      else ctx.moveTo(X(d), Y(r.v[k]));
    }
    ctx.stroke();
    ctx.setLineDash([]);
  };
  if (ghost) curve(ghost, true, css('--ghost'), 1.3);
  curve(run, false, css('--accent'), 1.8);
  const vline = (d: number, color: string) => {
    if (d < 0) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(X(d), pad.t);
    ctx.lineTo(X(d), pad.t + H + 90);
    ctx.stroke();
  };
  vline(cursorDist, 'rgba(241,163,59,.7)');
  vline(hoverDist, '#ffffff');
  return { pad, W, L };
}
