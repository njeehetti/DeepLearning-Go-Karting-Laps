// "Comment le pilote apprend": the 771 weights of the driver network as a map, updated live, for both ways
// of learning. Evolution (sep-CMA-ES): the population mean, how far it moved, how widely it still explores.
// Backpropagation: a student copies the champion; every weight's value, gradient and Adam update.
import type { SepCmaEs } from '../ai/es';
import type { WorldSpec } from '../ai/evalCore';
import type { LessonReport, Attempt } from '../ai/imitation';
import { DRIVER_SHAPE } from '../ai/mlp';
import { OBS_LABELS } from '../ai/obs';
import type { Track } from '../track/types';
import type { FromImit, ToImit } from '../workers/imit.worker';
import { css, fmtLap, prep } from './charts';

const [N_IN, N_HID, N_OUT] = DRIVER_SHAPE.sizes;
const OUT_LABELS = ['Direction', 'Gaz', 'Frein'];
const fr = (x: number, d = 0) => x.toLocaleString('fr-FR', { minimumFractionDigits: d, maximumFractionDigits: d });
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

/** Flat index of the weight from input i (N_IN = bias) to hidden neuron j, and from hidden j (N_HID = bias) to output k. */
const wIn = (j: number, i: number) => j * (N_IN + 1) + i;
const wOut = (k: number, j: number) => N_HID * (N_IN + 1) + k * (N_HID + 1) + j;

/** Plain-words name of a weight, for the hover line. */
function weightName(idx: number): string {
  const base = N_HID * (N_IN + 1);
  if (idx < base) {
    const j = Math.floor(idx / (N_IN + 1));
    const i = idx % (N_IN + 1);
    return i === N_IN ? `biais du neurone ${j + 1}` : `${OBS_LABELS[i]} → neurone ${j + 1}`;
  }
  const k = Math.floor((idx - base) / (N_HID + 1));
  const j = (idx - base) % (N_HID + 1);
  return j === N_HID ? `biais de la sortie ${OUT_LABELS[k]}` : `neurone ${j + 1} → ${OUT_LABELS[k]}`;
}

/** Heatmap of the network: one column per hidden neuron, its input weights above, its output weights below. */
class WeightMap {
  private values: ArrayLike<number> | null = null;
  private signed = true;
  private geom = { x0: 0, cw: 0, ch: 0, yOut: 0 };
  hover = -1;

  constructor(private readonly cv: HTMLCanvasElement, private readonly onHover: (idx: number) => void) {
    cv.addEventListener('mousemove', (e) => this.pick(e.offsetX, e.offsetY));
    cv.addEventListener('click', (e) => this.pick(e.offsetX, e.offsetY));
    cv.addEventListener('mouseleave', () => {
      this.hover = -1;
      this.onHover(-1);
      this.draw();
    });
  }

  set(values: ArrayLike<number> | null, signed: boolean): void {
    this.values = values;
    this.signed = signed;
    this.draw();
    if (this.hover >= 0) this.onHover(this.hover);
  }

  private pick(x: number, y: number): void {
    const { x0, cw, ch, yOut } = this.geom;
    const c = Math.floor((x - x0) / cw);
    let idx = -1;
    if (c >= 0 && c <= N_HID) {
      const r = Math.floor(y / ch);
      if (y < yOut && r >= 0 && r <= N_IN && c < N_HID) idx = wIn(c, r);
      const ro = Math.floor((y - yOut) / ch);
      if (y >= yOut && ro >= 0 && ro < N_OUT) idx = wOut(ro, c);
    }
    if (idx !== this.hover) {
      this.hover = idx;
      this.onHover(idx);
      this.draw();
    }
  }

  draw(): void {
    const { ctx, w, h } = prep(this.cv);
    if (w < 40) return;
    const label = 112;
    const cw = (w - label) / (N_HID + 1);
    const rows = N_IN + 1 + N_OUT;
    const ch = Math.min(14, (h - 22) / rows);
    const yOut = (N_IN + 1) * ch + 18;
    this.geom = { x0: label, cw, ch, yOut };
    ctx.font = `10px ${css('--font-data')}`;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    ctx.fillStyle = css('--faint');
    for (let i = 0; i <= N_IN; i++) ctx.fillText(i === N_IN ? 'biais' : OBS_LABELS[i], label - 6, (i + 0.5) * ch);
    ctx.fillStyle = css('--muted');
    for (let k = 0; k < N_OUT; k++) ctx.fillText(OUT_LABELS[k], label - 6, yOut + (k + 0.5) * ch);
    ctx.textAlign = 'left';
    ctx.fillStyle = css('--faint');
    ctx.fillText('↑ entrées → neurone   ·   colonnes : neurones cachés 1 à 32, puis biais   ·   neurone → sorties ↓', label, yOut - 9);
    const v = this.values;
    if (!v) return;
    // colour scale: 95th percentile of |value|, so a few large weights do not wash out the map
    const mags = Array.from(v, (x) => Math.abs(x)).sort((a, b) => a - b);
    const scale = mags[Math.floor(mags.length * 0.95)] || 1;
    const pos = css('--accent');
    const neg = css('--ghost');
    const cell = (idx: number, x: number, y: number) => {
      ctx.globalAlpha = 0.08 + 0.92 * Math.min(1, Math.abs(v[idx]) / scale);
      ctx.fillStyle = !this.signed || v[idx] >= 0 ? pos : neg;
      ctx.fillRect(x + 0.5, y + 0.5, Math.max(1, cw - 1), Math.max(1, ch - 1));
    };
    for (let j = 0; j < N_HID; j++) for (let i = 0; i <= N_IN; i++) cell(wIn(j, i), label + j * cw, i * ch);
    for (let k = 0; k < N_OUT; k++) for (let j = 0; j <= N_HID; j++) cell(wOut(k, j), label + j * cw, yOut + k * ch);
    ctx.globalAlpha = 1;
    if (this.hover >= 0) {
      const base = N_HID * (N_IN + 1);
      const [x, y] = this.hover < base
        ? [label + Math.floor(this.hover / (N_IN + 1)) * cw, (this.hover % (N_IN + 1)) * ch]
        : [label + ((this.hover - base) % (N_HID + 1)) * cw, yOut + Math.floor((this.hover - base) / (N_HID + 1)) * ch];
      ctx.strokeStyle = css('--text');
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x, y, cw, ch);
    }
  }
}

/** Small line chart: series share the x axis (index), each on its own scale; `log` for quantities that shrink. */
function drawSeries(cv: HTMLCanvasElement, series: { v: number[]; color: string; log?: boolean }[], xLabel: string): void {
  const { ctx, w, h } = prep(cv);
  const pad = { l: 8, r: 8, t: 8, b: 18 };
  const n = Math.max(2, ...series.map((s) => s.v.length));
  ctx.strokeStyle = css('--line');
  ctx.beginPath();
  ctx.moveTo(pad.l, h - pad.b + 0.5);
  ctx.lineTo(w - pad.r, h - pad.b + 0.5);
  ctx.stroke();
  ctx.font = `10px ${css('--font-data')}`;
  ctx.fillStyle = css('--faint');
  ctx.textBaseline = 'top';
  ctx.fillText(xLabel, pad.l, h - pad.b + 4);
  for (const s of series) {
    const f = (x: number) => (s.log ? Math.log10(Math.max(x, 1e-6)) : x);
    const vals = s.v.map(f).filter(Number.isFinite);
    if (vals.length < 2) continue;
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    s.v.forEach((x, i) => {
      const px = pad.l + (i / (n - 1)) * (w - pad.l - pad.r);
      const py = pad.t + (1 - (f(x) - lo) / Math.max(1e-9, hi - lo)) * (h - pad.t - pad.b);
      if (i) ctx.lineTo(px, py);
      else ctx.moveTo(px, py);
    });
    ctx.stroke();
  }
}

/** Track outline with driven paths on top (teacher in ghost blue, student in amber), a dot where each ended. */
function drawPaths(cv: HTMLCanvasElement, track: Track, paths: { xy: Float32Array; color: string; width: number }[]): void {
  const { ctx, w, h } = prep(cv);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < track.x.length; i++) {
    x0 = Math.min(x0, track.x[i]); x1 = Math.max(x1, track.x[i]); y0 = Math.min(y0, track.y[i]); y1 = Math.max(y1, track.y[i]);
  }
  const k = Math.min((w - 16) / (x1 - x0), (h - 16) / (y1 - y0));
  const X = (x: number) => w / 2 + (x - (x0 + x1) / 2) * k;
  const Y = (y: number) => h / 2 - (y - (y0 + y1) / 2) * k;
  const line = (xs: ArrayLike<number>, ys: ArrayLike<number>, n: number, step = 1, closed = false) => {
    ctx.beginPath();
    for (let i = 0; i < n; i += step) {
      if (i) ctx.lineTo(X(xs[i]), Y(ys[i]));
      else ctx.moveTo(X(xs[i]), Y(ys[i]));
    }
    if (closed) ctx.closePath();
    ctx.stroke();
  };
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.strokeStyle = css('--control');
  ctx.lineWidth = Math.max(3, 2 * track.half * k);
  line(track.x, track.y, track.x.length, 1, true);
  for (const p of paths) {
    const n = p.xy.length / 2;
    if (n < 2) continue;
    const xs = Float32Array.from({ length: n }, (_, i) => p.xy[2 * i]);
    const ys = Float32Array.from({ length: n }, (_, i) => p.xy[2 * i + 1]);
    ctx.strokeStyle = p.color;
    ctx.lineWidth = p.width;
    line(xs, ys, n);
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.arc(X(xs[n - 1]), Y(ys[n - 1]), 4, 0, 2 * Math.PI);
    ctx.fill();
  }
}

export interface LearnDeps {
  track: Track;
  world: () => WorldSpec;
  /** Brains that may teach, named for sentences ("meilleur pilote de ce circuit"); the fastest here is picked. */
  teachers: () => { label: string; params: number[] }[];
}

export function initLearn(deps: LearnDeps) {
  const visible = () => !$('learn').closest('.view')!.hasAttribute('hidden');
  let tab: 'evo' | 'bp' = 'evo';

  // ---------- evolution ----------
  const evo = { m: null as Float64Array | null, prev: null as Float64Array | null, explore: null as Float64Array | null, moves: [] as number[], spreads: [] as number[], mode: 'value' as 'value' | 'change' | 'explore' };
  const evoMap = new WeightMap($<HTMLCanvasElement>('evo-map'), (idx) => {
    $('evo-hover').textContent = idx < 0 || !evo.m
      ? 'Survole une case pour lire ce poids.'
      : `${weightName(idx)} : ${fr(evo.m[idx], 3)}${evo.prev ? ` · a bougé de ${fr(evo.m[idx] - evo.prev[idx], 4)} à cette génération` : ''}${evo.explore ? ` · les variantes l’essaient à ±${fr(evo.explore[idx], 3)}` : ''}`;
  });
  function drawEvoPane(): void {
    if (!visible() || tab !== 'evo') return;
    const m = evo.m;
    const prev = evo.prev;
    const vals = !m ? null : evo.mode === 'value' ? m : evo.mode === 'change' ? (prev ? m.map((x, i) => x - prev[i]) : null) : evo.explore;
    evoMap.set(vals, evo.mode !== 'explore');
    drawSeries($<HTMLCanvasElement>('evo-curve'), [{ v: evo.moves, color: css('--accent'), log: true }, { v: evo.spreads, color: css('--ghost'), log: true }], 'générations →');
  }
  function onGen(es: SepCmaEs): void {
    evo.prev = evo.m;
    evo.m = Float64Array.from(es.m);
    evo.explore = es.D.map((d) => d * es.sigma);
    if (evo.prev) {
      let mv = 0;
      for (let k = 0; k < evo.m.length; k++) mv += Math.abs(evo.m[k] - evo.prev[k]);
      evo.moves.push(mv / evo.m.length);
    }
    const sp = evo.explore.reduce((a, b) => a + b, 0) / evo.explore.length;
    evo.spreads.push(sp);
    const mv = evo.moves[evo.moves.length - 1];
    $('evo-status').innerHTML = `Génération <b>${es.gen}</b> : en moyenne, chaque poids a bougé de <b>${mv === undefined ? '–' : fr(mv, 4)}</b> et les variantes l’essaient à <b>±${fr(sp, 3)}</b> autour${evo.spreads.length > 5 && sp < evo.spreads[0] * 0.5 ? ' : l’algorithme resserre ses essais, il devient sûr de lui' : ''}.`;
    drawEvoPane();
  }
  function resetEvo(): void {
    Object.assign(evo, { m: null, prev: null, explore: null, moves: [], spreads: [] });
    $('evo-status').textContent = 'Lance l’entraînement dans la vue Piste : la carte se met à jour à chaque génération.';
    drawEvoPane();
  }

  // ---------- backpropagation ----------
  const bp = {
    worker: null as Worker | null, running: false, busy: false, ready: false, id: 0,
    teacher: null as Attempt | null, last: null as LessonReport | null,
    losses: [] as number[], mode: 'value' as 'value' | 'grad' | 'step', firstLap: -1,
  };
  const bpMap = new WeightMap($<HTMLCanvasElement>('bp-map'), (idx) => {
    const r = bp.last;
    $('bp-hover').textContent = idx < 0 || !r
      ? 'Survole une case pour lire ce poids.'
      : `${weightName(idx)} : ${fr(r.params[idx], 3)} · gradient moyen ${fr(r.grad[idx], 5)} · dernier pas d’Adam ${fr(r.step[idx], 5)}`;
  });
  function drawBpPane(): void {
    if (!visible() || tab !== 'bp') return;
    const r = bp.last;
    bpMap.set(!r ? null : bp.mode === 'value' ? r.params : bp.mode === 'grad' ? r.grad : r.step, bp.mode !== 'grad');
    const paths: { xy: Float32Array; color: string; width: number }[] = [];
    if (bp.teacher) paths.push({ xy: bp.teacher.path, color: css('--ghost'), width: 2 });
    if (r) paths.push({ xy: r.attempt.path, color: css('--accent'), width: 2.5 });
    drawPaths($<HTMLCanvasElement>('bp-track'), deps.track, paths);
    drawSeries($<HTMLCanvasElement>('bp-loss'), [{ v: bp.losses, color: css('--accent'), log: true }], 'pas de gradient →');
  }
  function bpButton(): void {
    $('bp-run').textContent = bp.running ? 'Pause' : bp.last ? 'Reprendre les leçons' : 'Lancer les leçons';
  }
  function describe(r: LessonReport): string {
    const a = r.attempt.result;
    const drove = a.completed
      ? `a bouclé le tour en <b>${fmtLap(a.lapTime)}</b>${a.clean ? '' : ' (roue dans l’herbe)'}`
      : `a roulé <b>${fr(a.progress)} m</b> (${fr((a.progress / deps.track.L) * 100)} % du tour) avant de ${a.end === 'mur' ? 'taper le mur' : a.end === 'herbe' ? 'sortir dans l’herbe' : 's’arrêter'}`;
    return `Leçon <b>${r.lesson + 1}</b> : l’élève ${drove}. Son volant s’écarte de <b>${fr(r.attempt.steerGap, 2)}</b> de celui du professeur · ${fr(r.samples)} exemples · erreur ${fr(r.losses[r.losses.length - 1], 4)}.`;
  }
  function lesson(w: Worker): Promise<FromImit> {
    const id = ++bp.id;
    return new Promise((res) => {
      const h = (e: MessageEvent<FromImit>) => {
        const m = e.data;
        if ((m.type === 'lesson' && m.id === id) || m.type === 'error') {
          w.removeEventListener('message', h);
          res(m);
        }
      };
      w.addEventListener('message', h);
      w.postMessage({ type: 'lesson', id } satisfies ToImit);
    });
  }
  async function lessons(): Promise<void> {
    if (bp.busy) return;
    bp.busy = true;
    while (bp.running && bp.worker && bp.ready) {
      const w = bp.worker;
      const m = await lesson(w);
      if (w !== bp.worker) break;
      if (m.type === 'error') {
        bp.running = false;
        $('bp-status').textContent = 'Erreur : ' + m.message;
        break;
      }
      if (m.type !== 'lesson') continue;
      bp.last = m.report;
      bp.losses.push(...m.report.losses);
      const a = m.report.attempt.result;
      if (a.completed && a.clean && bp.firstLap < 0) bp.firstLap = m.report.lesson + 1;
      $('bp-status').innerHTML = describe(m.report) + (bp.firstLap > 0 ? ` Premier tour propre à la leçon <b>${bp.firstLap}</b>.` : '');
      drawBpPane();
      // a short pause so each lesson can be read
      await new Promise((r) => setTimeout(r, 400));
    }
    bp.busy = false;
    bpButton();
  }
  function lapText(a: Attempt): string {
    const r = a.result;
    return r.completed ? `${fmtLap(r.lapTime)}${r.clean ? '' : ' hors limites'}` : `sort à ${fr(r.progress)} m`;
  }
  function teacherLine(ts: { label: string }[], laps: Attempt[], pick: number): string {
    const others = ts.map((t, i) => (i === pick ? '' : `${t.label} : ${lapText(laps[i])}`)).filter(Boolean);
    return `Professeur : le ${ts[pick].label}, <b>${lapText(laps[pick])}</b> ici${others.length ? ` (plus rapide que le ${others.join(', ')})` : ''}`;
  }
  function startStudent(): void {
    const ts = deps.teachers();
    if (!ts.length) {
      bp.running = false;
      bpButton();
      $('bp-status').textContent = 'Il faut un professeur : entraîne d’abord un pilote (vue Piste).';
      return;
    }
    bp.worker?.terminate();
    Object.assign(bp, { ready: false, last: null, teacher: null, losses: [], firstLap: -1 });
    const w = new Worker(new URL('../workers/imit.worker.ts', import.meta.url), { type: 'module' });
    bp.worker = w;
    $('bp-status').textContent = ts.length > 1 ? 'Chaque candidat professeur fait un tour : le plus rapide ici enseignera…' : `Le ${ts[0].label} fait un tour de démonstration…`;
    w.addEventListener('message', (e: MessageEvent<FromImit>) => {
      const m = e.data;
      if (m.type !== 'ready' || w !== bp.worker) return;
      bp.ready = true;
      bp.teacher = m.laps[m.pick];
      $('bp-teacher').innerHTML = teacherLine(ts, m.laps, m.pick);
      $('bp-status').textContent = 'Tour de démonstration fait : chaque instant est devenu un exemple. L’élève, aux poids tirés au hasard, va essayer à son tour.';
      drawBpPane();
      void lessons();
    });
    w.postMessage({ type: 'start', world: deps.world(), teachers: ts.map((t) => t.params), seed: (Math.random() * 1e9) | 0 } satisfies ToImit);
  }
  $('bp-run').addEventListener('click', () => {
    bp.running = !bp.running;
    bpButton();
    if (!bp.running) return;
    if (!bp.worker) startStudent();
    else if (bp.ready) void lessons();
  });
  $('bp-reset').addEventListener('click', () => {
    bp.running = true;
    bpButton();
    startStudent();
  });

  // ---------- tabs, modes, redraw ----------
  document.querySelectorAll<HTMLButtonElement>('[data-learn]').forEach((b) =>
    b.addEventListener('click', () => {
      tab = b.dataset.learn as 'evo' | 'bp';
      document.querySelectorAll('[data-learn]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      document.querySelectorAll<HTMLElement>('[data-learn-pane]').forEach((p) => (p.hidden = p.dataset.learnPane !== tab));
      redraw();
    }),
  );
  const modes = (attr: string, set: (v: string) => void, draw: () => void) =>
    document.querySelectorAll<HTMLButtonElement>(`[${attr}]`).forEach((b) =>
      b.addEventListener('click', () => {
        set(b.getAttribute(attr)!);
        document.querySelectorAll(`[${attr}]`).forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
        draw();
      }),
    );
  modes('data-evo-mode', (v) => (evo.mode = v as typeof evo.mode), drawEvoPane);
  modes('data-bp-mode', (v) => (bp.mode = v as typeof bp.mode), drawBpPane);
  function redraw(): void {
    drawEvoPane();
    drawBpPane();
  }
  /** Kart, conditions or best driver changed: the student starts over with the new teacher. */
  function resetStudent(): void {
    bp.running = false;
    bp.worker?.terminate();
    Object.assign(bp, { worker: null, ready: false, last: null, teacher: null, losses: [], firstLap: -1 });
    $('bp-status').textContent = 'L’élève part de poids tirés au hasard : il ne sait rien.';
    const ts = deps.teachers();
    $('bp-teacher').textContent = ts.length > 1 ? `Professeur : le plus rapide ici entre le ${ts.map((t) => t.label).join(' et le ')}` : ts.length ? `Professeur : le ${ts[0].label}` : '';
    bpButton();
    drawBpPane();
  }
  new ResizeObserver(redraw).observe($('learn'));
  resetEvo();
  resetStudent();
  return { onGen, resetEvo, resetStudent, redraw };
}
