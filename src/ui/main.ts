// Kart Lap Lab v2 UI: training control, live view, replay, ghosts, telemetry, corner sheet, network.
import './style.css';
import lyl from '../../data/tracks/lyl.json';
import { buildTrack } from '../track/build';
import { TrackQuery, Proj } from '../track/query';
import type { TrackDef } from '../track/types';
import type { Calibration, CalibrationMode } from '../sim/calibrate';
import { DEFAULT_CALIBRATION, LAP_EVIDENCE } from '../sim/defaults';
import { ENGINES, wheelPower } from '../sim/engine';
import { KARTS, type KartId } from '../sim/karts';
import { World, type WorldSpec } from '../ai/evalCore';
import { DRIVER_SHAPE, paramCount } from '../ai/mlp';
import { N_OBS } from '../ai/obs';
import { WorkerPool, type WorkerLike } from '../ai/pool';
import { cornerSheet, indexAtDist, poseAt, type CornerSheetRow, type Pose, type Run } from '../ai/record';
import { Trainer, type Brain, type GenStats } from '../ai/trainer';
import type { ToolRequest, ToolResponse } from '../workers/tool.worker';
import { drawEvo, drawTel, fmtLap, type TelGeom } from './charts';
import { NnView } from './nnview';
import { Scene3D, type CamMode } from './scene';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const DEF = lyl as TrackDef;
const SCALE = 'official' as const;
const track = buildTrack(DEF, { scale: SCALE });
const q = new TrackQuery(track);
const G = 9.81;
const kmh = (v: number) => Math.round(v * 3.6);
const fr = (n: number, d = 0) => n.toLocaleString('fr-FR', { minimumFractionDigits: d, maximumFractionDigits: d });

// ---------- storage (per-browser conveniences; everything also exportable as JSON) ----------
const LS = {
  get<T>(k: string): T | null {
    try {
      const v = localStorage.getItem('klab2:' + k);
      return v ? (JSON.parse(v) as T) : null;
    } catch {
      return null;
    }
  },
  set(k: string, v: unknown) {
    try {
      localStorage.setItem('klab2:' + k, JSON.stringify(v));
    } catch {
      /* quota or privacy mode: keep working without persistence */
    }
  },
  del(k: string) {
    try {
      localStorage.removeItem('klab2:' + k);
    } catch {
      /* ignore */
    }
  },
};
// v2: 18-input network (wheel angle added); older brains are ignored
const brainKey = (k: KartId) => `brain2:${DEF.id}:${k}:${SCALE}`;
const calibKey = (k: KartId) => `calib:${DEF.id}:${k}`;

// ---------- state ----------
const S = {
  kartId: (LS.get<KartId>('kart') ?? 'k270') as KartId,
  pop: LS.get<number>('pop') ?? 256,
  running: false,
  trainer: null as Trainer | null,
  pool: null as WorkerPool | null,
  poolWorld: '',
  t0: 0,
  computeMs: 0,
  best: null as Brain | null,
  run: null as Run | null,
  sheet: [] as CornerSheetRow[],
  ghostMode: 'none' as 'none' | 'theory' | 'older' | 'other',
  ghost: null as Run | null,
  ghostLabel: '',
  ghostSheet: [] as CornerSheetRow[],
  theory: new Map<string, { lap: number; run: Run }>(),
  mode: 'live' as 'live' | 'replay',
  lineMode: 'speed' as 'speed' | 'pedals',
  theorySheet: [] as CornerSheetRow[],
  view: 'piste' as string,
  pose: null as { x: number; y: number } | null,
  cam: 'heli' as CamMode,
  play: { t: 0, playing: true, rate: 1 },
  live: {
    traj: [] as (Float32Array | undefined)[],
    best: -1,
    gen: -1,
    lap: NaN,
    prevLap: NaN,
    /** Playback time of the generation being shown (s). */
    el: 0,
    hints: new Int32Array(128),
    dist: new Float64Array(128),
    lastS: new Float64Array(128),
    init: new Uint8Array(128),
    /** Latest generation received while the current one is still racing. */
    pending: null as LiveGen | null,
  },
  hoverDist: -1,
  selCorner: -1,
};

function calibOf(k: KartId): Calibration {
  return LS.get<Calibration>(calibKey(k)) ?? DEFAULT_CALIBRATION[k];
}
function worldOf(k: KartId): WorldSpec {
  const c = calibOf(k);
  return { track: DEF, scale: SCALE, kart: { kartId: k, mu: c.mu, powerScale: c.powerScale, vTopKmh: c.vTopKmh, transmission: c.transmission, driver: c.driver } };
}

// ---------- workers ----------
const nWorkers = Math.max(1, Math.min(12, (navigator.hardwareConcurrency || 4) - 1));
$('workers-note').textContent = `${nWorkers} cœurs de calcul en parallèle ; l'interface reste fluide pendant l'entraînement.`;
function makePool(): WorkerPool {
  const ws: WorkerLike[] = Array.from({ length: nWorkers }, () => {
    const w = new Worker(new URL('../workers/sim.worker.ts', import.meta.url), { type: 'module' });
    return { post: (m) => w.postMessage(m), onMessage: (cb) => (w.onmessage = (e) => cb(e.data)), terminate: () => w.terminate() };
  });
  return new WorkerPool(ws);
}
const tool = new Worker(new URL('../workers/tool.worker.ts', import.meta.url), { type: 'module' });
let toolId = 1;
const toolWait = new Map<number, (r: ToolResponse) => void>();
tool.onmessage = (e: MessageEvent<ToolResponse>) => {
  toolWait.get(e.data.id)?.(e.data);
  toolWait.delete(e.data.id);
};
type DistOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;
function ask(req: DistOmit<ToolRequest, 'id'>): Promise<ToolResponse> {
  const id = toolId++;
  return new Promise((res) => {
    toolWait.set(id, res);
    tool.postMessage({ ...req, id });
  });
}

// ---------- 3D ----------
const scene = new Scene3D($<HTMLCanvasElement>('c3d'), q);
if (!scene.ok) $('no3d').hidden = false;
new ResizeObserver(() => {
  const r = $('view3d').getBoundingClientRect();
  scene.resize(r.width, r.height);
}).observe($('view3d'));
const nn = new NnView($('nn'));

// ---------- header / meta ----------
function updateHeader(): void {
  $('t-kart').textContent = KARTS[S.kartId].label;
  const best = S.best?.lap ?? NaN;
  $('t-best').textContent = fmtLap(best);
  const th = S.theory.get(S.kartId);
  $('t-theory').textContent = th ? fmtLap(th.lap) : '…';
  const you = S.kartId === 'k270' ? LAP_EVIDENCE.driverBest270 : NaN;
  $('t-you').textContent = fmtLap(you);
  const gap = Number.isFinite(best) && th ? best - th.lap : NaN;
  $('t-gap').innerHTML = Number.isFinite(gap)
    ? `<span class="neg">+${fr(gap, 2)} s</span> sur le tour parfait${Number.isFinite(you) ? ` · ${best <= you ? `<span class="pos">−${fr(you - best, 2)} s</span> sous ton record` : `+${fr(best - you, 2)} s sur ton record`}` : ''}`
    : 'Pas encore de tour propre';
  const gens = S.trainer?.history.length ?? 0;
  $('finding').innerHTML = !S.best
    ? `Lance l'entraînement : ${S.pop} pilotes vont apprendre la piste en parallèle.`
    : `Meilleur tour IA <b>${fmtLap(best)}</b>${th ? `, à <b>${fr(gap, 2)} s</b> du tour parfait` : ''}${S.sheet.length && S.theorySheet.length ? ` · le plus de temps perdu en <b>${worstCorner()}</b>` : ''}${gens ? ` · génération ${gens}` : ''}.`;
  const pill = $('status');
  pill.className = 'status' + (S.running ? ' on' : '');
  pill.querySelector('span')!.textContent = S.running ? 'Entraînement en cours' : 'En pause';
  $('run').textContent = S.running ? 'Mettre en pause' : S.trainer && S.trainer.history.length ? 'Reprendre l’entraînement' : 'Lancer l’entraînement';
  $('k270').setAttribute('aria-pressed', String(S.kartId === 'k270'));
  $('k390').setAttribute('aria-pressed', String(S.kartId === 'k390'));
}
function updateMeta(): void {
  const c = calibOf(S.kartId);
  const k = KARTS[S.kartId];
  const hp = (wheelPower(ENGINES[S.kartId]) * c.powerScale) / 0.85 / 745.7;
  const rows: [string, string, boolean?][] = [
    ['Moteur', `${k.engine}, ${k.hp} ch annoncés`],
    ['Puissance effective', `≈ ${fr(hp, 1)} ch (×${fr(c.powerScale, 2)})`, c.powerScale > 1.6],
    ['Adhérence µ', fr(c.mu, 2), c.mu > 1.7],
    ['Vitesse de pointe', `${c.vTopKmh} km/h (hypothèse)`],
    ['Transmission', c.transmission === 'cvt' ? 'variateur (hypothèse)' : 'directe, embrayage centrifuge'],
    ['Masse', `${k.kartMass} kg + pilote ${c.driver} kg`],
    ['Tour parfait visé', Number.isFinite(c.targetLap) ? fmtLap(c.targetLap) : 'pas de donnée'],
  ];
  $('kart-meta').innerHTML = rows.map(([a, b, w]) => `<div><dt>${a}</dt><dd${w ? ' class="warn"' : ''}>${b}</dd></div>`).join('');
  ($('mylap') as HTMLInputElement).value = Number.isFinite(c.targetLap) ? fmtLap(c.targetLap) : '';
  ($('calmode') as HTMLSelectElement).value = c.mode;
}

// ---------- best brain, replay, ghosts ----------
async function loadTheory(k: KartId): Promise<void> {
  if (S.theory.has(k)) return;
  const r = await ask({ type: 'theory', world: worldOf(k), gens: 300 });
  if (r.type === 'theory') {
    S.theory.set(k, { lap: r.lap, run: r.run });
    if (k === S.kartId) S.theorySheet = cornerSheet(track, r.run);
    renderStrip();
  }
  updateHeader();
  if (S.ghostMode === 'theory') void refreshGhost();
  if (S.trainer) drawEvoNow();
}

async function setBest(brain: Brain | null): Promise<void> {
  S.best = brain;
  updateHeader();
  nn.setParams(brain ? brain.params : null);
  if (!brain) {
    S.run = null;
    renderSheet();
    return;
  }
  const r = await ask({ type: 'record', world: worldOf(brain.kart.kartId), params: brain.params });
  if (r.type !== 'record') return;
  const oldT = S.run ? S.run.t[S.run.n - 1] : 0;
  S.run = r.run;
  if (oldT) S.play.t = (S.play.t / oldT) * r.run.t[r.run.n - 1];
  S.sheet = cornerSheet(track, r.run);
  drawLine();
  renderSheet();
  if (S.ghostMode === 'older') void refreshGhost();
}

async function refreshGhost(): Promise<void> {
  S.ghost = null;
  S.ghostLabel = '';
  if (S.ghostMode === 'theory') {
    const th = S.theory.get(S.kartId);
    if (th) {
      S.ghost = th.run;
      S.ghostLabel = `Tour théorique ${fmtLap(th.lap)}`;
    }
  } else if (S.ghostMode === 'older' && S.trainer && S.best) {
    const target = S.best.gen - 50;
    const old = [...S.trainer.perGen].reverse().find((p) => p.gen <= target);
    if (old) {
      const r = await ask({ type: 'record', world: worldOf(S.kartId), params: Array.from(old.params) });
      if (r.type === 'record') {
        S.ghost = r.run;
        S.ghostLabel = `Génération ${old.gen} : ${fmtLap(r.run.result.lapTime)}`;
      }
    } else S.ghostLabel = 'Pas encore 50 générations d’écart';
  } else if (S.ghostMode === 'other') {
    const other: KartId = S.kartId === 'k270' ? 'k390' : 'k270';
    const b = LS.get<Brain>(brainKey(other));
    if (b) {
      const r = await ask({ type: 'record', world: worldOf(other), params: b.params });
      if (r.type === 'record') {
        S.ghost = r.run;
        S.ghostLabel = `${KARTS[other].label} : ${fmtLap(r.run.result.lapTime)}`;
      }
    } else S.ghostLabel = `Aucun cerveau ${KARTS[other].label} encore`;
  }
  S.ghostSheet = S.ghost ? cornerSheet(track, S.ghost) : [];
  $('h-ghost').hidden = !S.ghostLabel;
  $('h-ghost').textContent = S.ghostLabel ? 'Fantôme · ' + S.ghostLabel : '';
  renderSheet();
}

/** Racing line of the best lap in 3D, coloured by speed or by pedals (green = throttle, red = brake). */
function drawLine(): void {
  const run = S.run;
  if (!run) return;
  const col = (k: number): [number, number, number] => {
    if (S.lineMode === 'pedals') {
      const b = run.brk[k];
      const t = run.thr[k];
      if (b > 0.03) return [0.35 + 0.6 * b, 0.12, 0.14];
      if (t > 0.03) return [0.12, 0.25 + 0.55 * t, 0.2 + 0.25 * t];
      return [0.45, 0.5, 0.55];
    }
    const u = Math.max(0, Math.min(1, (run.v[k] - 7) / 15));
    return [0.36 + 0.6 * u, 0.17 + 0.6 * u, 0.53 - 0.3 * u];
  };
  scene.setLine(run.x, run.y, run.z, col, run.n);
}

// ---------- training ----------
async function ensureTrainer(fresh = false): Promise<Trainer> {
  const world = worldOf(S.kartId);
  const key = JSON.stringify(world.kart) + S.pop;
  if (!S.pool) S.pool = makePool();
  if (S.poolWorld !== JSON.stringify(world.kart)) {
    await S.pool.init(world);
    S.poolWorld = JSON.stringify(world.kart);
  }
  if (!S.trainer || fresh || (S.trainer as Trainer & { key?: string }).key !== key) {
    const stored = fresh ? null : LS.get<Brain>(brainKey(S.kartId));
    // same rolling start as the replay recorder (tool worker), so a recorded lap reproduces the trained one
    const local = new World(world);
    const startSpeed = (s: number) => local.startSpeed(s);
    const tr = new Trainer(S.pool, { world, lambda: S.pop, seed: (Math.random() * 1e9) | 0, L: track.L, startSpeed, x0: stored?.params, sigma0: stored ? 0.15 : 0.5 });
    (tr as Trainer & { key?: string }).key = key;
    S.trainer = tr;
    S.computeMs = 0;
  }
  return S.trainer;
}
async function loop(): Promise<void> {
  const tr = await ensureTrainer();
  S.t0 = performance.now();
  while (S.running && S.trainer === tr) {
    tr.recordN = S.mode === 'live' ? 96 : 0;
    const before = tr.best?.lap ?? Infinity;
    const s = await tr.step();
    S.computeMs += s.wallMs;
    onGen(tr, s);
    if ((tr.best?.lap ?? Infinity) < before - 1e-6 && tr.best) {
      const brain = tr.brain()!;
      const prev = LS.get<Brain>(brainKey(S.kartId));
      if (!prev || brain.lap < prev.lap) LS.set(brainKey(S.kartId), brain);
      void setBest(brain);
    }
    await new Promise((r) => setTimeout(r, 0));
  }
}

interface LiveGen {
  traj: (Float32Array | undefined)[];
  best: number;
  gen: number;
  lap: number;
  prevLap: number;
}
function startLive(g: LiveGen): void {
  Object.assign(S.live, { traj: g.traj, best: g.best, gen: g.gen, lap: g.lap, prevLap: g.prevLap, el: 0, pending: null });
  S.live.hints.fill(0);
  S.live.dist.fill(0);
  S.live.init.fill(0);
}
/** A generation is shown until every driver has finished (or crashed), plus a short pause. */
function liveDone(): boolean {
  let n = 0;
  for (const tr of S.live.traj) if (tr) n = Math.max(n, tr.length / 3);
  return n === 0 || S.live.el * 10 >= n + 15;
}

function onGen(tr: Trainer, s: GenStats): void {
  $('s-gen').textContent = fr(s.gen + 1);
  $('s-comp').textContent = `${Math.round(s.completion * 100)} %`;
  $('s-clean').textContent = `${Math.round(s.cleanRate * 100)} %`;
  $('s-time').textContent = `${fr(S.computeMs / 1000)} s`;
  if (tr.lastTraj.some(Boolean)) {
    const h = tr.history;
    const g: LiveGen = { traj: tr.lastTraj, best: tr.lastBestIndex, gen: s.gen, lap: s.bestLap, prevLap: h.length > 1 ? h[h.length - 2].bestLap : NaN };
    if (!S.live.traj.some(Boolean) || liveDone()) startLive(g);
    else S.live.pending = g;
  }
  drawEvoNow();
  renderGens(tr);
  updateHeader();
}
function drawEvoNow(): void {
  if (!S.trainer) return;
  $('evo-read').textContent = drawEvo($<HTMLCanvasElement>('evo'), S.trainer.history, S.theory.get(S.kartId)?.lap ?? NaN);
}

// ---------- controls ----------
const pop = $<HTMLInputElement>('pop');
pop.value = String(S.pop);
$('pop-v').textContent = String(S.pop);
pop.addEventListener('input', () => ($('pop-v').textContent = pop.value));
pop.addEventListener('change', () => {
  S.pop = parseInt(pop.value, 10);
  LS.set('pop', S.pop);
});
$('run').addEventListener('click', () => {
  S.running = !S.running;
  updateHeader();
  if (S.running) void loop();
});
$('reset').addEventListener('click', async () => {
  S.running = false;
  LS.del(brainKey(S.kartId));
  await ensureTrainer(true);
  await setBest(null);
  drawEvoNow();
  updateHeader();
});
async function setKart(k: KartId): Promise<void> {
  if (k === S.kartId) return;
  S.running = false;
  S.kartId = k;
  LS.set('kart', k);
  S.trainer = null;
  updateHeader();
  updateMeta();
  await setBest(LS.get<Brain>(brainKey(k)));
  void loadTheory(k);
  void refreshGhost();
}
$('k270').addEventListener('click', () => void setKart('k270'));
$('k390').addEventListener('click', () => void setKart('k390'));

function parseLap(s: string): number {
  const t = s.trim().replace(',', '.');
  const p = t.split(':');
  return p.length === 2 ? parseInt(p[0], 10) * 60 + parseFloat(p[1]) : parseFloat(t);
}
$('calib').addEventListener('click', async () => {
  const target = parseLap(($('mylap') as HTMLInputElement).value);
  const msg = $('calib-msg');
  if (!Number.isFinite(target) || target < 40 || target > 300) {
    msg.textContent = 'Écris un temps au format 1:31.500.';
    return;
  }
  const c = calibOf(S.kartId);
  const mode = ($('calmode') as HTMLSelectElement).value as CalibrationMode;
  msg.textContent = 'Calage en cours (≈ 15 s, en arrière-plan)…';
  const r = await ask({ type: 'calibrate', world: worldOf(S.kartId), input: { kartId: S.kartId, targetLap: target, mode, mu: mode === 'power' ? c.mu : 1.15, powerScale: mode === 'grip' ? c.powerScale : 1, vTopKmh: c.vTopKmh, transmission: c.transmission, driver: c.driver } });
  if (r.type !== 'calibrate') {
    msg.textContent = 'Le calage a échoué.';
    return;
  }
  const res = r.result;
  LS.set(calibKey(S.kartId), res);
  msg.textContent = res.reachable
    ? `Calé : µ ${fr(res.mu, 2)}, puissance ×${fr(res.powerScale, 2)} → tour parfait ${fmtLap(res.lap)}. L'entraînement repart sur ce kart.`
    : `Inatteignable dans ce mode : au mieux ${fmtLap(res.lap)}. Essaie un autre paramètre.`;
  S.running = false;
  S.trainer = null;
  S.theory.delete(S.kartId);
  updateMeta();
  updateHeader();
  void loadTheory(S.kartId);
});

$('export').addEventListener('click', () => {
  const b = S.best ?? LS.get<Brain>(brainKey(S.kartId));
  if (!b) {
    $('brain-msg').textContent = 'Pas encore de cerveau à exporter : lance l’entraînement.';
    return;
  }
  const blob = new Blob([JSON.stringify({ brain: b, history: S.trainer?.history ?? [] })], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `cerveau-${b.track}-${b.kart.kartId}-${fmtLap(b.lap).replace(/[:.]/g, '')}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});
$<HTMLInputElement>('import').addEventListener('change', async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (!f) return;
  try {
    const o = JSON.parse(await f.text());
    const b: Brain = o.brain ?? o;
    if (!Array.isArray(b.params) || b.params.length !== paramCount(DRIVER_SHAPE) || !b.kart?.kartId) throw new Error('format');
    LS.set(brainKey(b.kart.kartId), b);
    $('brain-msg').textContent = `Cerveau ${KARTS[b.kart.kartId as KartId].label} importé (${fmtLap(b.lap)}).`;
    if (b.kart.kartId === S.kartId) await setBest(b);
  } catch {
    $('brain-msg').textContent = 'Fichier non reconnu : il faut un cerveau exporté par Kart Lap Lab.';
  }
});

document.querySelectorAll<HTMLButtonElement>('[data-mode]').forEach((b) =>
  b.addEventListener('click', () => {
    S.mode = b.dataset.mode as 'live' | 'replay';
    document.querySelectorAll('[data-mode]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  }),
);
document.querySelectorAll<HTMLButtonElement>('[data-line]').forEach((b) =>
  b.addEventListener('click', () => {
    S.lineMode = b.dataset.line as 'speed' | 'pedals';
    document.querySelectorAll('[data-line]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    drawLine();
  }),
);
document.querySelectorAll<HTMLButtonElement>('[data-cam]').forEach((b) =>
  b.addEventListener('click', () => setCam(b.dataset.cam as CamMode)),
);
function setCam(c: CamMode): void {
  S.cam = c;
  scene.setCam(c);
  document.querySelectorAll('[data-cam]').forEach((x) => x.setAttribute('aria-pressed', String(x.getAttribute('data-cam') === c)));
}
$('play').addEventListener('click', () => {
  S.play.playing = !S.play.playing;
  $('play').textContent = S.play.playing ? 'Pause' : 'Lecture';
});
$<HTMLSelectElement>('rate').addEventListener('change', (e) => (S.play.rate = parseFloat((e.target as HTMLSelectElement).value)));
$<HTMLSelectElement>('ghost').addEventListener('change', (e) => {
  S.ghostMode = (e.target as HTMLSelectElement).value as typeof S.ghostMode;
  void refreshGhost();
});
$<HTMLInputElement>('scrub').addEventListener('input', (e) => {
  if (S.run) S.play.t = (parseFloat((e.target as HTMLInputElement).value) / 1000) * S.run.t[S.run.n - 1];
  showReplay();
});
function showReplay(): void {
  if (S.mode !== 'replay') {
    S.mode = 'replay';
    document.querySelectorAll('[data-mode]').forEach((x) => x.setAttribute('aria-pressed', String(x.getAttribute('data-mode') === 'replay')));
  }
}
function seek(t: number): void {
  S.play.t = Math.max(0, t);
  showReplay();
  if (S.cam === 'free' || S.cam === 'heli') setCam('chase');
  if (S.view !== 'piste') location.hash = '#piste';
}

// ---------- corner sheet ----------
function renderSheet(): void {
  renderStrip();
  const tb = $('corners');
  const rows = S.sheet;
  $('th-delta').textContent = S.ghost ? 'Écart fantôme' : '';
  if (!rows.length) {
    tb.innerHTML = '';
    return;
  }
  const maxB = Math.max(1, ...rows.map((r) => r.brakeDist));
  tb.innerHTML = rows
    .map((r, i) => {
      const brake = r.brakeDist > 0 ? `<span class="bar" style="width:${Math.round((r.brakeDist / maxB) * 48)}px"></span>${Math.round(r.brakeDist)} m` : '<span class="pos">sans frein</span>';
      const place = r.apexFromInside >= 0 ? `${fr(r.apexFromInside, 1)} m du bord intérieur` : `vibreur intérieur (${fr(-r.apexFromInside, 1)} m)`;
      const g = S.ghostSheet[i];
      const dt = g ? r.sectorTime - g.sectorTime : NaN;
      const delta = Number.isFinite(dt) ? `<span class="${dt <= 0 ? 'pos' : 'neg'}">${dt <= 0 ? '−' : '+'}${fr(Math.abs(dt), 2)} s</span>` : '';
      return `<tr data-i="${i}"${i === S.selCorner ? ' class="sel"' : ''}><td class="nm">${r.name}</td><td class="kind"><span class="dir">${r.dir}</span>${r.kind} · ${Math.round(r.angle)}°</td><td class="num">${brake}</td><td class="num">${kmh(r.vEntry)} km/h</td><td class="num">${kmh(r.vApex)} km/h</td><td class="num">${kmh(r.vExit)} km/h</td><td class="num">${fr(r.latG, 2)} g</td><td class="num">${place}</td><td class="num">${delta}</td></tr>`;
    })
    .join('');
}
$('corners').addEventListener('click', (e) => {
  const tr = (e.target as HTMLElement).closest('tr');
  if (!tr) return;
  S.selCorner = parseInt(tr.dataset.i ?? '-1', 10);
  const r = S.sheet[S.selCorner];
  renderSheet();
  if (r) {
    seek(r.tBrake - 2);
    $('view3d').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
});

// ---------- telemetry ----------
let telGeom: TelGeom | null = null;
function drawTelNow(cursor: number): void {
  telGeom = drawTel($<HTMLCanvasElement>('tel'), S.run, S.ghost, track.L, cursor, S.hoverDist, S.sheet.map((r) => ({ dist: r.distApex, name: r.name })));
}
{
  const cv = $<HTMLCanvasElement>('tel');
  const distAt = (e: PointerEvent | MouseEvent) => {
    if (!telGeom) return -1;
    const r = cv.getBoundingClientRect();
    const d = ((e.clientX - r.left - telGeom.pad.l) / telGeom.W) * telGeom.L;
    return d >= 0 && d <= telGeom.L ? d : -1;
  };
  cv.addEventListener('pointermove', (e) => {
    S.hoverDist = distAt(e);
    const run = S.run;
    if (!run || S.hoverDist < 0) {
      $('tel-read').textContent = 'Survole pour lire, clique pour rejouer à cet endroit';
      return;
    }
    const k = indexAtDist(run, S.hoverDist);
    $('tel-read').textContent = `${fr(Math.round(run.dist[k]))} m · ${kmh(run.v[k])} km/h · ${run.brk[k] > 0.05 ? 'frein ' + Math.round(run.brk[k] * 100) + ' %' : 'gaz ' + Math.round(run.thr[k] * 100) + ' %'} · ${fr(Math.abs(run.ay[k]) / G, 2)} g lat.`;
  });
  cv.addEventListener('pointerleave', () => (S.hoverDist = -1));
  cv.addEventListener('click', (e) => {
    const d = distAt(e);
    if (d >= 0 && S.run) seek(S.run.t[indexAtDist(S.run, d)]);
  });
}

// ---------- HUD ----------
const gctx = $<HTMLCanvasElement>('h-g').getContext('2d')!;
function hud(p: Pose | null, label: string): void {
  if (!p) {
    $('h-spd').textContent = '–';
    $('h-lap').textContent = label;
    return;
  }
  $('h-spd').textContent = String(kmh(p.v));
  $('h-thr').style.height = `${Math.round(p.thr * 100)}%`;
  $('h-brk').style.height = `${Math.round(p.brk * 100)}%`;
  $('h-glat').textContent = `${fr(Math.abs(p.ay) / G, 2)} g`;
  $('h-glat-l').textContent = 'latéral';
  $('h-lap').textContent = label;
  const R = 40;
  gctx.clearRect(0, 0, 92, 92);
  gctx.strokeStyle = 'rgba(255,255,255,.18)';
  gctx.lineWidth = 1.5;
  for (const r of [R, R / 2]) {
    gctx.beginPath();
    gctx.arc(46, 46, r, 0, 6.283);
    gctx.stroke();
  }
  gctx.fillStyle = '#f1a33b';
  gctx.beginPath();
  gctx.arc(46 - (p.ay / G / 1.5) * R, 46 - (p.ax / G / 1.5) * R, 6, 0, 6.283);
  gctx.fill();
  const rows = S.sheet;
  if (!rows.length) return;
  let cur: CornerSheetRow | null = null;
  let nxt: CornerSheetRow | null = null;
  let nd = Infinity;
  for (const r of rows) {
    const d = r.distApex - p.dist;
    if (Math.abs(d) < 12) cur = r;
    if (d > 0 && d < nd) {
      nd = d;
      nxt = r;
    }
  }
  const show = cur ?? nxt;
  if (!show) return;
  $('h-corner').textContent = `${show.name} · ${show.kind} ${show.dir === 'G' ? 'gauche' : 'droite'}`;
  $('h-advice').textContent = cur
    ? `Corde à ${kmh(cur.vApex)} km/h, ${fr(Math.max(0, cur.apexFromInside), 1)} m du bord`
    : `${nxt!.brakeDist > 0 ? `Freiner ${Math.round(nxt!.brakeDist)} m avant la corde · ` : 'À fond · '}corde ${kmh(nxt!.vApex)} km/h dans ${Math.round(nd)} m`;
}

function liveHud(leader: number, v: number, el: number): void {
  const L = S.live;
  const n = L.traj.filter(Boolean).length;
  $('h-spd').textContent = leader >= 0 ? String(kmh(v)) : '–';
  $('h-thr').style.height = '0%';
  $('h-brk').style.height = '0%';
  const training = S.trainer && S.trainer.es.gen - 1 > L.gen ? ` · entraînement : génération ${S.trainer.es.gen}` : '';
  $('h-lap').textContent = `Génération ${L.gen} · ${n} pilotes · ${fmtLap(el).slice(0, -2)}${training}`;
  $('h-corner').textContent = leader >= 0 ? `En tête : pilote n°${leader + 1}${leader === L.best ? ' (meilleur tour de la génération)' : ''}` : '–';
  let delta = '';
  if (Number.isFinite(L.lap) && Number.isFinite(L.prevLap)) {
    const d = L.lap - L.prevLap;
    delta = d < -0.0005 ? ` · −${fr(-d, 3)} s vs génération ${L.gen - 1}` : ` · égal à la génération ${L.gen - 1}`;
  }
  $('h-advice').textContent = Number.isFinite(L.lap) ? `Meilleur tour de cette génération : ${fmtLap(L.lap)}${delta}` : 'Aucun tour propre dans cette génération pour l’instant';
  $('h-glat').textContent = `${Math.round(Math.max(0, Math.min(1, L.dist[leader] / track.L)) * 100)} %`;
  $('h-glat-l').textContent = 'du tour';
}

// ---------- main loop ----------
const proj = new Proj();
let last = performance.now();
let frame = 0;
function tick(now: number): void {
  requestAnimationFrame(tick);
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  frame++;
  if (document.hidden) return;
  let target: { x: number; y: number; z: number; psi: number } | null = null;
  let ahead: { x: number; y: number; z: number } | null = null;
  const live = S.mode === 'live' && S.live.traj.some(Boolean);
  if (live) {
    if (liveDone() && S.live.pending) startLive(S.live.pending);
    if (S.play.playing) S.live.el += dt * S.play.rate;
    const el = S.live.el;
    let leader = -1;
    let leaderV = 0;
    const poses = S.live.traj.map((tr, i) => {
      if (!tr || tr.length < 3) return { x: 0, y: 0, z: -50, psi: 0 };
      const n = tr.length / 3;
      const f = Math.max(0, Math.min(n - 1, el * 10));
      const k = Math.floor(f);
      const u = f - k;
      const k2 = Math.min(n - 1, k + 1);
      const x = tr[k * 3] + (tr[k2 * 3] - tr[k * 3]) * u;
      const y = tr[k * 3 + 1] + (tr[k2 * 3 + 1] - tr[k * 3 + 1]) * u;
      q.project(x, y, S.live.hints[i] || 0, proj, S.live.init[i] ? 6 : 400);
      S.live.hints[i] = proj.i;
      // unwrapped distance along the track, to know who leads the race
      if (!S.live.init[i]) {
        S.live.init[i] = 1;
        S.live.lastS[i] = proj.s;
      } else {
        let ds = proj.s - S.live.lastS[i];
        if (ds < -track.L / 2) ds += track.L;
        else if (ds > track.L / 2) ds -= track.L;
        S.live.dist[i] += ds;
        S.live.lastS[i] = proj.s;
      }
      if (leader < 0 || S.live.dist[i] > S.live.dist[leader]) {
        leader = i;
        leaderV = k2 > k ? Math.hypot(tr[k2 * 3] - tr[k * 3], tr[k2 * 3 + 1] - tr[k * 3 + 1]) * 10 : 0;
      }
      return { x, y, z: q.at(track.z, proj.s), psi: tr[k * 3 + 2] };
    });
    scene.setCrowd(poses, leader);
    scene.setHero(null);
    scene.setGhost(null);
    target = leader >= 0 ? poses[leader] : null;
    S.pose = target;
    if (frame % 3 === 0) liveHud(leader, leaderV, el);
  } else {
    scene.setCrowd([], -1);
  }
  if (!live && S.run) {
    const run = S.run;
    const T = run.t[run.n - 1];
    if (S.play.playing) S.play.t = (S.play.t + dt * S.play.rate) % T;
    const p = poseAt(run, S.play.t);
    scene.setHero(p);
    const g = S.ghost ? poseAt(S.ghost, S.play.t) : null;
    scene.setGhost(g);
    target = p;
    S.pose = p;
    const a = poseAt(run, S.play.t + Math.max(0.35, 13 / Math.max(5, p.v)));
    ahead = a;
    if (frame % 3 === 0) {
      hud(p, `Tour ${fmtLap(p.t).slice(0, -2)} / ${fmtLap(T)}`);
      ($('scrub') as HTMLInputElement).value = String(Math.round((S.play.t / T) * 1000));
      nn.show(run.obs.subarray(p.k * N_OBS, p.k * N_OBS + N_OBS));
    }
    if (frame % 6 === 0) drawTelNow(p.dist);
  }
  if (frame % 3 === 0 && S.view === 'piste') drawMinimap();
  scene.render(dt, target, ahead);
}

// ---------- views (icon rail) ----------
function showView(v: string): void {
  if (!document.querySelector(`.view[data-view="${v}"]`)) v = 'piste';
  S.view = v;
  document.querySelectorAll<HTMLElement>('.view').forEach((el) => (el.hidden = el.dataset.view !== v));
  document.querySelectorAll<HTMLElement>('.nav').forEach((a) => (a.dataset.view === v ? a.setAttribute('aria-current', 'page') : a.removeAttribute('aria-current')));
  requestAnimationFrame(() => {
    const r = $('view3d').getBoundingClientRect();
    scene.resize(r.width, r.height);
    sizeMinimap();
    drawEvoNow();
    drawTelNow(-1);
  });
}
window.addEventListener('hashchange', () => showView(location.hash.slice(1)));

// ---------- timing tower: generation list ----------
function renderGens(tr: Trainer): void {
  const h = tr.history;
  const rows: string[] = [];
  for (let i = h.length - 1; i >= 0 && rows.length < 6; i--) {
    const s = h[i];
    const prev = i > 0 ? h[i - 1].bestLap : NaN;
    const d = Number.isFinite(s.bestLap) && Number.isFinite(prev) ? s.bestLap - prev : NaN;
    const dTxt = Number.isFinite(d) ? (d < -0.0005 ? `−${fr(-d, 3)}` : '=') : '–';
    rows.push(`<li><span>Génération ${s.gen}</span><span>${fmtLap(s.bestLap)}</span><span class="d${d < -0.0005 ? ' pos' : ''}">${dTxt}</span></li>`);
  }
  $('gen-list').innerHTML = rows.join('') || '<li class="empty">Aucune génération pour l’instant</li>';
}

// ---------- signature: time lost per corner against the perfect lap ----------
function worstCorner(): string {
  let w = -1;
  let wd = -Infinity;
  S.sheet.forEach((r, i) => {
    const t = S.theorySheet[i];
    if (t && r.sectorTime - t.sectorTime > wd) {
      wd = r.sectorTime - t.sectorTime;
      w = i;
    }
  });
  return w >= 0 ? `${S.sheet[w].name} (+${fr(wd, 2)} s)` : '–';
}
function renderStrip(): void {
  const el = $('strip');
  if (!S.sheet.length) {
    el.innerHTML = '<p class="empty">Le premier tour propre remplira cette bande : un bloc par virage, rouge là où le pilote perd du temps.</p>';
    return;
  }
  const losses = S.sheet.map((r, i) => (S.theorySheet[i] ? r.sectorTime - S.theorySheet[i].sectorTime : NaN));
  const max = Math.max(0.3, ...losses.filter(Number.isFinite));
  el.innerHTML = S.sheet
    .map((r, i) => {
      const l = losses[i];
      const a = Number.isFinite(l) ? Math.max(0, Math.min(1, l / max)) : 0;
      // one ramp from the neutral raised tone to the negative colour, by share of the worst loss
      // faster than the perfect lap (the quasi-static bound is not exact corner by corner): green
      const bg = l < -0.02 ? 'color-mix(in srgb, var(--positive) 45%, var(--raised))' : `color-mix(in srgb, var(--negative) ${Math.round(a * 70)}%, var(--raised))`;
      return `<button data-i="${i}" aria-pressed="${i === S.selCorner}" style="background:${bg}" title="${r.name} · ${r.kind}"><b>${r.name}</b><span>${Number.isFinite(l) ? (l >= 0 ? '+' : '−') + fr(Math.abs(l), 2) : '–'}</span></button>`;
    })
    .join('');
  updateHeader();
}
$('strip').addEventListener('click', (e) => {
  const b = (e.target as HTMLElement).closest('button');
  if (!b) return;
  S.selCorner = parseInt(b.dataset.i ?? '-1', 10);
  const r = S.sheet[S.selCorner];
  renderSheet();
  if (r) seek(r.tBrake - 2);
});

// ---------- timing tower: track map with the kart ----------
const mm = { cv: $<HTMLCanvasElement>('minimap'), w: 0, h: 0, k: 1, cx: 0, cy: 0 };
function sizeMinimap(): void {
  const r = mm.cv.getBoundingClientRect();
  const d = Math.min(window.devicePixelRatio || 1, 2);
  if (r.width < 10) return;
  mm.cv.width = Math.round(r.width * d);
  mm.cv.height = Math.round(r.height * d);
  mm.w = r.width;
  mm.h = r.height;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < track.N; i++) {
    x0 = Math.min(x0, track.x[i]); x1 = Math.max(x1, track.x[i]); y0 = Math.min(y0, track.y[i]); y1 = Math.max(y1, track.y[i]);
  }
  mm.k = Math.min((mm.w - 24) / (x1 - x0), (mm.h - 16) / (y1 - y0));
  mm.cx = (x0 + x1) / 2;
  mm.cy = (y0 + y1) / 2;
}
function drawMinimap(): void {
  if (!mm.w) sizeMinimap();
  if (!mm.w) return;
  const ctx = mm.cv.getContext('2d')!;
  const d = mm.cv.width / mm.w;
  ctx.setTransform(d, 0, 0, d, 0, 0);
  ctx.clearRect(0, 0, mm.w, mm.h);
  const X = (x: number) => mm.w / 2 + (x - mm.cx) * mm.k;
  const Y = (y: number) => mm.h / 2 - (y - mm.cy) * mm.k;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  for (let i = 0; i <= track.N; i++) {
    const j = i % track.N;
    if (i) ctx.lineTo(X(track.x[j]), Y(track.y[j]));
    else ctx.moveTo(X(track.x[j]), Y(track.y[j]));
  }
  ctx.strokeStyle = '#3a3a3e';
  ctx.lineWidth = Math.max(3, track.width * mm.k);
  ctx.stroke();
  // start line and the selected corner
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(X(track.x[0]) - 2, Y(track.y[0]) - 2, 4, 4);
  const sel = S.sheet[S.selCorner];
  if (sel) {
    const i = track.corners[S.selCorner]?.iMid ?? 0;
    ctx.strokeStyle = '#f1a33b';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(X(track.x[i]), Y(track.y[i]), 8, 0, 6.283);
    ctx.stroke();
  }
  if (S.pose) {
    ctx.fillStyle = '#f1a33b';
    ctx.beginPath();
    ctx.arc(X(S.pose.x), Y(S.pose.y), 4.5, 0, 6.283);
    ctx.fill();
  }
}
new ResizeObserver(() => sizeMinimap()).observe(mm.cv);

// ---------- boot ----------
window.addEventListener('resize', () => {
  drawEvoNow();
  drawTelNow(-1);
});
showView(location.hash.slice(1) || 'piste');
updateHeader();
updateMeta();
setCam('heli');
void loadTheory(S.kartId);
void setBest(LS.get<Brain>(brainKey(S.kartId)));
requestAnimationFrame(tick);
if (import.meta.env.DEV) (window as unknown as { __klab: unknown }).__klab = { scene, S, q };
