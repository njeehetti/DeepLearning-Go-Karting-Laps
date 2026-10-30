// Kart Lap Lab v2 UI: training control, live view, replay, ghosts, telemetry, corner sheet, network.
import './style.css';
import { buildTrack } from '../track/build';
import { TrackQuery, Proj } from '../track/query';
import { TRACKS, trackById } from '../track/registry';
import type { Calibration, CalibrationMode } from '../sim/calibrate';
import { DEFAULT_CALIBRATION, LAP_EVIDENCE } from '../sim/defaults';
import { ENGINES, wheelPower } from '../sim/engine';
import { KARTS, type KartId } from '../sim/karts';
import { LIVE_STRIDE, World, type WorldSpec } from '../ai/evalCore';
import { DRIVER_SHAPE, toDriverShape } from '../ai/mlp';
import generalist from '../../data/brains/generaliste-k270.json';
import sansLyl from '../../data/brains/sans-lyl-k270.json';
import { N_OBS } from '../ai/obs';
import { WorkerPool, type WorkerLike } from '../ai/pool';
import { cornerSheet, indexAtDist, poseAt, type CornerSheetRow, type Pose, type Run } from '../ai/record';
import { Trainer, type Brain, type GenStats } from '../ai/trainer';
import type { ToolRequest, ToolResponse } from '../workers/tool.worker';
import { drawEvo, drawTel, fmtLap, type TelGeom } from './charts';
import { NnView } from './nnview';
import { initLearn } from './learn';
import { Scene3D, type CamMode } from './scene';
import { applyTheme, skylineCanvas, themeOf } from './themes';
import { WEATHERS, WEATHER_IDS, type Weather } from '../sim/weather';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const SCALE = 'official' as const;
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
// circuit: ?c=<id> in the URL, else the last one chosen in this browser
const DEF = trackById(new URLSearchParams(location.search).get('c') ?? LS.get<string>('track'));
const track = buildTrack(DEF, { scale: SCALE });
const q = new TrackQuery(track);
// the page takes the colours and the skyline of the circuit's place
const THEME = themeOf(DEF.id);
applyTheme(THEME);
{
  const host = $('place');
  const sky = skylineCanvas(THEME, 1600, 170, '#0b0b0d');
  sky.setAttribute('aria-hidden', 'true');
  host.appendChild(sky);
}
// v4: 20-input network that does not know the corner speeds; older brains are ignored
// a lap time belongs to the driver's mass: brains trained at another mass than the default 75 kg are kept apart
const brainKeyOf = (id: string, k: KartId, w: Weather) => {
  const kg = calibOf(k).driver;
  return `brain4:${id}:${k}:${SCALE}${w === 'sec' ? '' : ':' + w}${kg === 75 ? '' : `:${kg}kg`}`;
};
const brainKey = (k: KartId, w: Weather = S.weather) => brainKeyOf(DEF.id, k, w);
/** Theoretical laps are cached per kart and track conditions. */
const thKey = (k: KartId, w: Weather = S.weather) => `${k}:${w}`;
// grip and power belong to the kart, not to the circuit (the LYL key is the pre-multi-circuit one)
const calibKey = (k: KartId) => `calib:kart:${k}`;
const recordKey = (k: KartId) => `record:${DEF.id}:${k}`;
/** The driver's own best lap on this circuit and kart (s), NaN if unknown. */
function myRecord(k: KartId): number {
  const r = LS.get<number>(recordKey(k));
  if (r != null) return r > 0 ? r : NaN;
  return DEF.id === 'lyl' && k === 'k270' ? LAP_EVIDENCE.driverBest270 : NaN;
}

// ---------- state ----------
const S = {
  kartId: (LS.get<KartId>('kart') ?? 'k270') as KartId,
  weather: (WEATHERS[LS.get<Weather>('weather') ?? 'sec'] ? LS.get<Weather>('weather') ?? 'sec' : 'sec') as Weather,
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
  ghostMode: 'none' as 'none' | 'theory' | 'older' | 'other' | 'guest',
  /** Driver of another circuit raced as the witness ghost (also next to the live population). */
  guest: null as { label: string; run: Run } | null,
  /** Driver the next training starts from, chosen in the other circuits' table. */
  seed: null as { label: string; brain: Brain } | null,
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
  /** 'follow': training waits until the live view has shown each generation; 'fast': full speed. */
  pace: (LS.get<string>('pace') === 'fast' ? 'fast' : 'follow') as 'follow' | 'fast',
  live: {
    traj: [] as (Float32Array | undefined)[],
    best: -1,
    gen: -1,
    lap: NaN,
    prevLap: NaN,
    /** Playback time of the generation being shown (s). */
    el: 0,
    /** Start countdown of the first generation (s left before the green light; then negative while "go" shows). */
    countdown: 0,
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

/** A brain on the current network: older 16-neuron brains are widened (they drive exactly the same). */
function asDriver(b: Brain | null): Brain | null {
  const params = b && Array.isArray(b.params) ? toDriverShape(b.params, b.shape) : null;
  return b && params ? { ...b, shape: DRIVER_SHAPE.sizes, params } : null;
}
const brainOf = (k: string) => asDriver(LS.get<Brain>(k));
/** Driver trained on every circuit at once, offered as a guest (null if its network is out of date). */
const GENERALIST = asDriver(generalist as Brain);
/** Same training on the 8 other circuits, never on the LYL: shows what a driver brings to an unknown track. */
const NEVER_LYL = asDriver(sansLyl as Brain);

function calibOf(k: KartId): Calibration {
  return LS.get<Calibration>(calibKey(k)) ?? LS.get<Calibration>(`calib:lyl:${k}`) ?? DEFAULT_CALIBRATION[k];
}
function worldOf(k: KartId, weather: Weather = S.weather): WorldSpec {
  const c = calibOf(k);
  return { track: DEF, scale: SCALE, kart: { kartId: k, mu: c.mu, powerScale: c.powerScale, vTopKmh: c.vTopKmh, transmission: c.transmission, driver: c.driver, weather } };
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
const scene = new Scene3D($<HTMLCanvasElement>('c3d'), q, THEME);
scene.setWeather(S.weather);
$('weather-note').textContent = S.weather === 'sec' ? '' : WEATHERS[S.weather].note;
// time of day: looks only, the physics ignores it
{
  const el = $<HTMLInputElement>('hour');
  const show = (h: number) => {
    $('hour-v').textContent = `${String(Math.floor(h) % 24).padStart(2, '0')}:${String(Math.round((h % 1) * 60)).padStart(2, '0')}`;
    scene.setTime(h);
  };
  el.value = String(LS.get<number>('hour') ?? 21);
  show(parseFloat(el.value));
  el.addEventListener('input', () => {
    const h = parseFloat(el.value);
    LS.set('hour', h);
    show(h);
  });
}
if (!scene.ok) $('no3d').hidden = false;
new ResizeObserver(() => {
  const r = $('view3d').getBoundingClientRect();
  scene.resize(r.width, r.height);
}).observe($('view3d'));
const nn = new NnView($('nn'));
// how the driver learns: the evolving weights, and a student copying the best driver by backpropagation
const learn = initLearn({
  track,
  world: () => worldOf(S.kartId),
  teachers: () => [
    ...(S.best ? [{ label: 'meilleur pilote de ce circuit', params: S.best.params }] : []),
    ...(GENERALIST ? [{ label: 'pilote généraliste', params: GENERALIST.params }] : []),
  ],
});

// ---------- header / meta ----------
function updateHeader(): void {
  $('t-kart').textContent = KARTS[S.kartId].label + (S.weather === 'sec' ? '' : ` · ${WEATHERS[S.weather].label.toLowerCase()}`);
  void renderWeather();
  const best = S.best?.lap ?? NaN;
  $('t-best').textContent = fmtLap(best);
  const th = S.theory.get(thKey(S.kartId));
  $('t-theory').textContent = th ? fmtLap(th.lap) : '…';
  const you = myRecord(S.kartId);
  $('t-you').textContent = fmtLap(you);
  $('t-you-row').hidden = !Number.isFinite(you);
  const gap = Number.isFinite(best) && th ? best - th.lap : NaN;
  $('t-gap').innerHTML = Number.isFinite(gap)
    ? `${gap > 0 ? `<span class="neg">+${fr(gap, 2)} s</span> sur le tour parfait` : `<span class="pos">−${fr(-gap, 2)} s</span> sous le tour parfait`}${Number.isFinite(you) ? ` · ${best <= you ? `<span class="pos">−${fr(you - best, 2)} s</span> sous ton record` : `+${fr(best - you, 2)} s sur ton record`}` : ''}`
    : 'Pas encore de tour sans sortie de piste';
  const gens = S.trainer?.history.length ?? 0;
  $('finding').innerHTML = !S.best
    ? gens
      ? `Génération ${gens} : les pilotes apprennent encore à boucler un tour sans sortir de la piste.`
      : `${S.pop} pilotes débutants vont découvrir ce circuit. Lance l'entraînement pour les voir apprendre.`
    : `Meilleur tour IA <b>${fmtLap(best)}</b>${th ? (gap > 0 ? `, à <b>${fr(gap, 2)} s</b> du tour parfait` : `, <b>${fr(-gap, 2)} s</b> sous le tour parfait`) : ''}${S.sheet.length && S.theorySheet.length ? ` · le plus de temps perdu en <b>${worstCorner()}</b>` : ''}${gens ? ` · génération ${gens}` : ''}.`;
  const pill = $('status');
  pill.className = 'status' + (S.running ? ' on' : '');
  pill.querySelector('span')!.textContent = S.running ? 'Entraînement en cours' : 'En pause';
  $('save-pilot').hidden = !S.trainer?.history.length;
  $('run').textContent = S.running ? 'Mettre en pause' : S.trainer && S.trainer.history.length ? 'Reprendre l’entraînement' : 'Lancer l’entraînement';
  $('k270').setAttribute('aria-pressed', String(S.kartId === 'k270'));
  $('k390').setAttribute('aria-pressed', String(S.kartId === 'k390'));
  // the guide lights up the step the user is at: choose, train, then compare once a driver exists
  const step = S.best ? 3 : S.running || gens ? 2 : 1;
  document.querySelectorAll<HTMLElement>('#steps li').forEach((li) => li.classList.toggle('on', li.dataset.step === String(step)));
}
// ---------- circuit picker (switching rebuilds everything, so it reloads the page) ----------
{
  const sel = $<HTMLSelectElement>('track-sel');
  sel.innerHTML = TRACKS.map((t) => `<option value="${t.id}"${t.id === DEF.id ? ' selected' : ''}>${t.short ?? t.name}</option>`).join('');
  sel.addEventListener('change', () => {
    LS.set('track', sel.value);
    const u = new URL(location.href);
    u.searchParams.delete('c');
    // assigning the same URL with a #hash only scrolls to the anchor: force the reload
    if (u.href === location.href) location.reload();
    else location.href = u.toString();
  });
  $('t-track').textContent = DEF.short ?? DEF.name;
  document.title = `Kart Lap Lab · ${DEF.short ?? DEF.name}`;
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
    ['Circuit', `${DEF.name}, ${fr(track.L)} m${DEF.direction ? `, sens ${DEF.direction}` : ''}`],
    ['Longueur officielle', DEF.lengthSource ?? 'tracé OSM'],
    ['Décor', THEME.caption],
  ];
  $('kart-meta').innerHTML = rows.map(([a, b, w]) => `<div><dt>${a}</dt><dd${w ? ' class="warn"' : ''}>${b}</dd></div>`).join('');
  ($('mylap') as HTMLInputElement).value = Number.isFinite(c.targetLap) ? fmtLap(c.targetLap) : '';
  ($('calmode') as HTMLSelectElement).value = c.mode;
  ($('weight') as HTMLInputElement).value = String(c.driver);
  const you = myRecord(S.kartId);
  ($('myrec') as HTMLInputElement).value = Number.isFinite(you) ? fmtLap(you) : '';
  $('myrec-l').textContent = `Ton record en ${KARTS[S.kartId].label} sur ${DEF.short ?? DEF.name}`;
}

// ---------- best brain, replay, ghosts ----------
async function loadTheory(k: KartId, w: Weather = S.weather): Promise<void> {
  if (S.theory.has(thKey(k, w))) return;
  const r = await ask({ type: 'theory', world: worldOf(k, w), gens: 300 });
  if (r.type === 'theory') {
    S.theory.set(thKey(k, w), { lap: r.lap, run: r.run });
    if (k === S.kartId && w === S.weather) S.theorySheet = cornerSheet(track, r.run);
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
    // nothing trained for this kart / circuit / condition yet: clear the replay, the line and the HUD
    S.run = null;
    S.sheet = [];
    scene.setLine([], [], [], () => [0, 0, 0], 0);
    scene.setHero(null);
    hud(null, '–');
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
    const th = S.theory.get(thKey(S.kartId));
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
        S.ghostLabel = `Génération ${old.gen + 1} : ${fmtLap(r.run.result.lapTime)}`;
      }
    } else S.ghostLabel = 'Pas encore 50 générations d’écart';
  } else if (S.ghostMode === 'other') {
    const other: KartId = S.kartId === 'k270' ? 'k390' : 'k270';
    const b = brainOf(brainKey(other));
    if (b) {
      const r = await ask({ type: 'record', world: worldOf(other), params: b.params });
      if (r.type === 'record') {
        S.ghost = r.run;
        S.ghostLabel = `${KARTS[other].label} : ${fmtLap(r.run.result.lapTime)}`;
      }
    } else S.ghostLabel = `Aucun cerveau ${KARTS[other].label} encore`;
  } else if (S.ghostMode === 'guest') {
    S.ghost = S.guest?.run ?? null;
    S.ghostLabel = S.guest ? `Témoin ${S.guest.label} : ${outcome(S.guest.run.result)}` : 'Choisis un témoin dans l’onglet Cerveau';
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
    // a new condition starts from the dry driver: it already knows the circuit, it only has to find the grip
    const stored = fresh ? null : brainOf(brainKey(S.kartId)) ?? brainOf(brainKey(S.kartId, 'sec'));
    // nothing yet on this circuit: start from the generalist, which already drives a lap almost anywhere
    const seed = fresh ? null : S.seed;
    S.seed = null;
    // a circuit never trained starts from zero: the pilots learn to drive in front of you (the generalist
    // or another circuit's driver is only a starting point when picked with « Partir de lui »)
    const x0 = seed?.brain ?? stored;
    if (seed) $('brain-msg').textContent = `Départ depuis le ${seed.label}.`;
    // same rolling start as the replay recorder (tool worker), so a recorded lap reproduces the trained one
    const local = new World(world);
    const startSpeed = (s: number) => local.startSpeed(s);
    const tr = new Trainer(S.pool, { world, lambda: S.pop, seed: (Math.random() * 1e9) | 0, L: track.L, startSpeed, x0: x0?.params, sigma0: stored && !seed ? 0.15 : x0 ? 0.25 : 0.5 });
    (tr as Trainer & { key?: string }).key = key;
    S.trainer = tr;
    learn.resetEvo();
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
    await waitForViewer(tr);
    if ((tr.best?.lap ?? Infinity) < before - 1e-6 && tr.best) {
      const brain = tr.brain()!;
      const prev = brainOf(brainKey(S.kartId));
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
  // the first generation lines up on the grid and waits for 3, 2, 1, go
  Object.assign(S.live, { traj: g.traj, best: g.best, gen: g.gen, lap: g.lap, prevLap: g.prevLap, el: 0, pending: null, countdown: g.gen === 0 ? 3 : 0 });
  S.live.hints.fill(0);
  S.live.dist.fill(0);
  S.live.init.fill(0);
}
const countdownEl = $('countdown');
countdownEl.lang = THEME.lang;
let countdownShown = '';
/** 3, 2, 1, then "go" for a second, in the language of the circuit's country. */
function showCountdown(left: number): void {
  const word = left > 0 ? THEME.countdown[3 - Math.ceil(left)] : left > -1 && S.live.gen === 0 ? THEME.countdown[3] : '';
  if (word === countdownShown) return;
  countdownShown = word;
  countdownEl.hidden = !word;
  countdownEl.textContent = word;
  countdownEl.classList.toggle('go', left <= 0);
  // restart the pop animation on each new word
  countdownEl.style.animation = 'none';
  void countdownEl.offsetWidth;
  countdownEl.style.animation = '';
}
/** "Chaque génération": the next generation is only computed once the viewer has seen this one race.
 *  Fast mode, another view, the replay or a hidden tab let the training run at full speed. */
async function waitForViewer(tr: Trainer): Promise<void> {
  while (S.running && S.trainer === tr && S.pace === 'follow' && S.mode === 'live' && S.view === 'piste' && !document.hidden && !liveDone())
    await new Promise((r) => setTimeout(r, 100));
}
/** A generation is shown until every driver has finished (or crashed), plus a short pause. */
function liveDone(): boolean {
  let n = 0;
  for (const tr of S.live.traj) if (tr) n = Math.max(n, tr.length / LIVE_STRIDE);
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
  learn.onGen(tr.es);
}
function drawEvoNow(): void {
  if (!S.trainer) return;
  $('evo-read').textContent = drawEvo($<HTMLCanvasElement>('evo'), S.trainer.history, S.theory.get(thKey(S.kartId))?.lap ?? NaN);
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
// a new population from zero; the saved record is kept and only replaced if the new pilots beat it
$('reset').addEventListener('click', async () => {
  S.running = false;
  await ensureTrainer(true);
  drawEvoNow();
  updateHeader();
  $('brain-msg').textContent = S.best
    ? `Nouveaux pilotes partis de zéro. Ton record (${fmtLap(S.best.lap)}) reste gardé tant qu’ils ne le battent pas.`
    : 'Nouveaux pilotes partis de zéro.';
});
async function setKart(k: KartId): Promise<void> {
  if (k === S.kartId) return;
  S.running = false;
  S.kartId = k;
  LS.set('kart', k);
  S.trainer = null;
  S.guest = null;
  learn.resetStudent();
  updateHeader();
  updateMeta();
  await setBest(brainOf(brainKey(k)));
  void loadTheory(k);
  void refreshGhost();
}
$('k270').addEventListener('click', () => void setKart('k270'));

async function setWeather(w: Weather): Promise<void> {
  if (w === S.weather) return;
  S.running = false;
  S.weather = w;
  LS.set('weather', w);
  S.trainer = null;
  S.guest = null;
  learn.resetStudent();
  scene.setWeather(w);
  $('weather-note').textContent = w === 'sec' ? '' : WEATHERS[w].note;
  updateHeader();
  await setBest(brainOf(brainKey(S.kartId)));
  void loadTheory(S.kartId);
  void refreshGhost();
  renderWeather();
  void renderGuests();
}
{
  const sel = $<HTMLSelectElement>('weather-sel');
  sel.innerHTML = WEATHER_IDS.map((w) => `<option value="${w}"${w === S.weather ? ' selected' : ''}>${WEATHERS[w].label}</option>`).join('');
  sel.addEventListener('change', () => void setWeather(sel.value as Weather));
}

// ---------- weather table: theoretical and AI laps in every condition ----------
let weatherBusy = false;
async function renderWeather(compute = false): Promise<void> {
  const tb = $('weather-rows');
  const dry = S.theory.get(thKey(S.kartId, 'sec'))?.lap ?? NaN;
  const dryAi = brainOf(brainKey(S.kartId, 'sec'))?.lap ?? NaN;
  tb.innerHTML = WEATHER_IDS.map((w) => {
    const th = S.theory.get(thKey(S.kartId, w))?.lap ?? NaN;
    const ai = brainOf(brainKey(S.kartId, w))?.lap ?? NaN;
    const d = (a: number, b: number) => (Number.isFinite(a) && Number.isFinite(b) && w !== 'sec' ? `<span class="neg">+${fr(a - b, 2)} s</span> <small>(+${fr((a / b - 1) * 100, 0)} %)</small>` : '');
    return `<tr${w === S.weather ? ' class="sel"' : ''}><td class="nm">${WEATHERS[w].label}</td><td class="num">${Number.isFinite(th) ? fmtLap(th) : '…'}</td><td class="num">${d(th, dry)}</td><td class="num">${Number.isFinite(ai) ? fmtLap(ai) : '<span class="faint">à entraîner</span>'}</td><td class="num">${d(ai, dryAi)}</td><td class="note-cell">${WEATHERS[w].note}</td></tr>`;
  }).join('');
  if (!compute || weatherBusy) return;
  weatherBusy = true;
  for (const w of WEATHER_IDS) {
    if (S.theory.has(thKey(S.kartId, w))) continue;
    await loadTheory(S.kartId, w);
    void renderWeather();
  }
  weatherBusy = false;
}
$('k390').addEventListener('click', () => void setKart('k390'));

function parseLap(s: string): number {
  const t = s.trim().replace(',', '.');
  const p = t.split(':');
  return p.length === 2 ? parseInt(p[0], 10) * 60 + parseFloat(p[1]) : parseFloat(t);
}
$('myrec').addEventListener('change', () => {
  const v = ($('myrec') as HTMLInputElement).value.trim();
  const t = parseLap(v);
  if (v === '') LS.set(recordKey(S.kartId), -1);
  else if (Number.isFinite(t) && t > 20 && t < 300) LS.set(recordKey(S.kartId), t);
  updateHeader();
  updateMeta();
});

// the driver's mass in kit: it belongs to the driver, so it applies to both karts
$('weight').addEventListener('change', () => {
  const el = $<HTMLInputElement>('weight');
  const kg = Math.round(parseFloat(el.value.replace(',', '.')));
  const msg = $('weight-msg');
  if (!Number.isFinite(kg) || kg < 40 || kg > 150) {
    msg.textContent = 'Écris un poids entre 40 et 150 kg.';
    return;
  }
  for (const k of ['k270', 'k390'] as KartId[]) {
    LS.set(calibKey(k), { ...calibOf(k), driver: kg });
    for (const w of WEATHER_IDS) S.theory.delete(thKey(k, w));
  }
  S.running = false;
  S.trainer = null;
  guestRuns.clear();
  learn.resetStudent();
  msg.textContent = `Les pilotes roulent maintenant à ${kg} kg. Le tour parfait est recalculé ; les cerveaux entraînés à un autre poids sont gardés à part.`;
  updateMeta();
  void setBest(brainOf(brainKey(S.kartId)));
  void loadTheory(S.kartId);
  void renderGuests();
});
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
  for (const w of WEATHER_IDS) S.theory.delete(thKey(S.kartId, w));
  updateMeta();
  updateHeader();
  void loadTheory(S.kartId);
});

// ---------- drivers of the other circuits: tested here, raced as a witness, or used as a starting point ----------
/** Lap time, or how far the driver got before leaving the track. */
function outcome(r: Run['result']): string {
  if (r.completed) return fmtLap(r.lapTime) + (r.clean ? '' : ' (herbe)');
  return `sorti à ${fr(r.progress)} m (${fr((r.progress / track.L) * 100)} %)`;
}
interface Guest {
  id: string;
  /** "pilote de Genk", "pilote du Mans": used in sentences. */
  label: string;
  name: string;
  home: string;
  brain: Brain;
  /** Saved by the user from a training run (can be removed). */
  saved?: boolean;
}
/** "de Genk", "du Mans". */
const of = (n: string) => (n.startsWith('Le ') ? 'du ' + n.slice(3) : 'de ' + n);

// ---------- pilots saved from a training run, to race them on other circuits ----------
interface SavedPilot {
  id: string;
  trackName: string;
  brain: Brain;
}
const savedPilots = () => LS.get<SavedPilot[]>('saved-pilots') ?? [];
$('save-pilot').addEventListener('click', () => {
  const b = S.trainer?.snapshot();
  const btn = $('save-pilot');
  if (!b) return;
  const list = savedPilots();
  const n = DEF.short ?? DEF.name;
  if (!list.some((p) => p.brain.track === b.track && p.brain.gen === b.gen && p.brain.kart.kartId === b.kart.kartId)) {
    LS.set('saved-pilots', [...list, { id: `s${Date.now()}`, trackName: n, brain: b }]);
    if (LS.get<SavedPilot[]>('saved-pilots')?.length !== list.length + 1) {
      btn.textContent = 'Stockage plein : exporte-le depuis la vue Cerveau';
      return;
    }
  }
  btn.textContent = `Sauvé : ${n} · gén. ${fr(b.gen + 1)}`;
  setTimeout(() => (btn.textContent = 'Sauver ce pilote'), 2500);
  void renderGuests();
});
function guests(): Guest[] {
  const out: Guest[] = GENERALIST ? [{ id: 'generaliste', label: 'pilote généraliste', name: 'Généraliste', home: 'les 9 circuits à la fois', brain: GENERALIST }] : [];
  if (NEVER_LYL) out.push({ id: 'sans-lyl', label: 'pilote qui n’a jamais vu le LYL', name: 'Jamais vu le LYL', home: 'les 8 autres circuits, à 80 kg', brain: NEVER_LYL });
  for (const t of TRACKS) {
    if (t.id === DEF.id) continue;
    const b = brainOf(brainKeyOf(t.id, S.kartId, S.weather)) ?? brainOf(brainKeyOf(t.id, S.kartId, 'sec'));
    const n = t.short ?? t.name;
    if (b) out.push({ id: t.id, label: `pilote ${of(n)}`, name: n, home: `${n} (${fmtLap(b.lap)})`, brain: b });
  }
  for (const p of savedPilots()) {
    const b = asDriver(p.brain);
    if (!b) continue;
    const gen = `génération ${fr(p.brain.gen + 1)}`;
    const lap = Number.isFinite(b.lap) && b.lap ? fmtLap(b.lap) : 'pas encore de tour propre';
    out.push({ id: p.id, label: `pilote ${of(p.trackName)} à la ${gen}`, name: `${p.trackName} · gén. ${fr(p.brain.gen + 1)}`, home: `${p.trackName}, ${gen} (${lap})`, brain: b, saved: true });
  }
  return out;
}
/** Laps driven here, per guest brain and conditions. */
const guestRuns = new Map<string, Run>();
const guestKey = (g: Guest) => `${g.id}:${S.kartId}:${S.weather}:${g.brain.createdAt}`;
let guestsBusy = false;
async function renderGuests(): Promise<void> {
  const list = guests();
  $('guest-rows').innerHTML = list.map((g) => {
    const run = guestRuns.get(guestKey(g));
    const here = run ? (run.result.completed ? `<b>${outcome(run.result)}</b>` : `<span class="neg">${outcome(run.result)}</span>`) : '<span class="faint">test en cours…</span>';
    return `<tr><td class="nm">${g.name}</td><td>${g.home}</td><td class="num">${here}</td>` +
      `<td class="acts"><button class="btn small" data-witness="${g.id}"${run ? '' : ' disabled'}>Faire rouler ici</button> <button class="btn small" data-seed="${g.id}">Partir de lui</button>` +
      `${g.saved ? ` <button class="btn small" data-remove="${g.id}" title="Retirer ce pilote sauvé">Retirer</button>` : ''}</td></tr>`;
  }).join('');
  $('guest-empty').hidden = list.length > 0;
  $('guest-rows').closest('.tablewrap')!.toggleAttribute('hidden', list.length === 0);
  if (guestsBusy) return;
  // one lap at a time, re-rendering after each so results appear as they come
  const todo = list.find((g) => !guestRuns.has(guestKey(g)));
  if (!todo) return;
  guestsBusy = true;
  const key = guestKey(todo);
  const r = await ask({ type: 'record', world: worldOf(S.kartId), params: todo.brain.params });
  guestsBusy = false;
  if (r.type !== 'record') return;
  guestRuns.set(key, r.run);
  void renderGuests();
}
$('guest-rows').addEventListener('click', (e) => {
  const btn = (e.target as HTMLElement).closest('button');
  if (!btn) return;
  if (btn.dataset.remove) {
    LS.set('saved-pilots', savedPilots().filter((p) => p.id !== btn.dataset.remove));
    void renderGuests();
    return;
  }
  const g = guests().find((x) => x.id === (btn.dataset.witness ?? btn.dataset.seed));
  if (!g) return;
  if (btn.dataset.witness) {
    const run = guestRuns.get(guestKey(g));
    if (!run) return;
    S.guest = { label: g.name, run };
    S.ghostMode = 'guest';
    $<HTMLSelectElement>('ghost').value = 'guest';
    void refreshGhost();
    $('guest-msg').textContent = `Le ${g.label} roule ici en fantôme jaune, à côté de tes pilotes en direct comme en replay.`;
    $('view3d').scrollIntoView({ behavior: 'smooth', block: 'center' });
  } else {
    S.running = false;
    S.trainer = null;
    S.seed = { label: g.label, brain: g.brain };
    updateHeader();
    $('guest-msg').textContent = `Le prochain entraînement partira du ${g.label} au lieu de débutants. Lance-le en haut de la page.`;
  }
});

$('export').addEventListener('click', () => {
  const b = S.best ?? brainOf(brainKey(S.kartId));
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
    const b = asDriver(o.brain ?? o);
    if (!b || !b.kart?.kartId) throw new Error('format');
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
function setPace(p: 'follow' | 'fast'): void {
  S.pace = p;
  LS.set('pace', p);
  document.querySelectorAll('[data-pace]').forEach((x) => x.setAttribute('aria-pressed', String(x.getAttribute('data-pace') === p)));
}
setPace(S.pace);
document.querySelectorAll<HTMLButtonElement>('[data-pace]').forEach((b) => b.addEventListener('click', () => setPace(b.dataset.pace as 'follow' | 'fast')));
$('h-catch').addEventListener('click', () => {
  if (S.live.pending) startLive(S.live.pending);
  else S.live.el = 1e6; // ends the generation on screen, the training computes the next one
});
$('go-invites').addEventListener('click', () => $('invites').scrollIntoView({ behavior: 'smooth', block: 'start' }));
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
      const trail = Number.isFinite(r.trail) ? `${Math.round(r.trail * 100)} %<small> · ${Math.round(r.brakePeak * 100)} %</small>` : '–';
      const bal = Math.abs(r.steerExcess) < 1 ? 'neutre' : r.steerExcess > 0 ? `sous-vire ${fr(r.steerExcess, 0)}°` : `survire ${fr(-r.steerExcess, 0)}°`;
      const steer = `<span class="${r.drift > 10 || Math.abs(r.steerExcess) >= 3 ? 'neg' : ''}">${bal}</span><small> · ${fr(r.drift, 0)}°</small>`;
      const g = S.ghostSheet[i];
      const dt = g ? r.sectorTime - g.sectorTime : NaN;
      const delta = Number.isFinite(dt) ? `<span class="${dt <= 0 ? 'pos' : 'neg'}">${dt <= 0 ? '−' : '+'}${fr(Math.abs(dt), 2)} s</span>` : '';
      return `<tr data-i="${i}"${i === S.selCorner ? ' class="sel"' : ''}><td class="nm">${r.name}</td><td class="kind"><span class="dir">${r.dir}</span>${r.kind} · ${Math.round(r.angle)}°</td><td class="num">${brake}</td><td class="num">${kmh(r.vEntry)} km/h</td><td class="num">${kmh(r.vApex)} km/h</td><td class="num">${kmh(r.vExit)} km/h</td><td class="num">${fr(r.latG, 2)} g</td><td class="num">${place}</td><td class="num">${trail}</td><td class="num">${steer}</td><td class="num">${delta}</td></tr>`;
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
function pedals(thr: number, brk: number): void {
  const t = Math.round(Math.min(1, Math.max(0, thr)) * 100);
  const b = Math.round(Math.min(1, Math.max(0, brk)) * 100);
  $('h-thr').style.height = `${t}%`;
  $('h-brk').style.height = `${b}%`;
  $('h-thr-v').textContent = String(t);
  $('h-brk-v').textContent = String(b);
}
const gctx = $<HTMLCanvasElement>('h-g').getContext('2d')!;
function hud(p: Pose | null, label: string): void {
  if (!p) {
    $('h-spd').textContent = '–';
    $('h-lap').textContent = label;
    pedals(0, 0);
    return;
  }
  $('h-spd').textContent = String(kmh(p.v));
  pedals(p.thr, p.brk);
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
  gctx.fillStyle = THEME.accent;
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

function liveHud(leader: number, v: number, el: number, thr: number, brk: number): void {
  const L = S.live;
  const n = L.traj.filter(Boolean).length;
  $('h-spd').textContent = leader >= 0 ? String(kmh(v)) : '–';
  pedals(leader >= 0 ? thr : 0, leader >= 0 ? brk : 0);
  // generations are numbered from 1 everywhere in the interface
  const behind = !!S.trainer && S.trainer.es.gen - 1 > L.gen;
  const training = behind ? ` · l’entraînement en est à la génération ${S.trainer!.es.gen}` : '';
  $('h-lap').textContent = `Génération ${L.gen + 1} · ${n} pilotes · ${fmtLap(el).slice(0, -2)}${training}`;
  // follow mode: skip the slow drivers of this generation; fast mode: jump to the latest one computed
  const btn = $('h-catch');
  btn.hidden = S.pace === 'follow' ? !S.running : !L.pending;
  btn.textContent = S.pace === 'follow' ? 'Génération suivante' : 'Voir la dernière génération';
  $('h-corner').textContent = leader >= 0 ? `En tête : pilote n°${leader + 1}${leader === L.best ? ' (meilleur tour de la génération)' : ''}` : '–';
  let delta = '';
  if (Number.isFinite(L.lap) && Number.isFinite(L.prevLap)) {
    const d = L.lap - L.prevLap;
    delta = d < -0.0005 ? ` · −${fr(-d, 3)} s sur la génération ${L.gen}` : ` · égal à la génération ${L.gen}`;
  }
  $('h-advice').textContent = Number.isFinite(L.lap)
    ? `Meilleur tour de cette génération : ${fmtLap(L.lap)}${delta}`
    : L.gen === 0
      ? 'Première génération : ils découvrent la piste et ne savent pas encore rouler'
      : 'Personne n’a encore bouclé un tour sans sortir de la piste';
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
    if (S.play.playing) {
      const was = S.live.countdown;
      S.live.countdown = Math.max(-1, was - dt);
      if (was <= 0) S.live.el += dt * S.play.rate;
    }
    showCountdown(S.live.countdown);
    const el = S.live.el;
    let leader = -1;
    let leaderV = 0;
    let leaderThr = 0;
    let leaderBrk = 0;
    const poses = S.live.traj.map((tr, i) => {
      if (!tr || tr.length < 3) return { x: 0, y: 0, z: -50, psi: 0 };
      const n = tr.length / LIVE_STRIDE;
      const f = Math.max(0, Math.min(n - 1, el * 10));
      const k = Math.floor(f);
      const u = f - k;
      const k2 = Math.min(n - 1, k + 1);
      const x = tr[k * LIVE_STRIDE] + (tr[k2 * LIVE_STRIDE] - tr[k * LIVE_STRIDE]) * u;
      const y = tr[k * LIVE_STRIDE + 1] + (tr[k2 * LIVE_STRIDE + 1] - tr[k * LIVE_STRIDE + 1]) * u;
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
        leaderV = k2 > k ? Math.hypot(tr[k2 * LIVE_STRIDE] - tr[k * LIVE_STRIDE], tr[k2 * LIVE_STRIDE + 1] - tr[k * LIVE_STRIDE + 1]) * 10 : 0;
        leaderThr = tr[k * LIVE_STRIDE + 3];
        leaderBrk = tr[k * LIVE_STRIDE + 4];
      }
      return { x, y, z: q.at(track.z, proj.s), psi: tr[k * LIVE_STRIDE + 2] };
    });
    scene.setCrowd(poses, leader);
    scene.setHero(null);
    // the witness races the population from the same rolling start; the other ghosts belong to the replay
    const g = S.ghostMode === 'guest' ? S.ghost : null;
    scene.setGhost(g && el <= g.t[g.n - 1] ? poseAt(g, el) : null);
    target = leader >= 0 ? poses[leader] : null;
    S.pose = target;
    if (frame % 3 === 0) liveHud(leader, leaderV, el, leaderThr, leaderBrk);
  } else {
    scene.setCrowd([], -1);
    showCountdown(0);
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
  if (v === 'analyse') void renderWeather(true);
  if (!document.querySelector(`.view[data-view="${v}"]`)) v = 'piste';
  if (v === 'piste') void renderGuests();
  S.view = v;
  document.querySelectorAll<HTMLElement>('.view').forEach((el) => (el.hidden = el.dataset.view !== v));
  document.querySelectorAll<HTMLElement>('.nav').forEach((a) => (a.dataset.view === v ? a.setAttribute('aria-current', 'page') : a.removeAttribute('aria-current')));
  requestAnimationFrame(() => {
    const r = $('view3d').getBoundingClientRect();
    scene.resize(r.width, r.height);
    sizeMinimap();
    drawEvoNow();
    drawTelNow(-1);
    learn.redraw();
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
    rows.push(`<li><span>Génération ${s.gen + 1}</span><span>${fmtLap(s.bestLap)}</span><span class="d${d < -0.0005 ? ' pos' : ''}">${dTxt}</span></li>`);
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
    el.innerHTML = '<p class="empty">Le premier tour sans sortie de piste remplira cette bande : un bloc par virage, rouge là où le pilote perd du temps.</p>';
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
    ctx.strokeStyle = THEME.accent;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(X(track.x[i]), Y(track.y[i]), 8, 0, 6.283);
    ctx.stroke();
  }
  if (S.pose) {
    ctx.fillStyle = THEME.accent;
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
void setBest(brainOf(brainKey(S.kartId)));
requestAnimationFrame(tick);
if (import.meta.env.DEV) (window as unknown as { __klab: unknown }).__klab = { scene, S, q };
