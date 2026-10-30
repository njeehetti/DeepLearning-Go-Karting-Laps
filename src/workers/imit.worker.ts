// Backpropagation lessons off the UI thread: the student copies the teacher, one lesson per request.
import { World, type WorldSpec } from '../ai/evalCore';
import { Imitation, bestLap, type Attempt, type LessonReport } from '../ai/imitation';

export type ToImit =
  | { type: 'start'; world: WorldSpec; teachers: number[][]; seed: number }
  | { type: 'lesson'; id: number };

export type FromImit =
  /** `pick`: which candidate teaches; `laps`: every candidate's own lap here. */
  | { type: 'ready'; pick: number; laps: Attempt[]; dim: number }
  | { type: 'lesson'; id: number; report: LessonReport }
  | { type: 'error'; message: string };

const post = (m: FromImit) => (self as unknown as Worker).postMessage(m);
let im: Imitation | null = null;

self.onmessage = (e: MessageEvent<ToImit>) => {
  const m = e.data;
  try {
    if (m.type === 'start') {
      const world = new World(m.world);
      // every candidate drives a lap here; the fastest clean one teaches (its lap is the line to copy)
      const probe = new Imitation(world, m.teachers[0], m.seed);
      const laps = m.teachers.map((t) => probe.attempt(false, t));
      const pick = bestLap(laps);
      im = new Imitation(world, m.teachers[pick], m.seed);
      post({ type: 'ready', pick, laps, dim: im.bp.dim });
    } else if (m.type === 'lesson') {
      if (!im) throw new Error('élève non initialisé');
      post({ type: 'lesson', id: m.id, report: im.step() });
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
