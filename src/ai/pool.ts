// Evaluation pools. Work is cut in small chunks handed out on demand (episodes have very uneven
// lengths), results are written back by candidate index so the outcome never depends on scheduling.
import { Evaluator, World, type CandidateResult, type FromWorker, type GenJob, type ToWorker, type WorldSpec } from './evalCore';

export interface EvalPool {
  readonly size: number;
  init(world: WorldSpec): Promise<void>;
  run(job: GenJob, lambda: number): Promise<CandidateResult[]>;
  close(): void;
}

/** Everything in the calling thread (tests, tiny runs). */
export class SyncPool implements EvalPool {
  readonly size = 1;
  private ev: Evaluator | null = null;

  async init(world: WorldSpec): Promise<void> {
    this.ev = new Evaluator(new World(world));
  }

  async run(job: GenJob, lambda: number): Promise<CandidateResult[]> {
    if (!this.ev) throw new Error('pool non initialisé');
    const out: CandidateResult[] = [];
    for (let i = 0; i < lambda; i++) out.push(this.ev.candidate(job, i));
    return out;
  }

  close(): void {}
}

/** Minimal worker surface shared by Node worker_threads and browser Web Workers. */
export interface WorkerLike {
  post(m: ToWorker): void;
  onMessage(cb: (m: FromWorker) => void): void;
  terminate(): void;
}

export class WorkerPool implements EvalPool {
  readonly size: number;
  private readonly workers: WorkerLike[];
  private readonly listeners: ((m: FromWorker) => void)[] = [];
  private nextId = 1;
  chunk = 8;

  constructor(workers: WorkerLike[]) {
    this.workers = workers;
    this.size = workers.length;
    workers.forEach((w, k) => w.onMessage((m) => this.listeners[k]?.(m)));
  }

  init(world: WorldSpec): Promise<void> {
    return Promise.all(
      this.workers.map(
        (w, k) =>
          new Promise<void>((resolve, reject) => {
            this.listeners[k] = (m) => (m.type === 'ready' ? resolve() : m.type === 'error' ? reject(new Error(m.message)) : undefined);
            w.post({ type: 'init', world });
          }),
      ),
    ).then(() => undefined);
  }

  run(job: GenJob, lambda: number): Promise<CandidateResult[]> {
    const id = this.nextId++;
    const out: CandidateResult[] = new Array(lambda);
    let next = 0;
    let pending = 0;
    return new Promise((resolve, reject) => {
      const feed = (k: number) => {
        if (next >= lambda) {
          if (pending === 0) resolve(out);
          return;
        }
        const from = next;
        const to = Math.min(lambda, from + this.chunk);
        next = to;
        pending++;
        this.workers[k].post({ type: 'job', id, job, from, to });
      };
      this.workers.forEach((_, k) => {
        this.listeners[k] = (m) => {
          if (m.type === 'error') return reject(new Error(m.message));
          if (m.type !== 'result' || m.id !== id) return;
          m.results.forEach((r, j) => (out[m.from + j] = r));
          pending--;
          feed(k);
          if (next >= lambda && pending === 0) resolve(out);
        };
        feed(k);
      });
    });
  }

  close(): void {
    this.workers.forEach((w) => w.terminate());
  }
}
