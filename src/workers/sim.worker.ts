// Browser simulation worker (pool member). Same protocol as the Node worker.
import { makeHandler, type ToWorker } from '../ai/evalCore';

const handle = makeHandler((m) => (self as unknown as Worker).postMessage(m));
self.onmessage = (e: MessageEvent<ToWorker>) => handle(e.data);
