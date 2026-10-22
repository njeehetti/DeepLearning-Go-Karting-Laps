// Node worker_threads entry point (headless training). Same protocol as the browser worker.
import { parentPort } from 'node:worker_threads';
import { makeHandler, type ToWorker } from './evalCore';

if (!parentPort) throw new Error('à lancer comme worker');
const port = parentPort;
const handle = makeHandler((m) => port.postMessage(m));
port.on('message', (m: ToWorker) => handle(m));
