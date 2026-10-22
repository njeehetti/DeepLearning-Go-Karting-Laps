// Temporary entry until the v2 UI lands (J4); shows the v1 theoretical lap computed by the TS port.
import lyl from '../data/tracks/lyl.json';
import { buildTrack } from './track/build';
import type { TrackDef } from './track/types';
import { kartParams } from './sim/karts';
import { LineEvaluator } from './sim/qss';

const track = buildTrack(lyl as TrackDef);
const ev = new LineEvaluator(track, 16);
const T = ev.lapTime(new Float64Array(ev.M), kartParams('k390'));
const app = document.getElementById('app');
if (app) {
  app.innerHTML = `<h1>Kart Lap Lab v2</h1>
  <p>${lyl.name} : ${Math.round(track.L)} m, ${track.corners.length} virages.</p>
  <p>Tour sur l'axe central (390cc, solveur v1) : ${T.toFixed(2)} s.</p>
  <p><a href="/legacy/kart-lap-lab.html">Ouvrir la v1</a></p>`;
}
