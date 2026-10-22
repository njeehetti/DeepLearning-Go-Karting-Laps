// Driver observations: what the neural pilot "sees". All inputs are normalised to roughly [−1, 1].
import type { KartState } from '../sim/kart';
import type { TrackQuery } from '../track/query';

export const RAY_ANGLES = Float64Array.from({ length: 9 }, (_, k) => ((k - 4) / 4) * (Math.PI / 2));
export const RAY_RANGE = 40;
export const LOOKAHEAD = [10, 25, 50] as const;
export const N_OBS = RAY_ANGLES.length + 6 + LOOKAHEAD.length;

/** Labels, in input order (for the network view). */
export const OBS_LABELS = [
  ...Array.from(RAY_ANGLES, (a) => `Rayon ${Math.round((a * 180) / Math.PI)}°`),
  'Vitesse', 'Dérive', 'Lacet', 'Écart latéral', 'Erreur de cap', 'Braquage',
  ...LOOKAHEAD.map((d) => `Courbure à ${d} m`),
];

const clamp = (x: number, a: number) => (x > a ? a : x < -a ? -a : x);

/** Fills `out` (length N_OBS) and the raw ray distances `rays` (m). */
export function observe(q: TrackQuery, st: KartState, out: Float64Array, rays?: Float64Array, deltaMax = 0.4): void {
  const t = q.track;
  const c = Math.cos(st.psi);
  const s = Math.sin(st.psi);
  let k = 0;
  for (let r = 0; r < RAY_ANGLES.length; r++) {
    const a = RAY_ANGLES[r];
    const ca = Math.cos(a);
    const sa = Math.sin(a);
    const dist = q.rayEdges(st.x, st.y, c * ca - s * sa, s * ca + c * sa, st.idx, RAY_RANGE);
    if (rays) rays[r] = dist;
    out[k++] = dist / RAY_RANGE;
  }
  out[k++] = st.vx / 25;
  out[k++] = clamp(Math.atan2(st.vy, Math.max(Math.abs(st.vx), 1)) / 0.3, 3);
  out[k++] = clamp(st.r / 2, 3);
  out[k++] = clamp(st.d / t.half, 3);
  let he = st.psi - Math.atan2(t.ty[st.idx], t.tx[st.idx]);
  he -= Math.round(he / (2 * Math.PI)) * 2 * Math.PI;
  out[k++] = clamp(he / 0.5, 3);
  // current front wheel angle: without it the policy cannot know where its wheels are and saws the wheel
  out[k++] = st.delta / deltaMax;
  for (const ahead of LOOKAHEAD) out[k++] = clamp(q.curvature(st.s + ahead) * 8, 2);
}
