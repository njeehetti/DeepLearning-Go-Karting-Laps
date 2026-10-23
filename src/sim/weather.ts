// Track conditions. Rental karts run the same slick-ish tyres whatever the weather, so conditions act on
// the grip of the asphalt, of the kerbs (painted, the first to go in the wet) and of the grass, and on
// rolling resistance (water on the track). Orders of magnitude from kart practice: a "green" or greasy
// track (dust, oil, no rubber laid down) costs ~10 % of grip, a damp drying track ~25 %, rain on slicks
// ~40 %; kerbs lose far more than the asphalt.

export type Weather = 'sec' | 'gras' | 'humide' | 'pluie';

export interface WeatherDef {
  label: string;
  /** Short explanation shown in the UI. */
  note: string;
  /** Multipliers of the dry asphalt grip, of the kerb and grass grips, and of rolling resistance. */
  grip: number;
  kerb: number;
  grass: number;
  crr: number;
}

export const WEATHERS: Record<Weather, WeatherDef> = {
  sec: { label: 'Sec', note: 'Piste sèche et gommée.', grip: 1, kerb: 1, grass: 1, crr: 1 },
  gras: { label: 'Piste grasse', note: 'Poussière, huile ou piste verte sans gomme : environ 12 % d’adhérence en moins, vibreurs glissants.', grip: 0.88, kerb: 0.8, grass: 0.9, crr: 1 },
  humide: { label: 'Humide', note: 'Piste qui sèche après la pluie : environ 25 % d’adhérence en moins, vibreurs très glissants.', grip: 0.75, kerb: 0.55, grass: 0.7, crr: 1.15 },
  pluie: { label: 'Pluie', note: 'Pluie sur pneus slicks : environ 40 % d’adhérence en moins, l’eau freine les roues, vibreurs et herbe comme de la glace.', grip: 0.6, kerb: 0.4, grass: 0.55, crr: 1.4 },
};

export const WEATHER_IDS = Object.keys(WEATHERS) as Weather[];

/** Applies the conditions to dry chassis parameters (returns a modified copy). */
export function withWeather<P extends { mu: number; kerbGrip: number; grassGrip: number; Crr: number; grassCrr: number }>(p: P, w: Weather | undefined): P {
  const d = WEATHERS[w ?? 'sec'];
  if (d.grip === 1 && d.kerb === 1 && d.grass === 1 && d.crr === 1) return p;
  return { ...p, mu: p.mu * d.grip, kerbGrip: p.kerbGrip * d.kerb, grassGrip: p.grassGrip * d.grass, Crr: p.Crr * d.crr, grassCrr: p.grassCrr * d.crr };
}
