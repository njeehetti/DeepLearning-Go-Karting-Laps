/** Generic, serialisable circuit definition (what lives in data/tracks/*.json or comes from an import). */
export interface TrackDef {
  id: string;
  name: string;
  short?: string;
  place?: string;
  /** [lat, lon] in driving order; the loop is implicitly closed. */
  coords: [number, number][];
  /** Human label only ("horaire", "anti-horaire"); the order of coords is what counts. */
  direction?: string;
  /** Default track width in metres. */
  width: number;
  /** Official lap length in metres, used by the "official" scale option. */
  officialLength?: number | null;
  /** A flyover where the loop crosses itself; the later pass goes over unless lowerIsBridge. */
  bridge?: BridgeDef | null;
  /** Start line as a fraction of the loop; default = middle of the longest straight. */
  startFrac?: number;
  source?: 'osm' | 'import' | string;
  attribution?: string;
}

export interface BridgeDef {
  height: number;
  plateau?: number;
  ramp?: number;
  lowerIsBridge?: boolean;
}

/** 'osm' keeps OSM geometry as-is; 'official' rescales the smoothed centreline to officialLength. */
export type TrackScale = 'osm' | 'official';

export interface BuildOptions {
  width?: number;
  scale?: TrackScale;
}

export interface Crossing {
  sLow: number;
  sHigh: number;
  x: number;
  y: number;
}

export interface Corner {
  i0: number;
  i1: number;
  iMid: number;
  dir: 'G' | 'D';
  angle: number;
  radius: number;
  name: string;
  kind: 'Épingle' | 'Serré' | 'Moyen' | 'Rapide';
}

/** Resampled centreline every ~2 m, start line at index 0. x east, y north, z up (m). */
export interface Track {
  N: number;
  L: number;
  x: Float64Array;
  y: Float64Array;
  z: Float64Array;
  grade: Float64Array;
  tx: Float64Array;
  ty: Float64Array;
  /** Left normal. */
  nx: Float64Array;
  ny: Float64Array;
  /** Cumulative distance, length N + 1 (s[N] = L). */
  s: Float64Array;
  /** Smoothed signed centreline curvature (1/m, + = left). */
  kc: Float64Array;
  width: number;
  half: number;
  crossings: Crossing[];
  origin: { lat0: number; lon0: number };
  corners: Corner[];
}
