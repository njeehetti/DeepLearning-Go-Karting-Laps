// Place themes: each circuit gets its own accent, flag stripe, night sky, ground and a skyline drawn from
// something that belongs to the place (the Petronas towers seen from Kajang, the Dunlop bridge at Le Mans,
// the mine headframes and slag heaps of Genk, Vesuvius above Sarno, the Heckington windmill near PFi…).
// The skyline is drawn in code, used as the header backdrop and as the horizon of the 3D scene.

type Draw = (c: CanvasRenderingContext2D, w: number, h: number, rnd: () => number) => void;

export interface PlaceTheme {
  /** One-line caption of what the backdrop shows. */
  caption: string;
  accent: string;
  /** Ghost colour, chosen to stay apart from the accent. */
  ghost: string;
  /** Flag stripe, left to right. */
  flag: string[];
  /** Night sky: zenith and horizon. */
  sky: [string, string];
  /** Grass and tree base colours (0–1, before the baked floodlight). */
  grass: [number, number, number];
  trees: [number, number, number];
  /** Share of the usual tree count (dry places have fewer). */
  treeDensity: number;
  /** Start countdown in the local language: 3, 2, 1, go, and its language tag. */
  countdown: [string, string, string, string];
  lang: string;
  skyline: Draw;
}

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- drawing helpers (y grows downwards, the ground line is y = h) ----------
function hills(c: CanvasRenderingContext2D, w: number, h: number, rnd: () => number, base: number, amp: number, waves: number): void {
  const ph = rnd() * 6.28;
  const ph2 = rnd() * 6.28;
  c.beginPath();
  c.moveTo(0, h);
  for (let x = 0; x <= w; x += 4) {
    const u = x / w;
    c.lineTo(x, h - base - amp * (0.6 * Math.sin(u * waves * 6.28 + ph) + 0.4 * Math.sin(u * waves * 2.3 * 6.28 + ph2)) * 0.5 - amp * 0.5);
  }
  c.lineTo(w, h);
  c.closePath();
  c.fill();
}
function rect(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
  c.fillRect(x, y, w, h);
}
function palm(c: CanvasRenderingContext2D, x: number, g: number, s: number): void {
  c.beginPath();
  c.moveTo(x - s * 0.04, g);
  c.quadraticCurveTo(x + s * 0.1, g - s * 0.5, x + s * 0.06, g - s);
  c.lineTo(x + s * 0.1, g - s);
  c.quadraticCurveTo(x + s * 0.16, g - s * 0.5, x + s * 0.04, g);
  c.fill();
  c.lineWidth = s * 0.05;
  for (let k = 0; k < 7; k++) {
    const a = -Math.PI + (k / 6) * Math.PI;
    c.beginPath();
    c.moveTo(x + s * 0.08, g - s);
    c.quadraticCurveTo(x + s * 0.08 + Math.cos(a) * s * 0.3, g - s - s * 0.18, x + s * 0.08 + Math.cos(a) * s * 0.45, g - s + Math.abs(Math.sin(a)) * s * 0.05 + s * 0.12);
    c.stroke();
  }
}
function pine(c: CanvasRenderingContext2D, x: number, g: number, s: number): void {
  c.beginPath();
  c.moveTo(x, g - s);
  for (let k = 0; k < 4; k++) {
    const y = g - s + (s * (k + 1)) / 4.6;
    const r = (s * (k + 1)) / 9;
    c.lineTo(x + r, y);
    c.lineTo(x + r * 0.45, y);
  }
  c.lineTo(x + s * 0.06, g);
  c.lineTo(x - s * 0.06, g);
  for (let k = 3; k >= 0; k--) {
    const y = g - s + (s * (k + 1)) / 4.6;
    const r = (s * (k + 1)) / 9;
    c.lineTo(x - r * 0.45, y);
    c.lineTo(x - r, y);
  }
  c.closePath();
  c.fill();
}
function cypress(c: CanvasRenderingContext2D, x: number, g: number, s: number): void {
  c.beginPath();
  c.ellipse(x, g - s * 0.5, s * 0.09, s * 0.5, 0, 0, 6.29);
  c.fill();
}
function spire(c: CanvasRenderingContext2D, x: number, g: number, s: number): void {
  rect(c, x - s * 0.06, g - s * 0.55, s * 0.12, s * 0.55);
  c.beginPath();
  c.moveTo(x - s * 0.07, g - s * 0.55);
  c.lineTo(x, g - s);
  c.lineTo(x + s * 0.07, g - s * 0.55);
  c.fill();
}
function turbine(c: CanvasRenderingContext2D, x: number, g: number, s: number, a: number): void {
  c.beginPath();
  c.moveTo(x - s * 0.015, g);
  c.lineTo(x - s * 0.007, g - s);
  c.lineTo(x + s * 0.007, g - s);
  c.lineTo(x + s * 0.015, g);
  c.fill();
  c.lineWidth = s * 0.018;
  for (let k = 0; k < 3; k++) {
    const b = a + (k * 2 * Math.PI) / 3;
    c.beginPath();
    c.moveTo(x, g - s);
    c.lineTo(x + Math.cos(b) * s * 0.45, g - s + Math.sin(b) * s * 0.45);
    c.stroke();
  }
}
function blob(c: CanvasRenderingContext2D, x: number, y: number, rx: number, ry: number): void {
  c.beginPath();
  c.ellipse(x, y, rx, ry, 0, 0, 6.29);
  c.fill();
}

// ---------- one skyline per place ----------
const skylines: Record<string, Draw> = {
  // Kajang at night: low jungle hills, oil palms, and the Petronas Twin Towers on the Kuala Lumpur horizon
  lyl: (c, w, h, rnd) => {
    c.globalAlpha = 0.55;
    hills(c, w, h, rnd, h * 0.12, h * 0.18, 2);
    c.globalAlpha = 1;
    const tx = w * 0.68;
    const g = h - h * 0.1;
    for (let b = 0; b < 9; b++) rect(c, tx - w * 0.16 + b * w * 0.035 + rnd() * 6, g - h * (0.12 + rnd() * 0.22), w * 0.02, h * 0.4);
    for (const dx of [-1, 1]) {
      const x = tx + dx * w * 0.022;
      const W = w * 0.022;
      for (let k = 0; k < 6; k++) rect(c, x - (W * (1 - k * 0.1)) / 2, g - h * (0.12 * k + 0.12), W * (1 - k * 0.1), h * 0.13);
      c.beginPath();
      c.moveTo(x - 2, g - h * 0.84);
      c.lineTo(x, g - h * 0.98);
      c.lineTo(x + 2, g - h * 0.84);
      c.fill();
    }
    rect(c, tx - w * 0.02, g - h * 0.46, w * 0.04, h * 0.02);
    for (let k = 0; k < 14; k++) palm(c, rnd() * w, h, h * (0.28 + rnd() * 0.18));
  },
  // Le Mans: the tyre-shaped Dunlop bridge arching over the track, grandstands, the Sarthe woods
  lemans: (c, w, h, rnd) => {
    c.globalAlpha = 0.5;
    hills(c, w, h, rnd, h * 0.14, h * 0.1, 3);
    c.globalAlpha = 1;
    const bx = w * 0.5;
    const R = h * 0.62;
    const cy = h + R * 0.12;
    c.lineWidth = h * 0.11;
    c.beginPath();
    c.arc(bx, cy, R, Math.PI * 1.08, Math.PI * 1.92);
    c.stroke();
    // tread blocks across the tyre
    c.lineWidth = h * 0.02;
    for (let k = -5; k <= 5; k++) {
      const a = Math.PI * 1.5 + k * 0.075;
      c.beginPath();
      c.moveTo(bx + Math.cos(a) * (R - h * 0.08), cy + Math.sin(a) * (R - h * 0.08));
      c.lineTo(bx + Math.cos(a) * (R + h * 0.08), cy + Math.sin(a) * (R + h * 0.08));
      c.stroke();
    }
    rect(c, bx - R * 0.95, h - h * 0.32, R * 1.9, h * 0.035);
    for (const gx of [w * 0.12, w * 0.78]) {
      c.beginPath();
      c.moveTo(gx, h);
      c.lineTo(gx, h - h * 0.3);
      c.lineTo(gx + w * 0.12, h - h * 0.38);
      c.lineTo(gx + w * 0.12, h);
      c.fill();
    }
    for (let k = 0; k < 40; k++) blob(c, rnd() * w, h - h * 0.08, h * (0.06 + rnd() * 0.08), h * (0.06 + rnd() * 0.08));
  },
  // Genk: a mining town, its winding-tower headframes and the slag heaps (terrils) left by the coal
  genk: (c, w, h, rnd) => {
    c.globalAlpha = 0.6;
    for (const [x, s] of [[0.18, 0.55], [0.44, 0.4], [0.88, 0.5]] as const) {
      c.beginPath();
      c.moveTo(w * x - h * s * 1.4, h);
      c.quadraticCurveTo(w * x, h - h * s * 1.25, w * x + h * s * 1.4, h);
      c.fill();
    }
    c.globalAlpha = 1;
    for (const [x, s] of [[0.32, 0.85], [0.62, 0.7], [0.7, 0.55]] as const) {
      const X = w * x;
      const H = h * s;
      c.lineWidth = h * 0.025;
      c.beginPath();
      c.moveTo(X - H * 0.18, h);
      c.lineTo(X, h - H);
      c.lineTo(X + H * 0.18, h);
      c.moveTo(X, h - H);
      c.lineTo(X - H * 0.42, h);
      c.stroke();
      for (let k = 1; k < 5; k++) {
        const y = h - (H * k) / 5;
        const r = (H * 0.18 * (5 - k)) / 5;
        c.beginPath();
        c.moveTo(X - r, y);
        c.lineTo(X + r, y);
        c.stroke();
      }
      c.beginPath();
      c.arc(X, h - H, H * 0.09, 0, 6.29);
      c.lineWidth = h * 0.02;
      c.stroke();
      rect(c, X + H * 0.12, h - H * 0.3, H * 0.4, H * 0.3);
    }
    for (let k = 0; k < 30; k++) blob(c, rnd() * w, h, h * (0.05 + rnd() * 0.07), h * (0.05 + rnd() * 0.07));
  },
  // Kristianstad: Scanian pine forest and the twin-spired Holy Trinity church
  kristianstad: (c, w, h, rnd) => {
    c.globalAlpha = 0.5;
    for (let k = 0; k < 60; k++) pine(c, rnd() * w, h - h * 0.04, h * (0.25 + rnd() * 0.2));
    c.globalAlpha = 1;
    const cx = w * 0.58;
    rect(c, cx - w * 0.05, h - h * 0.32, w * 0.1, h * 0.32);
    c.beginPath();
    c.moveTo(cx - w * 0.055, h - h * 0.32);
    c.lineTo(cx, h - h * 0.45);
    c.lineTo(cx + w * 0.055, h - h * 0.32);
    c.fill();
    spire(c, cx - w * 0.04, h - h * 0.3, h * 0.6);
    spire(c, cx + w * 0.04, h - h * 0.3, h * 0.6);
    for (let k = 0; k < 70; k++) pine(c, rnd() * w, h, h * (0.3 + rnd() * 0.32));
  },
  // Lonato del Garda: Monte Baldo above the lake, cypress rows and the Rocca di Lonato on its hill
  lonato: (c, w, h, rnd) => {
    c.globalAlpha = 0.45;
    c.beginPath();
    c.moveTo(0, h - h * 0.3);
    for (let x = 0; x <= w; x += 6) c.lineTo(x, h - h * (0.42 + 0.3 * Math.exp(-(((x / w - 0.3) / 0.22) ** 2)) + 0.03 * Math.sin(x * 0.05)));
    c.lineTo(w, h - h * 0.3);
    c.lineTo(w, h);
    c.lineTo(0, h);
    c.fill();
    c.globalAlpha = 1;
    hills(c, w, h, rnd, h * 0.06, h * 0.12, 2);
    const rx = w * 0.76;
    c.beginPath();
    c.moveTo(rx - w * 0.14, h);
    c.quadraticCurveTo(rx, h - h * 0.5, rx + w * 0.14, h);
    c.fill();
    rect(c, rx - w * 0.06, h - h * 0.5, w * 0.12, h * 0.16);
    for (let k = 0; k < 7; k++) rect(c, rx - w * 0.06 + k * w * 0.019, h - h * 0.55, w * 0.01, h * 0.05);
    rect(c, rx - w * 0.015, h - h * 0.68, w * 0.03, h * 0.18);
    for (let k = 0; k < 4; k++) rect(c, rx - w * 0.015 + k * w * 0.009, h - h * 0.72, w * 0.005, h * 0.04);
    for (let k = 0; k < 22; k++) cypress(c, rnd() * w, h, h * (0.3 + rnd() * 0.25));
  },
  // Adria: the Po delta lagoon, Venetian mooring piles (bricole), reeds and a bell tower
  adria: (c, w, h, rnd) => {
    rect(c, 0, h - h * 0.05, w, h * 0.05);
    spire(c, w * 0.3, h, h * 0.7);
    rect(c, w * 0.22, h - h * 0.16, w * 0.16, h * 0.16);
    for (let k = 0; k < 9; k++) {
      const x = w * (0.45 + k * 0.06) + rnd() * 10;
      const s = h * (0.25 + rnd() * 0.1);
      for (const a of [-0.12, 0, 0.12]) {
        c.save();
        c.translate(x, h);
        c.rotate(a);
        rect(c, -2, -s, 4, s);
        c.restore();
      }
    }
    c.lineWidth = 1.5;
    for (let k = 0; k < 160; k++) {
      const x = rnd() * w;
      if (x > w * 0.42 && x < w * 0.98) continue;
      c.beginPath();
      c.moveTo(x, h);
      c.quadraticCurveTo(x + 3, h - h * 0.1, x + 4 + rnd() * 6, h - h * (0.12 + rnd() * 0.12));
      c.stroke();
    }
  },
  // Sarno: Vesuvius and the Somma ridge above the plain, lemon and olive trees
  sarno: (c, w, h, rnd) => {
    c.beginPath();
    c.moveTo(0, h);
    c.lineTo(w * 0.12, h - h * 0.1);
    c.lineTo(w * 0.36, h - h * 0.7);
    c.lineTo(w * 0.41, h - h * 0.64);
    c.lineTo(w * 0.44, h - h * 0.86);
    c.lineTo(w * 0.5, h - h * 0.92);
    c.lineTo(w * 0.53, h - h * 0.88);
    c.lineTo(w * 0.76, h - h * 0.12);
    c.lineTo(w, h);
    c.fill();
    c.globalAlpha = 0.3;
    blob(c, w * 0.5, h - h * 0.99, w * 0.03, h * 0.05);
    c.globalAlpha = 1;
    for (let k = 0; k < 40; k++) blob(c, rnd() * w, h - h * 0.05, h * (0.06 + rnd() * 0.05), h * (0.05 + rnd() * 0.04));
  },
  // Zuera: the flat-topped badlands of the Ebro valley and the wind farms of Aragon
  zuera: (c, w, h, rnd) => {
    c.globalAlpha = 0.55;
    c.beginPath();
    c.moveTo(0, h);
    let x = 0;
    while (x < w) {
      const top = h - h * (0.25 + rnd() * 0.25);
      const span = w * (0.08 + rnd() * 0.12);
      c.lineTo(x + span * 0.15, top);
      c.lineTo(x + span * 0.85, top);
      c.lineTo(x + span, h - h * 0.12);
      x += span;
    }
    c.lineTo(w, h);
    c.fill();
    c.globalAlpha = 1;
    for (let k = 0; k < 8; k++) turbine(c, w * (0.08 + k * 0.12) + rnd() * 20, h - h * 0.3, h * (0.4 + rnd() * 0.15), rnd() * 6.28);
    for (let k = 0; k < 30; k++) blob(c, rnd() * w, h, h * 0.04, h * 0.03);
  },
  // PFi, Lincolnshire: flat fenland hedgerows, the eight-sailed Heckington windmill and a church spire
  pfi: (c, w, h, rnd) => {
    rect(c, 0, h - h * 0.06, w, h * 0.06);
    for (let k = 0; k < 50; k++) blob(c, rnd() * w, h - h * 0.07, h * (0.05 + rnd() * 0.08), h * (0.04 + rnd() * 0.05));
    const mx = w * 0.62;
    c.beginPath();
    c.moveTo(mx - w * 0.03, h);
    c.lineTo(mx - w * 0.018, h - h * 0.5);
    c.lineTo(mx + w * 0.018, h - h * 0.5);
    c.lineTo(mx + w * 0.03, h);
    c.fill();
    c.beginPath();
    c.ellipse(mx, h - h * 0.5, w * 0.02, h * 0.06, 0, Math.PI, 0);
    c.fill();
    c.lineWidth = h * 0.02;
    for (let k = 0; k < 8; k++) {
      const a = 0.2 + (k * Math.PI) / 4;
      c.beginPath();
      c.moveTo(mx, h - h * 0.52);
      c.lineTo(mx + Math.cos(a) * h * 0.36, h - h * 0.52 + Math.sin(a) * h * 0.36);
      c.stroke();
    }
    spire(c, w * 0.22, h, h * 0.62);
    rect(c, w * 0.16, h - h * 0.18, w * 0.08, h * 0.18);
  },
};

type Colours = Omit<PlaceTheme, 'skyline'>;
const COLOURS: Record<string, Colours> = {
  lyl: { caption: 'Kajang la nuit, les tours Petronas à l’horizon', accent: '#f1a33b', ghost: '#4a8ad6', flag: ['#cc0001', '#ffffff', '#010066', '#ffcc00'], sky: ['#05080d', '#14202a'], grass: [0.12, 0.2, 0.13], trees: [0.05, 0.11, 0.07], treeDensity: 1, countdown: ['Tiga', 'Dua', 'Satu', 'Mula!'], lang: 'ms' },
  lemans: { caption: 'Le pont Dunlop et les tribunes du circuit des 24 Heures', accent: '#6c9bff', ghost: '#efe3c8', flag: ['#0055a4', '#ffffff', '#ef4135'], sky: ['#05070f', '#151d33'], grass: [0.11, 0.19, 0.12], trees: [0.05, 0.1, 0.07], treeDensity: 1, countdown: ['Trois', 'Deux', 'Un', 'Partez !'], lang: 'fr' },
  genk: { caption: 'Les chevalements et les terrils de la mine de Genk', accent: '#e9c46a', ghost: '#7fb0ff', flag: ['#000000', '#fdda24', '#ef3340'], sky: ['#070708', '#1d1b18'], grass: [0.11, 0.17, 0.11], trees: [0.05, 0.09, 0.06], treeDensity: 0.9, countdown: ['Drie', 'Twee', 'Eén', 'Start!'], lang: 'nl' },
  kristianstad: { caption: 'Les pins de Scanie et l’église de la Trinité', accent: '#8fd3ff', ghost: '#ffd36b', flag: ['#006aa7', '#fecc02', '#006aa7'], sky: ['#030810', '#0d2230'], grass: [0.09, 0.17, 0.12], trees: [0.03, 0.09, 0.06], treeDensity: 1.6, countdown: ['Tre', 'Två', 'Ett', 'Kör!'], lang: 'sv' },
  lonato: { caption: 'Le Monte Baldo au-dessus du lac de Garde et la Rocca di Lonato', accent: '#dfe05a', ghost: '#7fb0ff', flag: ['#009246', '#ffffff', '#ce2b37'], sky: ['#05080e', '#17222e'], grass: [0.12, 0.2, 0.12], trees: [0.04, 0.1, 0.06], treeDensity: 1, countdown: ['Tre', 'Due', 'Uno', 'Via!'], lang: 'it' },
  adria: { caption: 'La lagune du delta du Pô et ses bricole', accent: '#2ec4d6', ghost: '#ffb37a', flag: ['#009246', '#ffffff', '#ce2b37'], sky: ['#040a0e', '#11252b'], grass: [0.12, 0.2, 0.14], trees: [0.05, 0.11, 0.08], treeDensity: 0.6, countdown: ['Tre', 'Due', 'Uno', 'Via!'], lang: 'it' },
  sarno: { caption: 'Le Vésuve et le Somma au-dessus de la plaine de Sarno', accent: '#ff7a3d', ghost: '#7fb0ff', flag: ['#009246', '#ffffff', '#ce2b37'], sky: ['#0a0607', '#2a1712'], grass: [0.13, 0.19, 0.11], trees: [0.06, 0.1, 0.05], treeDensity: 0.9, countdown: ['Tre', 'Due', 'Uno', 'Via!'], lang: 'it' },
  zuera: { caption: 'Les plateaux de l’Èbre et les éoliennes d’Aragon', accent: '#e8a46a', ghost: '#7fb0ff', flag: ['#aa151b', '#f1bf00', '#aa151b'], sky: ['#08070b', '#241c1a'], grass: [0.22, 0.18, 0.11], trees: [0.09, 0.1, 0.06], treeDensity: 0.25, countdown: ['Tres', 'Dos', 'Uno', '¡Ya!'], lang: 'es' },
  pfi: { caption: 'Les haies du Lincolnshire et le moulin à huit ailes de Heckington', accent: '#b9a7ff', ghost: '#ffd36b', flag: ['#012169', '#ffffff', '#c8102e'], sky: ['#05060c', '#161a2a'], grass: [0.11, 0.2, 0.12], trees: [0.05, 0.11, 0.07], treeDensity: 0.8, countdown: ['Three', 'Two', 'One', 'Go!'], lang: 'en' },
};

export function themeOf(id: string): PlaceTheme {
  const key = COLOURS[id] ? id : 'lyl';
  return { ...COLOURS[key], skyline: skylines[key] };
}

/** Draws the place's skyline on a transparent canvas, as a silhouette in `ink`. */
export function skylineCanvas(theme: PlaceTheme, w: number, h: number, ink: string): HTMLCanvasElement {
  const cv = document.createElement('canvas');
  cv.width = w;
  cv.height = h;
  const c = cv.getContext('2d')!;
  c.fillStyle = ink;
  c.strokeStyle = ink;
  c.lineCap = 'round';
  theme.skyline(c, w, h, rng(11));
  return cv;
}

/** Applies the theme's colours to the page (CSS tokens read by the stylesheet and the canvas charts). */
export function applyTheme(theme: PlaceTheme): void {
  const r = document.documentElement.style;
  r.setProperty('--accent', theme.accent);
  r.setProperty('--ghost', theme.ghost);
  const n = theme.flag.length;
  r.setProperty('--flag', `linear-gradient(90deg, ${theme.flag.map((c, i) => `${c} ${(i / n) * 100}% ${((i + 1) / n) * 100}%`).join(', ')})`);
}
