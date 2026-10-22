// Night scene of the circuit (Three.js, imperative): floodlit track with baked sodium light pools,
// bridge, kerbs, tyre walls, a hero kart, a ghost kart and an instanced population for the live view.
import * as THREE from 'three';
import type { TrackQuery } from '../track/query';
import type { Pose } from '../ai/record';

export type CamMode = 'onboard' | 'chase' | 'heli' | 'free';

/** Track coords (x east, y north, z up) → three (X, Y up, Z south). */
const P = (x: number, y: number, z: number) => new THREE.Vector3(x, z, -y);

interface KartModel {
  root: THREE.Group;
  body: THREE.Group;
  paint: THREE.MeshLambertMaterial | THREE.MeshBasicMaterial;
  driver: THREE.Group;
}

function makeKart(ghost: boolean, color: number): KartModel {
  const root = new THREE.Group();
  const body = new THREE.Group();
  root.add(body);
  const mk = (c: number) => (ghost ? new THREE.MeshBasicMaterial({ color: 0x9fd3ff, transparent: true, opacity: 0.32, depthWrite: false }) : new THREE.MeshLambertMaterial({ color: c }));
  const paint = mk(color);
  const dark = mk(0x202428);
  const tyre = mk(0x111213);
  const box = (w: number, h: number, d: number, m: THREE.Material, x: number, y: number, z: number) => {
    const o = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m);
    o.position.set(x, y, z);
    body.add(o);
  };
  box(1.7, 0.05, 1.0, dark, 0, 0.09, 0);
  box(0.42, 0.22, 0.95, paint, 0.82, 0.17, 0);
  box(0.95, 0.16, 0.16, paint, 0, 0.16, 0.56);
  box(0.95, 0.16, 0.16, paint, 0, 0.16, -0.56);
  box(0.12, 0.14, 1.32, dark, -0.98, 0.16, 0);
  box(0.42, 0.34, 0.42, dark, -0.28, 0.3, 0);
  box(0.22, 0.24, 0.36, dark, -0.62, 0.3, -0.3);
  for (const w of [[0.6, 0.6, 0.15], [0.6, -0.6, 0.15], [-0.62, 0.64, 0.22], [-0.62, -0.64, 0.22]]) {
    const c = new THREE.Mesh(new THREE.CylinderGeometry(0.135, 0.135, w[2], 14), tyre);
    c.rotation.x = Math.PI / 2;
    c.position.set(w[0], 0.135, w[1]);
    body.add(c);
  }
  const driver = new THREE.Group();
  body.add(driver);
  const torso = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.52, 0.42), mk(0x2b4a7a));
  torso.position.set(-0.22, 0.62, 0);
  torso.rotation.z = -0.25;
  driver.add(torso);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.155, 14, 10), mk(0xf4f4f2));
  head.position.set(-0.12, 0.98, 0);
  driver.add(head);
  return { root, body, paint, driver };
}

let glowTex: THREE.Texture | null = null;
/** Soft radial halo for the floodlights. */
function glowTexture(): THREE.Texture {
  if (glowTex) return glowTex;
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d')!;
  const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, 'rgba(255,220,160,1)');
  grd.addColorStop(0.25, 'rgba(255,180,90,.45)');
  grd.addColorStop(1, 'rgba(255,160,60,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, 64, 64);
  glowTex = new THREE.CanvasTexture(c);
  return glowTex;
}

function mulberry(a: number) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Scene3D {
  readonly ok: boolean = false;
  private renderer!: THREE.WebGLRenderer;
  private scene!: THREE.Scene;
  readonly cam!: THREE.PerspectiveCamera;
  private hero!: KartModel;
  private ghost!: KartModel;
  private crowd!: THREE.InstancedMesh;
  private crowdBest!: KartModel;
  private center = new THREE.Vector3();
  private orbit = { th: 0.7, ph: 0.9, r: 260 };
  private look: THREE.Vector3 | null = null;
  private chaseInit = false;
  mode: CamMode = 'heli';
  private readonly m4 = new THREE.Matrix4();
  private readonly q4 = new THREE.Quaternion();
  private readonly v3 = new THREE.Vector3();
  private readonly s3 = new THREE.Vector3(1, 1, 1);
  private readonly up = new THREE.Vector3(0, 1, 0);

  constructor(private readonly canvas: HTMLCanvasElement, private readonly q: TrackQuery) {
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    } catch (e) {
      console.error('WebGL indisponible', e);
      return;
    }
    (this as { ok: boolean }).ok = true;
    this.renderer = renderer;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    const scene = (this.scene = new THREE.Scene());
    scene.background = new THREE.Color(0x070b10);
    scene.fog = new THREE.Fog(0x070b10, 120, 520);
    (this as { cam: THREE.PerspectiveCamera }).cam = new THREE.PerspectiveCamera(60, 1, 0.05, 1600);
    scene.add(new THREE.HemisphereLight(0x5a6e88, 0x0b120d, 0.55));
    const moon = new THREE.DirectionalLight(0x8fa6c8, 0.35);
    moon.position.set(-80, 160, 60);
    scene.add(moon);
    const sodium = new THREE.DirectionalLight(0xffc27a, 0.45);
    sodium.position.set(60, 90, -40);
    scene.add(sodium);
    this.hero = makeKart(false, 0xf1a33b);
    scene.add(this.hero.root);
    this.ghost = makeKart(true, 0);
    this.ghost.root.visible = false;
    scene.add(this.ghost.root);
    this.crowd = new THREE.InstancedMesh(new THREE.BoxGeometry(1.6, 0.35, 1.05), new THREE.MeshBasicMaterial({ color: 0xc9d6df, transparent: true, opacity: 0.28, depthWrite: false }), 128);
    this.crowd.count = 0;
    this.crowd.frustumCulled = false;
    scene.add(this.crowd);
    this.crowdBest = makeKart(false, 0xf1a33b);
    this.crowdBest.root.visible = false;
    scene.add(this.crowdBest.root);
    this.buildTrack();
    this.bindOrbit();
  }

  private buildTrack(): void {
    const q = this.q;
    const t = q.track;
    const N = t.N;
    const grp = new THREE.Group();
    const e = (i: number, off: number, lift = 0) => P(t.x[i] + t.nx[i] * off, t.y[i] + t.ny[i] * off, t.z[i] + lift);
    // floodlight poles along the track, outside, away from other branches
    const poles: THREE.Vector3[] = [];
    for (let k = 0; k < 22; k++) {
      const i = Math.floor((k / 22) * N);
      const side = k % 2 ? 1 : -1;
      const off = side * (t.half + 4.5);
      const px = t.x[i] + t.nx[i] * off;
      const py = t.y[i] + t.ny[i] * off;
      let ok = true;
      for (let j = 0; j < N; j += 2) if (Math.hypot(px - t.x[j], py - t.y[j]) < t.half + 2.5) ok = false;
      if (ok) poles.push(new THREE.Vector3(px, py, 0));
    }
    /** Baked sodium light at a track-plane point (0.18 = night ambient). */
    const light = (x: number, y: number) => {
      let l = 0.1;
      for (const p of poles) {
        const d2 = (x - p.x) ** 2 + (y - p.y) ** 2;
        l += 0.95 * Math.exp(-d2 / (2 * 22 * 22));
      }
      return Math.min(1.25, l);
    };
    const sodium = (l: number, base: [number, number, number]): [number, number, number] => [base[0] * l * 1.08, base[1] * l * 0.92, base[2] * l * 0.7];
    // ground with baked light pools
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < N; i++) {
      x0 = Math.min(x0, t.x[i]); x1 = Math.max(x1, t.x[i]); y0 = Math.min(y0, t.y[i]); y1 = Math.max(y1, t.y[i]);
    }
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;
    const size = 900;
    const gg = new THREE.PlaneGeometry(size, size, 140, 140);
    gg.rotateX(-Math.PI / 2);
    gg.translate(cx, -0.04, -cy);
    const gpos = gg.getAttribute('position');
    const gcol = new Float32Array(gpos.count * 3);
    const rnd = mulberry(5);
    for (let v = 0; v < gpos.count; v++) {
      const c = sodium(light(gpos.getX(v), -gpos.getZ(v)), [0.12 + rnd() * 0.02, 0.2 + rnd() * 0.03, 0.13]);
      gcol.set(c, v * 3);
    }
    gg.setAttribute('color', new THREE.BufferAttribute(gcol, 3));
    grp.add(new THREE.Mesh(gg, new THREE.MeshBasicMaterial({ vertexColors: true })));
    // asphalt
    const pos: number[] = [];
    const col: number[] = [];
    const idx: number[] = [];
    for (let i = 0; i <= N; i++) {
      const ii = i % N;
      for (const sd of [1, -1]) {
        const p = e(ii, sd * t.half);
        pos.push(p.x, p.y, p.z);
        const c = sodium(light(t.x[ii] + t.nx[ii] * sd * t.half, t.y[ii] + t.ny[ii] * sd * t.half), [0.33, 0.35, 0.38]);
        col.push(...c);
      }
      if (i < N) {
        const a = i * 2;
        idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
      }
    }
    const ag = new THREE.BufferGeometry();
    ag.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    ag.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    ag.setIndex(idx);
    grp.add(new THREE.Mesh(ag, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide })));
    // strips (edge lines, kerbs, bridge walls) as quads between i and i+1
    const strip = (fn: (i: number) => { a: THREE.Vector3; b: THREE.Vector3; c: [number, number, number] } | null, mat: THREE.Material) => {
      const ps: number[] = [];
      const cs: number[] = [];
      for (let i = 0; i < N; i++) {
        const p = fn(i);
        const n = fn((i + 1) % N);
        if (!p || !n) continue;
        for (const v of [p.a, p.b, n.a, p.b, n.b, n.a]) ps.push(v.x, v.y, v.z);
        for (let k = 0; k < 6; k++) cs.push(...p.c);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(ps, 3));
      g.setAttribute('color', new THREE.Float32BufferAttribute(cs, 3));
      grp.add(new THREE.Mesh(g, mat));
    };
    const vc = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
    for (const sd of [1, -1]) {
      strip((i) => ({ a: e(i, sd * (t.half - 0.1), 0.012), b: e(i, sd * (t.half - 0.25), 0.012), c: sodium(light(t.x[i], t.y[i]), [0.85, 0.86, 0.87]) }), vc);
      strip((i) => (q.kerb[i] > 0 ? { a: e(i, sd * t.half, 0.02), b: e(i, sd * (t.half + q.kerb[i]), 0.05), c: sodium(light(t.x[i], t.y[i]), i % 2 ? [0.9, 0.15, 0.16] : [0.95, 0.95, 0.92]) } : null), vc);
    }
    const concrete = new THREE.MeshLambertMaterial({ color: 0x8c9399, side: THREE.DoubleSide });
    for (const sd of [1, -1]) strip((i) => (t.z[i] < 0.15 ? null : { a: e(i, sd * (t.half + 0.15), -0.75), b: e(i, sd * (t.half + 0.15), 0.85), c: [1, 1, 1] }), concrete);
    strip((i) => (t.z[i] < 0.3 ? null : { a: e(i, t.half + 0.15, -0.75), b: e(i, -t.half - 0.15, -0.75), c: [1, 1, 1] }), concrete);
    const pil: THREE.Vector3[] = [];
    for (let i = 0; i < N; i += 3) {
      if (t.z[i] <= 1.6) continue;
      if (t.crossings.some((c) => Math.hypot(t.x[i] - c.x, t.y[i] - c.y) < t.half + 1.5)) continue;
      for (const f of [0.75, -0.75]) pil.push(new THREE.Vector3(t.x[i] + t.nx[i] * t.half * f, t.y[i] + t.ny[i] * t.half * f, t.z[i] - 0.75));
    }
    if (pil.length) {
      const pm = new THREE.InstancedMesh(new THREE.BoxGeometry(0.6, 1, 0.6), concrete, pil.length);
      pil.forEach((p, k) => {
        this.m4.makeScale(1, p.z, 1);
        this.m4.setPosition(p.x, p.z / 2, -p.y);
        pm.setMatrixAt(k, this.m4);
      });
      grp.add(pm);
    }
    // tyre walls
    const tyres: THREE.Vector3[] = [];
    for (let i = 0; i < N; i += 2) {
      if (t.z[i] > 0.15) continue;
      for (const sd of [1, -1]) {
        const off = t.half + q.wall[i];
        const px = t.x[i] + t.nx[i] * sd * off;
        const py = t.y[i] + t.ny[i] * sd * off;
        let ok = true;
        for (let j = 0; j < N; j++) if (Math.hypot(px - t.x[j], py - t.y[j]) < t.half + 1) { ok = false; break; }
        if (ok) tyres.push(new THREE.Vector3(px, py, light(px, py)));
      }
    }
    if (tyres.length) {
      const tm = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.32, 0.32, 0.72, 10), new THREE.MeshBasicMaterial({ color: 0xffffff }), tyres.length);
      const c = new THREE.Color();
      tyres.forEach((p, k) => {
        this.m4.makeTranslation(p.x, 0.36, -p.y);
        tm.setMatrixAt(k, this.m4);
        const l = p.z;
        c.setRGB(...(k % 6 === 0 ? sodium(l, [0.7, 0.15, 0.17]) : sodium(l, [0.16, 0.17, 0.18])));
        tm.setColorAt(k, c);
      });
      grp.add(tm);
    }
    // poles and lamps
    const poleMat = new THREE.MeshLambertMaterial({ color: 0x3a4248 });
    const lampMat = new THREE.MeshBasicMaterial({ color: 0xffd08a });
    for (const p of poles) {
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.16, 12, 6), poleMat);
      pole.position.set(p.x, 6, -p.y);
      grp.add(pole);
      const lamp = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.3, 0.7), lampMat);
      lamp.position.set(p.x, 12, -p.y);
      grp.add(lamp);
      const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(), color: 0xffb35c, transparent: true, opacity: 0.9, depthWrite: false, blending: THREE.AdditiveBlending }));
      glow.scale.set(6, 6, 1);
      glow.position.set(p.x, 12, -p.y);
      grp.add(glow);
    }
    // dark tree line
    const trees: THREE.Vector3[] = [];
    const r = mulberry(7);
    for (let k = 0; k < 3000 && trees.length < 320; k++) {
      const tx = x0 - 90 + r() * (x1 - x0 + 180);
      const ty = y0 - 90 + r() * (y1 - y0 + 180);
      let ok = true;
      for (let j = 0; j < N; j += 2) if (Math.hypot(tx - t.x[j], ty - t.y[j]) < t.half + 9) { ok = false; break; }
      if (ok) trees.push(new THREE.Vector3(tx, ty, 0.7 + r() * 0.8));
    }
    const cone = new THREE.InstancedMesh(new THREE.ConeGeometry(2.2, 7, 7), new THREE.MeshBasicMaterial({ color: 0xffffff }), trees.length);
    const tc = new THREE.Color();
    trees.forEach((p, k) => {
      this.m4.makeScale(p.z, p.z, p.z);
      this.m4.setPosition(p.x, 3.5 * p.z, -p.y);
      cone.setMatrixAt(k, this.m4);
      tc.setRGB(...sodium(light(p.x, p.y), [0.05 + r() * 0.03, 0.11 + r() * 0.05, 0.07]));
      cone.setColorAt(k, tc);
    });
    grp.add(cone);
    this.scene.add(grp);
    this.center.set(cx, 0, -cy);
    this.orbit.r = Math.max(x1 - x0, y1 - y0) * 1.1;
  }

  /** Race line of the best lap, coloured by speed. */
  private lineMesh: THREE.Mesh | null = null;
  setLine(xs: ArrayLike<number>, ys: ArrayLike<number>, zs: ArrayLike<number>, colors: (k: number) => [number, number, number], n: number): void {
    if (!this.ok) return;
    if (this.lineMesh) {
      this.scene.remove(this.lineMesh);
      this.lineMesh.geometry.dispose();
    }
    const ps: number[] = [];
    const cs: number[] = [];
    for (let k = 0; k < n - 1; k++) {
      const dx = xs[k + 1] - xs[k];
      const dy = ys[k + 1] - ys[k];
      const l = Math.hypot(dx, dy) || 1;
      const nx = (-dy / l) * 0.18;
      const ny = (dx / l) * 0.18;
      const a = P(xs[k] + nx, ys[k] + ny, zs[k] + 0.035);
      const b = P(xs[k] - nx, ys[k] - ny, zs[k] + 0.035);
      const c = P(xs[k + 1] + nx, ys[k + 1] + ny, zs[k + 1] + 0.035);
      const d = P(xs[k + 1] - nx, ys[k + 1] - ny, zs[k + 1] + 0.035);
      for (const v of [a, b, c, b, d, c]) ps.push(v.x, v.y, v.z);
      const col = colors(k);
      for (let j = 0; j < 6; j++) cs.push(...col);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(ps, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(cs, 3));
    this.lineMesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.85 }));
    this.scene.add(this.lineMesh);
  }

  private place(k: KartModel, p: { x: number; y: number; z: number; psi: number; ay?: number }): void {
    k.root.position.copy(P(p.x, p.y, p.z));
    k.root.rotation.set(0, p.psi, 0);
    k.body.rotation.set(-(p.ay ?? 0) / 9.81 * 0.03, 0, 0);
  }

  setHero(p: Pose | null): void {
    if (!this.ok) return;
    this.hero.root.visible = !!p;
    if (p) this.place(this.hero, p);
  }

  setGhost(p: Pose | null): void {
    if (!this.ok) return;
    this.ghost.root.visible = !!p;
    if (p) this.place(this.ghost, p);
  }

  /** Live view: poses of the population (x, y, z, psi) and which one is the best. */
  setCrowd(poses: { x: number; y: number; z: number; psi: number }[], best: number): void {
    if (!this.ok) return;
    let n = 0;
    poses.forEach((p, i) => {
      if (i === best || n >= 128) return;
      this.q4.setFromAxisAngle(this.up, p.psi);
      this.v3.set(p.x, p.z + 0.25, -p.y);
      this.m4.compose(this.v3, this.q4, this.s3);
      this.crowd.setMatrixAt(n++, this.m4);
    });
    this.crowd.count = n;
    this.crowd.instanceMatrix.needsUpdate = true;
    const b = poses[best];
    this.crowdBest.root.visible = !!b;
    if (b) this.place(this.crowdBest, b);
  }

  setCam(mode: CamMode): void {
    this.mode = mode;
    this.chaseInit = false;
    this.look = null;
    this.canvas.style.cursor = mode === 'free' ? 'grab' : 'default';
  }

  /** Moves the camera around `target` (the followed kart) and `ahead` (a point further on its line). */
  render(dt: number, target: { x: number; y: number; z: number; psi: number } | null, ahead: { x: number; y: number; z: number } | null): void {
    if (!this.ok) return;
    const cam = this.cam;
    if (target && !(Number.isFinite(target.x) && Number.isFinite(target.y) && Number.isFinite(target.z) && Number.isFinite(target.psi))) target = null;
    if (!Number.isFinite(cam.position.x + cam.position.y + cam.position.z)) {
      cam.position.set(0, 50, 0);
      this.chaseInit = false;
      this.look = null;
    }
    this.hero.driver.visible = this.mode !== 'onboard';
    if (target && this.mode !== 'free') {
      const f = new THREE.Vector3(Math.cos(target.psi), 0, -Math.sin(target.psi));
      const Q = P(target.x, target.y, target.z);
      if (this.mode === 'onboard') {
        cam.fov = 78;
        cam.position.set(Q.x - f.x * 0.1, Q.y + 0.95, Q.z - f.z * 0.1);
        const L = ahead ? P(ahead.x, ahead.y, ahead.z + 0.55) : Q.clone().addScaledVector(f, 10);
        this.look = this.look ? this.look.lerp(L, 1 - Math.exp(-dt * 10)) : L;
        cam.lookAt(this.look);
      } else if (this.mode === 'chase') {
        cam.fov = 60;
        const want = new THREE.Vector3(Q.x - f.x * 5.2, Q.y + 2.3, Q.z - f.z * 5.2);
        if (!this.chaseInit) {
          cam.position.copy(want);
          this.chaseInit = true;
        } else cam.position.lerp(want, 1 - Math.exp(-dt * 5));
        cam.position.y = Math.max(cam.position.y, Q.y + 1.2);
        cam.lookAt(Q.x + f.x * 3, Q.y + 0.6, Q.z + f.z * 3);
      } else {
        cam.fov = 50;
        const want = new THREE.Vector3(Q.x - f.x * 16, Q.y + 24, Q.z - f.z * 16);
        if (!this.chaseInit) {
          cam.position.copy(want);
          this.chaseInit = true;
        } else cam.position.lerp(want, 1 - Math.exp(-dt * 1.5));
        cam.lookAt(Q);
      }
    } else {
      const o = this.orbit;
      cam.fov = 50;
      cam.position.set(this.center.x + o.r * Math.sin(o.ph) * Math.cos(o.th), o.r * Math.cos(o.ph), this.center.z + o.r * Math.sin(o.ph) * Math.sin(o.th));
      cam.lookAt(this.center);
    }
    cam.updateProjectionMatrix();
    this.renderer.render(this.scene, cam);
  }

  resize(w: number, h: number): void {
    if (!this.ok) return;
    this.renderer.setSize(w, h, false);
    this.cam.aspect = w / Math.max(1, h);
    this.cam.updateProjectionMatrix();
  }

  private bindOrbit(): void {
    const cv = this.canvas;
    let drag: { x: number; y: number; th: number; ph: number } | null = null;
    cv.addEventListener('pointerdown', (e) => {
      if (this.mode !== 'free') return;
      drag = { x: e.clientX, y: e.clientY, th: this.orbit.th, ph: this.orbit.ph };
      cv.setPointerCapture(e.pointerId);
    });
    cv.addEventListener('pointermove', (e) => {
      if (!drag) return;
      this.orbit.th = drag.th + (e.clientX - drag.x) * 0.006;
      this.orbit.ph = Math.max(0.15, Math.min(1.45, drag.ph - (e.clientY - drag.y) * 0.005));
    });
    cv.addEventListener('pointerup', () => (drag = null));
    cv.addEventListener('wheel', (e) => {
      if (this.mode !== 'free') return;
      e.preventDefault();
      this.orbit.r = Math.max(30, Math.min(700, this.orbit.r * Math.exp(e.deltaY * 0.001)));
    }, { passive: false });
  }
}
