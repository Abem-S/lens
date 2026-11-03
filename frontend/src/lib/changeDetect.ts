// Free in-browser checks that decide when a frame is worth sending to Spark:
// new ink has appeared (or ink was erased) and the hand is out of the way.
//
// Works on small grayscale frames (e.g. 160×120). Pure logic, no DOM, so it is
// unit-tested with synthetic frames.

export interface Frame {
  width: number;
  height: number;
  gray: Uint8Array; // width*height, 0 = black
  skin?: number; // fraction of pixels that look like skin, if the caller measured it
}

export type Trigger = "first_view" | "new_ink" | "ink_removed" | "page_changed";

export type DetectorState = "waiting" | "moving" | "hand" | "settling" | "steady";

export interface DetectorResult {
  state: DetectorState;
  trigger: Trigger | null;
  motion: number; // fraction of pixels that changed since the last frame
  newInk: number; // fraction of pixels that are new ink since the last check
}

export interface DetectorOptions {
  motionThreshold: number; // a frame with more changed pixels than this is "moving"
  pixelDelta: number; // per-pixel brightness change that counts as motion
  skinThreshold: number; // more skin than this means a hand is over the page
  stableFrames: number; // quiet frames needed before we look for new ink
  inkContrast: number; // how much darker than its neighbourhood a pixel must be to be ink
  inkThreshold: number; // fraction of the frame that must be new ink to trigger
  pageChangeThreshold: number; // this much new ink means a different page or a big move
  triggerOnFirstView: boolean;
}

export const DEFAULTS: DetectorOptions = {
  motionThreshold: 0.01,
  pixelDelta: 28,
  skinThreshold: 0.04,
  stableFrames: 4,
  inkContrast: 35,
  inkThreshold: 0.0015,
  pageChangeThreshold: 0.12,
  triggerOnFirstView: true,
};

/** Ink = pixels clearly darker than the local average (robust to shadows). */
export function inkMap(f: Frame, contrast = DEFAULTS.inkContrast, radius = 4): Uint8Array {
  const { width: w, height: h, gray } = f;
  // Integral image for a fast box mean.
  const integral = new Float64Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y++) {
    let row = 0;
    for (let x = 0; x < w; x++) {
      row += gray[y * w + x];
      integral[(y + 1) * (w + 1) + (x + 1)] = integral[y * (w + 1) + (x + 1)] + row;
    }
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - radius), y1 = Math.min(h, y + radius + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - radius), x1 = Math.min(w, x + radius + 1);
      const sum =
        integral[y1 * (w + 1) + x1] - integral[y0 * (w + 1) + x1] -
        integral[y1 * (w + 1) + x0] + integral[y0 * (w + 1) + x0];
      const mean = sum / ((x1 - x0) * (y1 - y0));
      out[y * w + x] = gray[y * w + x] < mean - contrast ? 1 : 0;
    }
  }
  return out;
}

function dilate(m: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(m.length);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      if (!m[y * w + x]) continue;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const yy = y + dy, xx = x + dx;
          if (yy >= 0 && yy < h && xx >= 0 && xx < w) out[yy * w + xx] = 1;
        }
    }
  return out;
}

/** `m` moved by (dx, dy) pixels; what moves in from outside is empty. */
export function shifted(m: Uint8Array, w: number, h: number, dx: number, dy: number): Uint8Array {
  if (!dx && !dy) return m;
  const out = new Uint8Array(m.length);
  for (let y = Math.max(0, dy); y < Math.min(h, h + dy); y++)
    for (let x = Math.max(0, dx); x < Math.min(w, w + dx); x++) out[y * w + x] = m[(y - dy) * w + (x - dx)];
  return out;
}

/** The shift (within ±r px) that lays `base` best over `ink`: a hand-held page or phone
 * drifts a little between looks, and drifted ink is not new ink. */
export function bestShift(base: Uint8Array, ink: Uint8Array, w: number, h: number, r = 4): [number, number] {
  const pts: number[] = [];
  for (let i = 0; i < base.length; i++) if (base[i]) pts.push(i);
  let best: [number, number] = [0, 0];
  let bestN = -1;
  for (let dy = -r; dy <= r; dy++)
    for (let dx = -r; dx <= r; dx++) {
      let n = 0;
      for (const i of pts) {
        const x = (i % w) + dx, y = ((i / w) | 0) + dy;
        if (x >= 0 && x < w && y >= 0 && y < h && ink[y * w + x]) n++;
      }
      // Prefer no shift on a tie, so a still page stays exactly as compared before.
      if (n > bestN || (n === bestN && Math.abs(dx) + Math.abs(dy) < Math.abs(best[0]) + Math.abs(best[1]))) {
        best = [dx, dy];
        bestN = n;
      }
    }
  return best;
}

/** How the page sits now relative to the last check: a small turn (degrees, about the
 * frame centre) and then a shift (pixels). */
export interface Pose {
  angle: number;
  dx: number;
  dy: number;
}

// A hand-held notebook tilts as well as drifts; a turn of a degree or two moves the ink at
// the edge of a 160 px frame by several pixels, which a shift alone cannot undo.
export const POSE_ANGLES = [0, -0.75, 0.75, -1.5, 1.5, -2.25, 2.25, -3, 3];

/** The turn (within ±3°) and shift (within ±r px) that lay `base` best over `ink`. */
export function bestPose(base: Uint8Array, ink: Uint8Array, w: number, h: number, r = 4): Pose {
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < base.length; i++)
    if (base[i]) {
      xs.push(i % w);
      ys.push((i / w) | 0);
    }
  const cx = (w - 1) / 2, cy = (h - 1) / 2;
  let best: Pose = { angle: 0, dx: 0, dy: 0 };
  let bestN = -1;
  const rx = new Int32Array(xs.length), ry = new Int32Array(xs.length);
  for (const angle of POSE_ANGLES) {
    const c = Math.cos((angle * Math.PI) / 180), s = Math.sin((angle * Math.PI) / 180);
    for (let k = 0; k < xs.length; k++) {
      const x = xs[k] - cx, y = ys[k] - cy;
      rx[k] = Math.round(cx + c * x - s * y);
      ry[k] = Math.round(cy + s * x + c * y);
    }
    for (let dy = -r; dy <= r; dy++)
      for (let dx = -r; dx <= r; dx++) {
        let n = 0;
        for (let k = 0; k < rx.length; k++) {
          const x = rx[k] + dx, y = ry[k] + dy;
          if (x >= 0 && x < w && y >= 0 && y < h && ink[y * w + x]) n++;
        }
        // A turn must clearly beat no turn (2 % more ink laid over), so a still page stays as it was.
        const cost = Math.abs(angle) > 0 ? Math.max(1, Math.round(0.02 * xs.length)) : 0;
        const score = n - cost;
        const tieBetter = Math.abs(angle) + Math.abs(dx) + Math.abs(dy) <
          Math.abs(best.angle) + Math.abs(best.dx) + Math.abs(best.dy);
        if (score > bestN || (score === bestN && tieBetter)) {
          best = { angle, dx, dy };
          bestN = score;
        }
      }
  }
  return best;
}

/** `m` turned by `pose.angle` about the frame centre, then moved by (dx, dy). */
export function posed(m: Uint8Array, w: number, h: number, pose: Pose): Uint8Array {
  if (!pose.angle) return shifted(m, w, h, pose.dx, pose.dy);
  const out = new Uint8Array(m.length);
  const cx = (w - 1) / 2, cy = (h - 1) / 2;
  const c = Math.cos((pose.angle * Math.PI) / 180), s = Math.sin((pose.angle * Math.PI) / 180);
  for (let i = 0; i < m.length; i++) {
    if (!m[i]) continue;
    const x = (i % w) - cx, y = ((i / w) | 0) - cy;
    const nx = Math.round(cx + c * x - s * y) + pose.dx, ny = Math.round(cy + s * x + c * y) + pose.dy;
    if (nx >= 0 && nx < w && ny >= 0 && ny < h) out[ny * w + nx] = 1;
  }
  return out;
}

function onlyIn(a: Uint8Array, notNear: Uint8Array): number {
  let n = 0;
  for (let i = 0; i < a.length; i++) if (a[i] && !notNear[i]) n++;
  return n / a.length;
}

export class ChangeDetector {
  private opts: DetectorOptions;
  private prev: Frame | null = null;
  private baseline: Uint8Array | null = null; // ink at the last check
  private quiet = 0;
  private armed = true; // false right after a trigger until something changes again

  constructor(opts: Partial<DetectorOptions> = {}) {
    this.opts = { ...DEFAULTS, ...opts };
  }

  /** Call after a manual check so the next automatic check waits for new ink. */
  markChecked(f: Frame): void {
    this.baseline = inkMap(f, this.opts.inkContrast);
    this.armed = false;
  }

  reset(): void {
    this.prev = null;
    this.baseline = null;
    this.quiet = 0;
    this.armed = true;
  }

  push(f: Frame): DetectorResult {
    const o = this.opts;
    let motion = 0;
    if (this.prev && this.prev.gray.length === f.gray.length) {
      let changed = 0;
      for (let i = 0; i < f.gray.length; i++) if (Math.abs(f.gray[i] - this.prev.gray[i]) > o.pixelDelta) changed++;
      motion = changed / f.gray.length;
    } else if (this.prev) {
      this.reset(); // resolution changed: start over
    }
    this.prev = f;

    const result = (state: DetectorState, trigger: Trigger | null = null, newInk = 0): DetectorResult => ({
      state, trigger, motion, newInk,
    });

    if (motion > o.motionThreshold) {
      this.quiet = 0;
      this.armed = true;
      return result("moving");
    }
    if ((f.skin ?? 0) > o.skinThreshold) {
      this.quiet = 0;
      this.armed = true;
      return result("hand");
    }
    this.quiet++;
    if (this.quiet < o.stableFrames) return result("settling");
    if (!this.armed) return result("steady");

    const ink = inkMap(f, o.inkContrast);
    if (!this.baseline) {
      this.baseline = ink;
      this.armed = false;
      return result("steady", o.triggerOnFirstView ? "first_view" : null);
    }
    const base = posed(this.baseline, f.width, f.height, bestPose(this.baseline, ink, f.width, f.height));
    const added = onlyIn(ink, dilate(base, f.width, f.height));
    const removed = onlyIn(base, dilate(ink, f.width, f.height));
    this.armed = false; // look once per quiet spell
    // Most of the checked page's ink is gone and little came instead: the page left
    // the view (a notebook lowered to write, a phone lifted away). Nothing to read;
    // keep the baseline so the page coming back with a new line is new ink.
    let baseInk = 0;
    for (const v of this.baseline) baseInk += v;
    baseInk /= this.baseline.length;
    if (baseInk > o.inkThreshold && removed > 0.6 * baseInk && added < 0.5 * baseInk) return result("steady", null, added);
    let trigger: Trigger | null = null;
    if (added > o.pageChangeThreshold || removed > o.pageChangeThreshold) trigger = "page_changed";
    else if (added > o.inkThreshold) trigger = "new_ink";
    else if (removed > o.inkThreshold) trigger = "ink_removed";
    if (trigger) this.baseline = ink;
    return result("steady", trigger, added);
  }
}

/** The student's live state for the tutor when live tracking is off, from the detector:
 * movement over the page reads as "writing", stillness as "paused", and a page that has
 * been still for `awayAfterMs` as "away" (nobody is working at it). */
export type Activity = "writing" | "paused" | "away";

export class ActivityTracker {
  private lastMove: number | null = null;
  constructor(private awayAfterMs = 20_000) {}

  update(state: DetectorState, nowMs: number): Activity {
    if (state === "moving" || state === "hand") {
      this.lastMove = nowMs;
      return "writing";
    }
    return this.lastMove !== null && nowMs - this.lastMove < this.awayAfterMs ? "paused" : "away";
  }

  private sent: { phase: Activity; at: number } | null = null;

  /** The phase to send to the tutor now, or null: on every change, and every `everyMs` as a
   * heartbeat (the server stops trusting a report after a few seconds). */
  report(state: DetectorState, nowMs: number, everyMs = 2000): Activity | null {
    const phase = this.update(state, nowMs);
    if (this.sent && this.sent.phase === phase && nowMs - this.sent.at < everyMs) return null;
    this.sent = { phase, at: nowMs };
    return phase;
  }
}
