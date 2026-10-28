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
    const added = onlyIn(ink, dilate(this.baseline, f.width, f.height));
    const removed = onlyIn(this.baseline, dilate(ink, f.width, f.height));
    this.armed = false; // look once per quiet spell
    let trigger: Trigger | null = null;
    if (added > o.pageChangeThreshold || removed > o.pageChangeThreshold) trigger = "page_changed";
    else if (added > o.inkThreshold) trigger = "new_ink";
    else if (removed > o.inkThreshold) trigger = "ink_removed";
    if (trigger) this.baseline = ink;
    return result("steady", trigger, added);
  }
}
