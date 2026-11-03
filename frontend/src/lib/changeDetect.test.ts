// D1 done criterion (logic half): writing a new line triggers exactly one upload.
import { describe, expect, it } from "vitest";
import { ActivityTracker, ChangeDetector, inkMap, type Frame, type Trigger } from "./changeDetect";

const W = 160, H = 120;

function rng(seed: number) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

/** A page with `lines` written lines, optional hand blob, plus sensor noise. */
function frame(
  lines: number,
  opts: { hand?: [number, number]; seed?: number; light?: number; shift?: [number, number]; angle?: number } = {},
): Frame {
  const [sx, sy] = opts.shift ?? [0, 0];
  const r = rng(opts.seed ?? 1);
  const gray = new Uint8Array(W * H);
  for (let i = 0; i < gray.length; i++) gray[i] = (opts.light ?? 200) + Math.round((r() - 0.5) * 8);
  // A page turned by `angle` degrees about the frame centre (a hand-held notebook tilting).
  const a = ((opts.angle ?? 0) * Math.PI) / 180, cx = (W - 1) / 2, cy = (H - 1) / 2;
  const ink = (x: number, y: number) => {
    const X = Math.round(cx + Math.cos(a) * (x - cx) - Math.sin(a) * (y - cy));
    const Y = Math.round(cy + Math.sin(a) * (x - cx) + Math.cos(a) * (y - cy));
    if (X >= 0 && X < W && Y >= 0 && Y < H) gray[Y * W + X] = 40;
  };
  for (let l = 0; l < lines; l++) {
    const y = 15 + l * 18 + sy;
    // a wavy pen stroke, like a line of handwriting
    for (let x = 20 + sx; x < 120 + sx; x++) {
      const yy = y + Math.round(Math.sin(x / 3) * 2);
      ink(x, yy);
      ink(x, yy + 1);
    }
  }
  let skin = 0;
  if (opts.hand) {
    const [hx, hy] = opts.hand;
    let n = 0;
    for (let y = hy; y < H; y++)
      for (let x = hx; x < Math.min(W, hx + 40); x++) {
        gray[y * W + x] = 150;
        n++;
      }
    skin = n / (W * H);
  }
  return { width: W, height: H, gray, skin };
}

function run(det: ChangeDetector, frames: Frame[]): Trigger[] {
  return frames.map((f) => det.push(f).trigger).filter((t): t is Trigger => t !== null);
}

const still = (lines: number, n: number, seed0 = 0) =>
  Array.from({ length: n }, (_, i) => frame(lines, { seed: seed0 + i + 1 }));

describe("ChangeDetector", () => {
  it("reads the page once when it first settles", () => {
    expect(run(new ChangeDetector(), still(3, 20))).toEqual(["first_view"]);
  });

  it("ignores sensor noise and lighting drift on an unchanged page", () => {
    const det = new ChangeDetector();
    run(det, still(3, 8));
    const drift = Array.from({ length: 30 }, (_, i) => frame(3, { seed: 100 + i, light: 200 - i }));
    expect(run(det, drift)).toEqual([]);
  });

  it("writing a new line triggers exactly one upload, after the hand leaves", () => {
    const det = new ChangeDetector();
    expect(run(det, still(3, 8))).toEqual(["first_view"]);
    // Hand comes in and writes line 4 over several frames, moving around.
    const writing = [
      frame(3, { hand: [100, 90], seed: 50 }),
      frame(3, { hand: [80, 80], seed: 51 }),
      frame(3, { hand: [60, 75], seed: 52 }),
      frame(4, { hand: [90, 75], seed: 53 }),
      frame(4, { hand: [110, 80], seed: 54 }),
      frame(4, { hand: [110, 80], seed: 55 }), // hand rests: no motion, but skin is visible
      frame(4, { hand: [110, 80], seed: 56 }),
      frame(4, { hand: [110, 80], seed: 57 }),
      frame(4, { hand: [110, 80], seed: 58 }),
      frame(4, { hand: [120, 100], seed: 59 }),
    ];
    expect(run(det, writing)).toEqual([]);
    // Hand gone, page still for a while.
    expect(run(det, still(4, 25, 60))).toEqual(["new_ink"]);
  });

  it("erasing a line triggers once", () => {
    const det = new ChangeDetector();
    run(det, still(4, 8));
    run(det, [frame(4, { hand: [60, 60], seed: 9 }), frame(3, { hand: [70, 60], seed: 10 })]);
    expect(run(det, still(3, 10, 200))).toEqual(["ink_removed"]);
  });

  it("does not read an empty view when the page leaves it, and reads it when it comes back", () => {
    const det = new ChangeDetector();
    expect(run(det, still(3, 8))).toEqual(["first_view"]);
    // Notebook lowered out of view: a moving frame, then a quiet view with almost no ink.
    run(det, [frame(0, { hand: [0, 0], seed: 400 })]);
    expect(run(det, still(0, 12, 400))).toEqual([]);
    // Held up again with a fourth line.
    run(det, [frame(4, { hand: [0, 60], seed: 500 })]);
    expect(run(det, still(4, 12, 500))).toEqual(["new_ink"]);
  });

  it("a page that drifted a few pixels between looks is not new ink", () => {
    const det = new ChangeDetector();
    expect(run(det, still(3, 8))).toEqual(["first_view"]);
    run(det, [frame(3, { hand: [0, 60], seed: 600 })]); // a bump: the held page moves
    const drifted = Array.from({ length: 12 }, (_, i) => frame(3, { seed: 600 + i, shift: [3, -2] }));
    expect(run(det, drifted)).toEqual([]);
    run(det, [frame(3, { hand: [0, 60], seed: 700 })]);
    const withLine = Array.from({ length: 12 }, (_, i) => frame(4, { seed: 700 + i, shift: [-2, 3] }));
    expect(run(det, withLine)).toEqual(["new_ink"]);
  });

  it("a page that tilted slightly between looks is not new ink (no second check of the same ink)", () => {
    for (const angle of [1.5, -2, 2.5]) {
      const det = new ChangeDetector();
      expect(run(det, still(5, 8))).toEqual(["first_view"]);
      run(det, [frame(5, { hand: [0, 60], seed: 800 })]); // the held page sways
      const tilted = Array.from({ length: 12 }, (_, i) => frame(5, { seed: 800 + i, angle, shift: [1, -1] }));
      expect(run(det, tilted), `tilt ${angle}°`).toEqual([]);
    }
  });

  it("a new line on a tilted page is still new ink", () => {
    const det = new ChangeDetector();
    expect(run(det, still(4, 8))).toEqual(["first_view"]);
    run(det, [frame(4, { hand: [0, 60], seed: 900 })]);
    const withLine = Array.from({ length: 12 }, (_, i) => frame(5, { seed: 900 + i, angle: -2 }));
    expect(run(det, withLine)).toEqual(["new_ink"]);
  });

  it("a manual check re-baselines, so the same ink does not trigger again", () => {
    const det = new ChangeDetector();
    run(det, still(3, 8));
    run(det, [frame(4, { hand: [60, 60], seed: 9 })]);
    det.markChecked(frame(4, { seed: 300 }));
    expect(run(det, still(4, 10, 300))).toEqual([]);
  });

  it("finds ink only where strokes are", () => {
    const ink = inkMap(frame(1));
    let n = 0;
    for (const v of ink) n += v;
    expect(n).toBeGreaterThan(150);
    expect(n).toBeLessThan(400);
  });
});

describe("ActivityTracker", () => {
  it("reads movement as writing, stillness as paused, and a long-still page as away", () => {
    const a = new ActivityTracker(20_000);
    expect(a.update("steady", 0)).toBe("away"); // nothing has moved yet
    expect(a.update("moving", 1000)).toBe("writing");
    expect(a.update("steady", 5000)).toBe("paused");
    expect(a.update("settling", 20_999)).toBe("paused");
    expect(a.update("steady", 21_001)).toBe("away");
    expect(a.update("hand", 22_000)).toBe("writing");
  });

  it("reports on every change and as a heartbeat every 2 s, not every frame", () => {
    const a = new ActivityTracker();
    expect(a.report("moving", 0)).toBe("writing");
    expect(a.report("moving", 250)).toBeNull();
    expect(a.report("steady", 500)).toBe("paused");
    expect(a.report("steady", 1500)).toBeNull();
    expect(a.report("steady", 2600)).toBe("paused");
  });
});
