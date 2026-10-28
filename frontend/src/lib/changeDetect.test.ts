// D1 done criterion (logic half): writing a new line triggers exactly one upload.
import { describe, expect, it } from "vitest";
import { ChangeDetector, inkMap, type Frame, type Trigger } from "./changeDetect";

const W = 160, H = 120;

function rng(seed: number) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

/** A page with `lines` written lines, optional hand blob, plus sensor noise. */
function frame(lines: number, opts: { hand?: [number, number]; seed?: number; light?: number } = {}): Frame {
  const r = rng(opts.seed ?? 1);
  const gray = new Uint8Array(W * H);
  for (let i = 0; i < gray.length; i++) gray[i] = (opts.light ?? 200) + Math.round((r() - 0.5) * 8);
  for (let l = 0; l < lines; l++) {
    const y = 15 + l * 18;
    // a wavy pen stroke, like a line of handwriting
    for (let x = 20; x < 120; x++) {
      const yy = y + Math.round(Math.sin(x / 3) * 2);
      gray[yy * W + x] = 40;
      gray[(yy + 1) * W + x] = 40;
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
