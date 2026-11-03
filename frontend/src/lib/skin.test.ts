import { describe, expect, it } from "vitest";
import { SkinTracker } from "./skin";

const N = 100;
/** A view where pixels [0, desk) are a skin-coloured desk and [from, to) a hand. */
function view(desk: number, hand?: [number, number]): Uint8Array {
  const m = new Uint8Array(N);
  for (let i = 0; i < desk; i++) m[i] = 1;
  if (hand) for (let i = hand[0]; i < hand[1]; i++) m[i] = 1;
  return m;
}

describe("SkinTracker", () => {
  it("stops counting a skin-coloured desk as a hand once it has stayed put", () => {
    const t = new SkinTracker(4);
    const seen = Array.from({ length: 6 }, () => t.push(view(30)));
    expect(seen[0]).toBeCloseTo(0.3); // not known yet
    expect(seen[5]).toBe(0); // background now
  });

  it("still sees a hand that comes in over the page", () => {
    const t = new SkinTracker(4);
    for (let i = 0; i < 6; i++) t.push(view(30));
    expect(t.push(view(30, [50, 60]))).toBeCloseTo(0.1);
    // a moving hand keeps covering new pixels and keeps counting
    expect(t.push(view(30, [55, 65]))).toBeCloseTo(0.1);
  });

  it("forgets a pixel as soon as it stops looking like skin", () => {
    const t = new SkinTracker(2);
    for (let i = 0; i < 3; i++) t.push(view(0, [10, 20]));
    expect(t.push(view(0, [10, 20]))).toBe(0);
    t.push(view(0)); // the hand left
    expect(t.push(view(0, [10, 20]))).toBeCloseTo(0.1); // and came back: a hand again
  });
});
