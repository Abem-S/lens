import { afterEach, describe, expect, it, vi } from "vitest";
import { FrameBurst, LOOK_EVERY_MS, LOOK_MAX_BUFFERED } from "./look";

afterEach(() => vi.useRealTimers());

describe("FrameBurst", () => {
  it("sends a picture every half second while talking and a moment after, then stops", async () => {
    vi.useFakeTimers();
    let t = 0;
    const sent: number[] = [];
    const b = new FrameBurst(async () => new Blob([String(t)]), () => sent.push(t), () => 0, () => t);
    b.start();
    for (let i = 0; i < 6; i++) {
      t += LOOK_EVERY_MS;
      await vi.advanceTimersByTimeAsync(LOOK_EVERY_MS);
    }
    expect(sent).toHaveLength(6);
    b.stopAfter(1500);
    for (let i = 0; i < 6; i++) {
      t += LOOK_EVERY_MS;
      await vi.advanceTimersByTimeAsync(LOOK_EVERY_MS);
    }
    expect(sent.length).toBeGreaterThanOrEqual(8);
    expect(sent.length).toBeLessThanOrEqual(9);
    expect(b.running).toBe(false);
  });

  it("never queues pictures behind a busy socket, and a dropped turn stops at once", async () => {
    vi.useFakeTimers();
    let backlog = LOOK_MAX_BUFFERED + 1;
    const sent: Blob[] = [];
    const b = new FrameBurst(async () => new Blob(["x"]), (x) => sent.push(x), () => backlog);
    b.start();
    await vi.advanceTimersByTimeAsync(LOOK_EVERY_MS * 3);
    expect(sent).toHaveLength(0);
    backlog = 0;
    await vi.advanceTimersByTimeAsync(LOOK_EVERY_MS);
    expect(sent).toHaveLength(1);
    b.stopAfter(0);
    await vi.advanceTimersByTimeAsync(LOOK_EVERY_MS * 3);
    expect(sent).toHaveLength(1);
  });
});
