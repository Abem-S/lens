// A tab left open must stop the GPU services: presence decides when Lens rests.
import { describe, expect, it } from "vitest";
import { backoffMs, cameraMessage, micMessage } from "./errors";
import { PresenceTracker } from "./presence";
import { trackingNote } from "./tracking";

const MIN = 60_000;

describe("PresenceTracker", () => {
  it("rests after 5 minutes with no movement, speech or taps, and movement wakes it", () => {
    const p = new PresenceTracker(0, 5 * MIN);
    expect(p.state(4 * MIN)).toBe("active");
    expect(p.state(5 * MIN)).toBe("idle");
    p.motion(6 * MIN);
    expect(p.state(6 * MIN)).toBe("active");
  });

  it("keeps going while the camera sees the student move", () => {
    const p = new PresenceTracker(0, 5 * MIN);
    for (let t = MIN; t <= 15 * MIN; t += MIN) p.motion(t);
    expect(p.state(15 * MIN)).toBe("active");
  });

  it("a webcam facing a busy room cannot keep Lens awake forever: after 20 min without a tap, word or check it rests", () => {
    const p = new PresenceTracker(0, 5 * MIN);
    for (let t = MIN; t <= 25 * MIN; t += MIN) p.motion(t);
    expect(p.state(20 * MIN)).toBe("idle");
    p.motion(25 * MIN); // movement alone does not wake it from here
    expect(p.state(25 * MIN)).toBe("idle");
    p.interact(26 * MIN); // a tap, speech or a page check does
    expect(p.state(26 * MIN)).toBe("active");
  });

  it("a hidden tab rests at once and coming back wakes it", () => {
    const p = new PresenceTracker(0);
    p.setHidden(true, 1000);
    expect(p.state(1000)).toBe("hidden");
    p.setHidden(false, 60 * MIN);
    expect(p.state(60 * MIN)).toBe("active");
  });
});

describe("friendly errors", () => {
  it("explains camera and mic problems in plain words", () => {
    expect(cameraMessage({ name: "NotAllowedError" })).toMatch(/permission/);
    expect(cameraMessage({ name: "NotReadableError" })).toMatch(/another app/);
    expect(micMessage({ name: "NotAllowedError" })).toMatch(/still checks your page/);
  });

  it("reconnects with growing waits, capped at 30 s", () => {
    const mid = () => 0.5;
    expect([0, 1, 2, 3, 10].map((n) => backoffMs(n, mid))).toEqual([1000, 2000, 4000, 8000, 30000]);
  });

  it("says when tracking is waking up or off, never nothing while connecting", () => {
    expect(trackingNote("connecting", null)).toMatch(/Waking up/);
    expect(trackingNote("unavailable", "busy msg")).toBe("busy msg");
    expect(trackingNote("live", null)).toBeNull();
  });
});
