import { describe, expect, it } from "vitest";
import { TurnClock } from "./latency";

describe("TurnClock", () => {
  it("measures from the real end of speech, counting the VAD's silence wait", () => {
    const c = new TurnClock();
    c.ended(10_000, 0.4);
    expect(c.audio(11_500, 3)).toEqual({ end_to_audio_s: 1.9, vad_wait_s: 0.4, send_to_audio_s: 1.5 });
    expect(c.audio(11_600, 3)).toBeNull(); // later chunks of the same reply
  });

  it("ignores tutor audio that is not a reply, and cancelled turns", () => {
    const c = new TurnClock();
    expect(c.audio(500, 1)).toBeNull(); // the opening
    c.ended(1000, 0.4);
    c.cancel();
    expect(c.audio(2000, 2)).toBeNull();
  });
});
