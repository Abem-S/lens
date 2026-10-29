import { describe, expect, it } from "vitest";
import { EnergyVad, encodeWav, type VadEvent } from "./vad";

const SR = 16000, FRAME = 320; // 20 ms

function noise(level: number, seed = 1): Float32Array {
  let s = seed;
  const f = new Float32Array(FRAME);
  for (let i = 0; i < FRAME; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    f[i] = ((s / 2 ** 32) * 2 - 1) * level;
  }
  return f;
}

function voice(level: number, t0: number): Float32Array {
  const f = new Float32Array(FRAME);
  for (let i = 0; i < FRAME; i++) f[i] = level * Math.sin((2 * Math.PI * 180 * (t0 + i)) / SR);
  return f;
}

function feed(vad: EnergyVad, frames: Float32Array[]): VadEvent[] {
  return frames.map((f) => vad.push(f)).filter((e): e is VadEvent => e !== null);
}

const quiet = (n: number, lvl = 0.004) => Array.from({ length: n }, (_, i) => noise(lvl, i + 1));
const talk = (n: number, lvl = 0.2) => Array.from({ length: n }, (_, i) => voice(lvl, i * FRAME));

describe("EnergyVad", () => {
  it("cuts one utterance out of speech between silences", () => {
    const vad = new EnergyVad();
    const events = feed(vad, [...quiet(50), ...talk(75), ...quiet(60)]);
    expect(events.map((e) => e.type)).toEqual(["start", "end"]);
    const end = events[1] as Extract<VadEvent, { type: "end" }>;
    expect(end.seconds).toBeGreaterThan(1.5); // 1.5 s of speech plus preroll and tail
    expect(end.seconds).toBeLessThan(2.8);
  });

  it("ignores background noise and short clicks", () => {
    const vad = new EnergyVad();
    expect(feed(vad, [...quiet(100, 0.006), ...talk(2), ...quiet(100, 0.006)])).toEqual([]);
  });

  it("drops a burst too short to be speech", () => {
    const vad = new EnergyVad();
    const events = feed(vad, [...quiet(50), ...talk(6), ...quiet(60)]);
    expect(events.map((e) => e.type)).toEqual(["start", "discard"]);
  });

  it("start fires within 100 ms of speech, for barge-in", () => {
    const vad = new EnergyVad();
    feed(vad, quiet(50));
    const firstFrames = talk(10).map((f) => vad.push(f));
    expect(firstFrames.findIndex((e) => e?.type === "start")).toBeLessThanOrEqual(4);
  });

  it("tutor's own voice at speaker level does not trigger during playback", () => {
    const vad = new EnergyVad();
    feed(vad, quiet(50));
    vad.playback = true;
    expect(feed(vad, talk(40, 0.025))).toEqual([]); // echo leaking into the mic
    expect(feed(vad, talk(10, 0.3)).map((e) => e.type)).toEqual(["start"]); // the student talking over it
  });

  it("encodes a valid 16-bit WAV", () => {
    const wav = encodeWav(new Float32Array([0, 0.5, -0.5, 1]), SR);
    expect(String.fromCharCode(...wav.slice(0, 4))).toBe("RIFF");
    expect(wav.length).toBe(44 + 8);
    expect(new DataView(wav.buffer).getUint32(24, true)).toBe(SR);
  });
});
