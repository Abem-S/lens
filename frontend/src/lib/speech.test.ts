/// <reference types="node" />
// Recorded speech with pauses (two halves of one spoken sentence, fixtures/*.wav, 16 kHz)
// through the VAD + streamer: a short pause never cuts the student off; a longer pause
// cuts early but the rest is sent as a continuation of the same turn; a long silence
// makes two turns.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { SpeechStreamer, type Outgoing } from "./speech";

const FRAME = 320; // 20 ms at 16 kHz

function wav(name: string): Float32Array {
  const b = readFileSync(`${process.cwd()}/src/lib/fixtures/${name}.wav`); // tests run from frontend/
  const ints = new Int16Array(b.buffer.slice(b.byteOffset + 44, b.byteOffset + b.length));
  return Float32Array.from(ints, (v) => v / 32768);
}

function room(seconds: number, seed = 7): Float32Array {
  let s = seed;
  return Float32Array.from({ length: Math.round(seconds * 16000) }, () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return ((s / 2 ** 32) * 2 - 1) * 0.003; // quiet room noise
  });
}

function join(parts: Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((a, p) => a + p.length, 0));
  let o = 0;
  for (const p of parts) out.set(p, (o += p.length) - p.length);
  return out;
}

/** Stream a recording through the streamer; returns every message sent. */
function run(audio: Float32Array): Outgoing[] {
  const st = new SpeechStreamer();
  const sent: Outgoing[] = [];
  for (let i = 0; i + FRAME <= audio.length; i += FRAME) {
    const noise = room(0.02, i + 1);
    const f = audio.slice(i, i + FRAME).map((v, k) => v + noise[k]);
    sent.push(...st.push(f, (i / FRAME) * 20).out);
  }
  return sent;
}

const types = (m: Outgoing[]) => m.map((o) => o.msg.type).filter((t) => t !== "speech");
const A = wav("speech_a"), B = wav("speech_b");
// `gap` is inserted silence; the halves add ~0.14 s of their own soft edges to the pause the VAD hears.
const said = (gap: number) => join([room(0.5), A, room(gap), B, room(1.5)]);

describe("SpeechStreamer on recorded speech with pauses", () => {
  it("does not cut a sentence at a short pause, and streams the audio as it comes", () => {
    const sent = run(said(0.12));
    const t = types(sent);
    expect(t.filter((x) => x === "speech_start")).toHaveLength(1);
    expect(t.filter((x) => x === "speech_end")).toHaveLength(1);
    expect(t.at(-1)).toBe("speech_end");
    // Nearly all of it went up before the end: only the last chunk is sent with speech_end.
    const chunks = sent.filter((o) => o.msg.type === "speech");
    const bytes = chunks.reduce((a, o) => a + (o.pcm?.length ?? 0), 0);
    expect(bytes / 32000).toBeGreaterThan(4.5); // both halves + pause + preroll + tail
    expect(chunks.at(-1)!.pcm!.length).toBeLessThanOrEqual(10 * FRAME * 2);
  });

  it("ends the turn early at a longer pause, then sends the rest as its continuation", () => {
    const t = types(run(said(0.7)));
    const starts = run(said(0.7)).filter((o) => o.msg.type === "speech_start").map((o) => o.msg.continues);
    expect(starts).toEqual([false, true]); // one turn, heard as one sentence on the server
    expect(t.filter((x) => x === "speech_end")).toHaveLength(2);
    expect(t).toContain("speech_pause");
  });

  it("makes two turns when the student really stopped", () => {
    const starts = run(said(2.0)).filter((o) => o.msg.type === "speech_start").map((o) => o.msg.continues);
    expect(starts).toEqual([false, false]);
  });

  it("tells the server to transcribe early at a pause, and takes it back if speech goes on", () => {
    const t = types(run(said(0.12)));
    const p = t.indexOf("speech_pause");
    expect(p).toBeGreaterThan(0);
    expect(t.slice(p)).toContain("speech_resume");
    expect(t.at(-2)).toBe("speech_pause"); // the final silence: transcript ready before speech_end
  });

  it("never continues a turn over the tutor's voice (that is a barge-in)", () => {
    const st = new SpeechStreamer();
    const audio = said(0.7);
    const starts: (boolean | undefined)[] = [];
    for (let i = 0; i + FRAME <= audio.length; i += FRAME) {
      const r = st.push(audio.slice(i, i + FRAME), (i / FRAME) * 20, i / 16000 > 2.6);
      if (r.event?.type === "start") starts.push(r.continues);
    }
    expect(starts).toEqual([false, false]);
  });
});
