/// <reference types="node" />
// Lens must finish what it says unless the student really talks. A desk session while Lens
// speaks, built from recorded speech (fixtures/*.wav) and synthesised room sounds: pen taps,
// clicks, page rustle, a chair scrape, Lens's own voice leaking back into the mic, and one real
// interruption. Counted: how often Lens is cut off for good (a false barge-in) before the
// voicing test and ducking (old: every VAD start stopped Lens) and after.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { SpeechStreamer, type Barge } from "./speech";
import { EnergyVad, periodicity, rms } from "./vad";

const SR = 16000, FRAME = 320;
const DUCK = 0.25; // the Player's ducked gain (audio.ts DUCK_GAIN)

function wav(name: string): Float32Array {
  const b = readFileSync(`${process.cwd()}/src/lib/fixtures/${name}.wav`);
  const ints = new Int16Array(b.buffer.slice(b.byteOffset + 44, b.byteOffset + b.length));
  return Float32Array.from(ints, (v) => v / 32768);
}
const A = wav("speech_a"), B = wav("speech_b");

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return (s / 2 ** 32) * 2 - 1;
  };
}
const len = (s: number) => Math.round(s * SR);

/** A pen tap or a knock on a desk: a few decaying resonances, a noisy attack and the room's
 * reverberation tail (~150 ms above the noise floor on a laptop mic). */
function tap(seed: number, level = 0.3): Float32Array {
  const r = rng(seed), out = new Float32Array(len(0.25));
  const modes = [700 + 400 * Math.abs(r()), 2100 + 600 * Math.abs(r()), 3900 + 800 * Math.abs(r())];
  for (let i = 0; i < out.length; i++) {
    const t = i / SR, env = Math.exp(-t / 0.008);
    let v = 0.6 * r() * Math.exp(-t / 0.002) + 0.35 * r() * Math.exp(-t / 0.05);
    for (const f of modes) v += Math.sin(2 * Math.PI * f * t) * env / modes.length;
    out[i] = level * v;
  }
  return out;
}

/** A sharp click (a pen cap, a key): a couple of milliseconds. */
function click(seed: number, level = 0.5): Float32Array {
  const r = rng(seed), out = new Float32Array(len(0.004));
  for (let i = 0; i < out.length; i++) out[i] = level * r() * Math.exp(-i / 20);
  return out;
}

/** Turning or sliding a page: band-limited noise with a ragged envelope. */
function rustle(seed: number, seconds: number, level = 0.15): Float32Array {
  const r = rng(seed), out = new Float32Array(len(seconds));
  let lp = 0, hp = 0, env = 0.5;
  for (let i = 0; i < out.length; i++) {
    const x = r();
    lp = 0.5 * lp + 0.5 * x;
    const band = x - lp - hp;
    hp = 0.98 * hp + 0.02 * (x - lp);
    if (i % 160 === 0) env = Math.min(1, Math.max(0.1, env + 0.5 * r()));
    out[i] = level * env * band * 2;
  }
  return out;
}

/** A chair scrape or a bump: low rumble. */
function rumble(seed: number, seconds: number, level = 0.1): Float32Array {
  const r = rng(seed), out = new Float32Array(len(seconds));
  let lp = 0;
  for (let i = 0; i < out.length; i++) out[i] = level * (lp = 0.97 * lp + 0.25 * r());
  return out;
}

/** Fidgeting: the pen tapped on the desk several times a second. */
function tapping(seed: number, seconds: number): Float32Array {
  const out = new Float32Array(len(seconds) + len(0.25));
  for (let k = 0; k * 0.12 < seconds; k++) {
    const t = tap(seed + k, 0.4), at = len(k * 0.12);
    for (let i = 0; i < t.length; i++) out[at + i] += t[i];
  }
  return out;
}

/** Writing a step: pen strokes scratching the paper, close to the mic. */
function scratching(seed: number, strokes: number): Float32Array {
  const out = new Float32Array(len(strokes * 0.35));
  for (let k = 0; k < strokes; k++) out.set(rustle(seed + k, 0.25, 0.1), len(k * 0.35));
  return out;
}

const scale = (x: Float32Array, k: number) => Float32Array.from(x, (v) => v * k);

interface Event { at: number; audio: Float32Array; kind: "noise" | "echo" | "student" }

/** Plays a session through the streamer frame by frame. Lens talks until stopped for good; its
 * echo follows the player's volume (ducked = quieter). Returns the barge decisions with times. */
function session(events: Event[], seconds: number, streamer: SpeechStreamer, oldBehaviour = false,
  restartS: number | null = null) {
  const room = rng(99);
  const mic = new Float32Array(len(seconds));
  for (let i = 0; i < mic.length; i++) mic[i] = 0.003 * room();
  const echo = new Float32Array(len(seconds));
  for (const e of events) {
    const into = e.kind === "echo" ? echo : mic, at = len(e.at);
    if (at >= into.length) continue;
    for (let k = 0; k < e.audio.length && at + k < into.length; k++) into[at + k] += e.audio[k];
  }
  let audible = true, ducked = false, silentSince = 0;
  const out: { t: number; barge: Barge }[] = [];
  for (let i = 0; i + FRAME <= mic.length; i += FRAME) {
    const t = i / SR;
    if (!audible && restartS !== null && t - silentSince >= restartS) audible = true; // Lens says something new
    const gain = !audible ? 0 : ducked ? DUCK : 1;
    const f = mic.slice(i, i + FRAME);
    for (let k = 0; k < FRAME; k++) f[k] += echo[i + k] * gain;
    streamer.vad.playback = audible;
    const r = streamer.push(f, t * 1000, audible);
    // Before: every VAD start stopped Lens in the browser, and every turn end stopped it on the
    // server (before the transcript was known), whatever the sound was.
    const cut = r.event?.type === "start" || r.event?.type === "end";
    const barge: Barge = oldBehaviour ? (cut && audible ? "stop" : null) : r.barge;
    if (barge) out.push({ t, barge });
    if (barge === "stop") [audible, ducked, silentSince] = [false, false, t];
    else if (barge === "duck") ducked = true;
    else if (barge === "restore") ducked = false;
  }
  return out;
}

// The VAD as it was: loudness only (no voicing test), 12 frames to keep an utterance.
const oldVad = () => new EnergyVad({ voicing: 0, startVoiced: 0, minVoicedFrames: 0, minSpeechFrames: 12 });

/** 40 s at the desk while Lens explains something: a desk sound every 1.5 s (writing, taps,
 * fidgeting, clicks, page turns, a chair, the notebook slid), Lens's own voice leaking back
 * through the echo canceller at `echo` of its level, and no one talks to Lens until 36 s. */
function deskSession(echo = 0.05): { events: Event[]; seconds: number; studentAt: number } {
  const events: Event[] = [];
  for (let t = 0.3; t < 40; t += 2.3) events.push({ at: t, audio: scale(t % 4.6 < 2.3 ? A : B, echo), kind: "echo" });
  const sounds = [
    (k: number) => scratching(k, 4), (k: number) => tap(k), (k: number) => tapping(k, 0.8),
    (k: number) => click(k), (k: number) => rustle(k, 0.6), (k: number) => tap(k, 0.5),
    (k: number) => rumble(k, 0.5), (k: number) => rustle(k, 1.2, 0.12),
  ];
  let k = 0;
  for (let t = 1; t < 34; t += 1.5, k++) events.push({ at: t, audio: sounds[k % sounds.length](k + 1), kind: "noise" });
  const studentAt = 36;
  events.push({ at: studentAt, audio: A, kind: "student" }); // "wait, ..." at full level
  return { events, seconds: 40, studentAt };
}

describe("voicing test", () => {
  it("finds vowels in recorded speech and not in taps, clicks, rustle or rumble", () => {
    const voicedShare = (x: Float32Array) => {
      let loud = 0, voiced = 0;
      for (let i = 0; i + FRAME <= x.length; i += FRAME) {
        const f = x.slice(i, i + FRAME);
        if (rms(f) < 0.012) continue;
        loud++;
        if (periodicity(f) >= 0.75) voiced++;
      }
      return voiced / Math.max(1, loud);
    };
    expect(voicedShare(A)).toBeGreaterThan(0.6);
    expect(voicedShare(B)).toBeGreaterThan(0.6);
    expect(voicedShare(rustle(3, 2))).toBeLessThan(0.05);
    expect(voicedShare(rumble(4, 2))).toBeLessThan(0.1);
    const taps = new Float32Array(len(3.5));
    for (let k = 0; k < 20; k++) taps.set(tap(k + 10), len(k * 0.15));
    // A tap rings, so a frame or two of it can look voiced; it is too short to duck Lens (bargein: duckVoiced).
    expect(voicedShare(taps)).toBeLessThan(0.8);
  });
});

describe("barge-in over Lens's voice", () => {
  it.each([0.05, 0.15])("a desk full of noise no longer cuts Lens off, and the student still stops it (echo %s)", (echo) => {
    const { events, seconds, studentAt } = deskSession(echo);
    const before = session(events, seconds, new SpeechStreamer(oldVad()), true);
    const after = session(events, seconds, new SpeechStreamer());
    const falseBefore = before.filter((b) => b.barge === "stop" && b.t < studentAt).length;
    const falseAfter = after.filter((b) => b.barge === "stop" && b.t < studentAt).length;
    console.log(`echo ${echo}: false barge-ins in a 36 s noisy desk session: before ${falseBefore}, after ${falseAfter}`);
    expect(falseBefore).toBeGreaterThanOrEqual(1); // the old VAD stopped Lens at the first loud desk sound
    expect(falseAfter).toBe(0);
    // ...and the real interruption stops Lens well under a second into the student's speech.
    const stop = after.find((b) => b.barge === "stop");
    expect(stop).toBeDefined();
    console.log(`echo ${echo}: the student's real interruption stopped Lens ${(stop!.t - studentAt).toFixed(2)} s in`);
    expect(stop!.t - studentAt).toBeLessThan(0.7);
  });

  it.each([[0.05, 5], [0.15, 1]])("counts how often desk noise cuts Lens off when it keeps finding new things to say (echo %s)", (echo, least) => {
    // The same desk; after each cut-off Lens speaks again 1.5 s later (as when the student
    // asks again): how many times it is cut off before anyone talks to it.
    const { events, studentAt } = deskSession(echo);
    const quietDesk = events.filter((e) => e.kind !== "student");
    const cuts = (st: SpeechStreamer, old: boolean) =>
      session(quietDesk, studentAt, st, old, 1.5).filter((b) => b.barge === "stop").length;
    const before = cuts(new SpeechStreamer(oldVad()), true);
    const after = cuts(new SpeechStreamer(), false);
    console.log(`echo ${echo}: times Lens is cut off in a 36 s noisy desk session (it speaks again 1.5 s after each): before ${before}, after ${after}`);
    // With a louder echo the old VAD sat in one long "utterance" (the 20 s noise turn in the logs).
    expect(before).toBeGreaterThanOrEqual(least);
    expect(after).toBe(0);
  });

  it("a short noise ducks Lens and then restores it", () => {
    const st = new SpeechStreamer();
    const events: Event[] = [{ at: 0.5, audio: rustle(1, 0.4, 0.1), kind: "noise" },
      { at: 0.55, audio: A.slice(0, len(0.12)), kind: "noise" }]; // a voiced blip inside it
    const out = session(events, 3, st);
    expect(out.map((b) => b.barge)).not.toContain("stop");
    if (out.length) expect(out.at(-1)!.barge).toBe("restore");
  });

  it("a short spoken word over Lens: ducked until the transcript says whether it had words", () => {
    // A short word ("okay"): voiced, but stopped short of the barge length (set long here).
    const short = () => new SpeechStreamer(new EnergyVad(), { bargeVoiced: 40 });
    const st = short();
    const word = A.slice(len(0.1), len(0.5));
    const out = session([{ at: 0.5, audio: word, kind: "student" }], 2, st);
    expect(out.map((b) => b.barge)).toEqual(["duck"]); // not stopped yet, not restored
    expect(st.heard("okay")).toBe("stop");
    const st2 = short();
    session([{ at: 0.5, audio: word, kind: "student" }], 2, st2);
    expect(st2.heard("")).toBe("restore");
    expect(st2.heard("anything")).toBeNull(); // nothing ducked any more
  });

  it("restores by itself when no transcript comes", () => {
    const st = new SpeechStreamer(new EnergyVad(), { bargeVoiced: 40 });
    const word = A.slice(len(0.1), len(0.5));
    const out = session([{ at: 0.5, audio: word, kind: "student" }], 5, st);
    expect(out.map((b) => b.barge)).toEqual(["duck", "restore"]);
    expect(out[1].t).toBeGreaterThan(3);
  });

  it("speech over Lens never stops it when Lens is silent (nothing to stop)", () => {
    const st = new SpeechStreamer();
    const frames: Barge[] = [];
    const audio = new Float32Array(len(3));
    audio.set(A, len(0.5));
    for (let i = 0; i + FRAME <= audio.length; i += FRAME) frames.push(st.push(audio.slice(i, i + FRAME), i / 16, false).barge);
    expect(frames.filter(Boolean)).toEqual([]);
  });

  it("desk noise sends far fewer empty turns to ASR, and real speech still starts and ends on time", () => {
    const { events, studentAt } = deskSession(0.05);
    const turns = (st: SpeechStreamer) => {
      const room = rng(7);
      const mic = new Float32Array(len(studentAt));
      for (let i = 0; i < mic.length; i++) mic[i] = 0.003 * room();
      for (const e of events) {
        if (e.kind !== "noise") continue; // Lens silent: no echo guard, as between replies
        for (let k = 0, at = len(e.at); k < e.audio.length && at + k < mic.length; k++) mic[at + k] += e.audio[k];
      }
      let n = 0;
      for (let i = 0; i + FRAME <= mic.length; i += FRAME) if (st.push(mic.slice(i, i + FRAME), i / 16).event?.type === "end") n++;
      return n;
    };
    const before = turns(new SpeechStreamer(oldVad())), after = turns(new SpeechStreamer());
    console.log(`noise turns sent to ASR in 36 s at the desk: before ${before}, after ${after}`);
    expect(before).toBeGreaterThanOrEqual(5);
    expect(after).toBe(0);
    // Real speech: "start" within 0.12 s of the first loud frame, "end" 0.4 s after the last voice.
    for (const [x, onset] of [[A, 0.06], [B, 0.04]] as const) {
      const st = new SpeechStreamer();
      const audio = new Float32Array(len(4));
      audio.set(x, len(1));
      const room = rng(3);
      for (let i = 0; i < audio.length; i++) audio[i] += 0.003 * room();
      const times: Record<string, number> = {};
      for (let i = 0; i + FRAME <= audio.length; i += FRAME) {
        const ev = st.push(audio.slice(i, i + FRAME), i / 16).event;
        if (ev && !(ev.type in times)) times[ev.type] = i / SR;
      }
      expect(times.start - (1 + onset)).toBeLessThan(0.12);
      expect(times.end).toBeDefined();
      expect(times.end - (1 + x.length / SR)).toBeLessThan(0.45);
    }
  });
});
