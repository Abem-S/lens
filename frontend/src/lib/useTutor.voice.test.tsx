/// <reference types="node" />
// Barge-in through the whole hook: speech over Lens first only lowers its voice; real speech
// stops it and tells the server; noise, or a turn the server heard no words in, lets it finish.
import { readFileSync } from "node:fs";
import { act, cleanup, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useTutor } from "./useTutor";

const audio = vi.hoisted(() => ({
  onFrame: null as ((f: Float32Array) => void) | null,
  player: null as null | { playing: boolean; ducks: boolean[]; stops: number },
}));

vi.mock("./audio", () => {
  class Player {
    playing = false;
    ducks: boolean[] = [];
    stops = 0;
    onIdle: (() => void) | null = null;
    constructor() {
      audio.player = this;
    }
    async resume() {}
    enqueue() {
      this.playing = true;
    }
    duck(on: boolean) {
      this.ducks.push(on);
    }
    stop() {
      this.playing = false;
      this.stops++;
    }
    close() {}
  }
  return {
    Player,
    startMic: async (cb: (f: Float32Array) => void) => {
      audio.onFrame = cb;
      return () => undefined;
    },
  };
});

class FakeSocket {
  static all: FakeSocket[] = [];
  static OPEN = 1;
  static CLOSED = 3;
  readyState = 0;
  sent: unknown[] = [];
  binaryType = "";
  onopen: (() => void) | null = null;
  onclose: ((ev: { code: number }) => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  constructor(public url: string) {
    FakeSocket.all.push(this);
  }
  send(m: unknown) {
    this.sent.push(typeof m === "string" ? JSON.parse(m) : m);
  }
  close() {
    this.readyState = 3;
  }
  open() {
    this.readyState = 1;
    this.onopen?.();
  }
  message(m: object) {
    this.onmessage?.({ data: JSON.stringify(m) });
  }
}

let seen: ReturnType<typeof useTutor> | null = null;
function Probe() {
  seen = useTutor(undefined, undefined, true);
  return null;
}

const SR = 16000, FRAME = 320;
function wav(name: string): Float32Array {
  const b = readFileSync(`${process.cwd()}/src/lib/fixtures/${name}.wav`);
  const ints = new Int16Array(b.buffer.slice(b.byteOffset + 44, b.byteOffset + b.length));
  return Float32Array.from(ints, (v) => v / 32768);
}

function room(seconds: number, seed = 1): Float32Array {
  let s = seed;
  return Float32Array.from({ length: Math.round(seconds * SR) }, () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return ((s / 2 ** 32) * 2 - 1) * 0.003;
  });
}

function rustle(seconds: number, seed = 5): Float32Array {
  let s = seed, lp = 0;
  return Float32Array.from({ length: Math.round(seconds * SR) }, () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    const x = (s / 2 ** 32) * 2 - 1;
    lp = 0.5 * lp + 0.5 * x;
    return 0.2 * (x - lp);
  });
}

function feed(x: Float32Array) {
  act(() => {
    for (let i = 0; i + FRAME <= x.length; i += FRAME) audio.onFrame!(x.slice(i, i + FRAME));
  });
}

async function talkingLens() {
  render(<Probe />);
  const s = FakeSocket.all[0];
  act(() => s.open());
  await act(async () => void (await seen!.startVoice()));
  act(() => {
    s.message({ type: "audio", utt: 1, sample_rate: 24000 });
    s.onmessage?.({ data: new ArrayBuffer(8) }); // Lens's voice arrives and plays
  });
  feed(room(1));
  return s;
}

const sentTypes = (s: FakeSocket) => s.sent.map((m) => (m as { type?: string }).type).filter(Boolean);

beforeEach(() => {
  FakeSocket.all = [];
  audio.onFrame = null;
  audio.player = null;
  vi.stubGlobal("WebSocket", FakeSocket);
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("useTutor barge-in", () => {
  it("a page rustle while Lens talks neither stops nor interrupts it", async () => {
    const s = await talkingLens();
    feed(rustle(0.8));
    feed(room(1));
    expect(audio.player!.playing).toBe(true);
    expect(audio.player!.stops).toBe(0);
    expect(sentTypes(s)).not.toContain("interrupt");
    expect(sentTypes(s)).not.toContain("speech_end"); // nothing went to ASR either
  });

  it("the student talking over Lens ducks it, then stops it and tells the server", async () => {
    const s = await talkingLens();
    feed(wav("speech_a").slice(0, Math.round(0.6 * SR)));
    expect(audio.player!.ducks[0]).toBe(true);
    expect(audio.player!.stops).toBe(1);
    expect(sentTypes(s)).toContain("interrupt");
  });

  it("a short turn over Lens waits for the transcript: no words restores it, words stop it", async () => {
    const s = await talkingLens();
    const blip = wav("speech_b").slice(Math.round(0.03 * SR), Math.round(0.36 * SR)); // a short word: under the barge length
    feed(blip);
    feed(room(0.6));
    expect(audio.player!.ducks).toEqual([true]);
    act(() => s.message({ type: "transcript", text: "" }));
    expect(audio.player!.ducks).toEqual([true, false]);
    expect(audio.player!.stops).toBe(0);
    feed(blip);
    feed(room(0.6));
    act(() => s.message({ type: "transcript", text: "okay" }));
    expect(audio.player!.stops).toBe(1);
    expect(sentTypes(s)).not.toContain("interrupt"); // the server stopped already on hearing words
  });
});
