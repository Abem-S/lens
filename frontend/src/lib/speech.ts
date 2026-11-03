// Turns microphone frames into the tutor socket's speech messages (protocol in
// backend/src/lens/tutor.py). Two things make the reply come sooner:
//
// - The audio goes up while the student talks (~200 ms chunks), so when they stop
//   there is nothing left to upload; at ~200 ms of quiet the server is told to start
//   transcribing what it has.
// - The VAD waits only 400 ms of silence before ending a turn. A student who pauses
//   longer mid-sentence and goes on within MERGE_MS is not cut off: the next speech
//   is sent as a continuation, and the server drops its reply to the first half and
//   hears both halves as one turn.
//
// Barge-in without cutting Lens off for a noise: when speech starts while Lens is talking,
// Lens only gets quieter ("duck"). It stops for real ("stop": the caller silences the player
// and tells the server) once the student has been voiced for BARGE_VOICED frames, or when
// the server's transcript of a short turn has words in it. A discarded or empty turn
// brings the volume back ("restore") and Lens finishes what it was saying.
import { EnergyVad, toPcm16, type VadEvent } from "./vad";

export interface SpeechOptions {
  chunkFrames: number; // frames per uploaded chunk (20 ms each)
  pauseFrames: number; // quiet frames before "speech_pause" (early transcription)
  mergeMs: number; // speech starting this soon after a turn ended continues that turn
  duckVoiced: number; // voiced frames over Lens before Lens gets quieter (a pen tap rings for 1-3)
  bargeVoiced: number; // voiced frames (20 ms) of the student over Lens before Lens stops for real
  duckWaitMs: number; // after a short turn over Lens, wait this long for its transcript, then restore
}

export const SPEECH_DEFAULTS: SpeechOptions = {
  chunkFrames: 10, pauseFrames: 10, mergeMs: 1200,
  duckVoiced: 4,
  bargeVoiced: 15, // 0.3 s of voiced speech: about 0.4-0.5 s into a real sentence (speech.test.ts)
  duckWaitMs: 3000,
};

/** What to do with Lens's voice: lower it, stop it for good, or bring it back. */
export type Barge = "duck" | "stop" | "restore" | null;

export interface Outgoing {
  msg: { type: string; continues?: boolean };
  pcm?: Uint8Array<ArrayBuffer>; // binary payload sent right after `msg`
}

export interface PushResult {
  out: Outgoing[];
  event: VadEvent | null;
  continues: boolean; // set on a "start" that continues the last turn
  barge: Barge;
}

export class SpeechStreamer {
  readonly vad: EnergyVad;
  private o: SpeechOptions;
  private pending: Float32Array[] = [];
  private paused = false;
  private lastEnd: number | null = null;
  private ducked = false; // Lens is quieter, waiting to learn whether the student really talks
  private stopped = false; // this utterance already stopped Lens

  constructor(vad = new EnergyVad(), opts: Partial<SpeechOptions> = {}) {
    this.vad = vad;
    this.o = { ...SPEECH_DEFAULTS, ...opts };
  }

  /** One 20 ms frame at `nowMs`. `tutorAudible`: the tutor's voice is playing (then new
   * speech is a barge-in, never a continuation). */
  push(frame: Float32Array, nowMs: number, tutorAudible = false): PushResult {
    const ev = this.vad.push(frame);
    const out: Outgoing[] = [];
    let continues = false;
    if (ev?.type === "start") {
      this.stopped = false;
      continues = this.lastEnd !== null && nowMs - this.lastEnd <= this.o.mergeMs && !tutorAudible;
      out.push({ msg: { type: "speech_start", continues } });
      this.pending = this.vad.audioSoFar();
      this.paused = false;
    } else if (ev?.type === "end") {
      this.pending.push(frame);
      this.flush(out);
      out.push({ msg: { type: "speech_end" } });
      this.lastEnd = nowMs;
    } else if (ev?.type === "discard") {
      this.pending = [];
      out.push({ msg: { type: "speech_cancel" } });
    } else if (this.vad.inSpeech) {
      this.pending.push(frame);
      const quiet = this.vad.quietFrames;
      if (quiet === this.o.pauseFrames) {
        this.flush(out);
        out.push({ msg: { type: "speech_pause" } });
        this.paused = true;
      } else if (quiet === 0 && this.paused) {
        out.push({ msg: { type: "speech_resume" } });
        this.paused = false;
      }
      if (this.pending.length >= this.o.chunkFrames) this.flush(out);
    }
    return { out, event: ev, continues, barge: this.barge(ev, nowMs, tutorAudible) };
  }

  private barge(ev: VadEvent | null, nowMs: number, tutorAudible: boolean): Barge {
    if (this.vad.inSpeech && tutorAudible && !this.stopped) {
      if (this.vad.voicedFrames >= this.o.bargeVoiced) {
        this.stopped = true;
        this.ducked = false;
        return "stop";
      }
      if (!this.ducked && this.vad.voicedFrames >= this.o.duckVoiced) {
        this.ducked = true;
        return "duck";
      }
      return null;
    }
    if (!this.ducked) return null;
    // A noise, Lens already done, or no transcript in time: back to full volume.
    const waitedOut = !this.vad.inSpeech && this.lastEnd !== null && nowMs - this.lastEnd > this.o.duckWaitMs;
    if (ev?.type === "discard" || !tutorAudible || waitedOut) {
      this.ducked = false;
      return "restore";
    }
    return null;
  }

  /** The server's transcript of a spoken turn: words in it stop a ducked Lens, none restore it. */
  heard(text: string): Barge {
    if (!this.ducked || this.vad.inSpeech) return null;
    this.ducked = false;
    return /[\p{L}\p{N}]/u.test(text) ? "stop" : "restore";
  }

  private flush(out: Outgoing[]): void {
    if (!this.pending.length) return;
    out.push({ msg: { type: "speech" }, pcm: toPcm16(this.pending) });
    this.pending = [];
  }
}
