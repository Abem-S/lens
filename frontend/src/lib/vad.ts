// Voice activity detection by loudness against an adaptive noise floor.
// Plumbing, not a model: it only decides when an utterance starts (barge-in)
// and ends (send it to ASR). Pure logic over Float32 frames, unit-tested.

export interface VadOptions {
  sampleRate: number;
  frameMs: number; // frame length the caller pushes
  minLevel: number; // RMS below this is never speech
  ratio: number; // speech must be this many times louder than the noise floor
  startFrames: number; // consecutive loud frames to start an utterance
  endFrames: number; // consecutive quiet frames to end it
  prerollFrames: number; // audio kept from before the start, so the first word is not clipped
  minSpeechFrames: number; // shorter bursts are dropped (coughs, clicks)
  maxFrames: number; // force an end after this long
  playbackFactor: number; // while the tutor speaks, demand this much more level (echo guard)
}

export const VAD_DEFAULTS: VadOptions = {
  sampleRate: 16000,
  frameMs: 20,
  minLevel: 0.012,
  ratio: 3,
  startFrames: 4,
  endFrames: 35,
  prerollFrames: 15,
  minSpeechFrames: 12,
  maxFrames: 1250,
  playbackFactor: 2.5,
};

export type VadEvent =
  | { type: "start" }
  | { type: "end"; audio: Float32Array; seconds: number }
  | { type: "discard" };

export function rms(frame: Float32Array): number {
  let s = 0;
  for (let i = 0; i < frame.length; i++) s += frame[i] * frame[i];
  return Math.sqrt(s / Math.max(1, frame.length));
}

export class EnergyVad {
  private o: VadOptions;
  private floor = 0.003;
  private loudRun = 0;
  private quietRun = 0;
  private speaking = false;
  private preroll: Float32Array[] = [];
  private speech: Float32Array[] = [];
  playback = false; // set while the tutor's voice is playing

  constructor(opts: Partial<VadOptions> = {}) {
    this.o = { ...VAD_DEFAULTS, ...opts };
  }

  get inSpeech(): boolean {
    return this.speaking;
  }

  threshold(): number {
    const base = Math.max(this.o.minLevel, this.floor * this.o.ratio);
    return this.playback ? base * this.o.playbackFactor : base;
  }

  push(frame: Float32Array): VadEvent | null {
    const level = rms(frame);
    const loud = level > this.threshold();
    if (!this.speaking) {
      if (!loud) this.floor = 0.95 * this.floor + 0.05 * Math.max(level, 0.0005);
      this.preroll.push(frame);
      if (this.preroll.length > this.o.prerollFrames) this.preroll.shift();
      this.loudRun = loud ? this.loudRun + 1 : 0;
      if (this.loudRun >= this.o.startFrames) {
        this.speaking = true;
        this.quietRun = 0;
        this.speech = [...this.preroll];
        this.preroll = [];
        return { type: "start" };
      }
      return null;
    }
    this.speech.push(frame);
    this.quietRun = loud ? 0 : this.quietRun + 1;
    if (this.quietRun >= this.o.endFrames || this.speech.length >= this.o.maxFrames) {
      return this.finish();
    }
    return null;
  }

  private finish(): VadEvent {
    this.speaking = false;
    this.loudRun = 0;
    const voiced = this.speech.length - this.quietRun - this.o.prerollFrames;
    const frames = this.speech;
    this.speech = [];
    if (voiced < this.o.minSpeechFrames) return { type: "discard" };
    const n = frames.reduce((a, f) => a + f.length, 0);
    const audio = new Float32Array(n);
    let off = 0;
    for (const f of frames) {
      audio.set(f, off);
      off += f.length;
    }
    return { type: "end", audio, seconds: n / this.o.sampleRate };
  }
}

/** 16-bit mono WAV for the ASR service. */
export function encodeWav(samples: Float32Array, sampleRate: number): Uint8Array<ArrayBuffer> {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF");
  v.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, "data");
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Uint8Array(buf);
}
