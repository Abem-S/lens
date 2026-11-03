// Voice activity detection by loudness against an adaptive noise floor, plus a voicing test.
// Plumbing, not a model: it only decides when an utterance starts and ends (send it to ASR).
// Pure logic over Float32 frames, unit-tested.
//
// Loudness alone let pen taps, page rustle and chair noise "start" an utterance (that cut
// Lens off mid-sentence and sent dozens of empty turns to ASR). Speech has vowels: frames
// that repeat at the pitch of a voice (normalised autocorrelation peak at a 2.5-12.5 ms lag).
// Taps, clicks and rustle are noise-like and score low, so an utterance starts only once a
// couple of loud frames are voiced, and one with too little voicing is discarded.

export interface VadOptions {
  sampleRate: number;
  frameMs: number; // frame length the caller pushes
  minLevel: number; // RMS below this is never speech
  ratio: number; // speech must be this many times louder than the noise floor
  startFrames: number; // consecutive loud frames to start an utterance
  endFrames: number; // consecutive quiet frames to end it (a longer pause is merged back: lib/speech.ts)
  prerollFrames: number; // audio kept from before the start, so the first word is not clipped
  minSpeechFrames: number; // shorter bursts are dropped (coughs, clicks)
  maxFrames: number; // force an end after this long
  playbackFactor: number; // while the tutor speaks, demand this much more level (echo guard)
  voicing: number; // periodicity at or above this makes a loud frame "voiced"
  startVoiced: number; // voiced frames needed within the loud run to start an utterance
  minVoicedFrames: number; // an utterance with fewer voiced frames is discarded (noise, not words)
}

export const VAD_DEFAULTS: VadOptions = {
  sampleRate: 16000,
  frameMs: 20,
  minLevel: 0.012,
  ratio: 3,
  startFrames: 4,
  endFrames: 20, // 400 ms (was 700 ms); a student who goes on after a longer pause is merged back in
  prerollFrames: 15,
  minSpeechFrames: 8, // with the voicing test guarding against noise, a short "yes" or "no" gets through
  maxFrames: 1250,
  playbackFactor: 2.5,
  // Measured on the recorded speech fixtures: loud vowel frames score 0.85-1.0; white, pink and
  // rumble noise, clicks and rustle 0.0-0.63.
  voicing: 0.75,
  startVoiced: 2,
  minVoicedFrames: 6,
};

export type VadEvent =
  | { type: "start" }
  // silenceS: the quiet tail already waited out before "end" (speech really ended this long ago)
  | { type: "end"; audio: Float32Array; seconds: number; silenceS: number }
  | { type: "discard" };

export function rms(frame: Float32Array): number {
  let s = 0;
  for (let i = 0; i < frame.length; i++) s += frame[i] * frame[i];
  return Math.sqrt(s / Math.max(1, frame.length));
}

/** How strongly a frame repeats at a voice pitch, 0-1 (normalised autocorrelation peak for lags
 * minLag..maxLag samples, searched only after the autocorrelation first drops to zero so a slow
 * rumble does not count). 20 ms at 16 kHz: lags 32-200 cover 80-500 Hz. */
export function periodicity(frame: Float32Array, minLag = 32, maxLag = 200): number {
  const n = frame.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += frame[i];
  mean /= Math.max(1, n);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = frame[i] - mean;
  let best = 0;
  let dipped = false;
  for (let lag = 1; lag <= Math.min(maxLag, n - 16); lag++) {
    let xy = 0, xx = 0, yy = 0;
    for (let i = 0; i + lag < n; i++) {
      const a = x[i], b = x[i + lag];
      xy += a * b;
      xx += a * a;
      yy += b * b;
    }
    const r = xy / Math.sqrt(xx * yy + 1e-12);
    if (r <= 0) dipped = true;
    if (dipped && lag >= minLag && r > best) best = r;
  }
  return best;
}

export class EnergyVad {
  private o: VadOptions;
  private floor = 0.003;
  private loudRun = 0;
  private voicedRun = 0; // voiced frames within the current loud run (before a start)
  private voiced = 0; // voiced frames in the current utterance
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

  /** Voiced frames so far in the current utterance (0 outside one): how sure it is that someone talks. */
  get voicedFrames(): number {
    return this.speaking ? this.voiced : 0;
  }

  /** Quiet frames in a row so far in the current utterance (0 while they talk). */
  get quietFrames(): number {
    return this.speaking ? this.quietRun : 0;
  }

  /** The current utterance's audio so far (pre-roll included), for streaming it up as it comes. */
  audioSoFar(): Float32Array[] {
    return this.speaking ? [...this.speech] : [];
  }

  threshold(): number {
    const base = Math.max(this.o.minLevel, this.floor * this.o.ratio);
    return this.playback ? base * this.o.playbackFactor : base;
  }

  push(frame: Float32Array): VadEvent | null {
    const level = rms(frame);
    const loud = level > this.threshold();
    const voiced = loud && periodicity(frame) >= this.o.voicing;
    if (!this.speaking) {
      if (!loud) this.floor = 0.95 * this.floor + 0.05 * Math.max(level, 0.0005);
      this.preroll.push(frame);
      if (this.preroll.length > this.o.prerollFrames) this.preroll.shift();
      this.loudRun = loud ? this.loudRun + 1 : 0;
      this.voicedRun = loud ? this.voicedRun + (voiced ? 1 : 0) : 0;
      if (this.loudRun >= this.o.startFrames && this.voicedRun >= this.o.startVoiced) {
        this.speaking = true;
        this.voiced = this.voicedRun;
        this.quietRun = 0;
        this.speech = [...this.preroll];
        this.preroll = [];
        return { type: "start" };
      }
      return null;
    }
    this.speech.push(frame);
    if (voiced) this.voiced++;
    this.quietRun = loud ? 0 : this.quietRun + 1;
    if (this.quietRun >= this.o.endFrames || this.speech.length >= this.o.maxFrames) {
      return this.finish();
    }
    return null;
  }

  private finish(): VadEvent {
    this.speaking = false;
    this.loudRun = 0;
    this.voicedRun = 0;
    const enoughVoice = this.voiced >= this.o.minVoicedFrames;
    this.voiced = 0;
    const voiced = this.speech.length - this.quietRun - this.o.prerollFrames;
    const silenceS = (this.quietRun * this.o.frameMs) / 1000;
    const frames = this.speech;
    this.speech = [];
    if (voiced < this.o.minSpeechFrames || !enoughVoice) return { type: "discard" };
    const n = frames.reduce((a, f) => a + f.length, 0);
    const audio = new Float32Array(n);
    let off = 0;
    for (const f of frames) {
      audio.set(f, off);
      off += f.length;
    }
    return { type: "end", audio, seconds: n / this.o.sampleRate, silenceS };
  }
}

/** Raw 16-bit little-endian PCM of some frames (the streamed speech chunks). */
export function toPcm16(frames: Float32Array[]): Uint8Array<ArrayBuffer> {
  const n = frames.reduce((a, f) => a + f.length, 0);
  const out = new Uint8Array(n * 2);
  const v = new DataView(out.buffer);
  let o = 0;
  for (const f of frames)
    for (let i = 0; i < f.length; i++, o += 2) {
      const s = Math.max(-1, Math.min(1, f[i]));
      v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
  return out;
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
