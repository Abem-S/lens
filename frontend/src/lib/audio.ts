// Microphone capture (16 kHz mono frames) and tutor voice playback (PCM16 chunks).

const WORKLET = `
class Tap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("lens-tap", Tap);
`;

export const MIC_RATE = 16000;
export const FRAME = 320; // 20 ms at 16 kHz

/** Calls onFrame with 20 ms Float32 frames at 16 kHz. Echo cancellation keeps the tutor out of the mic. */
export async function startMic(onFrame: (f: Float32Array) => void): Promise<() => void> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  });
  const ctx = new AudioContext({ sampleRate: MIC_RATE });
  const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
  await ctx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);
  const src = ctx.createMediaStreamSource(stream);
  const tap = new AudioWorkletNode(ctx, "lens-tap");
  let buf = new Float32Array(0);
  tap.port.onmessage = (ev: MessageEvent<Float32Array>) => {
    const joined = new Float32Array(buf.length + ev.data.length);
    joined.set(buf);
    joined.set(ev.data, buf.length);
    let off = 0;
    for (; off + FRAME <= joined.length; off += FRAME) onFrame(joined.slice(off, off + FRAME));
    buf = joined.slice(off);
  };
  src.connect(tap);
  // The worklet must be pulled by the graph; route it through a muted gain.
  const mute = ctx.createGain();
  mute.gain.value = 0;
  tap.connect(mute).connect(ctx.destination);
  return () => {
    stream.getTracks().forEach((t) => t.stop());
    ctx.close().catch(() => undefined);
  };
}

export const DUCK_GAIN = 0.25; // Lens's volume while it waits to learn whether the student really talks

/** Plays PCM16 chunks back to back; duck() lowers the volume (maybe a barge-in), stop() silences
 * everything at once (a real one). */
export class Player {
  private ctx: AudioContext;
  private gain: GainNode;
  private next = 0;
  private sources = new Set<AudioBufferSourceNode>();
  onIdle: (() => void) | null = null;

  constructor() {
    this.ctx = new AudioContext();
    this.gain = this.ctx.createGain();
    this.gain.connect(this.ctx.destination);
  }

  /** Lower (true) or restore (false) the volume, over ~30 ms so it does not click. */
  duck(on: boolean): void {
    const g = this.gain.gain;
    g.cancelScheduledValues(this.ctx.currentTime);
    g.setTargetAtTime(on ? DUCK_GAIN : 1, this.ctx.currentTime, 0.01);
  }

  async resume(): Promise<void> {
    if (this.ctx.state !== "running") await this.ctx.resume();
  }

  get playing(): boolean {
    return this.sources.size > 0;
  }

  enqueue(pcm: ArrayBuffer, sampleRate: number): void {
    const ints = new Int16Array(pcm);
    if (!ints.length) return;
    const buffer = this.ctx.createBuffer(1, ints.length, sampleRate);
    const ch = buffer.getChannelData(0);
    for (let i = 0; i < ints.length; i++) ch[i] = ints[i] / 32768;
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.gain);
    const at = Math.max(this.ctx.currentTime + 0.02, this.next);
    src.start(at);
    this.next = at + buffer.duration;
    this.sources.add(src);
    src.onended = () => {
      this.sources.delete(src);
      if (!this.sources.size) this.onIdle?.();
    };
  }

  stop(): void {
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {
        /* already stopped */
      }
    }
    this.sources.clear();
    this.next = 0;
    this.duck(false);
    this.onIdle?.();
  }

  close(): void {
    this.stop();
    this.ctx.close().catch(() => undefined);
  }
}
