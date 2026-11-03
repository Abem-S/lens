/**
 * Camera pictures while the student talks, so Lens can see what they show over time
 * ("what numbers am I showing?" with 2, then 4, then 6 fingers). Spark reads still images
 * only; a run of them in order gives it the sequence. The server keeps the last few seconds
 * and attaches them only to turns about what the camera shows (lens/tutor.py `_look_frames`).
 */
export const LOOK_EVERY_MS = 500;
export const LOOK_AFTER_MS = 1500; // the hand often still moves just after they stop talking
export const LOOK_SIDE = 768; // smaller than a check frame: a picture every half second is a lot to upload
export const LOOK_MAX_BUFFERED = 400_000; // skip a picture while the socket is still sending this much

export class FrameBurst {
  private timer: ReturnType<typeof setInterval> | null = null;
  private stopAt = Infinity;
  private grabbing = false;

  constructor(
    private grab: () => Promise<Blob | null>,
    private send: (b: Blob) => void,
    private backlog: () => number = () => 0,
    private now: () => number = () => performance.now(),
  ) {}

  get running(): boolean {
    return this.timer !== null;
  }

  /** The student started talking: a picture every LOOK_EVERY_MS until stopped. */
  start(): void {
    this.stopAt = Infinity;
    if (this.timer) return;
    this.timer = setInterval(() => void this.tick(), LOOK_EVERY_MS);
  }

  /** The turn ended: keep looking a moment longer (0 stops now). */
  stopAfter(ms: number): void {
    if (!this.timer) return;
    if (ms <= 0) this.stop();
    else this.stopAt = this.now() + ms;
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.stopAt = Infinity;
  }

  async tick(): Promise<void> {
    if (this.now() >= this.stopAt) {
      this.stop();
      return;
    }
    if (this.grabbing || this.backlog() > LOOK_MAX_BUFFERED) return; // never queue pictures
    this.grabbing = true;
    try {
      const b = await this.grab();
      if (b) this.send(b);
    } catch {
      // a camera hiccup: the next tick tries again
    } finally {
      this.grabbing = false;
    }
  }
}
