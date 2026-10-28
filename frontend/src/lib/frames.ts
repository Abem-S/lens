// Browser glue: sample a video into small grayscale frames for change detection,
// and grab a full-resolution JPEG when a check is due.
import type { Frame } from "./changeDetect";

const SAMPLE_W = 160;

/** Skin in YCbCr: a rough but cheap "is a hand over the page" signal. */
function isSkin(r: number, g: number, b: number): boolean {
  const y = 0.299 * r + 0.587 * g + 0.114 * b;
  const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
  const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
  return y > 60 && cb > 85 && cb < 135 && cr > 138 && cr < 180 && cr - cb > 15;
}

export class FrameSampler {
  private canvas = document.createElement("canvas");
  private ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;

  sample(video: HTMLVideoElement): Frame | null {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return null;
    const w = SAMPLE_W, h = Math.round((SAMPLE_W * vh) / vw);
    this.canvas.width = w;
    this.canvas.height = h;
    this.ctx.drawImage(video, 0, 0, w, h);
    const { data } = this.ctx.getImageData(0, 0, w, h);
    const gray = new Uint8Array(w * h);
    let skin = 0;
    for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
      const r = data[p], g = data[p + 1], b = data[p + 2];
      gray[i] = (r * 77 + g * 150 + b * 29) >> 8;
      if (isSkin(r, g, b)) skin++;
    }
    return { width: w, height: h, gray, skin: skin / gray.length };
  }
}

/** Full frame as JPEG, capped in size so uploads stay quick. */
export function grabJpeg(video: HTMLVideoElement, maxSide = 1600, quality = 0.85): Promise<Blob | null> {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return Promise.resolve(null);
  const scale = Math.min(1, maxSide / Math.max(vw, vh));
  const c = document.createElement("canvas");
  c.width = Math.round(vw * scale);
  c.height = Math.round(vh * scale);
  c.getContext("2d")!.drawImage(video, 0, 0, c.width, c.height);
  return new Promise((resolve) => c.toBlob(resolve, "image/jpeg", quality));
}
