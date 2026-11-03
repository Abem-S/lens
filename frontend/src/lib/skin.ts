/**
 * Skin that belongs to a hand, not to the background. A wooden or tan desk is
 * skin-coloured too, and counted as a hand it kept the phone's detector waiting
 * for ever. A pixel that has looked like skin for `staticSamples` samples in a row
 * is background; a hand that comes in still counts until it has been perfectly still
 * that long (and a hand that still is not writing).
 */
export class SkinTracker {
  private age: Uint16Array = new Uint16Array(0);

  constructor(private staticSamples = 20) {} // 20 samples = 5 s at the detectors' 250 ms

  /** `skin`: 1 where the pixel looks like skin in this sample. Returns the hand's fraction. */
  push(skin: Uint8Array): number {
    if (this.age.length !== skin.length) this.age = new Uint16Array(skin.length);
    let hand = 0;
    for (let i = 0; i < skin.length; i++) {
      if (!skin[i]) {
        this.age[i] = 0;
        continue;
      }
      if (this.age[i] < this.staticSamples) {
        this.age[i]++;
        hand++;
      }
    }
    return hand / Math.max(1, skin.length);
  }
}
