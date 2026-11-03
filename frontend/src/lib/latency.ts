// End of speech -> first tutor audio, measured in the browser (what the student feels).
// The server reports its own stages (asr, first model text, first piece, first audio);
// this adds the parts only the browser sees: the VAD's silence wait and the network.

export interface TurnLatency {
  end_to_audio_s: number; // the student stopped talking -> the first reply audio arrived
  vad_wait_s: number; // silence waited out before the utterance was cut
  send_to_audio_s: number; // utterance sent -> first reply audio arrived (server + network)
}

export class TurnClock {
  private pending: { speechEnd: number; sent: number } | null = null;
  private seen = 0; // highest tutor utterance id already measured or not a reply

  /** The VAD cut an utterance at `nowMs` after waiting `silenceS` of quiet. */
  ended(nowMs: number, silenceS: number): void {
    this.pending = { speechEnd: nowMs - silenceS * 1000, sent: nowMs };
  }

  /** The turn was dropped or merged into the next one: nothing to measure yet. */
  cancel(): void {
    this.pending = null;
  }

  /** Audio for tutor utterance `utt` arrived at `nowMs`. Returns the turn's latency once. */
  audio(nowMs: number, utt: number): TurnLatency | null {
    if (utt <= this.seen) return null;
    this.seen = utt;
    const p = this.pending;
    if (!p) return null;
    this.pending = null;
    const r = (x: number) => Math.round(x) / 1000;
    return {
      end_to_audio_s: r(nowMs - p.speechEnd),
      vad_wait_s: r(p.sent - p.speechEnd),
      send_to_audio_s: r(nowMs - p.sent),
    };
  }
}
