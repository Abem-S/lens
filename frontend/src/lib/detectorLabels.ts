import type { DetectorState } from "./changeDetect";

export const STATE_LABEL: Record<DetectorState, string> = {
  waiting: "Starting camera…",
  moving: "Watching…",
  hand: "Waiting for your hand to move away",
  settling: "Holding still…",
  steady: "Watching the page",
};
