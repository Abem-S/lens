// Is someone actually using Lens right now? GPU services (live tracking, voice) only run
// while the answer is yes, so a tab left open overnight costs nothing.
//
// - "hidden": the tab is in the background or the screen is off.
// - "idle": no camera movement, speech, tap or key for `idleMs` (default 5 min), or no
//   speech, tap, key or page check for `hardMs` (default 4x idleMs: a webcam facing a busy
//   room keeps seeing movement, so movement alone cannot keep Lens awake forever).
// - "active": otherwise.
// From soft idle, camera movement wakes Lens; from hard idle only a tap, key or speech does.
import { useCallback, useEffect, useRef, useState } from "react";
import { fetchConfig } from "./api";

export type Presence = "active" | "idle" | "hidden";

export class PresenceTracker {
  private lastMotion: number;
  private lastInteract: number;
  private hidden = false;

  constructor(now: number, private idleMs = 5 * 60_000, private hardMs = 4 * idleMs) {
    this.lastMotion = now;
    this.lastInteract = now;
  }

  setIdle(idleMs: number, hardMs = 4 * idleMs): void {
    this.idleMs = idleMs;
    this.hardMs = hardMs;
  }

  /** Movement in front of the camera (the pixel detector, or tracking seeing a hand). */
  motion(now: number): void {
    if (now - this.lastInteract < this.hardMs) this.lastMotion = now;
  }

  /** A tap, key, speech, or a page check: the student is definitely here. */
  interact(now: number): void {
    this.lastInteract = now;
    this.lastMotion = now;
  }

  setHidden(hidden: boolean, now: number): void {
    if (this.hidden && !hidden) this.interact(now); // coming back to the tab counts as being here
    this.hidden = hidden;
  }

  state(now: number): Presence {
    if (this.hidden) return "hidden";
    if (now - this.lastInteract >= this.hardMs) return "idle";
    return now - Math.max(this.lastMotion, this.lastInteract) >= this.idleMs ? "idle" : "active";
  }
}

/** The page's presence, re-checked every few seconds and on visibility / input events. */
export function usePresence(idleMin = 5) {
  const tracker = useRef(new PresenceTracker(performance.now(), idleMin * 60_000));
  const [state, setState] = useState<Presence>(() => (document.hidden ? "hidden" : "active"));
  const refresh = useCallback(() => setState(tracker.current.state(performance.now())), []);

  useEffect(() => {
    tracker.current.setIdle(idleMin * 60_000);
    refresh();
  }, [idleMin, refresh]);

  useEffect(() => {
    const onVisible = () => {
      tracker.current.setHidden(document.hidden, performance.now());
      refresh();
    };
    const onInput = () => {
      tracker.current.interact(performance.now());
      refresh();
    };
    onVisible();
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("pagehide", onVisible);
    for (const e of ["pointerdown", "keydown", "touchstart"]) window.addEventListener(e, onInput, { passive: true });
    const timer = setInterval(refresh, 5000);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("pagehide", onVisible);
      for (const e of ["pointerdown", "keydown", "touchstart"]) window.removeEventListener(e, onInput);
      clearInterval(timer);
    };
  }, [refresh]);

  const motion = useCallback(() => {
    tracker.current.motion(performance.now());
    if (state === "idle") refresh();
  }, [state, refresh]);
  const interact = useCallback(() => {
    tracker.current.interact(performance.now());
    if (state !== "active") refresh();
  }, [state, refresh]);

  return { state, active: state === "active", motion, interact };
}

/** LENS_IDLE_MIN from the server (minutes without activity before Lens rests). */
export function useIdleMinutes(): number {
  const [min, setMin] = useState(5);
  useEffect(() => {
    let live = true;
    fetchConfig().then((c) => live && c.idle_min > 0 && setMin(c.idle_min));
    return () => {
      live = false;
    };
  }, []);
  return min;
}

export const RESTING_NOTE: Record<Exclude<Presence, "active">, string> = {
  idle: "Lens is resting to save power. Move, speak or tap to carry on.",
  hidden: "Lens is paused while this tab is in the background.",
};
