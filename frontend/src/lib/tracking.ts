// Live tracking: small frames go to SAM 3.1 (via the app's /ws/track) a few times a
// second; back come the student's phase (writing / paused / clear), where they point,
// "line_done" when a line is finished and the hand has left it, and where the outlined
// line is now. Tracking owns the check trigger while it is live; the still-frame
// ChangeDetector stays as the fallback (and still catches a first view or a new page).
import { useEffect, useRef, useState } from "react";
import type { CheckResult, Step } from "../contract";
import type { Trigger } from "./changeDetect";
import { backoffMs } from "./errors";

export type Box = [number, number, number, number]; // x, y, w, h normalised
export type Phase = "writing" | "paused" | "clear";
export type Point = [number, number];

/** Where the outlined step is now (lens/follow.py, run by the server on the tracked frames,
 * started from the frame the check read). `anchor`: the step's box on that check frame;
 * `quad`: its corners now (tl, tr, br, bl), so a turned page turns the outline too. */
export interface Followed {
  anchor: Box;
  quad: [Point, Point, Point, Point];
  box: Box;
  state: "ok" | "held" | "lost";
  score: number;
  lost: number;
}

export interface TrackMessage {
  type: "track";
  seq: number;
  phase: Phase;
  pointer: [number, number] | null;
  pointer_kind: "pen" | "finger" | null;
  events: { type: string; box?: Box; phase?: Phase }[];
  follow: Record<string, Followed | null>;
  objects: { concept: string; id: number; box: Box }[];
  gpu_ms?: number;
}

export const FRAME_WIDTH = 512; // px sent per frame (~15 kB JPEG)
export const FRAME_MIN_HEIGHT = 384; // a wide webcam frame keeps this many rows (the page is small in it)
export const FRAME_MAX_WIDTH = 768;

/** Size of a tracking frame: 512 px wide, but a wide (16:9 webcam) frame keeps at least 384 px
 * of height, so handwriting on a page held up in the middle stays legible for the outline. */
export function frameSize(videoWidth: number, videoHeight: number): [number, number] {
  const want = Math.max(FRAME_WIDTH, Math.round((FRAME_MIN_HEIGHT * videoWidth) / videoHeight));
  const w = Math.min(want, FRAME_MAX_WIDTH, videoWidth);
  return [w, Math.round((w * videoHeight) / videoWidth)];
}
export const FPS = 5; // frames offered per second
export const MAX_IN_FLIGHT = 2; // frames sent but not answered; the rest are skipped
export const STALE_MS = 2000; // an unanswered frame older than this no longer counts

export const PHASE_LABEL: Record<Phase, string> = {
  writing: "Watching you write…",
  paused: "Watching the page",
  clear: "Watching the page",
};

/** After a check, the still-frame detector's "new ink" waits this long before it may
 * start a check of its own while tracking is live. Measured in the owner's phone test
 * (2026-11-01): a line_done check's frame still had the hand over the page, so the ink
 * under the hand looked "new" once the hand moved, and a whole-page check (40 s) queued
 * behind every line check. */
export const STILL_AFTER_CHECK_MS = 30_000;

/** While tracking is live it decides when a line it saw being written is done; the
 * still-frame detector keeps the triggers tracking cannot see: a first look at a page,
 * a new page, and ink that appeared without tracking seeing it written (a notebook
 * lowered to write and held up to the webcam again), but not right after a check. */
export function stillTriggerAllowed(trackingLive: boolean, trigger: Trigger, sawWriting = true,
  sinceCheckMs = Infinity): boolean {
  if (!trackingLive || trigger === "first_view" || trigger === "page_changed") return true;
  return !sawWriting && sinceCheckMs >= STILL_AFTER_CHECK_MS;
}

/** Binary frame message: 4-byte big-endian frame number, then the JPEG. */
export function encodeFrame(seq: number, jpeg: Uint8Array): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(4 + jpeg.length);
  new DataView(out.buffer).setUint32(0, seq >>> 0, false);
  out.set(jpeg, 4);
  return out;
}

/** The affine map that takes the anchor box's corners onto the followed quad. */
export function quadMap(anchor: Box, quad: Followed["quad"]): (p: Point) => Point {
  const [[x0, y0], [x1, y1], , [x3, y3]] = quad;
  const [ax, ay, aw, ah] = anchor;
  return ([x, y]) => {
    const u = aw > 0 ? (x - ax) / aw : 0, v = ah > 0 ? (y - ay) / ah : 0;
    return [x0 + u * (x1 - x0) + v * (x3 - x0), y0 + u * (y1 - y0) + v * (y3 - y0)];
  };
}

/** A box carried by `map`, as the axis-aligned bounds of its moved corners. */
export function mapBox(b: Box, map: (p: Point) => Point): Box {
  const pts = [map([b[0], b[1]]), map([b[0] + b[2], b[1]]), map([b[0] + b[2], b[1] + b[3]]), map([b[0], b[1] + b[3]])];
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)];
}

const sameBox = (a: Box, b: Box) => a.every((v, i) => Math.abs(v - b[i]) < 2e-3);

/** The check result drawn where the page is now. Every box and the outline move with the
 * followed step (an affine map, so turns and zoom carry over). `outline` says how sure that
 * place is: "ok" (seen now), "held" (covered a moment, or tracking paused: faded), "lost"
 * (not found: hidden rather than drawn on the wrong place). A follow report for another
 * step (an older check) is not used: the outline then waits for its own ("lost"). */
export function moveResult(result: CheckResult | null, followed: Followed | null, live: boolean): CheckResult | null {
  if (!result?.finding) return result;
  const finding = result.finding;
  const step = result.page.steps.find((s) => s.id === finding.step_id);
  if (!step || !(step.bbox[2] > 0)) return result;
  if (!followed || !sameBox(followed.anchor, step.bbox)) {
    // Tracking that never reported (off, unavailable): draw where the check saw it. Live
    // tracking that has not located this step yet: hide it until it has.
    return live ? { ...result, outline: "lost" } : result;
  }
  const outline = followed.state === "lost" ? "lost" : live ? followed.state : "held";
  const map = quadMap(followed.anchor, followed.quad);
  const steps: Step[] = result.page.steps.map((s) => ({ ...s, bbox: mapBox(s.bbox, map) }));
  const b = step.bbox;
  const mask: Point[] = finding.mask && finding.mask.length >= 3
    ? finding.mask
    : [[b[0], b[1]], [b[0] + b[2], b[1]], [b[0] + b[2], b[1] + b[3]], [b[0], b[1] + b[3]]];
  return { page: { ...result.page, steps }, finding: { ...finding, mask: mask.map(map) }, outline };
}

/** The line a fingertip points at (a simpler cousin of lens/pointing.py:nearest_step). */
export function pointedStep(point: [number, number] | null, steps: Step[], maxDistance = 0.12): Step | null {
  if (!point) return null;
  const [px, py] = point;
  let best: Step | null = null;
  let bestD = Infinity;
  for (const s of steps) {
    const [x, y, w, h] = s.bbox;
    if (w <= 0 || h <= 0) continue;
    const dx = Math.max(x - px, 0, px - (x + w));
    const below = py - (y + h);
    const dy = py >= y && py <= y + h ? 0 : below > 0 ? Math.max(0, below - 0.5 * h) : y - py;
    const d = Math.hypot(dx, 2 * dy);
    if (d < bestD) [best, bestD] = [s, d];
  }
  return bestD <= maxDistance ? best : null;
}

export type TrackingStatus = "off" | "connecting" | "live" | "unavailable";

interface Options {
  /** A line is finished and the hand has left it: time for a page check. */
  onLineDone?: (box: Box) => void;
  /** Every processed frame (e.g. to forward to the laptop in phone mode). */
  onTrack?: (m: TrackMessage) => void;
  id?: string; // ties the tutor's "this line?" to this tracking session
}

/** Stream frames from `getVideo()` to the tracker while `enabled`. */
export function useTracking(getVideo: () => HTMLVideoElement | null, enabled: boolean, opts: Options = {}) {
  const [status, setStatus] = useState<TrackingStatus>("off");
  const [last, setLast] = useState<TrackMessage | null>(null);
  const ws = useRef<WebSocket | null>(null);
  const seq = useRef(0);
  const inflight = useRef<Map<number, number>>(new Map());
  const sawWriting = useRef(false); // tracking saw the pen writing since the last check
  const checkedAt = useRef(-Infinity); // performance.now() of the last check sent
  const optsRef = useRef(opts);
  const getVideoRef = useRef(getVideo);
  getVideoRef.current = getVideo;
  optsRef.current = opts;
  const canvas = useRef<HTMLCanvasElement | null>(null);

  // Reconnect after the service went away (a cold start that failed, the server's time cap,
  // a network drop), with growing waits; never after "Modal is not configured".
  const [attempt, setAttempt] = useState(0);
  const failures = useRef(0);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) return;
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const id = encodeURIComponent(optsRef.current.id ?? "");
    const sock = new WebSocket(`${proto}://${window.location.host}/ws/track?id=${id}`);
    sock.binaryType = "arraybuffer";
    ws.current = sock;
    setStatus("connecting");
    let retry = true;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let closedByUs = false;
    const scheduleRetry = () => {
      if (closedByUs || !retry || retryTimer) return;
      retryTimer = setTimeout(() => setAttempt((a) => a + 1), backoffMs(failures.current++ + 2));
    };
    sock.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      const m = JSON.parse(ev.data);
      if (m.type === "ready") {
        setStatus("live");
        setNote(null);
        failures.current = 0;
      } else if (m.type === "unavailable") {
        setStatus("unavailable");
        if (m.retry === false) retry = false;
        setNote(m.message ?? null);
        scheduleRetry();
      } else if (m.type === "track") {
        const t = m as TrackMessage;
        for (const k of [...inflight.current.keys()]) if (k <= t.seq) inflight.current.delete(k);
        if (t.phase === "writing") sawWriting.current = true;
        setLast(t);
        optsRef.current.onTrack?.(t);
        for (const e of t.events) if (e.type === "line_done" && e.box) optsRef.current.onLineDone?.(e.box);
      }
    };
    sock.onclose = () => {
      setStatus((s) => (s === "live" || s === "connecting" ? "unavailable" : s));
      scheduleRetry();
    };
    return () => {
      closedByUs = true;
      if (retryTimer) clearTimeout(retryTimer);
      sock.close();
      ws.current = null;
      inflight.current.clear();
      setStatus("off");
    };
  }, [enabled, attempt]);

  // Offer a frame FPS times a second; skip it while MAX_IN_FLIGHT frames are unanswered.
  useEffect(() => {
    if (!enabled) return;
    const timer = setInterval(() => {
      const sock = ws.current, v = getVideoRef.current();
      if (!sock || sock.readyState !== WebSocket.OPEN || !v || !v.videoWidth || document.hidden) return;
      const now = performance.now();
      for (const [k, t] of inflight.current) if (now - t > STALE_MS) inflight.current.delete(k);
      if (inflight.current.size >= MAX_IN_FLIGHT) return;
      const c = (canvas.current ??= document.createElement("canvas"));
      [c.width, c.height] = frameSize(v.videoWidth, v.videoHeight);
      c.getContext("2d")!.drawImage(v, 0, 0, c.width, c.height);
      const n = ++seq.current;
      inflight.current.set(n, now);
      c.toBlob(async (b) => {
        if (!b || sock.readyState !== WebSocket.OPEN) return inflight.current.delete(n);
        sock.send(encodeFrame(n, new Uint8Array(await b.arrayBuffer())));
      }, "image/jpeg", 0.65);
    }, 1000 / FPS);
    return () => clearInterval(timer);
  }, [enabled]);

  return { status, live: status === "live", last, seq, sawWriting, checkedAt, note };
}

/** A line for the screen about live tracking, or null when there is nothing to say. */
export function trackingNote(status: TrackingStatus, serverNote: string | null): string | null {
  if (status === "connecting") return "Waking up live tracking (up to 30 s). Lens still checks your page meanwhile.";
  if (status === "unavailable")
    return serverNote ?? "Live tracking is off for now, so Lens checks your page when it goes still.";
  return null;
}

/** Everything a Lens mode needs from tracking, in one hook: frames out, a check on
 * `line_done`, the result moved with the page, a live "pointing at" line and a status. */
export function useLensTracking(
  getVideo: () => HTMLVideoElement | null,
  enabled: boolean,
  result: CheckResult | null,
  onLineDone: (box: Box) => void,
  id?: string,
) {
  const t = useTracking(getVideo, enabled, { onLineDone, id });
  const shown = moveResult(result, t.last?.follow?.finding ?? null, t.live);
  const fingerStep =
    t.live && t.last?.pointer_kind === "finger" && shown ? pointedStep(t.last.pointer, shown.page.steps) : null;
  return {
    live: t.live,
    status: t.status,
    note: trackingNote(t.status, t.note),
    /** The latest processed frame (to tell presence that a hand is moving). */
    last: t.last,
    result: shown,
    pointingStep: fingerStep?.id ?? null,
    label: t.live && t.last ? PHASE_LABEL[t.last.phase] : null,
    /** May the still-frame detector's `trigger` start a check now? */
    stillAllowed: (trigger: Trigger) =>
      stillTriggerAllowed(t.live, trigger, t.sawWriting.current, performance.now() - t.checkedAt.current),
    /** Call when a frame is grabbed for a check: tracking starts a new line. (The outline
     * starts from that very frame on the server, which gets it with the check.) */
    markCheck: () => {
      t.sawWriting.current = false;
      t.checkedAt.current = performance.now();
    },
  };
}
