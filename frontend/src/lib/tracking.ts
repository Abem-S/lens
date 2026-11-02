// Live tracking: small frames go to SAM 3.1 (via the app's /ws/track) a few times a
// second; back come the student's phase (writing / paused / clear), where they point,
// "line_done" when a line is finished and the hand has left it, and where the outlined
// line is now. Tracking owns the check trigger while it is live; the still-frame
// ChangeDetector stays as the fallback (and still catches a first view or a new page).
import { useCallback, useEffect, useRef, useState } from "react";
import type { CheckResult, Step } from "../contract";
import type { Trigger } from "./changeDetect";

export type Box = [number, number, number, number]; // x, y, w, h normalised
export type Phase = "writing" | "paused" | "clear";

export interface TrackMessage {
  type: "track";
  seq: number;
  phase: Phase;
  pointer: [number, number] | null;
  pointer_kind: "pen" | "finger" | null;
  events: { type: string; box?: Box; phase?: Phase }[];
  follow: Record<string, { box: Box; score: number; lost: number } | null>;
  objects: { concept: string; id: number; box: Box }[];
  gpu_ms?: number;
}

export const FRAME_WIDTH = 512; // px sent per frame (~15 kB JPEG)
export const FPS = 5; // frames offered per second
export const MAX_IN_FLIGHT = 2; // frames sent but not answered; the rest are skipped
export const STALE_MS = 2000; // an unanswered frame older than this no longer counts

export const PHASE_LABEL: Record<Phase, string> = {
  writing: "Watching you write…",
  paused: "Watching the page",
  clear: "Watching the page",
};

/** While tracking is live it decides when a line is done; the still-frame detector
 * only keeps the triggers tracking cannot see (first look at a page, a new page). */
export function stillTriggerAllowed(trackingLive: boolean, trigger: Trigger): boolean {
  return !trackingLive || trigger === "first_view" || trigger === "page_changed";
}

/** Binary frame message: 4-byte big-endian frame number, then the JPEG. */
export function encodeFrame(seq: number, jpeg: Uint8Array): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(4 + jpeg.length);
  new DataView(out.buffer).setUint32(0, seq >>> 0, false);
  out.set(jpeg, 4);
  return out;
}

/** Map a box drawn on the check frame to where the tracked anchor box is now. */
export function mapBox(b: Box, from: Box, to: Box): Box {
  const sx = from[2] > 0 ? to[2] / from[2] : 1;
  const sy = from[3] > 0 ? to[3] / from[3] : 1;
  return [to[0] + (b[0] - from[0]) * sx, to[1] + (b[1] - from[1]) * sy, b[2] * sx, b[3] * sy];
}

/** The check result drawn where the page is now: every box and the outline move with
 * the tracked mistake line (`from` = its box on the check frame, `to` = its box now). */
export function moveResult(result: CheckResult | null, from: Box | null, to: Box | null): CheckResult | null {
  if (!result || !from || !to) return result;
  const near = Math.abs(from[0] - to[0]) + Math.abs(from[1] - to[1]) + Math.abs(from[2] - to[2]) < 0.004;
  if (near) return result;
  const steps: Step[] = result.page.steps.map((s) => ({ ...s, bbox: mapBox(s.bbox, from, to) }));
  const finding = result.finding && {
    ...result.finding,
    mask: result.finding.mask?.map(([x, y]) => [
      to[0] + (x - from[0]) * (from[2] > 0 ? to[2] / from[2] : 1),
      to[1] + (y - from[1]) * (from[3] > 0 ? to[3] / from[3] : 1),
    ] as [number, number]) ?? null,
  };
  return { page: { ...result.page, steps }, finding };
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
  const optsRef = useRef(opts);
  const getVideoRef = useRef(getVideo);
  getVideoRef.current = getVideo;
  optsRef.current = opts;
  const canvas = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!enabled) return;
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const id = encodeURIComponent(optsRef.current.id ?? "");
    const sock = new WebSocket(`${proto}://${window.location.host}/ws/track?id=${id}`);
    sock.binaryType = "arraybuffer";
    ws.current = sock;
    setStatus("connecting");
    sock.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      const m = JSON.parse(ev.data);
      if (m.type === "ready") setStatus("live");
      else if (m.type === "unavailable") setStatus("unavailable");
      else if (m.type === "track") {
        const t = m as TrackMessage;
        for (const k of [...inflight.current.keys()]) if (k <= t.seq) inflight.current.delete(k);
        setLast(t);
        optsRef.current.onTrack?.(t);
        for (const e of t.events) if (e.type === "line_done" && e.box) optsRef.current.onLineDone?.(e.box);
      }
    };
    sock.onclose = () => setStatus((s) => (s === "live" || s === "connecting" ? "unavailable" : s));
    return () => {
      sock.close();
      ws.current = null;
      inflight.current.clear();
      setStatus("off");
    };
  }, [enabled]);

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
      c.width = Math.min(FRAME_WIDTH, v.videoWidth);
      c.height = Math.round((c.width * v.videoHeight) / v.videoWidth);
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

  /** Keep the line at `box` (on frame `atSeq`, default: the newest frame) located from now on. */
  const follow = useCallback((box: Box, atSeq?: number) => {
    const s = ws.current;
    if (s?.readyState === WebSocket.OPEN)
      s.send(JSON.stringify({ type: "follow", key: "finding", box, seq: atSeq ?? seq.current }));
  }, []);
  const unfollow = useCallback(() => {
    const s = ws.current;
    if (s?.readyState === WebSocket.OPEN) s.send(JSON.stringify({ type: "unfollow", key: "finding" }));
  }, []);

  return { status, live: status === "live", last, follow, unfollow, seq };
}

/** Outline following for a check result: remembers the frame a check read, asks the
 * tracker to follow the mistake line from there, and returns the result moved to
 * where the page is now. `followBox` is the tracker's latest box for that line. */
export function useFollowedResult(
  result: CheckResult | null,
  checkSeq: number | null,
  follow: (box: Box, atSeq?: number) => void,
  unfollow: () => void,
  followBox: Box | null,
  live: boolean,
): CheckResult | null {
  const anchor = useRef<Box | null>(null);
  const finding = result?.finding ?? null;
  const step = finding ? result?.page.steps.find((s) => s.id === finding.step_id) : undefined;
  const key = step ? `${finding?.step_id}:${step.bbox.join(",")}` : "";
  useEffect(() => {
    if (!live) return;
    if (step && step.bbox[2] > 0 && step.bbox[3] > 0) {
      anchor.current = step.bbox;
      follow(step.bbox, checkSeq ?? undefined);
    } else {
      anchor.current = null;
      unfollow();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, live]);
  return live ? moveResult(result, anchor.current, followBox) : result;
}

/** Everything a Lens mode needs from tracking, in one hook: frames out, a check on
 * `line_done`, the result moved with the page, a live "pointing at" line and a status. */
export function useLensTracking(
  getVideo: () => HTMLVideoElement | null,
  enabled: boolean,
  result: CheckResult | null,
  onLineDone: () => void,
  id?: string,
) {
  const checkSeq = useRef<number | null>(null);
  const t = useTracking(getVideo, enabled, { onLineDone, id });
  const followBox = t.last?.follow?.finding?.box ?? null;
  const shown = useFollowedResult(result, checkSeq.current, t.follow, t.unfollow, followBox, t.live);
  const fingerStep =
    t.live && t.last?.pointer_kind === "finger" && shown ? pointedStep(t.last.pointer, shown.page.steps) : null;
  return {
    live: t.live,
    status: t.status,
    result: shown,
    pointingStep: fingerStep?.id ?? null,
    label: t.live && t.last ? PHASE_LABEL[t.last.phase] : null,
    /** Call when a frame is grabbed for a check: its outline will start from this frame. */
    markCheck: () => void (checkSeq.current = t.seq.current),
  };
}
