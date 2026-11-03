import { useCallback, useEffect, useRef, useState } from "react";
import type { CheckResult } from "../contract";
import { checkPage } from "./api";
import { CheckQueue } from "./checkQueue";
import { withReading } from "./reading";
import type { Box } from "./tracking";
import type { Activity } from "./changeDetect";
import { Player, startMic } from "./audio";
import { TurnClock } from "./latency";
import { FrameBurst, LOOK_AFTER_MS, LOOK_SIDE } from "./look";
import { SpeechStreamer, type Barge } from "./speech";
import { useA11ySettings } from "./a11y";
import { localSummary, type SessionSummary } from "./summary";
import { ACCESS_EXPIRED, backoffMs, CLOSE_ACCESS, CLOSE_LIMIT, micMessage, OFFLINE, RECONNECTING } from "./errors";

type PendingCheck = { image: Blob; reason: string; box: Box | null };

export type TutorState = "offline" | "connecting" | "idle" | "listening" | "thinking" | "speaking" | "reading";

export interface Caption {
  who: "lens" | "you";
  text: string;
  utt?: number; // Lens: sentences of one reply share it
  typed?: boolean; // you: typed, not spoken
}

export const CAPTION_LOG = 60; // captions kept on screen (scrollable)
const SUMMARY_WAIT_MS = 4000;

const TRIGGER_NOTE: Record<string, string> = {
  first_view: "Reading your page…",
  new_ink: "New working spotted. Reading…",
  ink_removed: "Something changed. Reading…",
  page_changed: "New page. Reading…",
  line_done: "Step finished. Reading…",
  manual: "Reading your page…",
};

/**
 * The tutor session: page checks, the spoken conversation, barge-in and pointing.
 * Checks go over the tutor socket so the tutor knows the page; if the socket is
 * down they fall back to plain HTTP.
 */
export function useTutor(
  getFrame?: (maxSide?: number) => Promise<Blob | null>, // a camera picture (default size: a check frame)
  trackId?: string,
  active = true,
  onHeard?: () => void, // the student spoke: they are here (lib/presence.ts)
) {
  const onHeardRef = useRef(onHeard);
  onHeardRef.current = onHeard;
  const [state, setState] = useState<TutorState>("connecting");
  // A banner about the connection or a limit (reconnecting, access expired, today's cap…).
  const [notice, setNotice] = useState<string | null>(null);
  const [voiceReady, setVoiceReady] = useState(false); // the voice service answered its wake-up ping
  const [attempt, setAttempt] = useState(0); // bumps to reconnect the tutor socket
  const activeRef = useRef(active);
  activeRef.current = active;
  const voiceOnRef = useRef(false);
  const failures = useRef(0); // reconnect attempts in a row
  const dailyCap = useRef(false); // today's limit was hit: do not keep reconnecting
  const [result, setResult] = useState<CheckResult | null>(null);
  const [captions, setCaptions] = useState<Caption[]>([]);
  const [voiceOn, setVoiceOn] = useState(false);
  const [voiceAvailable, setVoiceAvailable] = useState(false);
  const [pointing, setPointing] = useState<{ step_id: string; point: [number, number] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  // Accessibility: captions, captions-only mode, speaking speed (lib/a11y.ts), and the session summary.
  const [settings, updateSettings] = useA11ySettings();
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const audioReady = useRef(false); // the browser allows audio once the student has clicked or typed
  const [summary, setSummary] = useState<SessionSummary | null>(null);
  const summaryWait = useRef<number | null>(null);
  const startedAt = useRef(performance.now());
  const resultRef = useRef<CheckResult | null>(null);
  resultRef.current = result;
  const ws = useRef<WebSocket | null>(null);
  const player = useRef<Player | null>(null);
  const speech = useRef(new SpeechStreamer());
  const vad = { current: speech.current.vad };
  const clock = useRef(new TurnClock());
  const stopMic = useRef<(() => void) | null>(null);
  const audioHeader = useRef<{ sample_rate: number; utt: number } | null>(null);
  const silencedUtt = useRef(0); // utterance ids at or below this are dropped (barge-in)
  // One check at a time; a step finished meanwhile waits instead of being dropped.
  const checks = useRef(new CheckQueue<PendingCheck>());
  const runCheck = useRef<(c: PendingCheck) => void>(() => undefined);
  const finishCheck = useCallback(() => {
    setNote(null);
    const next = checks.current.done();
    if (next) runCheck.current(next);
  }, []);
  const getFrameRef = useRef(getFrame);
  getFrameRef.current = getFrame;

  const send = useCallback((msg: object, payload?: Blob | Uint8Array<ArrayBuffer>) => {
    const s = ws.current;
    if (!s || s.readyState !== WebSocket.OPEN) return false;
    s.send(JSON.stringify(msg));
    if (payload) s.send(payload);
    return true;
  }, []);
  // Pictures every half second while the student talks, so Lens sees what they show over time.
  const burst = useRef<FrameBurst | null>(null);
  if (!burst.current)
    burst.current = new FrameBurst(
      () => getFrameRef.current?.(LOOK_SIDE) ?? Promise.resolve(null),
      (b) => void send({ type: "frame" }, b),
      () => ws.current?.bufferedAmount ?? 0,
    );
  useEffect(() => () => burst.current?.stop(), []);

  useEffect(() => {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const sock = new WebSocket(`${proto}://${window.location.host}/ws/tutor`);
    sock.binaryType = "arraybuffer";
    ws.current = sock;
    let closedByUs = false;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let limited = false; // a "limit" message came: the close that follows is on purpose
    // `track`: live tracking already knows where the student points (lib/tracking.ts).
    // After a reconnect the voice stays as it was, without a second greeting.
    sock.onopen = () => {
      failures.current = 0;
      setNotice(null);
      sock.send(JSON.stringify({ type: "hello", voice: voiceOnRef.current, opening: false, track: trackId,
        active: activeRef.current }));
      pushSettings();
    };
    sock.onclose = (ev) => {
      setState("offline");
      if (checks.current.busy) finishCheck(); // its answer will not come; a waiting check goes over HTTP
      if (closedByUs) return;
      if (ev.code === CLOSE_ACCESS) return setNotice(ACCESS_EXPIRED);
      if (ev.code === CLOSE_LIMIT || limited) {
        // Busy or an hourly limit: try again in a minute. Today's cap: stay closed.
        if (!dailyCap.current) retryTimer = setTimeout(() => setAttempt((a) => a + 1), 60_000);
        return;
      }
      setNotice(navigator.onLine === false ? OFFLINE : RECONNECTING);
      retryTimer = setTimeout(() => setAttempt((a) => a + 1), backoffMs(failures.current++));
    };
    sock.onmessage = (ev) => {
      if (typeof ev.data !== "string") {
        const h = audioHeader.current;
        if (h && h.utt > silencedUtt.current && player.current && settingsRef.current.voice) {
          const turn = clock.current.audio(performance.now(), h.utt);
          if (turn) {
            setLatency(turn.end_to_audio_s);
            sock.send(JSON.stringify({ type: "client_timing", ...turn }));
          }
          player.current.enqueue(ev.data as ArrayBuffer, h.sample_rate);
          vad.current.playback = true;
        }
        return;
      }
      const m = JSON.parse(ev.data);
      switch (m.type) {
        case "ready":
          setVoiceAvailable(Boolean(m.voice_available ?? m.voice));
          setState("idle");
          break;
        case "state":
          setState(m.state === "listening" ? (stopMic.current ? "listening" : "idle") : m.state);
          if (m.state === "reading") setNote((n) => n ?? TRIGGER_NOTE.manual);
          break;
        case "steps": // read, not yet verified: show the steps while the check goes on
          setResult((r) => withReading(r, m.page));
          break;
        case "result":
          setResult({ page: m.page, finding: m.finding });
          if (!m.cleared) finishCheck(); // `cleared`: a new problem, not the answer to a check
          break;
        case "say":
          if (m.utt > silencedUtt.current)
            setCaptions((c) => [...c.slice(1 - CAPTION_LOG), { who: "lens", text: m.text, utt: m.utt }]);
          break;
        case "transcript":
          // A short turn spoken over Lens: words stop it for good (the server has already
          // stopped speaking, so no "interrupt"), an empty one lets it finish at full volume.
          if (!m.typed) applyBarge(speech.current.heard(m.text ?? ""), false);
          // A typed turn is already on screen (added when sent); its echo only confirms it.
          if (m.text && !m.typed) setCaptions((c) => [...c.slice(1 - CAPTION_LOG), { who: "you", text: m.text }]);
          break;
        case "summary":
          if (summaryWait.current !== null) window.clearTimeout(summaryWait.current);
          summaryWait.current = null;
          setSummary(m as SessionSummary);
          break;
        case "audio":
          audioHeader.current = { sample_rate: m.sample_rate, utt: m.utt };
          break;
        case "pointing":
          setPointing({ step_id: m.step_id, point: m.point });
          setTimeout(() => setPointing(null), 4000);
          break;
        case "error":
          setError(m.message);
          finishCheck();
          break;
        case "note": // the model is slow: say so instead of a silent spinner
          setNote(m.text);
          break;
        case "limit":
          // "checks" / "utterances": this request was skipped, the session goes on.
          // "ip" / "busy" / "daily": the server closes the session next.
          if (m.reason !== "checks" && m.reason !== "utterances") limited = true;
          if (m.reason === "daily") dailyCap.current = true;
          setNotice(m.message);
          if (m.reason === "checks" || m.reason === "daily") finishCheck();
          if (!limited) setTimeout(() => setNotice((n) => (n === m.message ? null : n)), 15_000);
          break;
        case "warm":
          if (m.service === "voice") setVoiceReady(true);
          break;
      }
    };
    const online = () => {
      if (sock.readyState === WebSocket.CLOSED && !closedByUs) setAttempt((a) => a + 1);
    };
    window.addEventListener("online", online);
    return () => {
      closedByUs = true;
      if (retryTimer) clearTimeout(retryTimer);
      window.removeEventListener("online", online);
      sock.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attempt]);

  useEffect(
    () => () => {
      stopMic.current?.();
      player.current?.close();
      if (summaryWait.current !== null) window.clearTimeout(summaryWait.current);
    },
    [],
  );

  // Tell the server whether someone is here (it keeps GPU services warm only then), on every
  // change and as a heartbeat; it stops trusting a heartbeat after 90 s.
  useEffect(() => {
    send({ type: "presence", active });
    if (!active) return;
    const t = setInterval(() => send({ type: "presence", active: true }), 30_000);
    return () => clearInterval(t);
  }, [active, send, attempt]);

  /** Tell the server how Lens should speak. Audio only once the browser can play it. */
  function pushSettings() {
    const s = settingsRef.current;
    send({ type: "settings", aloud: s.voice && audioReady.current, speed: s.speed });
  }
  useEffect(pushSettings, [settings.voice, settings.speed, send]);

  /** Audio needs a click or key press first: call from one. */
  const ensurePlayer = useCallback(async () => {
    player.current ??= new Player();
    await player.current.resume();
    player.current.onIdle = () => (vad.current.playback = false);
    if (!audioReady.current) {
      audioReady.current = true;
      pushSettings();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  runCheck.current = async ({ image, reason, box }) => {
    setError(null);
    setNote(TRIGGER_NOTE[reason] ?? TRIGGER_NOTE.manual);
    // `box`: where the finished line is on this very frame (tracking's line_done), so the
    // fast line check still applies when the check waited behind another one.
    if (send(box ? { type: "check", reason, box } : { type: "check", reason }, image)) return;
    try {
      setResult(await checkPage(image, result?.page.problem));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      finishCheck();
    }
  };

  const check = useCallback((image: Blob, reason: string, box?: Box | null) => {
    const now = checks.current.offer({ image, reason, box: box ?? null });
    if (now) runCheck.current(now);
  }, []);

  /** Stop Lens at once (the student talks, types or asks for the page). `tell`: also tell the
   * server, unless it already stopped by itself. */
  const bargeIn = useCallback((tell = true) => {
    if (player.current?.playing) {
      player.current.stop();
      silencedUtt.current = audioHeader.current?.utt ?? silencedUtt.current;
      if (tell) send({ type: "interrupt" });
    }
    vad.current.playback = false;
  }, [send]);

  /** Speech over Lens first only lowers its voice; it stops once the student really talks (lib/speech.ts). */
  function applyBarge(b: Barge, tell = true) {
    if (b === "stop") bargeIn(tell);
    else if (b === "duck") player.current?.duck(true);
    else if (b === "restore") player.current?.duck(false);
  }

  /** Start the spoken conversation (needs a click: browsers only allow audio after one). */
  const startVoice = useCallback(async () => {
    try {
      await ensurePlayer();
      stopMic.current = await startMic((frame) => {
        const { out, event, continues, barge } = speech.current.push(frame, performance.now(), !!player.current?.playing);
        applyBarge(barge);
        if (event?.type === "start") {
          onHeardRef.current?.();
          if (continues) clock.current.cancel(); // the same turn goes on: measure from its new end
          else
            // A view of the page for "this step?" / "can you see…", sent while they talk so it is
            // up before they stop (live tracking, when on, gives the server a fresher pointer).
            (getFrameRef.current?.() ?? Promise.resolve(null)).then((f) => f && send({ type: "frame" }, f));
          burst.current?.start(); // ...then smaller ones every half second (lib/look.ts)
        }
        for (const o of out) send(o.msg, o.pcm);
        if (event?.type === "end") {
          clock.current.ended(performance.now(), event.silenceS);
          burst.current?.stopAfter(LOOK_AFTER_MS);
        }
        if (event?.type === "discard") {
          clock.current.cancel();
          burst.current?.stopAfter(0);
        }
      });
      setVoiceOn(true);
      voiceOnRef.current = true;
      setState("listening");
      send({ type: "voice", on: true });
    } catch (e) {
      setError(micMessage(e as Error));
    }
  }, [bargeIn, send, ensurePlayer]);

  const stopVoice = useCallback(() => {
    stopMic.current?.();
    stopMic.current = null;
    player.current?.stop();
    setVoiceOn(false);
    voiceOnRef.current = false;
    setState("idle");
    send({ type: "voice", on: false });
  }, [send]);

  /** What the student is doing, from the pixel detector when live tracking is off
   * (callers send it on change and as a heartbeat: lib/changeDetect.ts ActivityTracker.report). */
  const activity = useCallback((phase: Activity) => void send({ type: "activity", phase }), [send]);

  /** Type instead of speaking: the same reply path as a spoken turn, with the current view attached. */
  const sendText = useCallback(async (raw: string) => {
    const text = raw.trim();
    if (!text) return false;
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
      setError("Lens is offline, so it cannot answer right now.");
      return false;
    }
    bargeIn();
    setCaptions((c) => [...c.slice(1 - CAPTION_LOG), { who: "you", text, typed: true }]);
    if (settingsRef.current.voice) await ensurePlayer().catch(() => undefined);
    const frame = await (getFrameRef.current?.() ?? Promise.resolve(null)).catch(() => null);
    if (frame) send({ type: "frame" }, frame);
    return send({ type: "text", text });
  }, [bargeIn, ensurePlayer, send]);

  /** Read the steps already read back aloud (and in the captions). */
  const readPage = useCallback(async () => {
    bargeIn();
    if (settingsRef.current.voice) await ensurePlayer().catch(() => undefined);
    if (!send({ type: "read_page" })) setError("Lens is offline, so it cannot read the page right now.");
  }, [bargeIn, ensurePlayer, send]);

  /** End the session: Lens stops talking and the summary is shown. Nothing is stored. */
  const endSession = useCallback(() => {
    stopMic.current?.();
    stopMic.current = null;
    player.current?.stop();
    setVoiceOn(false);
    const fallback = () => {
      summaryWait.current = null;
      setSummary(localSummary(resultRef.current, startedAt.current, performance.now()));
    };
    if (!send({ type: "end" })) return fallback();
    summaryWait.current = window.setTimeout(fallback, SUMMARY_WAIT_MS);
  }, [send]);

  return {
    state, result, captions, voiceOn, voiceAvailable, pointing, error, note, latency,
    busy: note !== null, check, startVoice, stopVoice, activity, notice, voiceReady,
    settings, updateSettings, sendText, readPage, summary, endSession, closeSummary: () => setSummary(null),
  };
}
