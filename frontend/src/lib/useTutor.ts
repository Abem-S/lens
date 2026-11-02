import { useCallback, useEffect, useRef, useState } from "react";
import type { CheckResult } from "../contract";
import { checkPage } from "./api";
import { Player, startMic, MIC_RATE } from "./audio";
import { EnergyVad, encodeWav } from "./vad";

export type TutorState = "offline" | "connecting" | "idle" | "listening" | "thinking" | "speaking" | "reading";

export interface Caption {
  who: "lens" | "you";
  text: string;
}

const TRIGGER_NOTE: Record<string, string> = {
  first_view: "Reading your page…",
  new_ink: "New line spotted. Reading…",
  ink_removed: "Something changed. Reading…",
  page_changed: "New page. Reading…",
  line_done: "Line finished. Reading…",
  manual: "Reading your page…",
};

/**
 * The tutor session: page checks, the spoken conversation, barge-in and pointing.
 * Checks go over the tutor socket so the tutor knows the page; if the socket is
 * down they fall back to plain HTTP.
 */
export function useTutor(getFrame?: () => Promise<Blob | null>, trackId?: string) {
  const [state, setState] = useState<TutorState>("connecting");
  const [result, setResult] = useState<CheckResult | null>(null);
  const [captions, setCaptions] = useState<Caption[]>([]);
  const [voiceOn, setVoiceOn] = useState(false);
  const [voiceAvailable, setVoiceAvailable] = useState(false);
  const [pointing, setPointing] = useState<{ step_id: string; point: [number, number] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const ws = useRef<WebSocket | null>(null);
  const player = useRef<Player | null>(null);
  const vad = useRef(new EnergyVad());
  const stopMic = useRef<(() => void) | null>(null);
  const audioHeader = useRef<{ sample_rate: number; utt: number } | null>(null);
  const silencedUtt = useRef(0); // utterance ids at or below this are dropped (barge-in)
  const busy = useRef(false);
  const getFrameRef = useRef(getFrame);
  getFrameRef.current = getFrame;

  const send = useCallback((msg: object, payload?: Blob | Uint8Array<ArrayBuffer>) => {
    const s = ws.current;
    if (!s || s.readyState !== WebSocket.OPEN) return false;
    s.send(JSON.stringify(msg));
    if (payload) s.send(payload);
    return true;
  }, []);

  useEffect(() => {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const sock = new WebSocket(`${proto}://${window.location.host}/ws/tutor`);
    sock.binaryType = "arraybuffer";
    ws.current = sock;
    // `track`: live tracking already knows where the student points (lib/tracking.ts).
    sock.onopen = () => sock.send(JSON.stringify({ type: "hello", voice: false, opening: false, track: trackId }));
    sock.onclose = () => setState("offline");
    sock.onmessage = (ev) => {
      if (typeof ev.data !== "string") {
        const h = audioHeader.current;
        if (h && h.utt > silencedUtt.current && player.current) {
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
        case "result":
          busy.current = false;
          setNote(null);
          setResult({ page: m.page, finding: m.finding });
          break;
        case "say":
          if (m.utt > silencedUtt.current) setCaptions((c) => [...c.slice(-6), { who: "lens", text: m.text }]);
          break;
        case "transcript":
          if (m.text) setCaptions((c) => [...c.slice(-6), { who: "you", text: m.text }]);
          break;
        case "audio":
          audioHeader.current = { sample_rate: m.sample_rate, utt: m.utt };
          break;
        case "pointing":
          setPointing({ step_id: m.step_id, point: m.point });
          setTimeout(() => setPointing(null), 4000);
          break;
        case "timing":
          setLatency(m.first_audio_s);
          break;
        case "error":
          busy.current = false;
          setNote(null);
          setError(m.message);
          break;
      }
    };
    return () => {
      sock.close();
      stopMic.current?.();
      player.current?.close();
    };
  }, []);

  const check = useCallback(
    async (image: Blob, reason: string) => {
      if (busy.current) return;
      busy.current = true;
      setError(null);
      setNote(TRIGGER_NOTE[reason] ?? TRIGGER_NOTE.manual);
      if (send({ type: "check", reason }, image)) return;
      try {
        setResult(await checkPage(image, result?.page.problem));
      } catch (e) {
        setError((e as Error).message);
      } finally {
        busy.current = false;
        setNote(null);
      }
    },
    [send, result],
  );

  const bargeIn = useCallback(() => {
    if (player.current?.playing) {
      player.current.stop();
      silencedUtt.current = audioHeader.current?.utt ?? silencedUtt.current;
      send({ type: "interrupt" });
    }
    vad.current.playback = false;
  }, [send]);

  /** Start the spoken conversation (needs a click: browsers only allow audio after one). */
  const startVoice = useCallback(async () => {
    try {
      player.current ??= new Player();
      await player.current.resume();
      player.current.onIdle = () => (vad.current.playback = false);
      stopMic.current = await startMic((frame) => {
        const ev = vad.current.push(frame);
        if (!ev) return;
        if (ev.type === "start") bargeIn();
        if (ev.type === "end") {
          const wav = encodeWav(ev.audio, MIC_RATE);
          (getFrameRef.current?.() ?? Promise.resolve(null)).then((frame) => {
            if (frame) send({ type: "frame" }, frame);
            send({ type: "utterance" }, wav);
          });
        }
      });
      setVoiceOn(true);
      setState("listening");
      send({ type: "voice", on: true });
    } catch (e) {
      setError(`Microphone: ${(e as Error).message}`);
    }
  }, [bargeIn, send]);

  const stopVoice = useCallback(() => {
    stopMic.current?.();
    stopMic.current = null;
    player.current?.stop();
    setVoiceOn(false);
    setState("idle");
    send({ type: "voice", on: false });
  }, [send]);

  return {
    state, result, captions, voiceOn, voiceAvailable, pointing, error, note, latency,
    busy: note !== null, check, startVoice, stopVoice,
  };
}
