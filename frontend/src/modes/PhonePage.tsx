import { useEffect, useRef, useState } from "react";
import { socketUrl } from "../lib/api";
import { useCamera, useInterval, useWakeLock } from "../lib/camera";
import { ActivityTracker, ChangeDetector, type Trigger } from "../lib/changeDetect";
import { backoffMs } from "../lib/errors";
import { FrameSampler, grabJpeg } from "../lib/frames";
import { STATE_LABEL } from "../lib/detectorLabels";
import { RESTING_NOTE, useIdleMinutes, usePresence } from "../lib/presence";
import { type Box, PHASE_LABEL, stillTriggerAllowed, trackingNote, useTracking } from "../lib/tracking";
import { startSender, type RtcMessage } from "../lib/rtc";

type Link = "connecting" | "linked" | "lost" | "gone";

/** Phone side: film the page, decide when it changed, stream frames to the laptop. */
export function PhonePage({ sessionId }: { sessionId: string }) {
  const { videoRef, error, ready } = useCamera("environment");
  const [linked, setLinked] = useState<Link>("connecting");
  const [label, setLabel] = useState(STATE_LABEL.waiting);
  const ws = useRef<WebSocket | null>(null);
  const sampler = useRef<FrameSampler | null>(null);
  const detector = useRef(new ChangeDetector());
  const activity = useRef(new ActivityTracker());
  const sending = useRef(false);
  const [trackId, setTrackId] = useState<string | null>(null); // sent by the laptop, shared with its tutor
  // Live video to the laptop (WebRTC). While it flows, the JPEG preview stops.
  const rtc = useRef<ReturnType<typeof startSender> | null>(null);
  const rtcLive = useRef(false);
  const rtcWanted = useRef(false); // the laptop asked for video; start once the camera is ready
  const startRtc = () => {
    const stream = videoRef.current?.srcObject as MediaStream | null;
    const sock = ws.current;
    if (!rtcWanted.current || !stream || !sock || sock.readyState !== WebSocket.OPEN) return;
    rtcWanted.current = false; // one offer per request: a second offer would cross the first answer
    rtc.current?.stop();
    rtc.current = startSender(stream, (m) => sock.readyState === WebSocket.OPEN && sock.send(JSON.stringify(m)));
  };
  // Tracking (GPU) runs only while someone is at the page: the phone screen is on and the
  // camera sees movement now and then (lib/presence.ts).
  const presence = usePresence(useIdleMinutes());
  useWakeLock(presence.active);
  const toLaptop = (m: object) => ws.current?.readyState === WebSocket.OPEN && ws.current.send(JSON.stringify(m));
  // Live SAM 3.1 tracking: decides when a line is done; its events go on to the laptop.
  const track = useTracking(() => videoRef.current, ready && linked === "linked" && !!trackId && presence.active, {
    id: trackId ?? undefined,
    onLineDone: (box) => send("check", "line_done", box),
    onTrack: (m) => {
      if (m.objects.length) presence.motion();
      toLaptop({ type: "track", seq: m.seq, phase: m.phase, follow: m.follow, pointer: m.pointer,
        pointer_kind: m.pointer_kind });
    },
  });
  const trackRef = useRef(track);
  trackRef.current = track;

  // The laptop's tutor keeps its services warm while this phone says someone is here.
  useEffect(() => {
    if (linked !== "linked") return;
    toLaptop({ type: "presence", active: presence.active });
    if (!presence.active) return;
    const t = setInterval(() => toLaptop({ type: "presence", active: true }), 30_000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presence.active, linked]);
  useEffect(() => {
    if (linked === "linked") toLaptop({ type: "tracking", status: track.status, note: track.note });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [track.status, track.note, linked]);

  const send = async (kind: "preview" | "check", reason?: Trigger | "manual" | "line_done", box?: Box) => {
    const v = videoRef.current, sock = ws.current;
    if (!v || !sock || sock.readyState !== WebSocket.OPEN) return;
    // Never queue previews: a backlog is what makes the laptop's picture lag behind.
    if (kind === "preview" && (rtcLive.current || sending.current || sock.bufferedAmount > 0)) return;
    if (kind === "check") {
      // Whatever started this check, the detector's baseline becomes this frame and
      // tracking starts a new line: neither checks the same ink again.
      const f = (sampler.current ??= new FrameSampler()).sample(v);
      if (f) detector.current.markChecked(f);
      trackRef.current.sawWriting.current = false;
      trackRef.current.checkedAt.current = performance.now();
    }
    sending.current = true;
    const blob = kind === "check" ? await grabJpeg(v) : await grabJpeg(v, 720, 0.6);
    sending.current = false;
    if (!blob) return;
    // `box`: the finished line on this frame (the laptop's tutor checks just that line).
    sock.send(JSON.stringify({ type: "frame", kind, reason, seq: trackRef.current?.seq.current, box }));
    sock.send(blob);
  };

  // The pairing socket; it reconnects by itself after a network drop or the screen sleeping.
  useEffect(() => {
    let closed = false;
    let failures = 0;
    let retry: ReturnType<typeof setTimeout> | null = null;
    const connect = () => {
      const sock = new WebSocket(socketUrl(sessionId, "phone"));
      ws.current = sock;
      sock.onopen = () => {
        failures = 0;
        setLinked("linked");
      };
      sock.onclose = (ev) => {
        rtc.current?.stop();
        rtc.current = null;
        rtcLive.current = false;
        if (closed) return;
        if (ev.code === 4404 || ev.code === 4401) return setLinked("gone"); // pairing forgotten or expired
        setLinked("lost");
        retry = setTimeout(connect, backoffMs(failures++));
      };
      sock.onmessage = (ev) => {
        if (typeof ev.data !== "string") return;
        const msg = JSON.parse(ev.data);
        if (msg.type === "track_id") setTrackId(msg.id);
        else if (msg.type === "rtc") rtc.current?.handle(msg as RtcMessage);
        else if (msg.type === "rtc_request") {
          rtcWanted.current = true;
          startRtc();
        } else if (msg.type === "rtc_live") rtcLive.current = !!msg.on;
        else if (msg.type === "capture") send("check", "manual");
      };
    };
    connect();
    const online = () => {
      if (ws.current?.readyState === WebSocket.CLOSED && !closed) {
        if (retry) clearTimeout(retry);
        connect();
      }
    };
    window.addEventListener("online", online);
    return () => {
      closed = true;
      if (retry) clearTimeout(retry);
      window.removeEventListener("online", online);
      rtc.current?.stop();
      ws.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // The laptop's request may arrive before the camera is ready: offer as soon as it is.
  useEffect(() => {
    if (ready && linked === "linked") startRtc();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, linked]);

  useInterval(
    () => {
      const v = videoRef.current;
      if (!v) return;
      const f = (sampler.current ??= new FrameSampler()).sample(v);
      if (!f) return;
      const r = detector.current.push(f);
      if (r.state === "moving" || r.state === "hand") presence.motion();
      const next = track.live && track.last ? PHASE_LABEL[track.last.phase] : STATE_LABEL[r.state];
      if (next !== label) {
        setLabel(next);
        toLaptop({ type: "state", label: next });
      }
      // Without live tracking, the laptop's tutor learns what the student is doing from the detector.
      const act = activity.current.report(r.state, performance.now());
      if (!track.live && act) toLaptop({ type: "activity", phase: act });
      if (presence.active && r.trigger &&
        stillTriggerAllowed(track.live, r.trigger, track.sawWriting.current, performance.now() - track.checkedAt.current))
        send("check", r.trigger);
    },
    250,
    ready && linked === "linked",
  );
  useInterval(() => send("preview"), 500, ready && linked === "linked" && presence.active);

  const bar =
    error ??
    (linked === "linked"
      ? label
      : linked === "lost"
        ? "Lost the laptop. Reconnecting…"
        : linked === "gone"
          ? "This pairing has ended. On your laptop, choose phone mode again and scan the new code."
          : "Connecting…");
  const tracking = linked === "linked" && presence.active ? trackingNote(track.status, track.note) : null;
  return (
    <div className="phone">
      <video ref={videoRef} playsInline muted />
      <div className="phone-bar" role="status">
        <span className={`dot dot-${linked === "gone" ? "lost" : linked}`} />
        {bar}
      </div>
      {!presence.active && linked === "linked" && (
        <div className="phone-rest" role="status">
          <p>{RESTING_NOTE[presence.state as "idle" | "hidden"]}</p>
          <button className="primary" onClick={presence.interact}>
            I'm here
          </button>
        </div>
      )}
      <p className="phone-tip">{tracking ?? "Hold the phone over your page. Lens watches; your laptop talks."}</p>
    </div>
  );
}
