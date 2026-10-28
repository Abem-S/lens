import { useEffect, useRef, useState } from "react";
import { socketUrl } from "../lib/api";
import { useCamera, useInterval, useWakeLock } from "../lib/camera";
import { ChangeDetector, type Trigger } from "../lib/changeDetect";
import { FrameSampler, grabJpeg } from "../lib/frames";
import { STATE_LABEL } from "../lib/useChecker";

/** Phone side: film the page, decide when it changed, stream frames to the laptop. */
export function PhonePage({ sessionId }: { sessionId: string }) {
  const { videoRef, error, ready } = useCamera("environment");
  const [linked, setLinked] = useState<"connecting" | "linked" | "lost">("connecting");
  const [label, setLabel] = useState(STATE_LABEL.waiting);
  const ws = useRef<WebSocket | null>(null);
  const sampler = useRef<FrameSampler | null>(null);
  const detector = useRef(new ChangeDetector());
  const sending = useRef(false);
  useWakeLock(true);

  const send = async (kind: "preview" | "check", reason?: Trigger | "manual") => {
    const v = videoRef.current, sock = ws.current;
    if (!v || !sock || sock.readyState !== WebSocket.OPEN) return;
    if (kind === "preview" && (sending.current || sock.bufferedAmount > 256_000)) return;
    sending.current = true;
    const blob = kind === "check" ? await grabJpeg(v) : await grabJpeg(v, 720, 0.6);
    sending.current = false;
    if (!blob) return;
    sock.send(JSON.stringify({ type: "frame", kind, reason }));
    sock.send(blob);
  };

  useEffect(() => {
    const sock = new WebSocket(socketUrl(sessionId, "phone"));
    ws.current = sock;
    sock.onopen = () => setLinked("linked");
    sock.onclose = () => setLinked("lost");
    sock.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      const msg = JSON.parse(ev.data);
      if (msg.type === "capture") {
        const v = videoRef.current;
        const f = v && (sampler.current ??= new FrameSampler()).sample(v);
        if (f) detector.current.markChecked(f);
        send("check", "manual");
      }
    };
    return () => sock.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  useInterval(
    () => {
      const v = videoRef.current;
      if (!v) return;
      const f = (sampler.current ??= new FrameSampler()).sample(v);
      if (!f) return;
      const r = detector.current.push(f);
      const next = STATE_LABEL[r.state];
      if (next !== label) {
        setLabel(next);
        ws.current?.readyState === WebSocket.OPEN && ws.current.send(JSON.stringify({ type: "state", label: next }));
      }
      if (r.trigger) send("check", r.trigger);
    },
    250,
    ready && linked === "linked",
  );
  useInterval(() => send("preview"), 500, ready && linked === "linked");

  return (
    <div className="phone">
      <video ref={videoRef} playsInline muted />
      <div className="phone-bar">
        <span className={`dot dot-${linked}`} />
        {error ??
          (linked === "linked" ? label : linked === "lost" ? "Lost the laptop. Scan the code again." : "Connecting…")}
      </div>
      <p className="phone-tip">Hold the phone over your page. Lens watches; your laptop talks.</p>
    </div>
  );
}
