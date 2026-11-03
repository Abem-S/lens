import QRCode from "qrcode";
import { useEffect, useRef, useState } from "react";
import { LensScreen } from "../components/LensScreen";
import { TutorBar } from "../components/TutorBar";
import { SessionActions } from "../components/SessionActions";
import { createSession, socketUrl } from "../lib/api";
import type { Trigger } from "../lib/changeDetect";
import { backoffMs } from "../lib/errors";
import { RESTING_NOTE, useIdleMinutes, usePresence } from "../lib/presence";
import { useTutor } from "../lib/useTutor";
import { type Box, moveResult, trackingNote, type TrackMessage } from "../lib/tracking";
import { grabJpeg } from "../lib/frames";
import { startReceiver, type RtcMessage } from "../lib/rtc";

type Header = {
  type: "frame"; kind: "preview" | "check"; reason?: Trigger | "manual" | "line_done"; seq?: number; box?: Box;
};

const PHONE_FRESH_MS = 75_000; // the phone's "someone is here" heartbeat (every 30 s) is trusted this long

/** Laptop side of phone mode: show a QR, then the phone's view of the page. */
export function PhoneScreenMode({ onExit }: { onExit: () => void }) {
  const [qr, setQr] = useState<string | null>(null);
  const [phoneUrl, setPhoneUrl] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [everConnected, setEverConnected] = useState(false);
  const [pairedIn, setPairedIn] = useState<number | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [aspect, setAspect] = useState(4 / 3);
  const [phoneState, setPhoneState] = useState<string>("");
  const [setupError, setSetupError] = useState<string | null>(null);
  const [linkNote, setLinkNote] = useState<string | null>(null);
  const [pairing, setPairing] = useState(0); // bump to start a fresh pairing (new QR)
  const ws = useRef<WebSocket | null>(null);
  const lastFrame = useRef<Blob | null>(null);
  const [trackId] = useState(() => `phone-${Math.random().toString(36).slice(2, 10)}`);
  // Live video from the phone (WebRTC); the JPEG preview is the fallback.
  const [remote, setRemote] = useState<MediaStream | null>(null);
  const liveVideo = useRef<HTMLVideoElement | null>(null);
  // Someone is here if this laptop is in use or the phone says its student is at the page.
  const laptop = usePresence(useIdleMinutes());
  const [phoneSeen, setPhoneSeen] = useState<{ active: boolean; at: number } | null>(null);
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    const t = setInterval(() => setNow(performance.now()), 5000);
    return () => clearInterval(t);
  }, []);
  const phoneActive = !!phoneSeen?.active && now - phoneSeen.at < PHONE_FRESH_MS;
  const active = laptop.active || phoneActive;
  const tutor = useTutor(
    async (side = 1024) => (liveVideo.current?.videoWidth
      ? grabJpeg(liveVideo.current, side, side < 1024 ? 0.7 : 0.8) : lastFrame.current),
    trackId,
    active,
    laptop.interact,
  );
  // The phone tracks the page live (SAM 3.1) and forwards its events here.
  const [track, setTrack] = useState<{ msg: TrackMessage; at: number } | null>(null);
  const [phoneTracking, setPhoneTracking] = useState<{ status: string; note: string | null } | null>(null);
  const trackLive = !!track && performance.now() - track.at < 3000;
  // The outline moves with the page: the server follows the step from the very frame the
  // check read, and the phone forwards where it is now with each tracked frame.
  const shown = moveResult(tutor.result, track?.msg.follow?.finding ?? null, trackLive);

  useEffect(() => {
    let closed = false;
    const started = performance.now();
    let header: Header | null = null;
    let lastUrl: string | null = null;
    let stopRtc: (() => void) | null = null;
    let retry: ReturnType<typeof setTimeout> | null = null;
    let failures = 0;
    createSession()
      .then(async ({ id, phone_url }) => {
        if (closed) return;
        setSetupError(null);
        setPhoneUrl(phone_url);
        setQr(await QRCode.toDataURL(phone_url, { margin: 1, width: 320, color: { dark: "#1d2a3a" } }));
        // The pairing socket; it reconnects by itself if the network drops.
        const connect = () => {
          const sock = new WebSocket(socketUrl(id, "screen"));
          sock.binaryType = "blob";
          ws.current = sock;
          const send = (m: object) => sock.readyState === WebSocket.OPEN && sock.send(JSON.stringify(m));
          const receiver = startReceiver(send, (stream) => {
            setRemote(stream);
            send({ type: "rtc_live", on: !!stream }); // the phone stops the JPEG preview while video flows
          });
          stopRtc = receiver.stop;
          sock.onopen = () => {
            failures = 0;
            setLinkNote(null);
          };
          sock.onclose = (ev) => {
            receiver.stop();
            if (closed) return;
            setConnected(false);
            setRemote(null);
            if (ev.code === 4404) {
              // The server restarted and forgot this pairing: start a new one.
              setLinkNote("Lens restarted. Scan the new code with your phone.");
              setEverConnected(false);
              setPairing((p) => p + 1);
              return;
            }
            setLinkNote("Lost the connection to Lens. Reconnecting…");
            retry = setTimeout(connect, backoffMs(failures++));
          };
          sock.onmessage = (ev) => {
            if (typeof ev.data === "string") {
              const msg = JSON.parse(ev.data);
              if (msg.type === "rtc") {
                receiver.handle(msg as RtcMessage);
                return;
              }
              if (msg.type === "phone_connected") {
                sock.send(JSON.stringify({ type: "track_id", id: trackId }));
                send({ type: "rtc_request" });
                setConnected(true);
                setEverConnected(true);
                setLinkNote(null);
                setPhoneSeen({ active: true, at: performance.now() });
                const secs = (performance.now() - started) / 1000;
                setPairedIn((p) => p ?? secs);
                console.info(`[lens] phone paired in ${secs.toFixed(1)} s`);
              } else if (msg.type === "phone_disconnected") {
                setConnected(false);
                setRemote(null);
              } else if (msg.type === "frame") header = msg;
              else if (msg.type === "state") setPhoneState(msg.label);
              else if (msg.type === "activity") tutor.activity(msg.phase);
              else if (msg.type === "presence") setPhoneSeen({ active: !!msg.active, at: performance.now() });
              else if (msg.type === "tracking") setPhoneTracking({ status: msg.status, note: msg.note ?? null });
              else if (msg.type === "track") setTrack({ msg, at: performance.now() });
              return;
            }
            const blob = ev.data as Blob;
            lastFrame.current = blob;
            if (header?.kind === "check") tutor.check(blob, header.reason ?? "manual", header.box);
            const url = URL.createObjectURL(blob);
            if (lastUrl) URL.revokeObjectURL(lastUrl);
            lastUrl = url;
            setPreview(url);
            header = null;
          };
        };
        connect();
      })
      .catch((e) => setSetupError((e as Error).message));
    return () => {
      closed = true;
      if (retry) clearTimeout(retry);
      stopRtc?.();
      ws.current?.close();
      if (lastUrl) URL.revokeObjectURL(lastUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pairing]);

  const pairPanel = (
    <>
      {qr ? <img className="qr" src={qr} alt="Pairing QR code" /> : <div className="qr qr-empty" />}
      {phoneUrl && <code className="pair-url">{phoneUrl}</code>}
    </>
  );

  if (!connected && !everConnected) {
    return (
      <div className="pair">
        <button className="brand" onClick={onExit}>
          Muna <b>Lens</b>
        </button>
        <h1>Scan with your phone</h1>
        <p className="muted">Open the camera app and point it at this code. On the same Wi-Fi the video is smoothest.</p>
        {linkNote && <p className="notice">{linkNote}</p>}
        {pairPanel}
        {setupError && (
          <p className="error-note">
            {setupError}{" "}
            <button className="link" onClick={() => setPairing((p) => p + 1)}>
              Try again
            </button>
          </p>
        )}
      </div>
    );
  }

  const phoneGone = !connected
    ? "The phone disconnected. Open the link on the phone again (it reconnects by itself), or scan the code below."
    : null;
  const tracking = phoneTracking && active ? trackingNote(phoneTracking.status as never, phoneTracking.note) : null;
  const waking = tutor.voiceOn && !tutor.voiceReady ? "Waking up Lens's voice…" : null;
  return (
    <LensScreen
      aspect={aspect}
      media={
        remote ? (
          <video
            ref={(el) => {
              liveVideo.current = el;
              if (el && el.srcObject !== remote) el.srcObject = remote;
            }}
            autoPlay
            playsInline
            muted
            onLoadedMetadata={(e) => setAspect(e.currentTarget.videoWidth / e.currentTarget.videoHeight)}
          />
        ) : preview ? (
          <img
            src={preview}
            alt="Your page, seen by the phone"
            onLoad={(e) => setAspect(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight)}
          />
        ) : (
          <div className="media-empty">Waiting for the phone camera…</div>
        )
      }
      result={shown}
      status={tutor.note ?? waking ?? (phoneState || `Phone connected${pairedIn ? ` in ${pairedIn.toFixed(1)} s` : ""}`)}
      busy={tutor.busy}
      error={tutor.error}
      notice={linkNote ?? tutor.notice ?? phoneGone ?? tracking}
      resting={active || !connected ? null : RESTING_NOTE[laptop.state === "hidden" ? "hidden" : "idle"]}
      onWake={laptop.interact}
      onExit={onExit}
      pointingStep={tutor.pointing?.step_id}
      extra={
        <>
          {!connected && <section className="pair-again">{pairPanel}</section>}
          <TutorBar state={tutor.state} captions={tutor.captions} voiceOn={tutor.voiceOn}
            voiceAvailable={tutor.voiceAvailable} latency={tutor.latency}
            onStart={tutor.startVoice} onStop={tutor.stopVoice}
            showCaptions={tutor.settings.captions} speaksAloud={tutor.settings.voice} onSend={tutor.sendText} />
        </>
      }
      actions={<SessionActions tutor={tutor} onExit={onExit} />}
      onReadAloud={tutor.readPage}
    />
  );
}
