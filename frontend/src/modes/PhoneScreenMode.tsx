import QRCode from "qrcode";
import { useEffect, useRef, useState } from "react";
import { LensScreen } from "../components/LensScreen";
import { createSession, socketUrl } from "../lib/api";
import type { Trigger } from "../lib/changeDetect";
import { useChecker } from "../lib/useChecker";

type Header = { type: "frame"; kind: "preview" | "check"; reason?: Trigger | "manual" };

/** Laptop side of phone mode: show a QR, then the phone's view of the page. */
export function PhoneScreenMode({ onExit }: { onExit: () => void }) {
  const [qr, setQr] = useState<string | null>(null);
  const [phoneUrl, setPhoneUrl] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [pairedIn, setPairedIn] = useState<number | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [aspect, setAspect] = useState(4 / 3);
  const [phoneState, setPhoneState] = useState<string>("");
  const [setupError, setSetupError] = useState<string | null>(null);
  const ws = useRef<WebSocket | null>(null);
  const checker = useChecker();

  useEffect(() => {
    let closed = false;
    const started = performance.now();
    let header: Header | null = null;
    let lastUrl: string | null = null;
    createSession()
      .then(async ({ id, phone_url }) => {
        if (closed) return;
        setPhoneUrl(phone_url);
        setQr(await QRCode.toDataURL(phone_url, { margin: 1, width: 320, color: { dark: "#1d2a3a" } }));
        const sock = new WebSocket(socketUrl(id, "screen"));
        sock.binaryType = "blob";
        ws.current = sock;
        sock.onmessage = (ev) => {
          if (typeof ev.data === "string") {
            const msg = JSON.parse(ev.data);
            if (msg.type === "phone_connected") {
              setConnected(true);
              const secs = (performance.now() - started) / 1000;
              setPairedIn(secs);
              console.info(`[lens] phone paired in ${secs.toFixed(1)} s`);
            } else if (msg.type === "phone_disconnected") setConnected(false);
            else if (msg.type === "frame") header = msg;
            else if (msg.type === "state") setPhoneState(msg.label);
            return;
          }
          const blob = ev.data as Blob;
          if (header?.kind === "check") checker.run(blob, header.reason ?? "manual");
          const url = URL.createObjectURL(blob);
          if (lastUrl) URL.revokeObjectURL(lastUrl);
          lastUrl = url;
          setPreview(url);
          header = null;
        };
      })
      .catch((e) => setSetupError((e as Error).message));
    return () => {
      closed = true;
      ws.current?.close();
      if (lastUrl) URL.revokeObjectURL(lastUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!connected) {
    return (
      <div className="pair">
        <button className="brand" onClick={onExit}>
          Muna <b>Lens</b>
        </button>
        <h1>Scan with your phone</h1>
        <p className="muted">Open the camera app and point it at this code. Keep both on the same Wi-Fi.</p>
        {qr ? <img className="qr" src={qr} alt="Pairing QR code" /> : <div className="qr qr-empty" />}
        {phoneUrl && <code className="pair-url">{phoneUrl}</code>}
        {setupError && <p className="error-note">{setupError}</p>}
      </div>
    );
  }

  return (
    <LensScreen
      aspect={aspect}
      media={
        preview ? (
          <img
            src={preview}
            alt="Your page, seen by the phone"
            onLoad={(e) => setAspect(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight)}
          />
        ) : (
          <div className="media-empty">Waiting for the phone camera…</div>
        )
      }
      result={checker.result}
      status={checker.note ?? (phoneState || `Phone connected${pairedIn ? ` in ${pairedIn.toFixed(1)} s` : ""}`)}
      busy={checker.busy}
      error={checker.error}
      onCheck={() => ws.current?.send(JSON.stringify({ type: "capture" }))}
      onExit={onExit}
    />
  );
}
