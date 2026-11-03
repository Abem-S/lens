import { cameraMessage } from "./errors";
import { useEffect, useRef, useState } from "react";

export type Facing = "user" | "environment";

/** Attach a camera stream to a <video>. */
export function useCamera(facing: Facing, enabled = true) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    let stream: MediaStream | null = null;
    let cancelled = false;
    if (!navigator.mediaDevices?.getUserMedia) {
      setError(
        window.isSecureContext
          ? "This browser cannot open the camera."
          : "The camera needs HTTPS. Open this page over https:// (see README).",
      );
      return;
    }
    navigator.mediaDevices
      .getUserMedia({
        video: { facingMode: facing, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      })
      .then((s) => {
        if (cancelled) return s.getTracks().forEach((t) => t.stop());
        stream = s;
        setError(null);
        for (const t of s.getVideoTracks())
          t.onended = () => setError("The camera stopped (unplugged or taken by another app). Reload the page to start it again.");
        const v = videoRef.current;
        if (v) {
          v.srcObject = s;
          v.onloadedmetadata = () => {
            v.play().catch(() => undefined);
            setReady(true);
          };
        }
      })
      .catch((e: Error) => setError(cameraMessage(e)));
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
      setReady(false);
    };
  }, [facing, enabled]);

  return { videoRef, error, ready };
}

/** Keep the phone screen on while it films the page. */
export function useWakeLock(active: boolean) {
  useEffect(() => {
    if (!active || !("wakeLock" in navigator)) return;
    let lock: WakeLockSentinel | null = null;
    const acquire = () =>
      navigator.wakeLock.request("screen").then((l) => (lock = l)).catch(() => undefined);
    acquire();
    const onVisible = () => document.visibilityState === "visible" && acquire();
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      lock?.release().catch(() => undefined);
    };
  }, [active]);
}

/** Run `fn` every `ms` while `active`. */
export function useInterval(fn: () => void, ms: number, active: boolean) {
  const saved = useRef(fn);
  saved.current = fn;
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => saved.current(), ms);
    return () => clearInterval(id);
  }, [ms, active]);
}
