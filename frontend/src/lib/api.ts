import type { CheckResult } from "../contract";
import { ACCESS_EXPIRED } from "./errors";

/** The server's own message for a failed request (it is written for students), else a plain one. */
async function failure(r: Response, fallback: string): Promise<Error> {
  if (r.status === 401) return new Error(ACCESS_EXPIRED);
  const detail = await r.json().then((j) => j.detail).catch(() => null);
  return new Error(typeof detail === "string" && detail ? detail : fallback);
}

export async function checkPage(image: Blob, problem?: string | null): Promise<CheckResult> {
  const form = new FormData();
  form.append("image", image, "page.jpg");
  if (problem) form.append("problem", problem);
  let r: Response;
  try {
    r = await fetch("/api/check", { method: "POST", body: form });
  } catch {
    throw new Error("Lens couldn't reach the server to read your page. Check your internet; it will try again.");
  }
  if (!r.ok) throw await failure(r, "Lens couldn't read your page just now. It will try again when you write your next step.");
  return r.json();
}

export async function createSession(): Promise<{ id: string; phone_url: string }> {
  let r: Response;
  try {
    r = await fetch("/api/sessions", { method: "POST", headers: { "x-lens-origin": window.location.origin } });
  } catch {
    throw new Error("Lens couldn't reach the server to pair your phone. Check your internet and try again.");
  }
  if (!r.ok) throw await failure(r, "Pairing didn't start. Go back and try again.");
  return r.json();
}

/** The start page was touched: wake the GPU services now, so the session starts warm. */
export function warmUp(): void {
  fetch("/api/warm", { method: "POST" }).catch(() => undefined);
}

export async function fetchConfig(): Promise<{ idle_min: number }> {
  try {
    const r = await fetch("/api/config");
    if (r.ok) return r.json();
  } catch {
    /* defaults below */
  }
  return { idle_min: 5 };
}

export function socketUrl(sessionId: string, role: "phone" | "screen"): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws/${sessionId}/${role}`;
}
