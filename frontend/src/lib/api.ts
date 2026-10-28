import type { CheckResult } from "../contract";

export async function checkPage(image: Blob, problem?: string | null): Promise<CheckResult> {
  const form = new FormData();
  form.append("image", image, "page.jpg");
  if (problem) form.append("problem", problem);
  const r = await fetch("/api/check", { method: "POST", body: form });
  if (!r.ok) {
    const detail = await r.json().then((j) => j.detail).catch(() => r.statusText);
    throw new Error(detail || `check failed (${r.status})`);
  }
  return r.json();
}

export async function createSession(): Promise<{ id: string; phone_url: string }> {
  const r = await fetch("/api/sessions", {
    method: "POST",
    headers: { "x-lens-origin": window.location.origin },
  });
  if (!r.ok) throw new Error("could not start pairing");
  return r.json();
}

export function socketUrl(sessionId: string, role: "phone" | "screen"): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws/${sessionId}/${role}`;
}
