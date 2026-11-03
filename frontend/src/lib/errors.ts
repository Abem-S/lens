// What the student sees when something fails: short, friendly, and what to do next.

export function cameraMessage(e: { name?: string; message?: string }): string {
  switch (e.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Lens can't see: camera permission was turned off. Allow the camera in your browser's address bar, then reload.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "Lens couldn't find a camera. Plug one in or try another device, then reload.";
    case "NotReadableError":
    case "AbortError":
      return "The camera is busy in another app. Close it there, then reload this page.";
    case "NotSupportedError":
      return "This browser can't open the camera here. Try Chrome, Edge, Safari or Firefox over https://.";
    default:
      return `The camera didn't start${e.message ? ` (${e.message})` : ""}. Reload the page to try again.`;
  }
}

export function micMessage(e: { name?: string; message?: string }): string {
  switch (e.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Lens can't hear you: microphone permission was turned off. Allow the mic in the address bar and press Talk again. Lens still checks your page without it.";
    case "NotFoundError":
      return "No microphone found. Lens still checks your page; plug in a mic to talk.";
    case "NotReadableError":
      return "The microphone is busy in another app. Close it there and press Talk again.";
    default:
      return "The microphone didn't start. Press Talk to try again; Lens still checks your page meanwhile.";
  }
}

/** WebSocket close codes the server uses on purpose. */
export const CLOSE_LIMIT = 4429; // a rate limit or the daily cap (a "limit" message comes first)
export const CLOSE_ACCESS = 4401; // the tester access code is missing or expired

export const ACCESS_EXPIRED = "Your tester access has expired. Open the tester link again to carry on.";
export const RECONNECTING = "Connection lost. Reconnecting…";
export const OFFLINE = "You're offline. Lens will reconnect when the internet is back.";

/** Wait before reconnect attempt `n` (0-based): 1 s, 2 s, 4 s … capped at 30 s, with jitter. */
export function backoffMs(n: number, rand = Math.random): number {
  const base = Math.min(30_000, 1000 * 2 ** Math.min(n, 5));
  return Math.round(base * (0.8 + 0.4 * rand()));
}
