// F: what the student sees when the tutor connection fails, and presence heartbeats.
import { act, cleanup, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ACCESS_EXPIRED, RECONNECTING } from "./errors";
import { useTutor } from "./useTutor";

class FakeSocket {
  static all: FakeSocket[] = [];
  static OPEN = 1;
  static CLOSED = 3;
  readyState = 0;
  sent: unknown[] = [];
  binaryType = "";
  onopen: (() => void) | null = null;
  onclose: ((ev: { code: number }) => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  constructor(public url: string) {
    FakeSocket.all.push(this);
  }
  send(m: unknown) {
    this.sent.push(typeof m === "string" ? JSON.parse(m) : m);
  }
  close() {
    this.readyState = 3;
  }
  // test helpers
  open() {
    this.readyState = 1;
    this.onopen?.();
  }
  message(m: object) {
    this.onmessage?.({ data: JSON.stringify(m) });
  }
  drop(code = 1006) {
    this.readyState = 3;
    this.onclose?.({ code });
  }
}

let seen: ReturnType<typeof useTutor> | null = null;
function Probe({ active = true }: { active?: boolean }) {
  seen = useTutor(undefined, "t1", active);
  return null;
}

beforeEach(() => {
  FakeSocket.all = [];
  vi.useFakeTimers();
  vi.stubGlobal("WebSocket", FakeSocket);
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("useTutor connection", () => {
  it("says hello with presence, and sends presence when the student goes idle", () => {
    const { rerender } = render(<Probe />);
    const s = FakeSocket.all[0];
    act(() => s.open());
    expect(s.sent[0]).toMatchObject({ type: "hello", track: "t1", active: true });
    rerender(<Probe active={false} />);
    expect(s.sent.at(-1)).toEqual({ type: "presence", active: false });
  });

  it("shows 'Reconnecting…' on a dropped connection and reconnects by itself", () => {
    render(<Probe />);
    act(() => FakeSocket.all[0].open());
    act(() => FakeSocket.all[0].drop());
    expect(seen!.notice).toBe(RECONNECTING);
    expect(seen!.state).toBe("offline");
    act(() => void vi.advanceTimersByTime(2000));
    expect(FakeSocket.all).toHaveLength(2);
    act(() => FakeSocket.all[1].open());
    expect(seen!.notice).toBeNull();
  });

  it("explains an expired access code and does not hammer the server", () => {
    render(<Probe />);
    act(() => FakeSocket.all[0].drop(4401));
    expect(seen!.notice).toBe(ACCESS_EXPIRED);
    act(() => void vi.advanceTimersByTime(120_000));
    expect(FakeSocket.all).toHaveLength(1);
  });

  it("shows the server's limit message; busy retries in a minute, today's cap does not", () => {
    render(<Probe />);
    act(() => FakeSocket.all[0].message({ type: "limit", reason: "busy", message: "Lens is busy." }));
    act(() => FakeSocket.all[0].drop(4429));
    expect(seen!.notice).toBe("Lens is busy.");
    act(() => void vi.advanceTimersByTime(61_000));
    expect(FakeSocket.all).toHaveLength(2);
    act(() => FakeSocket.all[1].message({ type: "limit", reason: "daily", message: "Resting until tomorrow." }));
    act(() => FakeSocket.all[1].drop(4429));
    act(() => void vi.advanceTimersByTime(600_000));
    expect(FakeSocket.all).toHaveLength(2);
    expect(seen!.notice).toBe("Resting until tomorrow.");
  });

  it("a slow check shows the server's note instead of a silent spinner", () => {
    render(<Probe />);
    act(() => FakeSocket.all[0].open());
    act(() => FakeSocket.all[0].message({ type: "note", text: "This is taking longer than usual…" }));
    expect(seen!.note).toBe("This is taking longer than usual…");
    expect(seen!.busy).toBe(true);
  });
});
