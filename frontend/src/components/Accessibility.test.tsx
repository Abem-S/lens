// Accessibility: captions, type-to-talk, the settings panel, the summary, and their wiring in useTutor.
import { act, cleanup, fireEvent, render, renderHook, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DEMO_RESULT } from "../demo/fixture";
import { DEFAULT_A11Y, clampSpeed, loadA11y, saveA11y } from "../lib/a11y";
import { localSummary, summaryLines, summaryText, type SessionSummary } from "../lib/summary";
import { useTutor } from "../lib/useTutor";
import { AccessPanel } from "./AccessPanel";
import { LensScreen } from "./LensScreen";
import { SummaryDialog } from "./SummaryDialog";
import { CaptionLog, TutorBar } from "./TutorBar";

afterEach(cleanup);
beforeEach(() => window.localStorage.clear());

const SUMMARY: SessionSummary = {
  minutes: 12.4, problem: "Solve 2x + 5 = 11", steps_read: 4, steps_ok: 3, checks: 5,
  mistakes: [{ step: "2x = 11 + 5", spoken: "2 x equals 11 plus 5", kind: "sign error moving a term across =", status: "fixed" }],
};

describe("live captions", () => {
  it("shows what Lens said, sentence by sentence, and what it heard, newest last", () => {
    render(<CaptionLog captions={[
      { who: "you", text: "is this right?" },
      { who: "lens", text: "Good question,", utt: 3 },
      { who: "lens", text: "what did you do with the 5?", utt: 3 },
      { who: "you", text: "I added it", typed: true },
    ]} />);
    const log = screen.getByRole("log", { name: "Captions" });
    const items = log.querySelectorAll("li");
    expect(items).toHaveLength(4);
    expect(items[0].textContent).toBe("Youis this right?");
    expect(items[1].textContent).toBe("LensGood question,");
    expect(items[2].classList.contains("cap-cont")).toBe(true); // same reply: no second "Lens:"
    expect(items[3].textContent).toBe("You (typed)I added it");
  });

  it("can be hidden, and says where captions will appear when empty", () => {
    const base = { state: "idle" as const, voiceOn: false, voiceAvailable: true, latency: null,
      onStart: () => undefined, onStop: () => undefined };
    const { rerender } = render(<TutorBar {...base} captions={[]} />);
    expect(screen.getByText(/Captions appear here/)).toBeTruthy();
    rerender(<TutorBar {...base} captions={[{ who: "lens", text: "Hi" }]} showCaptions={false} speaksAloud={false} />);
    expect(screen.queryByRole("log")).toBeNull();
    expect(screen.getByText("Captions only")).toBeTruthy();
  });
});

describe("type to talk", () => {
  it("sends the typed text and clears the box", async () => {
    const onSend = vi.fn(() => true);
    render(<TutorBar state="idle" captions={[]} voiceOn={false} voiceAvailable={false} latency={null}
      onStart={() => undefined} onStop={() => undefined} onSend={onSend} />);
    const box = screen.getByLabelText("Type to Lens") as HTMLInputElement;
    const send = screen.getByRole("button", { name: "Send" }) as HTMLButtonElement;
    expect(send.disabled).toBe(true);
    fireEvent.change(box, { target: { value: "is my second step right?" } });
    await act(async () => fireEvent.submit(box.form!));
    expect(onSend).toHaveBeenCalledWith("is my second step right?");
    expect(box.value).toBe("");
  });

  it("is disabled while the tutor is offline", () => {
    render(<TutorBar state="offline" captions={[]} voiceOn={false} voiceAvailable={false} latency={null}
      onStart={() => undefined} onStop={() => undefined} onSend={() => true} />);
    expect((screen.getByLabelText("Type to Lens") as HTMLInputElement).disabled).toBe(true);
  });
});

describe("accessibility settings", () => {
  it("opens from the bar, changes settings by keyboard-reachable controls, closes on Escape", () => {
    const onChange = vi.fn();
    render(<AccessPanel settings={DEFAULT_A11Y} onChange={onChange} />);
    const button = screen.getByRole("button", { name: "Accessibility settings" });
    expect(button.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(button);
    expect(button.getAttribute("aria-expanded")).toBe("true");
    const captions = screen.getByRole("checkbox", { name: /Captions/ });
    expect(document.activeElement).toBe(captions); // focus moves into the panel
    fireEvent.click(captions);
    expect(onChange).toHaveBeenLastCalledWith({ captions: false });
    fireEvent.click(screen.getByRole("checkbox", { name: /Lens speaks aloud/ }));
    expect(onChange).toHaveBeenLastCalledWith({ voice: false });
    fireEvent.change(screen.getByRole("slider"), { target: { value: "1.2" } });
    expect(onChange).toHaveBeenLastCalledWith({ speed: 1.2 });
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("group", { name: "Accessibility settings" })).toBeNull();
    expect(document.activeElement).toBe(button);
  });

  it("disables the speed when Lens does not speak", () => {
    render(<AccessPanel settings={{ ...DEFAULT_A11Y, voice: false }} onChange={() => undefined} />);
    fireEvent.click(screen.getByRole("button", { name: "Accessibility settings" }));
    expect((screen.getByRole("slider") as HTMLInputElement).disabled).toBe(true);
  });

  it("remembers settings per browser and survives broken storage", () => {
    expect(loadA11y()).toEqual(DEFAULT_A11Y); // captions on by default
    saveA11y({ captions: false, voice: false, speed: 1.2 });
    expect(loadA11y()).toEqual({ captions: false, voice: false, speed: 1.2 });
    window.localStorage.setItem("lens.a11y.v1", "{not json");
    expect(loadA11y()).toEqual(DEFAULT_A11Y);
    const spy = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    expect(loadA11y()).toEqual(DEFAULT_A11Y);
    spy.mockRestore();
    const set = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    expect(() => saveA11y(DEFAULT_A11Y)).not.toThrow();
    set.mockRestore();
    expect(clampSpeed(5)).toBe(1.4);
    expect(clampSpeed(Number.NaN)).toBe(1);
  });
});

describe("read my page aloud", () => {
  it("offers the button once steps have been read", () => {
    const onRead = vi.fn();
    const { rerender } = render(<LensScreen aspect={1} media={null} result={null} status="" onReadAloud={onRead} />);
    expect(screen.queryByRole("button", { name: "Read my page aloud" })).toBeNull();
    rerender(<LensScreen aspect={1} media={null} result={DEMO_RESULT} status="" onReadAloud={onRead} />);
    fireEvent.click(screen.getByRole("button", { name: "Read my page aloud" }));
    expect(onRead).toHaveBeenCalled();
  });
});

describe("session summary", () => {
  it("says what happened in plain words, without answers", () => {
    const lines = summaryLines(SUMMARY);
    expect(lines).toEqual([
      "You worked for 12 minutes.",
      "Working on: Solve 2x + 5 = 11",
      "Lens read 4 steps; 3 steps checked out.",
      "Lens found 1 mistake:",
      '- In the step "2 x equals 11 plus 5" (sign error moving a term across =): you fixed it.',
    ]);
    const text = summaryText(SUMMARY, new Date(2026, 9, 31, 10, 0));
    expect(text).toMatch(/^Muna Lens session, /);
    expect(text).toContain("Nothing from this session is saved");
    expect(text).not.toMatch(/\bline\b/i);
  });

  it("handles a session with nothing read", () => {
    const empty = localSummary(null, 0, 20_000);
    expect(summaryLines(empty)).toEqual(["You worked for under a minute.", "Lens did not read any steps this time."]);
    const fromScreen = localSummary(DEMO_RESULT, 0, 120_000);
    expect(fromScreen.mistakes).toHaveLength(1);
    expect(fromScreen.mistakes[0].status).toBe("open");
  });

  it("is a focused dialog with a download, and Escape keeps working", () => {
    const keep = vi.fn();
    const done = vi.fn();
    const created: Blob[] = [];
    URL.createObjectURL = vi.fn((b: Blob) => {
      created.push(b);
      return "blob:x";
    }) as typeof URL.createObjectURL;
    URL.revokeObjectURL = vi.fn();
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
    render(<SummaryDialog summary={SUMMARY} onDone={done} onKeepGoing={keep} />);
    const dialog = screen.getByRole("dialog", { name: "Session summary" });
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    expect(document.activeElement?.id).toBe("summary-title");
    fireEvent.click(screen.getByRole("button", { name: "Download as text" }));
    expect(click).toHaveBeenCalled();
    expect(created[0].type).toBe("text/plain");
    fireEvent.click(screen.getByRole("button", { name: "Finish" }));
    expect(done).toHaveBeenCalled();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(keep).toHaveBeenCalled();
    click.mockRestore();
  });
});

// ---------- useTutor wiring, over a fake socket ----------

class FakeSocket {
  static last: FakeSocket | null = null;
  static OPEN = 1;
  readyState = 1;
  binaryType = "";
  sent: (string | Blob | Uint8Array)[] = [];
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  constructor() {
    FakeSocket.last = this;
  }
  send(d: string | Blob | Uint8Array) {
    this.sent.push(d);
  }
  close() {}
  json() {
    return this.sent.filter((d): d is string => typeof d === "string").map((d) => JSON.parse(d));
  }
  receive(m: object) {
    this.onmessage?.({ data: JSON.stringify(m) });
  }
}

describe("useTutor accessibility wiring", () => {
  beforeEach(() => {
    vi.stubGlobal("WebSocket", FakeSocket);
  });
  afterEach(() => vi.unstubAllGlobals());

  it("sends settings on connect, typed turns with the current view, and shows the summary", async () => {
    saveA11y({ captions: true, voice: false, speed: 1.2 }); // captions only: no audio needed
    const frame = new Blob(["jpeg"]);
    const { result } = renderHook(() => useTutor(async () => frame));
    const sock = FakeSocket.last!;
    act(() => sock.onopen!());
    expect(sock.json()).toContainEqual({ type: "settings", aloud: false, speed: 1.2 });

    await act(async () => {
      await result.current.sendText("  is this step right? ");
    });
    const i = sock.sent.indexOf(frame);
    expect(i).toBeGreaterThan(-1);
    expect(JSON.parse(sock.sent[i - 1] as string)).toEqual({ type: "frame" });
    expect(JSON.parse(sock.sent[i + 1] as string)).toEqual({ type: "text", text: "is this step right?" });
    expect(result.current.captions.at(-1)).toEqual({ who: "you", text: "is this step right?", typed: true });

    act(() => sock.receive({ type: "transcript", text: "is this step right?", typed: true }));
    expect(result.current.captions.filter((c) => c.who === "you")).toHaveLength(1); // echo not doubled
    act(() => sock.receive({ type: "say", text: "What did you do with the 5?", utt: 1 }));
    expect(result.current.captions.at(-1)).toEqual({ who: "lens", text: "What did you do with the 5?", utt: 1 });

    await act(async () => result.current.readPage());
    expect(sock.json().at(-1)).toEqual({ type: "read_page" });

    act(() => result.current.endSession());
    expect(sock.json().at(-1)).toEqual({ type: "end" });
    act(() => sock.receive({ type: "summary", ...SUMMARY }));
    expect(result.current.summary?.steps_read).toBe(4);
  });

  it("changing the speed tells the server", () => {
    const { result } = renderHook(() => useTutor());
    const sock = FakeSocket.last!;
    act(() => sock.onopen!());
    act(() => result.current.updateSettings({ speed: 0.8 }));
    expect(sock.json().at(-1)).toEqual({ type: "settings", aloud: false, speed: 0.8 });
    expect(loadA11y().speed).toBe(0.8);
  });
});
