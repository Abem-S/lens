import { useEffect, useRef, useState, type FormEvent } from "react";
import type { Caption, TutorState } from "../lib/useTutor";

const LABEL: Record<TutorState, string> = {
  offline: "Tutor offline",
  connecting: "Connecting…",
  idle: "Ready",
  listening: "Listening",
  thinking: "Thinking…",
  speaking: "Speaking",
  reading: "Reading your page…",
};

interface Props {
  state: TutorState;
  captions: Caption[];
  voiceOn: boolean;
  voiceAvailable: boolean;
  latency: number | null;
  onStart: () => void;
  onStop: () => void;
  showCaptions?: boolean;
  speaksAloud?: boolean;
  onSend?: (text: string) => Promise<boolean> | boolean;
}

/** The conversation: talk button, what Lens is doing, live captions and a box to type instead of talking. */
export function TutorBar({
  state, captions, voiceOn, voiceAvailable, latency, onStart, onStop, showCaptions = true, speaksAloud = true, onSend,
}: Props) {
  return (
    <section className="tutor" aria-label="Conversation with Lens">
      <div className="tutor-head">
        <span className={`tutor-dot tutor-${state}`} aria-hidden="true" />
        <span className="tutor-state">{LABEL[state]}</span>
        {!speaksAloud && <span className="tutor-mode">Captions only</span>}
        {latency !== null && <span className="tutor-latency" title="Time from the end of your question to Lens's first word">{latency.toFixed(1)} s</span>}
      </div>
      {showCaptions && <CaptionLog captions={captions} />}
      {voiceAvailable &&
        (voiceOn ? (
          <button className="talk talk-on" onClick={onStop}>
            Stop talking
          </button>
        ) : (
          <button className="talk" onClick={onStart}>
            Talk with Lens
          </button>
        ))}
      {onSend && <TypeBox onSend={onSend} disabled={state === "offline" || state === "connecting"} />}
    </section>
  );
}

/** What Lens says (sentence by sentence, as it speaks) and what it heard, newest at the bottom. */
export function CaptionLog({ captions }: { captions: Caption[] }) {
  const end = useRef<HTMLLIElement | null>(null);
  useEffect(() => {
    end.current?.scrollIntoView?.({ block: "nearest" });
  }, [captions]);
  if (captions.length === 0) {
    return <p className="captions-empty muted">Captions appear here: what Lens says, and what it heard from you.</p>;
  }
  return (
    <ol className="captions" role="log" aria-label="Captions" tabIndex={0}>
      {captions.map((c, i) => {
        const prev = captions[i - 1];
        const cont = c.who === "lens" && prev?.who === "lens" && c.utt !== undefined && prev.utt === c.utt;
        return (
          <li key={i} ref={i === captions.length - 1 ? end : undefined}
            className={`cap cap-${c.who}${cont ? " cap-cont" : ""}`}>
            {!cont && <span className="cap-who">{c.who === "lens" ? "Lens" : c.typed ? "You (typed)" : "You"}</span>}
            {c.text}
          </li>
        );
      })}
    </ol>
  );
}

/** Type to Lens instead of speaking: same answers, same rules. */
export function TypeBox({ onSend, disabled }: { onSend: (text: string) => Promise<boolean> | boolean; disabled?: boolean }) {
  const [text, setText] = useState("");
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;
    const sent = await onSend(text);
    if (sent) setText("");
  };
  return (
    <form className="typebox" onSubmit={submit}>
      <label className="sr-only" htmlFor="lens-type">Type to Lens</label>
      <input id="lens-type" type="text" value={text} maxLength={500} autoComplete="off" enterKeyHint="send"
        placeholder="Or type to Lens…" onChange={(e) => setText(e.target.value)} disabled={disabled} />
      <button type="submit" disabled={disabled || !text.trim()}>Send</button>
    </form>
  );
}
