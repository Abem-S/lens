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
}

/** The spoken conversation: talk button, what Lens is doing, and captions. */
export function TutorBar({ state, captions, voiceOn, voiceAvailable, latency, onStart, onStop }: Props) {
  return (
    <section className="tutor" aria-label="Conversation with Lens">
      <div className="tutor-head">
        <span className={`tutor-dot tutor-${state}`} aria-hidden="true" />
        <span className="tutor-state">{LABEL[state]}</span>
        {latency !== null && <span className="tutor-latency" title="Time from the end of your question to Lens's first word">{latency.toFixed(1)} s</span>}
      </div>
      {captions.length > 0 && (
        <ol className="captions" aria-live="polite">
          {captions.slice(-4).map((c, i) => (
            <li key={i} className={`cap cap-${c.who}`}>
              {c.text}
            </li>
          ))}
        </ol>
      )}
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
    </section>
  );
}
