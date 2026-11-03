import { useEffect, useRef } from "react";
import { summaryLines, summaryText, type SessionSummary } from "../lib/summary";

interface Props {
  summary: SessionSummary;
  onDone: () => void; // back to the start screen
  onKeepGoing: () => void;
}

/** End of session: what was done, in plain words, with a text file to keep. Nothing is stored. */
export function SummaryDialog({ summary, onDone, onKeepGoing }: Props) {
  const heading = useRef<HTMLHeadingElement | null>(null);
  useEffect(() => heading.current?.focus(), []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onKeepGoing();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onKeepGoing]);

  const download = () => {
    const url = URL.createObjectURL(new Blob([summaryText(summary)], { type: "text/plain" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `lens-session-${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return (
    <div className="scrim">
      <div className="summary" role="dialog" aria-modal="true" aria-labelledby="summary-title">
        <h1 id="summary-title" ref={heading} tabIndex={-1}>Session summary</h1>
        <ul className="summary-lines">
          {summaryLines(summary).map((l, i) => (
            <li key={i} className={l.startsWith("- ") ? "summary-mistake" : undefined}>
              {l.replace(/^- /, "")}
            </li>
          ))}
        </ul>
        <p className="muted summary-note">Nothing from this session is saved. Download it to keep a copy.</p>
        <div className="summary-actions">
          <button className="primary" onClick={download}>Download as text</button>
          <button className="talk" onClick={onDone}>Finish</button>
          <button className="link-btn" onClick={onKeepGoing}>Keep working</button>
        </div>
      </div>
    </div>
  );
}
