import type { ReactNode } from "react";
import type { CheckResult } from "../contract";
import { PageView } from "./PageView";

interface Props {
  aspect: number;
  media: ReactNode;
  result: CheckResult | null;
  status: string;
  busy?: boolean;
  error?: string | null;
  onExit?: () => void;
  extra?: ReactNode;
  pointingStep?: string | null;
  actions?: ReactNode; // top bar buttons (accessibility, end session)
  onReadAloud?: () => void; // "read my page aloud"
  /** A banner about the connection or a limit (reconnecting, access expired, today's cap…). */
  notice?: string | null;
  /** Set while Lens rests (idle or hidden tab): shown over the page, with a button to carry on. */
  resting?: string | null;
  onWake?: () => void;
}

/** D2: live page, outline on the mistake, tutor pointer. No list of the steps read: the student has
 * their paper, and a list of raw readings only showed misreads (owner's test, 2026-11-03). */
export function LensScreen({
  aspect, media, result, status, busy, error, onExit, extra, pointingStep, actions, onReadAloud,
  notice, resting, onWake,
}: Props) {
  const page = result?.page ?? null;
  const finding = result?.finding ?? null;
  return (
    <div className="lens">
      <header className="lens-bar">
        <button className="brand" onClick={onExit} title="Back to start">
          Muna <b>Lens</b>
        </button>
        <span className={`status${busy ? " status-busy" : ""}`} role="status">
          {status}
        </span>
        {actions && <div className="bar-actions">{actions}</div>}
      </header>
      {notice && (
        <p className="notice" role="alert">
          {notice}
        </p>
      )}
      <main className="lens-main">
        <div className="page-wrap">
          <PageView aspect={aspect} page={page} finding={finding} pointingStep={pointingStep} outline={result?.outline}>
            {media}
          </PageView>
          {resting && (
            <div className="resting" role="status">
              <p>{resting}</p>
              {onWake && (
                <button className="primary" onClick={onWake}>
                  I'm here
                </button>
              )}
            </div>
          )}
        </div>
        <aside className="panel">
          {extra}
          {finding && (
            <section className="hint" aria-live="polite">
              <h2>Lens asks</h2>
              <p data-testid="hint">{finding.hint}</p>
            </section>
          )}
          {error && <p className="error-note">{error}</p>}
          {onReadAloud && page && page.steps.length > 0 && (
            <button className="link-btn" onClick={onReadAloud}>Read my page aloud</button>
          )}
        </aside>
      </main>
    </div>
  );
}
