import type { ReactNode } from "react";
import type { CheckResult } from "../contract";
import { PageView } from "./PageView";
import { StepsPanel } from "./StepsPanel";

interface Props {
  aspect: number;
  media: ReactNode;
  result: CheckResult | null;
  status: string;
  busy?: boolean;
  error?: string | null;
  onCheck?: () => void;
  onExit?: () => void;
  extra?: ReactNode;
}

/** D2: live page, outline on the mistake, tutor pointer, steps panel. */
export function LensScreen({ aspect, media, result, status, busy, error, onCheck, onExit, extra }: Props) {
  const page = result?.page ?? null;
  const finding = result?.finding ?? null;
  const allOk = page && page.steps.length > 0 && !finding;
  return (
    <div className="lens">
      <header className="lens-bar">
        <button className="brand" onClick={onExit} title="Back to start">
          Muna <b>Lens</b>
        </button>
        <span className={`status${busy ? " status-busy" : ""}`} role="status">
          {status}
        </span>
      </header>
      <main className="lens-main">
        <PageView aspect={aspect} page={page} finding={finding}>
          {media}
        </PageView>
        <aside className="panel">
          {page?.problem && (
            <section>
              <h2>Working on</h2>
              <p className="problem">{page.problem}</p>
            </section>
          )}
          {finding && (
            <section className="hint" aria-live="polite">
              <h2>Lens asks</h2>
              <p data-testid="hint">{finding.hint}</p>
            </section>
          )}
          {allOk && <p className="all-ok">Every line checks out so far.</p>}
          {error && <p className="error-note">{error}</p>}
          <section>
            <h2>Lines</h2>
            <StepsPanel page={page} finding={finding} />
          </section>
          {extra}
          {onCheck && (
            <button className="primary" onClick={onCheck} disabled={busy}>
              {busy ? "Reading…" : "Check my work"}
            </button>
          )}
        </aside>
      </main>
    </div>
  );
}
