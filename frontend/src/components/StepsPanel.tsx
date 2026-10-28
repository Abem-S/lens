import type { Finding, Page } from "../contract";
import { MathText } from "./MathText";

const ICON = { ok: "✓", error: "✗", unverifiable: "·", unchecked: "·" } as const;
const LABEL = {
  ok: "checked",
  error: "look here",
  unverifiable: "not checked",
  unchecked: "not checked yet",
} as const;

export function StepsPanel({ page, finding }: { page: Page | null; finding: Finding | null }) {
  if (!page || page.steps.length === 0) {
    return <p className="muted">No lines read yet. Write on the page, or press “Check my work”.</p>;
  }
  return (
    <ol className="steps" aria-label="Lines read from the page">
      {page.steps.map((s) => (
        <li key={s.id} className={`step step-${s.status}${finding?.step_id === s.id ? " step-focus" : ""}`}>
          <span className="step-icon" aria-label={LABEL[s.status]} title={LABEL[s.status]}>
            {ICON[s.status]}
          </span>
          <MathText latex={s.latex} />
        </li>
      ))}
    </ol>
  );
}
