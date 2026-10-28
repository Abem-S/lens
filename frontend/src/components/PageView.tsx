import type { ReactNode } from "react";
import type { Finding, Page, Step } from "../contract";

interface Props {
  aspect: number; // width / height of the frame the boxes refer to
  page: Page | null;
  finding: Finding | null;
  children: ReactNode; // the <video> or <img> of the page
}

function outlinePoints(finding: Finding, step: Step | undefined): string {
  if (finding.mask && finding.mask.length >= 3) return finding.mask.map(([x, y]) => `${x},${y}`).join(" ");
  if (!step) return "";
  const [x, y, w, h] = step.bbox;
  return `${x},${y} ${x + w},${y} ${x + w},${y + h} ${x},${y + h}`;
}

/** The live page with the tutor's marks drawn on top, in frame coordinates. */
export function PageView({ aspect, page, finding, children }: Props) {
  const errorStep = finding ? page?.steps.find((s) => s.id === finding.step_id) : undefined;
  return (
    <div className="page-view" style={{ aspectRatio: String(aspect) }}>
      <div className="page-media">{children}</div>
      <svg className="page-overlay" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
        {finding && (
          <polygon className="outline" data-testid="finding-outline" points={outlinePoints(finding, errorStep)} />
        )}
      </svg>
      {page?.steps.map((s) => {
        const [x, y, w, h] = s.bbox;
        if (w <= 0 || h <= 0 || s.status === "unchecked") return null;
        return (
          <span
            key={s.id}
            className={`mark mark-${s.status}`}
            data-step={s.id}
            style={{ left: `${Math.min(0.97, x + w + 0.025) * 100}%`, top: `${(y + h / 2) * 100}%` }}
          >
            {s.status === "ok" ? "✓" : s.status === "error" ? "✗" : "·"}
          </span>
        );
      })}
      {errorStep && (
        <div
          className="tutor-pointer"
          data-testid="tutor-pointer"
          style={{
            left: `${Math.max(0, errorStep.bbox[0] - 0.045) * 100}%`,
            top: `${(errorStep.bbox[1] + errorStep.bbox[3] / 2) * 100}%`,
          }}
        >
          <svg viewBox="0 0 24 24" width="100%" height="100%">
            <path d="M3 12h13m0 0-5-5m5 5-5 5" fill="none" stroke="currentColor" strokeWidth="3"
              strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      )}
    </div>
  );
}
