import type { ReactNode } from "react";
import type { Finding, Page, Step } from "../contract";

interface Props {
  aspect: number; // width / height of the frame the boxes refer to
  page: Page | null;
  finding: Finding | null;
  children: ReactNode; // the <video> or <img> of the page
  pointingStep?: string | null; // the step the student is pointing at
  /** How sure the outline's place on the live page is: "held" fades it, "lost" hides it
   * (with the tutor pointer) rather than leave it on the wrong place. */
  outline?: "ok" | "held" | "lost";
}

function outlinePoints(finding: Finding, step: Step | undefined): string {
  if (finding.mask && finding.mask.length >= 3) return finding.mask.map(([x, y]) => `${x},${y}`).join(" ");
  if (!step) return "";
  const [x, y, w, h] = step.bbox;
  return `${x},${y} ${x + w},${y} ${x + w},${y + h} ${x},${y + h}`;
}

/** The live page with the tutor's marks drawn on top, in frame coordinates. Only the mistake
 * is marked on the page itself; the checked steps are listed beside it, not ticked over the writing. */
export function PageView({ aspect, page, finding, children, pointingStep, outline }: Props) {
  const shown = outline === "lost" ? null : finding;
  const errorStep = shown ? page?.steps.find((s) => s.id === shown.step_id) : undefined;
  const faded = outline === "held" ? " faded" : "";
  const pointed = pointingStep ? page?.steps.find((s) => s.id === pointingStep) : undefined;
  return (
    <div className="page-view" style={{ aspectRatio: String(aspect) }}>
      <div className="page-media">{children}</div>
      <svg className="page-overlay" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
        {pointed && (
          <rect className="pointed" data-testid="pointed-step" x={pointed.bbox[0] - 0.01} y={pointed.bbox[1] - 0.01}
            width={pointed.bbox[2] + 0.02} height={pointed.bbox[3] + 0.02} rx="0.01" />
        )}
        {shown && (
          <polygon className={`outline${faded}`} data-testid="finding-outline" points={outlinePoints(shown, errorStep)} />
        )}
      </svg>
      {errorStep && (
        <div
          className={`tutor-pointer${faded}`}
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
