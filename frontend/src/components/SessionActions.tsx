import type { A11ySettings } from "../lib/a11y";
import type { SessionSummary } from "../lib/summary";
import { AccessPanel } from "./AccessPanel";
import { SummaryDialog } from "./SummaryDialog";

interface TutorLike {
  settings: A11ySettings;
  updateSettings: (patch: Partial<A11ySettings>) => void;
  voiceAvailable: boolean;
  summary: SessionSummary | null;
  endSession: () => void;
  closeSummary: () => void;
}

/** Top bar of a live session: accessibility settings, and "End session" with its summary. */
export function SessionActions({ tutor, onExit }: { tutor: TutorLike; onExit: () => void }) {
  return (
    <>
      <AccessPanel settings={tutor.settings} onChange={tutor.updateSettings} voiceAvailable={tutor.voiceAvailable} />
      <button className="bar-btn" onClick={tutor.endSession}>
        End session
      </button>
      {tutor.summary && <SummaryDialog summary={tutor.summary} onDone={onExit} onKeepGoing={tutor.closeSummary} />}
    </>
  );
}
