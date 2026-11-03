import type { CheckResult } from "../contract";

/** The end-of-session summary (server: TutorSession.summary). Shown once, never stored. */
export interface SummaryMistake {
  step: string; // what the student wrote (LaTeX)
  spoken?: string; // the same in words
  kind: string; // what kind of mistake, no correction
  status: "fixed" | "changed" | "open";
}

export interface SessionSummary {
  minutes: number;
  problem: string | null;
  steps_read: number;
  steps_ok: number;
  checks: number;
  mistakes: SummaryMistake[];
}

export const STATUS_WORDS: Record<SummaryMistake["status"], string> = {
  fixed: "you fixed it",
  changed: "you crossed it out or rewrote it",
  open: "still open, worth another look",
};

/** When the tutor socket is down: what the screen itself knows. */
export function localSummary(result: CheckResult | null, startedAt: number, now: number): SessionSummary {
  const steps = result?.page.steps ?? [];
  const wrong = result?.finding ? steps.find((s) => s.id === result.finding!.step_id) : undefined;
  return {
    minutes: Math.round(Math.max(0, now - startedAt) / 6000) / 10,
    problem: result?.page.problem ?? null,
    steps_read: steps.length,
    steps_ok: steps.filter((s) => s.status === "ok").length,
    checks: result ? 1 : 0,
    mistakes: wrong ? [{ step: wrong.latex, kind: "a mistake Lens asked you about", status: "open" }] : [],
  };
}

function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}

/** The summary in plain sentences, for the screen and the downloaded text file. */
export function summaryLines(s: SessionSummary): string[] {
  const lines: string[] = [];
  const mins = s.minutes < 1 ? "under a minute" : plural(Math.round(s.minutes), "minute");
  lines.push(`You worked for ${mins}.`);
  if (s.problem) lines.push(`Working on: ${s.problem}`);
  if (s.steps_read === 0) lines.push("Lens did not read any steps this time.");
  else lines.push(`Lens read ${plural(s.steps_read, "step")}; ${plural(s.steps_ok, "step")} checked out.`);
  if (s.mistakes.length === 0) {
    if (s.steps_read > 0) lines.push("Lens found no mistakes in the steps it read.");
  } else {
    lines.push(`Lens found ${plural(s.mistakes.length, "mistake")}:`);
    for (const m of s.mistakes) {
      lines.push(`- In the step "${m.spoken || m.step}" (${m.kind}): ${STATUS_WORDS[m.status]}.`);
    }
  }
  return lines;
}

export function summaryText(s: SessionSummary, when: Date = new Date()): string {
  const date = when.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  return [
    `Muna Lens session, ${date}`,
    "",
    ...summaryLines(s),
    "",
    "Lens asks questions and never gives answers. Nothing from this session is saved anywhere but this file.",
    "",
  ].join("\n");
}
