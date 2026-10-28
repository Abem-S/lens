import { useCallback, useRef, useState } from "react";
import type { CheckResult } from "../contract";
import { checkPage } from "./api";
import type { DetectorState, Trigger } from "./changeDetect";

export const STATE_LABEL: Record<DetectorState, string> = {
  waiting: "Starting camera…",
  moving: "Watching…",
  hand: "Waiting for your hand to move away",
  settling: "Holding still…",
  steady: "Watching the page",
};

const TRIGGER_LABEL: Record<Trigger | "manual", string> = {
  first_view: "Reading your page…",
  new_ink: "New line spotted. Reading…",
  ink_removed: "Something changed. Reading…",
  page_changed: "New page. Reading…",
  manual: "Reading your page…",
};

/** One check at a time; remembers the problem once it is known. */
export function useChecker() {
  const [result, setResult] = useState<CheckResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const inFlight = useRef(false);
  const problem = useRef<string | null>(null);

  const run = useCallback(async (image: Blob, why: Trigger | "manual") => {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setNote(TRIGGER_LABEL[why]);
    setError(null);
    try {
      const r = await checkPage(image, problem.current);
      if (r.page.problem) problem.current = r.page.problem;
      setResult(r);
      setNote(null);
    } catch (e) {
      setError((e as Error).message);
      setNote(null);
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }, []);

  return { result, busy, error, note, run };
}
