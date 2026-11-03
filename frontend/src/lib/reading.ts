import type { CheckResult, Page } from "../contract";

/**
 * The steps as just read (server "steps" message), shown while they are verified.
 * The outline of the mistake on screen stays where it is if its step is still there and
 * reads the same (same id and writing); otherwise it goes until the check's result says.
 */
export function withReading(prev: CheckResult | null, page: Page): CheckResult {
  const f = prev?.finding ?? null;
  const old = f ? prev?.page.steps.find((s) => s.id === f.step_id) : undefined;
  const now = f ? page.steps.find((s) => s.id === f.step_id) : undefined;
  if (!f || !old || !now || now.latex !== old.latex) return { page, finding: null };
  return { page: { ...page, steps: page.steps.map((s) => (s.id === old.id ? old : s)) }, finding: f };
}
