// Mirror of the shared contract in AGENTS.md (backend: lens/contract.py).

export type StepStatus = "unchecked" | "ok" | "error" | "unverifiable";

export interface Step {
  id: string;
  latex: string;
  bbox: [number, number, number, number]; // x, y, w, h normalised 0–1
  status: StepStatus;
  verified_by: "sympy" | "model" | null;
}

export interface Page {
  problem: string | null;
  steps: Step[];
}

export type Polygon = [number, number][];

export interface Finding {
  step_id: string;
  kind: string;
  mask: Polygon | null;
  hint: string;
}

export interface CheckResult {
  page: Page;
  finding: Finding | null;
}
