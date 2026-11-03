import { describe, expect, it } from "vitest";
import type { CheckResult, Page, Step } from "../contract";
import { withReading } from "./reading";

const step = (id: string, latex: string, y: number, status: Step["status"] = "ok"): Step => ({
  id, latex, bbox: [0.1, y, 0.5, 0.05], status, verified_by: status === "unchecked" ? null : "sympy",
});

const prev: CheckResult = {
  page: { problem: null, steps: [step("s1", "2x+5=11", 0.1), step("s2", "2x=16", 0.2, "error")] },
  finding: { step_id: "s2", kind: "k", mask: [[0.1, 0.2], [0.6, 0.2], [0.6, 0.25]], hint: "h" },
};

describe("withReading", () => {
  it("shows the new reading and keeps the outline while its step reads the same", () => {
    const read: Page = { problem: null, steps: [step("s1", "2x+5=11", 0.12, "unchecked"),
      step("s2", "2x=16", 0.22, "unchecked"), step("s3", "x=8", 0.32, "unchecked")] };
    const shown = withReading(prev, read);
    expect(shown.page.steps.map((s) => s.latex)).toEqual(["2x+5=11", "2x=16", "x=8"]);
    expect(shown.finding?.step_id).toBe("s2");
    expect(shown.page.steps[1]).toBe(prev.page.steps[1]); // the outlined step keeps its box and mark
    expect(shown.page.steps[2].status).toBe("unchecked");
  });

  it("drops the outline when its step changed or is gone (a new problem)", () => {
    const changed: Page = { problem: null, steps: [step("s1", "2x+5=11", 0.1, "unchecked"),
      step("s2", "2x=6", 0.2, "unchecked")] };
    expect(withReading(prev, changed).finding).toBeNull();
    const fresh: Page = { problem: "Solve 3x-1=8", steps: [step("s1", "3x-1=8", 0.1, "unchecked")] };
    expect(withReading(prev, fresh)).toEqual({ page: fresh, finding: null });
    expect(withReading(null, fresh)).toEqual({ page: fresh, finding: null });
  });
});
