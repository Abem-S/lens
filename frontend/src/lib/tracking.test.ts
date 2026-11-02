import { describe, expect, it } from "vitest";
import type { CheckResult, Step } from "../contract";
import { encodeFrame, mapBox, moveResult, pointedStep, stillTriggerAllowed, type Box } from "./tracking";

const step = (id: string, bbox: Box): Step => ({ id, latex: "x", bbox, status: "ok", verified_by: "sympy" });

describe("tracking helpers", () => {
  it("lets tracking own the line trigger, keeping first view and new page", () => {
    expect(stillTriggerAllowed(false, "new_ink")).toBe(true);
    expect(stillTriggerAllowed(true, "new_ink")).toBe(false);
    expect(stillTriggerAllowed(true, "ink_removed")).toBe(false);
    expect(stillTriggerAllowed(true, "first_view")).toBe(true);
    expect(stillTriggerAllowed(true, "page_changed")).toBe(true);
  });

  it("prefixes a frame with its number", () => {
    const f = encodeFrame(258, new Uint8Array([9, 8]));
    expect([...f]).toEqual([0, 0, 1, 2, 9, 8]);
  });

  it("moves boxes and the outline with the tracked line", () => {
    const from: Box = [0.2, 0.4, 0.4, 0.05];
    const to: Box = [0.25, 0.3, 0.44, 0.055]; // page moved up-right and closer (10 % zoom)
    const moved = mapBox([0.2, 0.2, 0.4, 0.05], from, to);
    expect(moved[0]).toBeCloseTo(0.25);
    expect(moved[1]).toBeCloseTo(0.3 - 0.2 * 1.1);
    expect(moved[2]).toBeCloseTo(0.44);

    const result: CheckResult = {
      page: { problem: null, steps: [step("s1", [0.2, 0.2, 0.4, 0.05]), step("s2", from)] },
      finding: { step_id: "s2", kind: "sign", hint: "?", mask: [[0.2, 0.4], [0.6, 0.4], [0.6, 0.45]] },
    };
    const shown = moveResult(result, from, to)!;
    expect(shown.page.steps[1].bbox[0]).toBeCloseTo(0.25);
    expect(shown.page.steps[1].bbox[1]).toBeCloseTo(0.3);
    expect(shown.finding!.mask![1][0]).toBeCloseTo(0.25 + 0.4 * 1.1);
    expect(moveResult(result, from, from)).toBe(result); // no movement: same object
    expect(moveResult(result, null, to)).toBe(result);
  });

  it("finds the line a fingertip points at, or none", () => {
    const steps = [step("a", [0.1, 0.1, 0.5, 0.05]), step("b", [0.1, 0.3, 0.5, 0.05])];
    expect(pointedStep([0.3, 0.32], steps)?.id).toBe("b");
    expect(pointedStep([0.3, 0.36], steps)?.id).toBe("b"); // just under the line
    expect(pointedStep([0.9, 0.9], steps)).toBeNull();
    expect(pointedStep(null, steps)).toBeNull();
  });
});
