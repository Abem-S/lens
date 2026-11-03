import { describe, expect, it } from "vitest";
import type { CheckResult, Step } from "../contract";
import {
  encodeFrame, frameSize, mapBox, moveResult, pointedStep, quadMap, STILL_AFTER_CHECK_MS, stillTriggerAllowed,
  type Box, type Followed,
} from "./tracking";

const step = (id: string, bbox: Box): Step => ({ id, latex: "x", bbox, status: "ok", verified_by: "sympy" });

describe("tracking helpers", () => {
  it("lets tracking own the line trigger, keeping first view and new page", () => {
    expect(stillTriggerAllowed(false, "new_ink")).toBe(true);
    expect(stillTriggerAllowed(true, "new_ink")).toBe(false);
    expect(stillTriggerAllowed(true, "ink_removed")).toBe(false);
    expect(stillTriggerAllowed(true, "first_view")).toBe(true);
    expect(stillTriggerAllowed(true, "page_changed")).toBe(true);
  });

  it("lets the detector check ink that tracking never saw being written", () => {
    // A notebook lowered to write and held up to the webcam again: no pen seen writing.
    expect(stillTriggerAllowed(true, "new_ink", false)).toBe(true);
    expect(stillTriggerAllowed(true, "ink_removed", false)).toBe(true);
    // Tracking saw the line written: its line_done owns the check.
    expect(stillTriggerAllowed(true, "new_ink", true)).toBe(false);
  });

  it("does not let ink uncovered right after a check start a second, whole-page check", () => {
    // Owner's phone test: the hand left the page after a line_done check; the ink it had
    // covered looked new, and a 40 s page check queued behind every line check.
    expect(stillTriggerAllowed(true, "new_ink", false, 4_000)).toBe(false);
    expect(stillTriggerAllowed(true, "ink_removed", false, 4_000)).toBe(false);
    expect(stillTriggerAllowed(true, "new_ink", false, STILL_AFTER_CHECK_MS)).toBe(true);
    // A new page, and the detector alone (tracking off), are never held back.
    expect(stillTriggerAllowed(true, "page_changed", false, 1_000)).toBe(true);
    expect(stillTriggerAllowed(false, "new_ink", false, 1_000)).toBe(true);
  });

  it("sends wide webcam frames with enough rows for handwriting", () => {
    expect(frameSize(1080, 1920)).toEqual([512, 910]); // phone, portrait
    expect(frameSize(1280, 720)).toEqual([683, 384]); // webcam 16:9
    expect(frameSize(640, 480)).toEqual([512, 384]);
    expect(frameSize(400, 300)).toEqual([400, 300]); // never larger than the video
  });

  it("prefixes a frame with its number", () => {
    const f = encodeFrame(258, new Uint8Array([9, 8]));
    expect([...f]).toEqual([0, 0, 1, 2, 9, 8]);
  });

  it("moves boxes and the outline with the followed step, turns included", () => {
    const anchor: Box = [0.2, 0.4, 0.4, 0.05];
    // The page moved right, up and closer (10 % zoom) and turned 5 degrees about the step's centre.
    const a = (5 * Math.PI) / 180, k = 1.1, c: [number, number] = [0.45, 0.375];
    const carry = ([x, y]: [number, number]): [number, number] => {
      const dx = (x - 0.4) * k, dy = (y - 0.425) * k;
      return [c[0] + Math.cos(a) * dx - Math.sin(a) * dy, c[1] + Math.sin(a) * dx + Math.cos(a) * dy];
    };
    const quad = [carry([0.2, 0.4]), carry([0.6, 0.4]), carry([0.6, 0.45]), carry([0.2, 0.45])] as Followed["quad"];
    const followed: Followed = { anchor, quad, box: [0, 0, 0, 0], state: "ok", score: 0.9, lost: 0 };
    const result: CheckResult = {
      page: { problem: null, steps: [step("s1", [0.2, 0.2, 0.4, 0.05]), step("s2", anchor)] },
      finding: { step_id: "s2", kind: "sign", hint: "?", mask: [[0.2, 0.4], [0.6, 0.4], [0.6, 0.45]] },
    };
    const shown = moveResult(result, followed, true)!;
    expect(shown.outline).toBe("ok");
    shown.finding!.mask!.forEach((p, i) => {
      const want = carry(result.finding!.mask![i]);
      expect(p[0]).toBeCloseTo(want[0]);
      expect(p[1]).toBeCloseTo(want[1]);
    });
    const s1 = shown.page.steps[0].bbox, far = carry([0.6, 0.25]);
    expect(s1[0] + s1[2]).toBeCloseTo(Math.max(far[0], carry([0.6, 0.2])[0])); // bounds of the turned box
    expect(mapBox([0, 0, 1, 1], quadMap([0, 0, 1, 1], [[0, 0], [1, 0], [1, 1], [0, 1]]))).toEqual([0, 0, 1, 1]);
    // No mask: the step's box is outlined, turned with the page.
    const boxOnly = moveResult({ ...result, finding: { ...result.finding!, mask: null } }, followed, true)!;
    expect(boxOnly.finding!.mask![1][0]).toBeCloseTo(carry([0.6, 0.4])[0]);
  });

  it("fades a held outline, hides a lost one and never uses another step's report", () => {
    const anchor: Box = [0.2, 0.4, 0.4, 0.05];
    const result: CheckResult = {
      page: { problem: null, steps: [step("s2", anchor)] },
      finding: { step_id: "s2", kind: "sign", hint: "?", mask: null },
    };
    const quad: Followed["quad"] = [[0.2, 0.4], [0.6, 0.4], [0.6, 0.45], [0.2, 0.45]];
    const f = (state: Followed["state"], a: Box = anchor): Followed => ({ anchor: a, quad, box: a, state, score: 0, lost: 0 });
    expect(moveResult(result, f("held"), true)!.outline).toBe("held");
    expect(moveResult(result, f("lost"), true)!.outline).toBe("lost");
    expect(moveResult(result, f("ok"), false)!.outline).toBe("held"); // tracking paused: last place, faded
    expect(moveResult(result, f("ok", [0.1, 0.1, 0.3, 0.05]), true)!.outline).toBe("lost"); // an older check's step
    expect(moveResult(result, null, true)!.outline).toBe("lost"); // not located yet
    expect(moveResult(result, null, false)).toBe(result); // no tracking at all: as checked
    const fine = { ...result, finding: null };
    expect(moveResult(fine, f("ok"), true)).toBe(fine);
  });

  it("finds the line a fingertip points at, or none", () => {
    const steps = [step("a", [0.1, 0.1, 0.5, 0.05]), step("b", [0.1, 0.3, 0.5, 0.05])];
    expect(pointedStep([0.3, 0.32], steps)?.id).toBe("b");
    expect(pointedStep([0.3, 0.36], steps)?.id).toBe("b"); // just under the line
    expect(pointedStep([0.9, 0.9], steps)).toBeNull();
    expect(pointedStep(null, steps)).toBeNull();
  });
});
