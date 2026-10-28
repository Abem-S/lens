// D2 done criterion (structure): a recorded Page + Finding renders correctly.
// Visual check at laptop and phone widths: `npm run screenshots` (see README).
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { DEMO_RESULT } from "../demo/fixture";
import { LensScreen } from "./LensScreen";

afterEach(cleanup);

describe("LensScreen with a recorded page", () => {
  it("lists every read line with its status", () => {
    const { container } = render(<LensScreen aspect={4 / 3} media={null} result={DEMO_RESULT} status="demo" />);
    const items = container.querySelectorAll(".steps li");
    expect(items).toHaveLength(DEMO_RESULT.page.steps.length);
    expect(container.querySelectorAll(".step-ok")).toHaveLength(3);
    expect(container.querySelectorAll(".step-error")).toHaveLength(1);
    expect(container.querySelector(".step-focus .step-icon")?.textContent).toBe("✗");
  });

  it("outlines the mistake with the finding's mask and points at it", () => {
    render(<LensScreen aspect={4 / 3} media={null} result={DEMO_RESULT} status="demo" />);
    const outline = screen.getByTestId("finding-outline");
    expect(outline.getAttribute("points")!.split(" ")).toHaveLength(DEMO_RESULT.finding!.mask!.length);
    const pointer = screen.getByTestId("tutor-pointer");
    const errStep = DEMO_RESULT.page.steps.find((s) => s.id === DEMO_RESULT.finding!.step_id)!;
    expect(parseFloat(pointer.style.top)).toBeCloseTo((errStep.bbox[1] + errStep.bbox[3] / 2) * 100, 3);
  });

  it("shows the hint, and the problem", () => {
    render(<LensScreen aspect={4 / 3} media={null} result={DEMO_RESULT} status="demo" />);
    expect(screen.getByTestId("hint").textContent).toBe(DEMO_RESULT.finding!.hint);
    expect(screen.getByText(DEMO_RESULT.page.problem!)).toBeTruthy();
  });

  it("falls back to the step box when there is no mask", () => {
    const noMask = { ...DEMO_RESULT, finding: { ...DEMO_RESULT.finding!, mask: null } };
    render(<LensScreen aspect={4 / 3} media={null} result={noMask} status="demo" />);
    expect(screen.getByTestId("finding-outline").getAttribute("points")!.split(" ")).toHaveLength(4);
  });

  it("says all is well when nothing is wrong", () => {
    const clean = { page: DEMO_RESULT.page, finding: null };
    render(<LensScreen aspect={4 / 3} media={null} result={clean} status="demo" />);
    expect(screen.queryByTestId("finding-outline")).toBeNull();
    expect(screen.getByText(/Every line checks out/)).toBeTruthy();
  });
});
