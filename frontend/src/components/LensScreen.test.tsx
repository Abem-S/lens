// D2 done criterion (structure): a recorded Page + Finding renders correctly.
// Visual check at laptop and phone widths: `npm run screenshots` (see README).
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { DEMO_RESULT } from "../demo/fixture";
import { LensScreen } from "./LensScreen";

afterEach(cleanup);

describe("LensScreen with a recorded page", () => {
  it("shows no list of the steps read (raw readings only showed misreads)", () => {
    const { container } = render(<LensScreen aspect={4 / 3} media={null} result={DEMO_RESULT} status="demo" />);
    expect(container.querySelectorAll(".steps li")).toHaveLength(0);
    expect(screen.queryByText(DEMO_RESULT.page.problem!)).toBeNull();
  });

  it("outlines the mistake with the finding's mask and points at it", () => {
    render(<LensScreen aspect={4 / 3} media={null} result={DEMO_RESULT} status="demo" />);
    const outline = screen.getByTestId("finding-outline");
    expect(outline.getAttribute("points")!.split(" ")).toHaveLength(DEMO_RESULT.finding!.mask!.length);
    const pointer = screen.getByTestId("tutor-pointer");
    const errStep = DEMO_RESULT.page.steps.find((s) => s.id === DEMO_RESULT.finding!.step_id)!;
    expect(parseFloat(pointer.style.top)).toBeCloseTo((errStep.bbox[1] + errStep.bbox[3] / 2) * 100, 3);
  });

  it("shows the hint", () => {
    render(<LensScreen aspect={4 / 3} media={null} result={DEMO_RESULT} status="demo" />);
    expect(screen.getByTestId("hint").textContent).toBe(DEMO_RESULT.finding!.hint);
  });

  it("fades the outline while its place is held and hides it once lost, keeping the hint", () => {
    const { rerender } = render(<LensScreen aspect={4 / 3} media={null} result={{ ...DEMO_RESULT, outline: "held" }}
      status="demo" />);
    expect(screen.getByTestId("finding-outline").getAttribute("class")).toContain("faded");
    expect(screen.getByTestId("tutor-pointer").className).toContain("faded");
    rerender(<LensScreen aspect={4 / 3} media={null} result={{ ...DEMO_RESULT, outline: "lost" }} status="demo" />);
    expect(screen.queryByTestId("finding-outline")).toBeNull();
    expect(screen.queryByTestId("tutor-pointer")).toBeNull();
    expect(screen.getByTestId("hint").textContent).toBe(DEMO_RESULT.finding!.hint);
  });

  it("falls back to the step box when there is no mask", () => {
    const noMask = { ...DEMO_RESULT, finding: { ...DEMO_RESULT.finding!, mask: null } };
    render(<LensScreen aspect={4 / 3} media={null} result={noMask} status="demo" />);
    expect(screen.getByTestId("finding-outline").getAttribute("points")!.split(" ")).toHaveLength(4);
  });

  it("draws nothing on the page when nothing is wrong", () => {
    const clean = { page: DEMO_RESULT.page, finding: null };
    render(<LensScreen aspect={4 / 3} media={null} result={clean} status="demo" />);
    expect(screen.queryByTestId("finding-outline")).toBeNull();
  });
});

describe("the page overlay", () => {
  it("marks only the mistake on the page, no ticks over the writing", async () => {
    const { render } = await import("@testing-library/react");
    const { PageView } = await import("./PageView");
    const page = {
      problem: null,
      steps: [
        { id: "s1", latex: "a", bbox: [0.1, 0.1, 0.3, 0.1] as [number, number, number, number], status: "ok" as const, verified_by: "sympy" as const },
        { id: "s2", latex: "b", bbox: [0.1, 0.3, 0.3, 0.1] as [number, number, number, number], status: "error" as const, verified_by: "sympy" as const },
      ],
    };
    const finding = { step_id: "s2", kind: "k", mask: null, hint: "h" };
    const { container } = render(<PageView aspect={1} page={page} finding={finding}><div /></PageView>);
    expect(container.querySelectorAll(".mark").length).toBe(0);
    expect(container.querySelector("[data-testid=finding-outline]")).toBeTruthy();
  });
});
