import { describe, expect, it } from "vitest";
import { CheckQueue } from "./checkQueue";

describe("CheckQueue", () => {
  it("sends at once when idle", () => {
    const q = new CheckQueue<string>();
    expect(q.offer("a")).toBe("a");
    expect(q.busy).toBe(true);
    expect(q.done()).toBeNull();
    expect(q.busy).toBe(false);
  });

  it("keeps a step finished during a running check and sends it next", () => {
    const q = new CheckQueue<string>();
    q.offer("line 1");
    expect(q.offer("line 2")).toBeNull();
    expect(q.done()).toBe("line 2");
    expect(q.busy).toBe(true); // line 2 is now running
    expect(q.done()).toBeNull();
  });

  it("keeps only the newest waiting check", () => {
    const q = new CheckQueue<string>();
    q.offer("a");
    q.offer("b");
    q.offer("c");
    expect(q.done()).toBe("c");
    expect(q.done()).toBeNull();
  });
});
