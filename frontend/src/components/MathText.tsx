import katex from "katex";
import "katex/dist/katex.min.css";
import { useMemo } from "react";

export function MathText({ latex }: { latex: string }) {
  const html = useMemo(
    () => katex.renderToString(latex, { throwOnError: false, strict: "ignore", output: "html" }),
    [latex],
  );
  return <span className="math" dangerouslySetInnerHTML={{ __html: html }} />;
}
