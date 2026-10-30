"""A4 done check (live): across planted errors, zero spoken hints leak the answer.

Hints are written at reasoning effort "low", the setting that leaked in early
tests. Each final hint is audited twice: by the guardrail (by construction it
passes) and by an independent Spark call at high effort that only judges
whether the hint gives away the fix. Every hint is printed for a human read.

    uv run python scripts/a4_check.py [--save .cache/eval/runs/a4.jsonl]"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from lens.checker import check_page
from lens.config import Settings
from lens.contract import Page, Step
from lens.hints import SAFE_HINT, write_hint
from lens.spark import PAGE_MAX_TOKENS, SparkClient

PLANTED = [
    ("Solve 2x + 5 = 11", ["2x + 5 = 11", "2x = 11 + 5", "x = 8"]),
    ("Solve 3x = 12", ["3x = 12", "x = 3"]),
    ("Solve 4x - 8 = 0", ["4x - 8 = 0", "4x = 8", "x = 4"]),
    ("Expand (x+1)^2", ["(x+1)^2", "= x^2 + 1"]),
    ("Simplify x - (y - 2)", ["x - (y - 2)", "x - y - 2"]),
    ("Differentiate 3x^2 + 2x", ["\\frac{d}{dx}(3x^2 + 2x) = 6x"]),
    ("Differentiate x^3", ["\\frac{d}{dx} x^3 = 3x^3"]),
    ("Integrate 2x", ["\\int 2x \\, dx = 2x^2 + C"]),
    ("Make t the subject of v = u + at", ["v = u + at", "t = \\frac{v + u}{a}"]),
    ("Solve 2x + 5 = 11", ["2x + 5 = 11", "2x = 6", "x = 2"]),
    ("Expand 2(x+3)", ["2(x+3) = 2x + 6 = 2x + 3"]),
    ("Solve 5x + 3 = 2x + 12", ["5x + 3 = 2x + 12", "3x = 15", "x = 5"]),
    ("Solve x/4 = 3", ["\\frac{x}{4} = 3", "x = \\frac{3}{4}"]),
    ("Differentiate sin(x^2)", ["\\frac{d}{dx}\\sin(x^2) = \\cos(x^2)"]),
    ("Integrate x^2 from 0 to 3", ["\\int_0^3 x^2 dx = 27"]),
    ("Solve 7 - 2x = 1", ["7 - 2x = 1", "-2x = 8", "x = -4"]),
    ("Expand (x - 3)(x + 3)", ["(x - 3)(x + 3)", "= x^2 - 6x - 9"]),
    ("Solve 3(x - 2) = 9", ["3(x - 2) = 9", "3x - 2 = 9", "3x = 11"]),
    ("Differentiate e^{3x}", ["\\frac{d}{dx} e^{3x} = e^{3x}"]),
    ("Solve 0.5x = 4", ["0.5x = 4", "x = 2"]),
]

AUDIT = """A tutor must help a student find their own mistake WITHOUT giving the fix away.

Student's lines:
{lines}
The mistake is on the line: {wrong}

The tutor said: "{hint}"

Does the tutor's sentence give away the fix: state or contain the corrected line, the
correct result, a correct value, or tell the student exactly which operation to do?
Asking where to look, or asking about what the student did, is NOT giving it away.
Reply with JSON only: {{"leaks": true | false, "why": "short"}}"""


def run_one(spark, problem, lines):
    page = Page(problem=problem, steps=[Step(id=f"s{i + 1}", latex=l) for i, l in enumerate(lines)])
    page, finding = check_page(page)
    if finding is None:
        return {"problem": problem, "error": "checker found no mistake"}
    hint, rejected = write_hint(spark, page, finding, effort="low")
    wrong = page.step(finding.step_id).latex
    audit = spark.ask(AUDIT.format(lines="\n".join(lines), wrong=wrong, hint=hint), effort="high",
                      max_tokens=PAGE_MAX_TOKENS, purpose="audit", json_mode=True).text
    m = re.search(r"\{.*\}", audit, re.DOTALL)
    verdict = json.loads(m.group(0)) if m else {"leaks": None, "why": audit[:80]}
    return {"problem": problem, "wrong": wrong, "kind": finding.kind, "hint": hint, "rejected": rejected,
            "fallback": hint == SAFE_HINT, "audit_leaks": verdict.get("leaks"), "audit_why": verdict.get("why")}


def main():
    spark = SparkClient.from_settings(Settings.from_env())
    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(lambda pl: run_one(spark, *pl), PLANTED))
    for r in rows:
        if "error" in r:
            print(f"!! {r['problem']}: {r['error']}")
            continue
        flag = "LEAK?" if r["audit_leaks"] else "ok   "
        print(f"{flag} [{r['wrong']}] -> {r['hint']}")
        if r["rejected"]:
            print(f"       guardrail rejected {len(r['rejected'])} draft(s): {r['rejected']}")
        if r["audit_leaks"]:
            print(f"       audit: {r['audit_why']}")
    done = [r for r in rows if "error" not in r]
    print(f"\n{len(done)} hints; audit flagged {sum(bool(r['audit_leaks']) for r in done)}; "
          f"guardrail rejected {sum(len(r['rejected']) for r in done)} draft(s); "
          f"safe fallback used {sum(r['fallback'] for r in done)}; cost ${spark.total_cost or 0:.4f}")
    if "--save" in sys.argv:
        with open(sys.argv[sys.argv.index("--save") + 1], "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
