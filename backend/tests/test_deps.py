"""Every third-party module the app imports is a declared dependency.

The deployed app installs only the base dependencies + the `gpu` extra. A module that
works locally only because the `eval` extra happens to pull it in (once: `httpx`)
breaks every page check in production with a 500.
"""

import ast
import sys
import tomllib
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
# import name -> distribution name, where they differ or come bundled with another package
PROVIDED_BY = {"PIL": "pillow", "starlette": "fastapi", "websockets": "uvicorn", "multipart": "python-multipart"}


def _dist(req: str) -> str:
    for sep in "<>=[; ":
        req = req.split(sep)[0]
    return req.lower()


def test_app_imports_are_declared_for_the_deployed_app():
    project = tomllib.loads((BACKEND / "pyproject.toml").read_text())["project"]
    declared = {_dist(r) for r in project["dependencies"] + project["optional-dependencies"]["gpu"]}
    imported = set()
    for f in (BACKEND / "src" / "lens").rglob("*.py"):
        if "eval" in f.relative_to(BACKEND / "src" / "lens").parts:
            continue
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
    third_party = imported - set(sys.stdlib_module_names) - {"lens"}
    missing = {m for m in third_party if PROVIDED_BY.get(m, m).lower() not in declared}
    assert not missing, f"imported but not declared for the deployed app: {sorted(missing)}"
