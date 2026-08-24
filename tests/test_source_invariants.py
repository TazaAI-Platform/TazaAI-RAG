"""Static checks for defect classes that recurred in this codebase.

These are not style rules. Each one here has already shipped as a real bug, and each is
invisible until the exact moment things are going wrong — which is the worst time to find
out that the error handler itself is broken.
"""

import ast
import builtins
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "taza_rag"


def _modules() -> list[tuple[str, Path]]:
    out = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name == "__main__.py":
            continue
        rel = path.relative_to(ROOT).with_suffix("")
        out.append((".".join(rel.parts), path))
    return out


def _handler_names(tree: ast.AST) -> list[tuple[str, int]]:
    """Every exception name referenced by an `except` clause."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue
        targets = (
            node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
        )
        for t in targets:
            if isinstance(t, ast.Name):
                found.append((t.id, t.lineno))
    return found


def test_every_caught_exception_is_actually_resolvable():
    """Caught-but-not-imported raises NameError precisely when the handler is needed.

    This shipped twice: FactivaRetrieveError was caught in both eval modules while only
    FactivaRetrievalClient was imported, so a run that hit an upstream failure would have
    died with NameError instead of recording the failure and continuing.
    """
    problems: list[str] = []
    for mod_name, path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = importlib.import_module(mod_name)
        for name, lineno in _handler_names(tree):
            if hasattr(builtins, name) or hasattr(module, name):
                continue
            problems.append(f"{mod_name}:{lineno} catches undefined {name!r}")
    assert not problems, "unresolvable exception handlers:\n" + "\n".join(problems)


def test_no_handler_swallows_an_error_without_recording_it():
    """A bare `except: pass` hides exactly the failures an eval needs to report."""
    problems: list[str] = []
    for mod_name, path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            body = [n for n in node.body if not isinstance(n, ast.Pass)]
            if not body:
                problems.append(f"{mod_name}:{node.lineno} silently passes")
    assert not problems, "handlers that discard the error:\n" + "\n".join(problems)


def test_model_text_is_never_printed_as_rich_markup():
    """Rich reads [c1] as a style tag and deletes it, so printing an answer as markup
    strips its citations. Any console.print of answer or article text must opt out."""
    risky = ("result.answer", "row['answer']", 'row["answer"]', "chunk.text", "c.text")
    problems: list[str] = []
    for mod_name, path in _modules():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "console.print" not in line:
                continue
            if any(token in line for token in risky) and "markup=False" not in line:
                problems.append(f"{mod_name}:{lineno} prints model text as markup")
    assert not problems, "citation markers would be stripped:\n" + "\n".join(problems)
