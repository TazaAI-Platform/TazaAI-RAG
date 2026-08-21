#!/usr/bin/env python3
"""Stdlib test runner — works without pytest installed.

Usage: python scripts/run_tests.py
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODULES = [
    "tests.test_retrieval_quality",
    "tests.test_contextual",
    "tests.test_smoke",
]


def main() -> int:
    passed = 0
    failed: list[str] = []
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(mod, name)()
                passed += 1
                print(f"PASS {name}")
            except Exception:
                failed.append(f"{mod_name}.{name}")
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {len(failed)} failed")
    for f in failed:
        print(f"  - {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
