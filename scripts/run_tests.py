#!/usr/bin/env python3
"""Stdlib test runner — works without pytest installed.

Usage: python scripts/run_tests.py
       python scripts/run_tests.py --allow-network   (diagnostics only)

Every test here is meant to be offline, so the runner blocks outbound sockets and fails
any test that reaches for one. This is not hygiene for its own sake: two tests in this
suite were passing for the wrong reason, because the network call they made failed and the
error was caught as an expected condition. A silent network call turns a real assertion
into a coin flip that happens to land right.
"""

from __future__ import annotations

import importlib
import socket
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class NetworkAccessInTest(BaseException):
    """Derives from BaseException so it survives `except Exception`.

    The library wraps connection failures into LLMError, and callers legitimately catch
    that to degrade gracefully — which is precisely how a stray network call inside a test
    turns into a silent pass. Bypassing those handlers makes the call impossible to ignore.
    """


def _block_network() -> None:
    """Block outbound connections while keeping socket.socket a class.

    `ssl.SSLSocket` subclasses it, so replacing it with a function breaks importing ssl.
    Blocking `connect` instead leaves the type hierarchy intact.
    """

    class GuardedSocket(socket.socket):
        def connect(self, *args, **kwargs):
            raise NetworkAccessInTest(
                "offline test attempted a network connection; stub the client instead"
            )

        def connect_ex(self, *args, **kwargs):
            raise NetworkAccessInTest("offline test attempted a network connection")

    def blocked(*args, **kwargs):
        raise NetworkAccessInTest("offline test attempted to resolve a hostname")

    socket.socket = GuardedSocket
    socket.create_connection = blocked
    socket.getaddrinfo = blocked

MODULES = [
    "tests.test_retrieval_quality",
    "tests.test_contextual",
    "tests.test_a1_eval",
    "tests.test_verify",
    "tests.test_llm_transport",
    "tests.test_repair_loop",
    "tests.test_cli_output",
    "tests.test_gold_quality",
    "tests.test_source_invariants",
    "tests.test_factiva_retry",
    "tests.test_smoke",
]


def main() -> int:
    if "--allow-network" not in sys.argv:
        _block_network()
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
