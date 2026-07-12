"""
run_all_tests.py — Part 3 test runner (house Rule 7 compliant).

Exit codes:
    0 — every test file ran and passed
    1 — at least one test file failed
    2 — infrastructure problem (tests dir missing, no test files found)

Each test file is executed as its own process from the package directory so
sibling imports resolve. stdout AND stderr are surfaced verbatim — a test
that dies with a traceback must be visible, never swallowed (the exact
failure mode of the original Part 2 audit finding D1).
"""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


def main() -> int:
    package_dir = Path(__file__).resolve().parent
    tests_dir = package_dir / "tests"
    if not tests_dir.is_dir():
        print(f"INFRASTRUCTURE ERROR: tests directory not found at {tests_dir}",
              file=sys.stderr)
        return 2
    test_files = sorted(tests_dir.glob("test_*.py"))
    if not test_files:
        print(f"INFRASTRUCTURE ERROR: no test_*.py files found in {tests_dir}",
              file=sys.stderr)
        return 2

    failures: list[str] = []
    for test_file in test_files:
        print(f"\n{'=' * 70}\nRUNNING {test_file.name}\n{'=' * 70}")
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(package_dir)
            if not existing_pythonpath
            else str(package_dir) + os.pathsep + existing_pythonpath
        )
        proc = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=str(package_dir),
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        if proc.returncode != 0:
            failures.append(test_file.name)
            print(f"FAILED: {test_file.name} (exit {proc.returncode})")
        else:
            print(f"PASSED: {test_file.name}")

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {len(test_files)} test files, {len(failures)} failed")
    if failures:
        for name in failures:
            print(f"  FAILED: {name}")
        return 1
    print("ALL PART 3 TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
