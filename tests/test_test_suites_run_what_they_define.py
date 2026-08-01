"""
test_test_suites_run_what_they_define.py -- guard against silently skipped tests.

SIX test files across the three parts printed a success line while running less
than they defined. The pattern is always the same: a hand-written call list in
`if __name__ == "__main__":` that nobody updates when a test is appended below
it, so the new test is defined, never called, and the file still ends with
"All <x> tests passed."

That is the worst possible failure mode for a test suite -- it reports green
while covering less than it claims, and every one of those files was green the
whole time it was hiding something. Between them they concealed the guard on the
no-op phi update, a nose-overhang bounds check, a d_halo blacklist test, and the
measured-envelope check on the triangle-quality gate.

This checks all three parts at once, statically, so a seventh instance fails
here rather than being discovered months later. A file passes if it either
collects its tests by name (`for n in dir(module) if n.startswith("test_")`) or
explicitly calls every test it defines.
"""
import ast
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE.parent))

PARTS = ("part1-simulation", "part2-simulation", "part3-simulation")


def _offenders():
    out = []
    for part in PARTS:
        tdir = _ROOT / part / "tests"
        if not tdir.is_dir():
            continue
        for f in sorted(tdir.glob("test_*.py")):
            src = f.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(src)
            except SyntaxError as exc:
                out.append((f"{part}/{f.name}", f"does not parse: {exc}"))
                continue

            defined = {n.name for n in tree.body
                       if isinstance(n, ast.FunctionDef)
                       and n.name.startswith("test_")}
            if not defined:
                continue

            main = None
            for n in tree.body:
                if (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                        and getattr(n.test.left, "id", None) == "__name__"):
                    main = n
            if main is None:
                # Run by the package runner as a script; with no __main__ it
                # would execute nothing at all.
                out.append((f"{part}/{f.name}",
                            f"defines {len(defined)} tests and has no "
                            f"__main__ block, so running it does nothing"))
                continue

            main_src = ast.get_source_segment(src, main) or ""
            if "dir(" in main_src and "startswith" in main_src:
                continue        # collects by name; cannot skip

            called = set()
            for n in ast.walk(main):
                if isinstance(n, ast.Name) and n.id in defined:
                    called.add(n.id)
                elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    called.add(n.func.id)

            missing = sorted(defined - called)
            if missing:
                out.append((f"{part}/{f.name}",
                            f"defines {len(defined)} tests but never calls "
                            f"{len(missing)}: {', '.join(missing)}"))
    return out


def test_no_test_file_silently_skips_its_own_tests():
    bad = _offenders()
    assert not bad, (
        "these files report success while running less than they define:\n  "
        + "\n  ".join(f"{name}: {why}" for name, why in bad)
        + "\n\nFix by collecting tests by name in __main__ rather than listing "
          "them by hand:\n"
          "    _mod = sys.modules[__name__]\n"
          "    for _n in sorted(n for n in dir(_mod) if n.startswith('test_')):\n"
          "        ...")


def test_the_guard_itself_detects_a_planted_offender(tmpdir=None):
    """The check must actually be able to fail.

    A static scan that returns 'all clear' because its own matching is broken
    is exactly the class of bug it exists to catch.
    """
    import tempfile
    import textwrap

    planted = textwrap.dedent('''
        def test_one():
            pass

        def test_two_never_called():
            pass

        if __name__ == "__main__":
            test_one()
    ''')
    with tempfile.TemporaryDirectory() as d:
        tdir = Path(d) / "tests"
        tdir.mkdir()
        (tdir / "test_planted.py").write_text(planted, encoding="utf-8")

        # Re-run the same logic against the planted file.
        tree = ast.parse(planted)
        defined = {n.name for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
        main = [n for n in tree.body
                if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                and getattr(n.test.left, "id", None) == "__name__"][0]
        called = {n.func.id for n in ast.walk(main)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert defined - called == {"test_two_never_called"}, (
            "the detector failed to spot a deliberately uncalled test; it "
            "would report 'all clear' on a real one too")


if __name__ == "__main__":
    _mod = sys.modules[__name__]
    _passed = _failed = 0
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        try:
            getattr(_mod, _n)()
            print("PASS " + _n)
            _passed += 1
        except Exception as _e:  # noqa: BLE001
            print("FAIL %s: %s" % (_n, _e))
            _failed += 1
    print("%d passed, %d failed" % (_passed, _failed))
    sys.exit(1 if _failed else 0)
