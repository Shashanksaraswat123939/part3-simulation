"""Keep GRID_SPACING_MM from leaking between tests. Mirrors part1's conftest.

Part 3 has the same defect part1 had. test_orchestrator_multivariable sets
2.0 mm at import and test_unified_pipeline_end_to_end sets 3.0 mm at import;
pytest imports every test module during collection, alphabetically, so the 3.0
lands last and EVERY test then runs at 3.0 -- including the orchestrator tests
written for 2.0. test_sweep_actually_runs_every_d_halo_candidate_and_round
failed only in a full-suite run and passed alone, which reads as flakiness
rather than as the wrong-resolution bug it is.

See part1-simulation/tests/conftest.py for the full write-up.
"""
import os
import sys
from pathlib import Path

import pytest

_P3 = Path(__file__).resolve().parent.parent
_ROOT = _P3.parent
for _p in (_P3, _ROOT / "part1-simulation", _ROOT / "part2-simulation",
           _ROOT / "part1-simulation" / "sandbox"):
    sys.path.insert(0, str(_p))
os.environ.setdefault("PART2_PATH", str(_ROOT / "part2-simulation"))

import geometry_contract  # noqa: E402

_PRISTINE_MM = geometry_contract.GRID_SPACING_MM


def _reset():
    if geometry_contract.GRID_SPACING_MM != _PRISTINE_MM:
        import coarse
        coarse.use_spacing(_PRISTINE_MM)


@pytest.fixture(autouse=True)
def _restore_grid_spacing():
    yield
    _reset()


def pytest_collectstart(collector):
    """Reset before each test module is imported, so a file that freezes
    GRID_SPACING_M with a module-level `from ... import` gets its own value
    rather than whichever file pytest happened to collect first."""
    if isinstance(collector, pytest.Module):
        _reset()


def pytest_collection_finish(session):
    _reset()
