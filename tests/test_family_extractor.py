"""Unit tests for the pure helpers in src/data/family_extractor.py.

The module imports torch-dependent training utilities at import time; they are
stubbed out here so the domain-normalisation and pattern-matching helpers can
be tested without the heavyweight graph stack.
"""
from __future__ import annotations

import importlib.util
import sys
import types

from conftest import REPO_ROOT

_stub = types.ModuleType("src.train.utils")
for _name in ("GraphBundle", "build_primary_task_frame", "load_graph_bundle", "save_dataframe"):
    setattr(_stub, _name, object())
sys.modules.setdefault("src.train.utils", _stub)

spec = importlib.util.spec_from_file_location(
    "family_extractor", REPO_ROOT / "src" / "data" / "family_extractor.py"
)
fam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fam)

# NOTE: order matters — patterns are tried in dict insertion order, and the
# compact form of "bc." degrades to plain "bc", which would greedily swallow
# bcgame domains if bcdot were listed first. Production config lists bcgame
# before bcdot for exactly this reason (see extract_denmark_families defaults).
PATTERNS = {
    "tsars": ["tsars"],
    "bcgame": ["bcgame", "bc-game"],
    "bcdot": ["bc."],
}


def test_normalise_domain_strips_www_case_and_whitespace():
    assert fam._normalise_domain("  WWW.Example.COM ") == "example.com"


def test_compact_domain_drops_non_letters():
    assert fam._compact_domain("bc-game.io") == "bcgameio"


def test_pattern_family_matches_substring():
    family, source = fam._pattern_family("tsars77.com", PATTERNS)
    assert family == "tsars"
    assert source == "pattern:tsars"


def test_pattern_family_compact_match_bridges_hyphens():
    family, _ = fam._pattern_family("bcgame-mirror.io", PATTERNS)
    assert family == "bcgame"


def test_pattern_family_returns_none_when_no_match():
    assert fam._pattern_family("plainexample.org", PATTERNS) == (None, None)


def test_pattern_order_dependence_of_greedy_compact_bc_pattern():
    """Documents a review finding: with bcdot first, its compact 'bc' pattern
    captures bcgame domains, so correctness depends on insertion order."""
    bcdot_first = {"bcdot": ["bc."], "bcgame": ["bcgame"]}
    family, _ = fam._pattern_family("bcgame-mirror.io", bcdot_first)
    assert family == "bcdot"
