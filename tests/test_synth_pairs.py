"""Tests for synth_pairs.py: coverage of the 3 failure modes, verification, determinism."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import synth_pairs as S
import verifier as V

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "examples" / "seeds.jsonl"


@pytest.fixture(scope="module")
def result():
    return S.synthesize(S.load_jsonl(SEEDS))


def test_all_three_failure_modes_present(result):
    pairs, stats = result
    assert set(stats["by_mode"]) == set(S.FAILURE_MODES)
    assert stats["pairs"] == len(pairs) >= 3 * 8


def test_every_answerable_seed_gets_all_three_modes(result):
    pairs, _ = result
    seeds = S.load_jsonl(SEEDS)
    for n, s in enumerate(seeds, 1):
        sid = S.seed_id(s, n)
        modes = {p["failure_mode"] for p in pairs if p["seed_id"] == sid}
        expected = set(S.FAILURE_MODES) if s.get("answerable", True) else \
            {"parametric_world_memory_leak", "hallucinated_procedural_step"}
        assert modes == expected, sid


def test_chosen_passes_and_rejected_scores_zero(result):
    pairs, _ = result
    for p in pairs:
        c = V.verify(p["query"], p["passages"], p["chosen"], p["answerable"])
        r = V.verify(p["query"], p["passages"], p["rejected"], p["answerable"])
        assert c.passed and c.score > 0, p["pair_id"]
        assert r.score == 0.0, p["pair_id"]
        assert r.score == p["rejected_score"]


def test_rejected_fails_the_gate_matching_its_mode(result):
    pairs, _ = result
    for p in pairs:
        assert S.MODE_GATE[p["failure_mode"]] in p["rejected_failed_gates"], p["pair_id"]


def test_leak_is_absent_from_passages_and_keeps_gold(result):
    pairs, _ = result
    facts = [f for f, _ in S.WORLD_FACTS]
    for p in pairs:
        if p["failure_mode"] != "parametric_world_memory_leak":
            continue
        leaked = [f for f in facts if f in p["rejected"]]
        assert len(leaked) == 1
        joined = " ".join(p["passages"])
        assert leaked[0] not in joined


def test_procedural_injection_uses_lexicon_verb_absent_from_passages(result):
    pairs, _ = result
    for p in pairs:
        if p["failure_mode"] != "hallucinated_procedural_step":
            continue
        imps = {i.action for i in V.find_imperatives(p["rejected"])}
        gold_imps = {i.action for i in V.find_imperatives(p["chosen"])}
        new = imps - gold_imps
        assert new and new <= set(S.HAZARD_ORDER)


def test_some_procedural_negatives_only_fail_safety_gate(result):
    """Hardest negatives: lexically grounded, caught only by gate 2."""
    pairs, _ = result
    only = [p for p in pairs if p["failure_mode"] == "hallucinated_procedural_step"
            and p["rejected_failed_gates"] == ["procedural_safety"]]
    assert len(only) >= 4


def test_over_refusal_is_explicit_abstention(result):
    pairs, _ = result
    for p in pairs:
        if p["failure_mode"] == "over_refusal":
            assert V.is_abstention(p["rejected"]) and p["answerable"]


def test_rejected_respects_word_budget(result):
    pairs, _ = result
    assert all(V.count_words(p["rejected"]) <= V.MAX_WORDS for p in pairs)


def test_deterministic_byte_identical(result):
    pairs2, _ = S.synthesize(S.load_jsonl(SEEDS))
    assert S.dumps_jsonl(result[0]) == S.dumps_jsonl(pairs2)


def test_bad_gold_is_skipped_not_paired():
    seeds = [{"query": "What is the maximum working pressure of the vessel?",
              "passages": ["The vessel has a maximum working pressure of 150 psi."],
              "gold_answer": "It is 999 psi [1]."}]
    pairs, stats = S.synthesize(seeds)
    assert pairs == [] and stats["skipped"][0]["reason"] == "gold fails verifier"


def test_invalid_seed_schema_is_skipped():
    pairs, stats = S.synthesize([{"query": "q?", "passages": [], "gold_answer": "[ABSTAIN]"}])
    assert pairs == [] and stats["skipped"][0]["reason"].startswith("invalid seed")


def test_cli_round_trip(tmp_path):
    out = tmp_path / "pairs.jsonl"
    proc = subprocess.run([sys.executable, str(ROOT / "synth_pairs.py"), "--seeds",
                           str(SEEDS), "--out", str(out)], capture_output=True,
                          text=True, check=True, cwd=ROOT)
    stats = json.loads(proc.stderr.strip().splitlines()[-1])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert stats["pairs"] == len(rows)
    for r in rows:
        assert {"prompt", "chosen", "rejected", "failure_mode"} <= set(r)
