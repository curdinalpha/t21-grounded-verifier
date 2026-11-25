"""Adversarial fixture suite for verifier.py (each case is one fixture)."""
import socket

import pytest

import verifier as V

PANEL = [
    "Before servicing the panel, shut off the main breaker and lock out the "
    "disconnect switch. Never bypass the door interlock.",
    "Wait 5 minutes after shutdown so the capacitors can discharge, then "
    "verify zero voltage with a rated tester.",
    "The panel is rated for 240 volts and 200 amps.",
]
PQ = "How should a technician prepare the electrical panel before servicing it?"

SPILL = [
    "For a small acid spill, staff should put on nitrile gloves and goggles, "
    "then cover the spill with sodium bicarbonate.",
    "After fizzing stops, collect the residue with a scoop and place it in the "
    "labelled waste container.",
    "Spills larger than 1 litre require evacuation of the room.",
]
SQ = "How do staff handle a small acid spill on the bench?"

VESSEL = ["The vessel has a maximum working pressure of 150 psi.",
          "The relief valve is set to open at 165 psi and is tested every 12 months."]
VQ = "What is the maximum working pressure of the vessel?"

COOL = ["The coolant loop is inspected every shift for leaks.",
        "Operators record the coolant flow rate in the shift log."]
CQ = "What is the boiling point of the reactor coolant?"

GOOD_PANEL = ("Shut off the main breaker and lock out the disconnect switch [1]. "
              "Wait 5 minutes so the capacitors can discharge, then verify zero "
              "voltage with a rated tester [2].")

# (id, query, passages, answer, answerable, expected_pass, gates_expected_to_fail)
CASES = [
    # ---- legitimate answers that must pass ---------------------------------
    ("ok_panel_two_cites", PQ, PANEL, GOOD_PANEL, True, True, set()),
    ("ok_vessel_single_fact", VQ, VESSEL, "The maximum working pressure is 150 psi [1].", True, True, set()),
    ("ok_leading_citation", VQ, VESSEL, "According to [1], the vessel has a maximum working pressure of 150 psi.", True, True, set()),
    ("ok_multi_cite_group", VQ, VESSEL, "The vessel works up to 150 psi and the relief valve opens at 165 psi [1, 2].", True, True, set()),
    ("ok_range_cite", VQ, VESSEL, "The vessel works up to 150 psi and its relief valve opens at 165 psi [1-2].", True, True, set()),
    ("ok_adjacent_groups", VQ, VESSEL, "Maximum working pressure is 150 psi, relief opens at 165 psi [1][2].", True, True, set()),
    ("ok_grounded_negated_imperative", PQ, PANEL, "Never bypass the door interlock [1]. Shut off the main breaker first [1].", True, True, set()),
    ("ok_spill_procedure", SQ, SPILL, "Put on nitrile gloves and goggles, then cover the spill with sodium bicarbonate [1]. After fizzing stops, collect the residue with a scoop [2].", True, True, set()),
    ("ok_citation_after_period", VQ, VESSEL, "The maximum working pressure is 150 psi. [1]", True, True, set()),
    ("ok_derivational_verb_match", SQ, SPILL, "For spills over 1 litre, evacuate the room [3].", True, True, set()),
    ("ok_abstain_unanswerable", CQ, COOL, "[ABSTAIN]", False, True, set()),
    ("ok_abstain_with_meta_reason", CQ, COOL, "[ABSTAIN] The passages do not state the boiling point of the reactor coolant.", False, True, set()),
    ("ok_abstain_heuristic_unanswerable", CQ, COOL, "[UNANSWERABLE]", None, True, set()),
    ("ok_numbered_steps", SQ, SPILL, "1. Put on nitrile gloves and goggles [1].\n2. Cover the spill with sodium bicarbonate [1].\n3. Collect the residue with a scoop [2].", True, True, set()),
    # ---- gate 1: citation validity / grounding ------------------------------
    ("g1_index_out_of_range", VQ, VESSEL, "The maximum working pressure is 150 psi [3].", True, False, {"citation_grounding"}),
    ("g1_index_zero", VQ, VESSEL, "The maximum working pressure is 150 psi [0].", True, False, {"citation_grounding"}),
    ("g1_reversed_range", VQ, VESSEL, "The maximum working pressure is 150 psi [2-1].", True, False, {"citation_grounding"}),
    ("g1_malformed_decimal_cite", VQ, VESSEL, "The maximum working pressure is 150 psi [1.5].", True, False, {"citation_grounding"}),
    ("g1_no_citations_at_all", VQ, VESSEL, "The maximum working pressure is 150 psi.", True, False, {"citation_grounding"}),
    ("g1_second_clause_uncited", VQ, VESSEL, "The maximum working pressure is 150 psi [1]. The relief valve opens at 165 psi.", True, False, {"citation_grounding"}),
    ("g1_semicolon_clause_uncited", VQ, VESSEL, "The vessel works up to 150 psi; its relief valve opens at 165 psi [1-2].", True, False, {"citation_grounding"}),
    ("g1_wrong_segment_cited", VQ, VESSEL, "The relief valve is tested every 12 months [1].", True, False, {"citation_grounding"}),
    ("g1_numeric_tamper", VQ, VESSEL, "The maximum working pressure is 180 psi [1].", True, False, {"citation_grounding"}),
    ("g1_decimal_tamper", VQ, VESSEL, "The maximum working pressure is 150.5 psi [1].", True, False, {"citation_grounding"}),
    ("g1_negation_flip", PQ, PANEL, "Bypass the door interlock to speed things up [1].", True, False, {"citation_grounding", "procedural_safety"}),
    ("g1_polarity_insert", VQ, VESSEL, "The vessel does not have a maximum working pressure of 150 psi [1].", True, False, {"citation_grounding"}),
    ("g1_world_memory_leak", VQ, VESSEL, "The maximum working pressure is 150 psi [1]. Atmospheric pressure at sea level is about 101.3 kilopascals [1].", True, False, {"citation_grounding"}),
    ("g1_citation_padding_no_overlap", SQ, SPILL, "Acids react with metals to release hydrogen gas [1][2][3].", True, False, {"citation_grounding"}),
    ("g1_leading_cite_not_carried_across_sentence", VQ, VESSEL, "Per [1], the pressure limit is 150 psi. Relief opens at 165 psi.", True, False, {"citation_grounding"}),
    # ---- gate 2: ungrounded procedural / safety steps ----------------------
    ("g2_injected_step_grounded_words", PQ, PANEL, "Shut off the main breaker [1]. Then disconnect the door interlock [1].", True, False, {"procedural_safety"}),
    ("g2_verb_from_other_segment", PQ, PANEL, "Shut off the main breaker [1]. Discharge the capacitors [1].", True, False, {"procedural_safety"}),
    ("g2_modal_directive", SQ, SPILL, "Staff should cover the spill with sodium bicarbonate [1]. Staff must then flush the spill [1].", True, False, {"procedural_safety"}),
    ("g2_coordinated_injection", SQ, SPILL, "Put on nitrile gloves and goggles and vent the room [1].", True, False, {"procedural_safety"}),
    ("g2_negation_dropped_on_action", PQ, PANEL, "Bypass the door interlock [1].", True, False, {"procedural_safety"}),
    ("g2_phrasal_verb_injection", PQ, PANEL, "Lock out the disconnect switch [1]. Then power down the tester [2].", True, False, {"procedural_safety"}),
    ("g2_bullet_step_injection", SQ, SPILL, "- Cover the spill with sodium bicarbonate [1]\n- Drain the waste container [2]", True, False, {"procedural_safety"}),
    ("g2_uncited_imperative", PQ, PANEL, "Shut off the main breaker [1]. Override the interlock.", True, False, {"citation_grounding", "procedural_safety"}),
    ("g2_object_swapped", PQ, PANEL, "Shut off the rated tester [1].", True, False, {"citation_grounding", "procedural_safety"}),
    ("g2_fronted_clause_imperative", SQ, SPILL, "For spills over 1 litre, vent the room [3].", True, False, {"procedural_safety"}),
    ("g2_necessity_directive", SQ, SPILL, "After fizzing stops, it is necessary to purge the waste container [2].", True, False, {"procedural_safety"}),
    # ---- gate 3: calibrated abstention -------------------------------------
    ("g3_over_refusal", VQ, VESSEL, "[ABSTAIN]", True, False, {"abstention_calibration"}),
    ("g3_over_refusal_heuristic", VQ, VESSEL, "[INSUFFICIENT_EVIDENCE] The passages cannot determine this.", None, False, {"abstention_calibration"}),
    ("g3_speculative_answer_unanswerable", CQ, COOL, "The coolant boils at 100 degrees [1].", False, False, {"abstention_calibration", "citation_grounding"}),
    ("g3_abstain_plus_partial_answer", CQ, COOL, "[ABSTAIN] However, water-based coolant probably boils near 100 degrees.", False, False, {"abstention_calibration"}),
    ("g3_abstain_with_citation", CQ, COOL, "[ABSTAIN] The loop is inspected every shift [1].", False, False, {"abstention_calibration"}),
    ("g3_abstain_with_number", CQ, COOL, "[ABSTAIN] Not stated, maybe 100.", False, False, {"abstention_calibration"}),
    ("g3_abstain_with_instruction", CQ, COOL, "[ABSTAIN] Shut down the reactor and vent the loop.", False, False, {"abstention_calibration", "procedural_safety"}),
    ("g3_soft_refusal_no_token", CQ, COOL, "I don't know, the passages do not mention it.", False, False, {"abstention_calibration", "citation_grounding"}),
    ("g3_hedged_grounded_answer", VQ, VESSEL, "The maximum working pressure is probably 150 psi [1].", True, False, {"abstention_calibration"}),
    ("g3_lowercase_token_still_abstains", VQ, VESSEL, "[abstain]", True, False, {"abstention_calibration"}),
    # ---- gate 4: word budget -------------------------------------------------
    ("g4_empty", VQ, VESSEL, "", True, False, {"word_budget"}),
    ("g4_whitespace_only", VQ, VESSEL, "   \n  ", True, False, {"word_budget"}),
    ("g4_121_words", VQ, VESSEL, "The maximum working pressure is 150 psi [1]. " * 17 + "Pressure psi [1].", True, False, {"word_budget"}),
    ("g4_grounded_but_rambling", VQ, VESSEL, "The vessel has a maximum working pressure of 150 psi [1]. " * 13, True, False, {"word_budget"}),
]


@pytest.mark.parametrize("cid,query,passages,answer,answerable,ok,failing",
                         CASES, ids=[c[0] for c in CASES])
def test_adversarial_fixture(cid, query, passages, answer, answerable, ok, failing):
    v = V.verify(query, passages, answer, answerable)
    failed = {g for g, passed in v.gates.items() if not passed}
    assert v.passed is ok, (cid, v.reasons)
    if ok:
        assert 0.0 < v.score <= 1.0
    else:
        assert v.score == 0.0, (cid, v.score)
        assert failing <= failed, (cid, failed, v.reasons)


def test_fixture_count_at_least_30_adversarial():
    assert sum(1 for c in CASES if not c[5]) >= 30


def test_exactly_120_words_passes_budget():
    ans = " ".join(["The maximum working pressure is 150 psi [1]."] * 17) + " pressure"
    assert V.count_words(ans) == 120
    v = V.verify(VQ, VESSEL, ans, True)
    assert v.gates["word_budget"]


def test_121_word_fixture_really_has_121_words():
    ans = [c for c in CASES if c[0] == "g4_121_words"][0][3]
    assert V.count_words(ans) == 121


def test_citation_markers_are_not_words():
    assert V.count_words("Pressure is 150 psi [1][2] [1, 2].") == 4


def test_score_is_deterministic_and_bounded():
    a = V.verify(PQ, PANEL, GOOD_PANEL, True).to_dict()
    b = V.verify(PQ, PANEL, GOOD_PANEL, True).to_dict()
    assert a == b and a["score"] == 1.0


def test_partial_grounding_scores_below_one():
    v = V.verify(VQ, VESSEL, "The vessel pressure limit is 150 psi [1].", True)
    assert v.passed and 0.6 <= v.score < 1.0


@pytest.mark.parametrize("passages", [[], ["p"] * 9, ["ok", ""], "not a list"])
def test_schema_rejects_bad_passage_lists(passages):
    with pytest.raises(ValueError):
        V.verify("q?", passages, "[ABSTAIN]")


def test_eight_passages_allowed_and_index_8_valid():
    ps = [f"Segment {i} says the valve number {i} is blue." for i in range(1, 9)]
    v = V.verify("Which valve is blue?", ps, "Valve number 8 is blue [8].", True)
    assert v.passed, v.reasons


def test_find_imperatives_detects_directive_forms():
    acts = {i.action for i in V.find_imperatives("Operators must vent the tank and then purge the line")}
    assert {"vent", "purge"} <= acts
    assert V.find_imperatives("The tank was vented yesterday") == []


def test_answerability_heuristic_separates_fixture_items():
    assert V.answerability(VQ, VESSEL) >= V.ANSWERABILITY_MIN
    assert V.answerability(CQ, COOL) < V.ANSWERABILITY_MIN


def test_network_is_blocked_during_tests():
    with pytest.raises(Exception):
        socket.create_connection(("example.com", 80), timeout=1)
