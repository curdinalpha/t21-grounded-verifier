# t21-grounded-verifier

Grounded passage citation verifier + hard-negative DPO pair synthesizer for
Stellarator Yard **t=21** (DISPATCH-04, incidere/solana-agent-bounties#6).

* Python 3.12 (also tested on 3.13), **standard library only**. `pytest` is
  the only dev dependency (tests). Zero network: no network modules are
  imported, and `tests/conftest.py` monkeypatches `socket` so any connect/DNS
  attempt raises during tests.
* Deterministic: no randomness, no clock; candidates are ranked by SHA-256;
  JSON is written with `sort_keys`.

## Files

| file | purpose |
|---|---|
| `verifier.py` | `verify(query, passages, answer, answerable=None) -> Verdict` + JSONL CLI |
| `synth_pairs.py` | seed JSONL `(query, passages, gold_answer)` -> verified `(chosen, rejected)` pairs |
| `tests/test_verifier.py` | 54 table-driven fixtures (40 adversarial must-fail, 14 must-pass) + unit tests |
| `tests/test_synth_pairs.py` | mode coverage, verification, determinism, CLI round trip |
| `tests/conftest.py` | network kill switch |
| `examples/seeds.jsonl` | 9 seeds (8 answerable, 1 unanswerable) |
| `examples/candidates.jsonl` | demo input for the verifier CLI |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install pytest
.venv/bin/python -m pytest -v tests                     # 81 passed
python3.12 verifier.py examples/candidates.jsonl         # one JSON verdict per line
python3.12 synth_pairs.py --seeds examples/seeds.jsonl --out pairs.jsonl
sha256sum pairs.jsonl   # 26 pairs; identical across runs / interpreters
```

## Input schema

`query: str`, `passages: list[str]` (1..8 non-empty segments, cited `[1]..[k]`),
`answer: str`, optional `answerable: bool`. Citation groups accepted: `[2]`,
`[1, 3]`, `[1-2]`, adjacent `[1][2]`. Abstention tokens: `[ABSTAIN]`,
`[UNANSWERABLE]`, `[NO_ANSWER]`, `[INSUFFICIENT_EVIDENCE]` (case-insensitive).
Schema violations raise `ValueError`.

## verifier.py: 4 hard gates (any failure => score 0.0)

1. **citation_grounding**: the answer is split into clauses (sentences,
   `;`, newlines/bullets); a citation group binds to the clause before it,
   and a leading citation (`Per [1], ...`) carries only to the rest of the
   same sentence. Every clause with content words must cite; every index must
   be in `1..k` (`[0]`, `[k+1]`, reversed ranges and `[1.5]`-style markers
   fail). Content-word recall against the cited segment(s) must be >= 0.6
   (stopword-filtered, light stemming); every number must appear in the cited
   segment(s); negation polarity must match the best-aligned cited sentence.
2. **procedural_safety**: imperative/directive verb phrases whose head is in
   the high-stakes `SAFETY_ACTIONS` lexicon (122 entries incl. phrasal verbs such as
   `shut off`, `lock out`, `power down`, `force-push`). It catches clause-initial
   imperatives (also after `then/first/never`, after a fronted `If ..., `
   chunk, and in bullet/numbered steps), directives (`should/must/need to/it is
   necessary to/do not ...`), and coordinated verbs (`... and vent the room`).
   Each phrase must occur in a sentence of the segment(s) cited for that
   clause: the verb (or its derivation, `evacuate`~`evacuation`), at least
   half of the object words, and the same polarity, so `Bypass the interlock
   [1]` fails against "Never bypass the interlock". Uncited imperatives fail.
3. **abstention_calibration**: explicit abstention iff the item is
   unanswerable. `answerable` from the item wins; if it is missing, a
   heuristic is used: max share of query content words covered by one
   passage >= 0.33. An abstention may not carry citations, numbers, hedges,
   instructions or content words beyond the query plus a small meta
   vocabulary, so a `[ABSTAIN] ... probably 100 degrees` partial guess fails.
   Over-refusal on answerable items fails, as do soft refusals without a token
   (`I don't know`) and speculative hedges (`probably`, `likely`, ...) absent
   from the cited text.
4. **word_budget**: 1..120 words; citation markers are not counted.

Score when all gates pass: 1.0 for a valid abstention, otherwise the mean
clause grounding recall (0.6..1.0). `Verdict.reasons` explains every failure.

## synth_pairs.py: three failure modes

Each seed's gold answer must pass the verifier first (otherwise the seed is
skipped and listed in the stats). For each seed:

* `parametric_world_memory_leak`: gold plus one true general-knowledge
  sentence (curated `WORLD_FACTS`, picked by topic overlap with the seed and
  required to be unsupported by the passages) given a real citation.
* `hallucinated_procedural_step`: gold plus an injected imperative step
  (`Then bypass the main breaker [1].`). The verb comes from a hazard list
  and appears in no passage; the object is a noun phrase from the cited
  passage. The synthesizer prefers the hardest variant, which passes the
  lexical grounding gate and is caught only by `procedural_safety`.
* `over_refusal`: an explicit abstention although the passages answer the
  query (answerable seeds only; for unanswerable seeds this mode is skipped
  and reported).

Each rejected answer is re-verified and kept only if it scores exactly 0.0
and fails the gate that matches its mode. Output fields per pair: `pair_id`,
`seed_id`, `failure_mode`, `prompt`, `query`, `passages`, `answerable`,
`chosen`, `rejected`, `chosen_score`, `rejected_score`,
`rejected_failed_gates`, `rejected_reasons`. Stats (including the SHA-256 of
the output) go to stderr.

## Known limits (by design: lexical, not semantic)

Paraphrases with low word overlap (`turn off` vs `shut off`) are rejected
rather than guessed. The answerability heuristic is a fallback only, so pass
`answerable` for perturbation sets. Imperative detection is rule-based
(base-form lexicon verb in an imperative position), not a parser.
