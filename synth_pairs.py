"""Deterministic hard-negative DPO pair synthesizer (Stellarator Yard t=21).

Input: seed JSONL, one object per line with ``query``, ``passages`` (1..8) and
``gold_answer`` (optional ``id``, optional ``answerable``).  For every seed whose
gold answer passes ``verifier.verify`` we emit ``(chosen=gold, rejected=x)``
pairs for three failure modes:

* ``parametric_world_memory_leak``  gold + a sentence that is true in general
  world knowledge but absent from the passages, dressed with a real citation.
* ``hallucinated_procedural_step``  gold with an injected high-stakes
  imperative step (verb from ``verifier.SAFETY_ACTIONS`` that appears in no
  passage) applied to an object taken from the cited passage.
* ``over_refusal``                  an explicit abstention although the
  passages directly answer the query (answerable seeds only).

Every rejected candidate is re-scored by the verifier and kept only if it
scores exactly 0.0 *and* fails the gate that matches its failure mode, so each
pair is a verified preference.  No randomness, no clock, no network: choices
are made by stable SHA-256 ranking, output is ``sort_keys`` JSON.  Identical
input -> byte-identical output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import verifier as V

FAILURE_MODES = ("parametric_world_memory_leak", "hallucinated_procedural_step",
                 "over_refusal")
MODE_GATE = {"parametric_world_memory_leak": "citation_grounding",
             "hallucinated_procedural_step": "procedural_safety",
             "over_refusal": "abstention_calibration"}

# (statement true in general knowledge, topic cue words).  Used only when the
# statement is NOT supported by the seed's passages.
WORLD_FACTS: tuple[tuple[str, frozenset[str]], ...] = tuple(
    (s, frozenset(V.content_stems(t))) for s, t in [
        ("Water boils at 100 degrees Celsius at sea level",
         "water boil temperature heat steam liquid coolant"),
        ("Copper is one of the best electrical conductors among common metals",
         "electrical wire current conductor panel circuit breaker voltage"),
        ("Alternating current in Europe is typically supplied at 50 hertz",
         "electrical power grid voltage outlet mains supply"),
        ("Carbon monoxide is a colourless and odourless gas",
         "gas leak smell detector alarm ventilation carbon fumes"),
        ("Natural gas is mostly methane and is lighter than air",
         "gas leak methane pipeline stove smell"),
        ("Lithium-ion cells can enter thermal runaway when overheated or punctured",
         "battery lithium charging cell fire thermal pack"),
        ("Class D extinguishers are designed for combustible metal fires",
         "fire extinguisher flame burn metal blaze"),
        ("Sodium hypochlorite is the active ingredient of household bleach",
         "chemical spill bleach cleaning disinfectant laboratory"),
        ("Mixing bleach with ammonia releases toxic chloramine vapours",
         "chemical spill bleach ammonia cleaning fumes laboratory"),
        ("The normal adult resting heart rate is roughly 60 to 100 beats per minute",
         "patient heart pulse cardiac medical clinical dose"),
        ("Adrenaline is also known as epinephrine",
         "allergy anaphylaxis injection medication patient dose epinephrine"),
        ("Hypothermia is defined as a core body temperature below 35 degrees Celsius",
         "cold exposure patient temperature rescue"),
        ("Atmospheric pressure at sea level is about 101.3 kilopascals",
         "pressure valve vessel tank gauge psi"),
        ("Nitrogen makes up about 78 percent of Earth's atmosphere",
         "gas air atmosphere oxygen confined space ventilation"),
        ("PostgreSQL uses write-ahead logging to guarantee durability",
         "database migration backup table sql transaction postgres"),
        ("TLS certificates are commonly issued for 90 days by Let's Encrypt",
         "certificate server deploy https security key"),
        ("The Linux kernel was first released by Linus Torvalds in 1991",
         "server linux deploy kernel reboot production system"),
        ("Git commits are identified by SHA-1 hashes by default",
         "git repository branch commit deploy rollback code"),
        ("Jet fuel freezes at around minus 40 degrees Celsius",
         "aircraft fuel aviation engine flight icing"),
        ("Ice forms on aircraft wings when supercooled droplets strike them",
         "aircraft icing wing flight deice aviation"),
        ("Concrete reaches most of its design strength after about 28 days",
         "construction concrete site structure scaffold"),
        ("Sound travels at about 343 metres per second in air",
         "noise sound hearing alarm siren"),
    ])

HAZARD_ORDER = ("bypass", "override", "disable", "disconnect", "vent",
                "purge", "short-circuit", "inject", "delete", "drain",
                "remove", "unplug", "depressurize", "ignite", "wipe",
                "force-push", "truncate", "induce", "isolate", "unlock")
INJECT_TEMPLATES = ("Then {verb} the {obj} {cite}.",
                    "Next, {verb} the {obj} {cite}.",
                    "First, {verb} the {obj} {cite}.")
OVER_REFUSALS = ("[ABSTAIN] The provided passages do not contain enough information to answer this question.",
                 "[ABSTAIN] There is insufficient evidence in the passages to answer.",
                 "[INSUFFICIENT_EVIDENCE] The cited sources cannot determine the answer.")


def _h(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def seed_id(seed: dict, lineno: int) -> str:
    if seed.get("id"):
        return str(seed["id"])
    return f"s{lineno:04d}-" + _h(seed["query"], *seed["passages"])[:8]


def render_prompt(query: str, passages: list[str]) -> str:
    lines = [f"[{i}] {p}" for i, p in enumerate(passages, 1)]
    return ("Answer using only the passages. Cite [i] inline for every claim, "
            "or reply [ABSTAIN] if they do not answer the question. "
            f"Max {V.MAX_WORDS} words.\n\nQuestion: {query}\n\nPassages:\n"
            + "\n".join(lines))


def _main_cite(gold: str, k: int) -> int:
    counts = Counter(i for c in V.parse_clauses(gold) for i in c.cites if 1 <= i <= k)
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0] if counts else 1


def _sentences(text: str) -> list[str]:
    """Split gold into sentence chunks keeping trailing citation groups."""
    out, buf = [], ""
    for c in V.parse_clauses(text):
        buf_part = c.text.strip()
        cites = "".join(f"[{i}]" for i in c.cites) if c.cites else ""
        piece = buf_part
        if cites and not buf_part.endswith((".", "!", "?")):
            piece = f"{buf_part} {cites}".strip()
        elif cites:
            piece = f"{buf_part[:-1]} {cites}{buf_part[-1]}"
        buf = f"{buf} {piece}".strip()
        if c.closed:
            if not buf.endswith((".", "!", "?", ";", ":")):
                buf += "."
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf if buf.endswith((".", "!", "?")) else buf + ".")
    return out


def _fits(ans: str) -> bool:
    return V.count_words(ans) <= V.MAX_WORDS


def _check(seed: dict, ans: str, mode: str) -> V.Verdict | None:
    v = V.verify(seed["query"], seed["passages"], ans, seed["answerable"])
    if v.score == 0.0 and not v.gates[MODE_GATE[mode]]:
        return v
    return None


def leak_candidates(seed: dict, sid: str) -> list[str]:
    ctx = set(V.content_stems(seed["query"] + " " + " ".join(seed["passages"])))
    pst = set(V.content_stems(" ".join(seed["passages"])))
    ranked = []
    for fact, topic in WORLD_FACTS:
        fs = V.content_stems(fact)
        support = sum(1 for s in fs if s in pst) / max(1, len(fs))
        if support >= V.GROUNDING_MIN:
            continue  # the passages actually support it -> not a leak
        ranked.append((-len(topic & ctx), _h(sid, fact), fact))
    return [f for _, _, f in sorted(ranked)]


def make_leak(seed: dict, sid: str) -> tuple[str, V.Verdict] | None:
    gold, k = seed["gold_answer"], len(seed["passages"])
    c = _main_cite(gold, k)
    base = _sentences(gold) if not V.is_abstention(gold) else []
    for fact in leak_candidates(seed, sid):
        leak = f"{fact} [{c}]."
        for pos in sorted(range(len(base) + 1), key=lambda p: _h(sid, fact, str(p))):
            parts = base[:pos] + [leak] + base[pos:]
            while len(parts) > 1 and not _fits(" ".join(parts)):
                parts.pop(-1 if pos < len(parts) - 1 else 0)
            ans = " ".join(parts)
            v = _check(seed, ans, "parametric_world_memory_leak")
            if v:
                return ans, v
    return None


def make_procedural(seed: dict, sid: str) -> tuple[str, V.Verdict] | None:
    gold, passages = seed["gold_answer"], seed["passages"]
    k = len(passages)
    c = _main_cite(gold, k)
    all_stems = set(V._stems(V.tokenize(" ".join(passages))))
    verbs = [v for v in HAZARD_ORDER
             if not set(V._stems(V.tokenize(v))) <= all_stems]
    phrases = noun_phrases(passages[c - 1]) or ["system"]
    base = _sentences(gold) if not V.is_abstention(gold) else []
    order = sorted(((verb, obj, tpl) for verb in verbs for obj in phrases
                    for tpl in INJECT_TEMPLATES),
                   key=lambda x: (-len(x[1].split()), HAZARD_ORDER.index(x[0]) // 4,
                                  _h(sid, *x)))
    fallback = None
    for verb, obj, tpl in order[:400]:
        step = tpl.format(verb=verb, obj=obj, cite=f"[{c}]")
        pos = min(1, len(base))
        parts = base[:pos] + [step] + base[pos:]
        while len(parts) > 1 and not _fits(" ".join(parts)):
            parts.pop(-1)
        ans = " ".join(parts)
        v = _check(seed, ans, "hallucinated_procedural_step")
        if v is None:
            continue
        # hardest negative: lexically grounded enough to pass gate 1, caught
        # only by the procedural-safety gate
        if v.gates["citation_grounding"]:
            return ans, v
        fallback = fallback or (ans, v)
    return fallback


def noun_phrases(text: str) -> list[str]:
    """Full determiner-led noun phrases (1-3 words) from a passage.

    'the door interlock' -> 'door interlock'.  Runs stop at punctuation and
    must end right before a function word / adverb / clause end, so partial
    phrases such as 'maximum working' are not produced.
    """
    out: list[str] = []
    for chunk in re.split(r"[.,;:!?()]", text):
        toks = V.tokenize(chunk)
        for i, t in enumerate(toks):
            if t not in {"the", "a", "an", "their", "its", "your", "each"}:
                continue
            run: list[str] = []
            j = i + 1
            while j < len(toks) and len(run) < 4:
                w = toks[j]
                if (not w.isalpha() or w in V.STOPWORDS or w in V.META
                        or w in V.NEGATIONS or w in V.SAFETY_ACTIONS
                        or w.endswith("ly")):
                    break
                run.append(w)
                j += 1
            if not run or len(run) > 3:
                continue
            nxt = toks[j] if j < len(toks) else None
            if nxt is None or nxt in V.STOPWORDS or nxt.endswith("ly") \
                    or nxt in V.SAFETY_ACTIONS:
                phrase = " ".join(run)
                if phrase not in out:
                    out.append(phrase)
    return out


def make_over_refusal(seed: dict, sid: str) -> tuple[str, V.Verdict] | None:
    if not seed["answerable"]:
        return None
    for ans in sorted(OVER_REFUSALS, key=lambda a: _h(sid, a)):
        v = _check(seed, ans, "over_refusal")
        if v:
            return ans, v
    return None


MAKERS = {"parametric_world_memory_leak": make_leak,
          "hallucinated_procedural_step": make_procedural,
          "over_refusal": make_over_refusal}


def synthesize(seeds: list[dict]) -> tuple[list[dict], dict]:
    pairs: list[dict] = []
    skipped: list[dict] = []
    for n, raw in enumerate(seeds, 1):
        seed = {"query": raw["query"], "passages": list(raw["passages"]),
                "gold_answer": raw["gold_answer"]}
        sid = seed_id(raw, n)
        seed["answerable"] = raw.get("answerable",
                                     not V.is_abstention(seed["gold_answer"]))
        try:
            gv = V.verify(seed["query"], seed["passages"], seed["gold_answer"],
                          seed["answerable"])
        except ValueError as e:
            skipped.append({"seed_id": sid, "reason": f"invalid seed: {e}"})
            continue
        if not gv.passed:
            skipped.append({"seed_id": sid, "reason": "gold fails verifier",
                            "details": gv.reasons})
            continue
        for mode in FAILURE_MODES:
            res = MAKERS[mode](seed, sid)
            if res is None:
                skipped.append({"seed_id": sid, "mode": mode,
                                "reason": "not applicable" if mode == "over_refusal"
                                and not seed["answerable"] else "no verified negative"})
                continue
            rejected, rv = res
            pairs.append({
                "pair_id": f"{sid}:{mode}", "seed_id": sid, "failure_mode": mode,
                "query": seed["query"], "passages": seed["passages"],
                "answerable": seed["answerable"],
                "prompt": render_prompt(seed["query"], seed["passages"]),
                "chosen": seed["gold_answer"], "rejected": rejected,
                "chosen_score": gv.score, "rejected_score": rv.score,
                "rejected_failed_gates": [g for g in V.GATES if not rv.gates[g]],
                "rejected_reasons": rv.reasons,
            })
    stats = {"seeds": len(seeds), "pairs": len(pairs),
             "by_mode": dict(sorted(Counter(p["failure_mode"] for p in pairs).items())),
             "skipped": skipped}
    return pairs, stats


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def dumps_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Synthesize verified hard-negative DPO pairs.")
    ap.add_argument("--seeds", default=str(Path(__file__).with_name("examples") / "seeds.jsonl"))
    ap.add_argument("--out", default="-", help="output JSONL path ('-' = stdout)")
    args = ap.parse_args(argv)
    pairs, stats = synthesize(load_jsonl(args.seeds))
    text = dumps_jsonl(pairs)
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text, encoding="utf-8")
    stats["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(json.dumps(stats, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
