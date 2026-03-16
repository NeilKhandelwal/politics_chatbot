"""
Unified eval suite

Single command that runs the full eval pipeline from scratch:
    1. run_eval      — runs swap-test pairs through the chatbot, writes results.json
                       (skipped if results.json exists and contains all current pairs)
    2. behavior_test — pipeline mechanics (steps, classification, scope)
    3. score         — anchor-based stance scoring (Bang et al. 2024)
    4. judge         — LLM framing audit (framing direction + mechanism)
    5. quality       — LLM argument quality (specificity, relevance, grounding)

Then cross-joins layers 3-5 on pair_id to flag compound issues.

Requires: GEMINI_API_KEY, TAVILY_API_KEY

Exit code: 0 if all behavior tests pass and no pipeline errors exist.
           1 if any behavior test fails or any pipeline error remains.

Slant/framing/quality metrics are informational — not pass/fail thresholds.
"""

import contextlib
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(override=True)

import behavior_test
import run_eval as run_eval_mod
import score as score_mod
import judge as judge_mod
import quality as quality_mod

from score import CATEGORY_THRESHOLDS, DEFAULT_THRESHOLD
from judge import CLEAR_FRAMING_THRESHOLD, MILD_FRAMING_THRESHOLD, DIMS, _count_dims
from quality import QUALITY_GAP_THRESHOLD

# Display order for compound outcomes — most actionable first.
# Used both in _cross_join documentation and run_suite() output sorting.
_OUTCOME_PRIORITY = {"COMPOUND_CONFIRMED": 0, "CONTRADICTION": 1, "LINGUISTIC_ONLY": 2, "SBERT_ONLY": 3}

_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")
_PAIRS_PATH = os.path.join(os.path.dirname(__file__), "pairs.json")


def _needs_rerun() -> tuple[bool, str]:
    if not os.path.exists(_RESULTS_PATH):
        return True, "results.json not found"
    with open(_PAIRS_PATH) as f:
        expected = {p["id"] for p in json.load(f)}
    with open(_RESULTS_PATH) as f:
        existing = {r["id"] for r in json.load(f)}
    new_ids = expected - existing
    if new_ids:
        return True, f"pairs.json has new pair(s): {', '.join(sorted(new_ids))}"
    stale_ids = existing - expected
    if stale_ids:
        # results.json contains pairs no longer in pairs.json; rerun to stay in sync.
        return True, f"results.json has stale pair(s) removed from pairs.json: {', '.join(sorted(stale_ids))}"
    return False, ""


def _cross_join(scores: list, judge_valid: list, quality_valid: list) -> list[dict]:
    """Merge score + judge + quality on pair_id. Classifies each pair with signal into
    one of 4 compound outcome types:

        COMPOUND_CONFIRMED — SBERT moderate/HIGH + judge flags in the SAME direction
        CONTRADICTION      — SBERT moderate/HIGH + judge flags in OPPOSITE direction
        LINGUISTIC_ONLY    — SBERT low/balanced  + judge flags clearly (|score| >= CLEAR threshold)
        SBERT_ONLY         — SBERT HIGH           + judge balanced (possible anchor artifact)

    Contradictions are always surfaced: they represent cases where vocabulary coverage
    (SBERT) and linguistic framing (judge) diverge — both may be genuine and their
    co-existence is itself a finding worth reviewing.

    Note on category thresholds: SBERT uses per-category HIGH thresholds (e.g.,
    historical_events HIGH=0.35 vs policy_positions HIGH=0.20). The judge threshold
    is global, so historical pairs face a stricter SBERT bar before compounding.
    """
    score_by_id   = {s["id"]: s for s in scores}
    judge_by_id   = {r["id"]: r for r in judge_valid}
    quality_by_id = {r["id"]: r for r in quality_valid}

    compound = []
    for pid, s in score_by_id.items():
        j = judge_by_id.get(pid)
        q = quality_by_id.get(pid)
        if not j or not q:
            continue

        category      = s["category"]
        threshold     = CATEGORY_THRESHOLDS.get(category, DEFAULT_THRESHOLD)
        high_sbert    = s["cosine_distance"] >= threshold["high"]
        sbert_lean    = s.get("lean", "balanced")
        sbert_flagged = s["slant_level"] in ("MODERATE", "HIGH")

        framing_score = j.get("framing_score", 0)
        judge_lean    = j.get("judge_lean", "balanced")
        judge_flagged = abs(framing_score) >= MILD_FRAMING_THRESHOLD

        qa = q["score_a"].get("overall_quality", 0) if "score_a" in q else 0
        qb = q["score_b"].get("overall_quality", 0) if "score_b" in q else 0
        quality_gap_val = abs(qa - qb)
        quality_gap     = quality_gap_val >= QUALITY_GAP_THRESHOLD

        # Classify outcome type and apply per-type entry conditions
        if sbert_flagged and judge_flagged:
            # sbert_lean == judge_lean checks directional agreement.
            # Both being "balanced" can't happen here: sbert_flagged requires slant_level
            # in ("MODERATE","HIGH") and judge_flagged requires abs(framing_score) >= 1,
            # so at least one side must have a non-balanced lean.
            outcome = "COMPOUND_CONFIRMED" if sbert_lean == judge_lean else "CONTRADICTION"
        elif judge_flagged and abs(framing_score) >= CLEAR_FRAMING_THRESHOLD:
            outcome = "LINGUISTIC_ONLY"
        elif high_sbert and not judge_flagged:
            # Anchor artifact: SBERT flagged because the generated anchor sentences were
            # asymmetric (style/vocabulary mismatch), not because bot responses were biased.
            # The judge is the tiebreaker — neutral judge here means treat with caution.
            outcome = "SBERT_ONLY"
        else:
            continue  # no meaningful signal in any layer

        evidence = j.get("evidence", [])
        fl, fr = _count_dims(j)
        compound.append({
            "id": pid,
            "category": category,
            "outcome": outcome,
            # Layer trigger booleans
            "high_sbert": high_sbert,
            "quality_gap": quality_gap,
            # SBERT metrics
            "cosine_distance": s["cosine_distance"],
            "slant_level": s["slant_level"],
            "sbert_lean": sbert_lean,
            "sbert_threshold": threshold["high"],
            # Judge metrics
            "framing_score": framing_score,
            "judge_lean": judge_lean,
            "judge_dims_left": fl,
            "judge_dims_right": fr,
            "overall": j.get("overall", "?"),
            "evidence_sample": evidence[0] if evidence else None,
            # Quality metrics
            "quality_a": qa,
            "quality_b": qb,
            "quality_gap_val": quality_gap_val,
        })

    return compound


def run_suite():
    print(f"EVAL SUITE  --  {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Generate/regenerate results.json silently if missing or stale
    needs_run, reason = _needs_rerun()
    if needs_run:
        print(f"  Regenerating results.json ({reason})...")
        with contextlib.redirect_stdout(io.StringIO()):
            run_eval_mod.run_pairs()

    results_age = datetime.fromtimestamp(os.path.getmtime(_RESULTS_PATH)).strftime("%Y-%m-%d %H:%M")
    print(f"  results.json last generated: {results_age}")

    # Behavior tests — keep verbose output (per-test pass/fail is the signal)
    print("\n--- BEHAVIOR TESTS ---")
    behavior_results = behavior_test.run_all()
    b_total  = len(behavior_results)
    b_passed = sum(1 for r in behavior_results if r["passed"])

    # Score / judge / quality — run silently; suite shows only distilled results
    _quiet = io.StringIO()
    with contextlib.redirect_stdout(_quiet):
        scores,        score_failures  = score_mod.score_results(_RESULTS_PATH)
        judge_valid,   judge_failures  = judge_mod.run_judge(results_path=_RESULTS_PATH, sbert_scores=scores)
        quality_valid, quality_failures = quality_mod.run_quality(results_path=_RESULTS_PATH)

    compound = _cross_join(scores, judge_valid, quality_valid)
    # Deduplicate by pair ID: a pair that fails in multiple eval layers would otherwise
    # appear multiple times in the failure output.
    seen_failure_ids: set[str] = set()
    all_failure_records = []
    for rec in score_failures + judge_failures + quality_failures:
        if rec["id"] not in seen_failure_ids:
            seen_failure_ids.add(rec["id"])
            all_failure_records.append(rec)
    all_failures = seen_failure_ids

    # --- Quality summary ---
    print("\n--- QUALITY + RUBRIC ---")
    if quality_valid:
        all_q = [r["score_a"]["overall_quality"] for r in quality_valid
                 if "score_a" in r and "overall_quality" in r.get("score_a", {})]
        all_q += [r["score_b"]["overall_quality"] for r in quality_valid
                  if "score_b" in r and "overall_quality" in r.get("score_b", {})]
        gaps = [r for r in quality_valid
                if "score_a" in r and "score_b" in r
                and abs(r["score_a"].get("overall_quality", 0)
                        - r["score_b"].get("overall_quality", 0)) >= QUALITY_GAP_THRESHOLD]
        if all_q:
            print(f"Quality:  mean={sum(all_q)/len(all_q):.2f}   gaps>={QUALITY_GAP_THRESHOLD}: {len(gaps)}")

    # --- Rubric findings (cross-layer) ---
    if compound:
        outcome_counts = Counter(c["outcome"] for c in compound)
        summary_parts  = [f"{k}: {v}" for k, v in sorted(outcome_counts.items(),
                           key=lambda x: _OUTCOME_PRIORITY.get(x[0], 99))]
        n_contradict   = outcome_counts.get("CONTRADICTION", 0)

        print(f"\nRUBRIC FINDINGS  ({len(compound)} pair(s) -- {', '.join(summary_parts)})")
        if n_contradict:
            print(f"  Caution: {n_contradict} contradiction(s) -- SBERT and judge disagree; review manually")
        print("=" * 58)

        for i, c in enumerate(sorted(compound, key=lambda x: _OUTCOME_PRIORITY.get(x["outcome"], 99))):
            note = ""
            if c["outcome"] == "CONTRADICTION":
                note = f"  -- SBERT lean={c['sbert_lean']} vs Judge lean={c['judge_lean']}"
            elif c["outcome"] == "SBERT_ONLY":
                note = "  -- verify anchor quality"
            print(f"[{c['outcome']}]  {c['id']}  [{c['category']}]{note}")

            sbert_str = f"SBERT {c['cosine_distance']:.4f}"
            if c["high_sbert"]:
                sbert_str += f" >= {c['sbert_threshold']}"
            sbert_str += f" [{c['slant_level']}] lean={c['sbert_lean']}"

            judge_str  = (f"Judge {c['framing_score']:+d}"
                          f" (L{c['judge_dims_left']}/R{c['judge_dims_right']})"
                          f" lean={c['judge_lean']}")
            gap_tag    = " [GAP]" if c["quality_gap"] else ""
            quality_str = f"quality gap={c['quality_gap_val']:.2f}{gap_tag}"

            print(f"  {sbert_str}  |  {judge_str}  |  {quality_str}")
            if i < len(compound) - 1:
                print("-" * 58)
        print("=" * 58)
    else:
        print("\nNo multi-layer threshold crossings.")

    # --- Exit summary ---
    overall_pass = (b_passed == b_total) and not all_failures
    b_status = "Pass" if b_passed == b_total else "FAIL"
    e_status = "Pass" if not all_failures else "Error"
    print(f"\nBehavior: {b_status} ({b_passed}/{b_total})   Pipeline errors: {e_status} ({len(all_failures)} pair(s))")
    if all_failure_records:
        for rec in all_failure_records:
            msg = rec.get("error") or rec.get("skip_reason") or "pipeline failure (check results.json)"
            print(f"  pair {rec['id']}: {msg}")
    print(f"EXIT: {'PASS' if overall_pass else 'FAIL'}")

    print("\nFull per-pair results (SBERT scores, framing dimensions, quality breakdowns):")
    print("  python eval/score.py     -- SBERT stance scoring")
    print("  python eval/judge.py     -- LLM framing audit")
    print("  python eval/quality.py   -- argument quality scores")
    print()

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    run_suite()
