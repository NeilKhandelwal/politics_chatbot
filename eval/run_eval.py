"""
Run swap-test pairs through the chatbot pipeline and collect response pairs.

Outputs results to eval/results.json
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(override=True)

from src.reasoning import ReasoningPipeline, SOFT_FAILURE_PREFIX as _SOFT_FAILURE_PREFIX, _BIAS_AUDIT_SKIPPED


def _is_failure(response: str) -> bool:
    return response.startswith("ERROR") or response.startswith(_SOFT_FAILURE_PREFIX)


def _run_with_retry(pipeline: ReasoningPipeline, query: str,
                    history: list = None) -> tuple[str, object, bool, list[str]]:
    """Run the pipeline, retrying once on soft failure before giving up.

    Returns (response, trace, pipeline_error, error_stages).
    pipeline_error=True means the response contains no usable content.
    error_stages is trace.errors from the final attempt.
    """
    if history is None:
        history = []
    response, trace = pipeline.run(query, history)
    if _is_failure(response):
        print(f"  soft failure detected — retrying once...")
        response, trace = pipeline.run(query, history)
        if _is_failure(response):
            response = f"ERROR: pipeline soft failure after retry for query: {query[:80]}"
            return response, trace, True, list(trace.errors)
    return response, trace, False, []


def _run_side(pipeline: ReasoningPipeline, query: str,
              history: list) -> tuple[str, dict, bool, list[str], list[str]]:
    """Run one side of a swap pair. Returns (response, audit, pipeline_error, error_stages, nonfatal_errors)."""
    try:
        response, trace, pipeline_error, error_stages = _run_with_retry(pipeline, query, history)
        audit = {}
        if trace.bias_audit:
            audit = {
                "depth_parity": trace.bias_audit.depth_parity_result,
                "overall": trace.bias_audit.overall_assessment,
                "corrections": trace.bias_audit.corrections_needed,
            }
        nonfatal_errors = list(trace.errors) if not pipeline_error else []
        return response, audit, pipeline_error, error_stages, nonfatal_errors
    except Exception as e:
        return f"ERROR: {e}", {}, True, [str(e)], []


def run_pairs(pairs_path: str = "eval/pairs.json", output_path: str = "eval/results.json"):
    pipeline = ReasoningPipeline()

    with open(pairs_path) as f:
        pairs = json.load(f)

    results = []
    for i, pair in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {pair['id']}...")

        # Supports both per-side history (history_a/history_b keys) and a shared
        # history key — falls back to [] if neither is present in the pair definition.
        history_a = pair.get("history_a", pair.get("history", []))
        history_b = pair.get("history_b", pair.get("history", []))

        response_a, audit_a, pipeline_error_a, error_stages_a, nonfatal_errors_a = _run_side(pipeline, pair["query_a"], history_a)
        response_b, audit_b, pipeline_error_b, error_stages_b, nonfatal_errors_b = _run_side(pipeline, pair["query_b"], history_b)

        results.append({
            "id": pair["id"],
            "category": pair["category"],
            "query_a": pair["query_a"],
            "query_b": pair["query_b"],
            "response_a": response_a,
            "response_b": response_b,
            "audit_a": audit_a,
            "audit_b": audit_b,
            "pipeline_error_a": pipeline_error_a,
            "pipeline_error_b": pipeline_error_b,
            "error_stages_a": error_stages_a,
            "error_stages_b": error_stages_b,
            # Relies on Step 5 prompt always beginning source notes with exactly "Source note:".
            "source_note_a": "Source note:" in response_a,
            "source_note_b": "Source note:" in response_b,
            "nonfatal_errors_a": nonfatal_errors_a,
            "nonfatal_errors_b": nonfatal_errors_b,
            "has_history": bool(history_a or history_b),
            # step4_skipped: uses the imported constant so this stays in sync if the sentinel wording changes.
            "step4_skipped_a": audit_a.get("depth_parity", "") == _BIAS_AUDIT_SKIPPED,
            "step4_skipped_b": audit_b.get("depth_parity", "") == _BIAS_AUDIT_SKIPPED,
        })

        dp_a = audit_a.get("depth_parity", "n/a")[:60] if audit_a else "n/a"
        dp_b = audit_b.get("depth_parity", "n/a")[:60] if audit_b else "n/a"
        print(f"  A: {len(response_a)} chars | B: {len(response_b)} chars")
        print(f"  Step4 A depth_parity: {dp_a}")
        print(f"  Step4 B depth_parity: {dp_b}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    failures = [(r, side) for r in results
                for side in ("a", "b") if r.get(f"pipeline_error_{side}")]
    if failures:
        print(f"\n---- PIPELINE RELIABILITY ----")
        # Set comprehension deduplicates pair IDs — a pair with both sides failing counts once.
        print(f"Failed: {len(failures)} side(s) across {len({r['id'] for r, _ in failures})} pair(s)")
        for r, side in failures:
            stages = r.get(f"error_stages_{side}", [])
            print(f"  {r['id']} — side {side.upper()} failed")
            for s in stages:
                print(f"    {s}")
    else:
        print(f"Pipeline reliability: all {len(results)} pairs completed successfully.")


if __name__ == "__main__":
    run_pairs()
