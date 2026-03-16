"""
Shared constants and helpers for eval modules.
"""

import os

# Eval-specific model constant — intentionally separate from src/reasoning.py's MODEL.
# Eval modules import from here; the pipeline imports from its own module.
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _has_pipeline_failure(pair: dict) -> bool:
    return (
        pair.get("pipeline_error_a") or pair.get("pipeline_error_b") or
        pair["response_a"].startswith("ERROR") or pair["response_b"].startswith("ERROR")
    )


def _print_pipeline_reliability(failures: list, total: int) -> None:
    if failures:
        print("\n---- PIPELINE RELIABILITY ----")
        print(f"Pairs skipped due to pipeline failure: {len(failures)} / {total}")
        for p in failures:
            sides = []
            if p.get("pipeline_error_a") or p["response_a"].startswith("ERROR"):
                stages = p.get("error_stages_a") or []
                sides.append("A" + (f" ({stages[0]})" if stages else ""))
            if p.get("pipeline_error_b") or p["response_b"].startswith("ERROR"):
                stages = p.get("error_stages_b") or []
                sides.append("B" + (f" ({stages[0]})" if stages else ""))
            print(f"  {p['id']} ({p['category']}) — side(s) {', '.join(sides)} failed")
        print("Note: these pairs are excluded from all metrics above.")
    else:
        print(f"\nPipeline reliability: all {total} pairs completed successfully.")
