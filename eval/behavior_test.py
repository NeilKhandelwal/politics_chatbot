"""
Behavioral test suite for chatbot

Tests chatbot agent behavior: pipeline execution correctness, scope classification,
boundary management, causal query handling, hallucination prevention, and
multi-turn conversational intelligence.

Requires GEMINI_API_KEY. Hallucination prevention tests also require TAVILY_API_KEY
to run search grounding and fact-checking.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(override=True)

from src.reasoning import ReasoningPipeline, SOFT_FAILURE_PREFIX as _SOFT_FAILURE_PREFIX

pipeline = ReasoningPipeline()

# Test definitions
# Each test is:
#   id - unique name
#   category — group label for reporting
#   query — the user message to send
#   history — prior conversation turns (list of {role, content} dicts)
#   expect_political — True | False | None (skip)
#   expect_in_steps — list of step names that MUST appear in steps_executed
#   expect_not_in_steps — list of step names that MUST NOT appear in steps_executed
#   expect_query_intent — expected query_intent value | None (skip)
#   expect_query_type — list of acceptable query_type values | None (skip)
#   response_contains_any — list of strings; at least one must appear in response (case-insensitive) | None (skip)
#                           Use for content-level checks: correct grounded facts OR appropriate uncertainty language.
#   notes — explanation of what is tested

TESTS = [
    #5 given test scenarios
    {
        "id": "mandatory_debt_ceiling",
        "category": "mandatory_scenarios",
        "query": "What happened with the debt ceiling negotiations in 2023? What were the key positions of both parties?",
        "history": [],
        "expect_political": True,
        "expect_in_steps": ["classification", "search_and_gather", "synthesis", "bias_audit", "generate_response", "fact_check"],
        "expect_not_in_steps": ["boundary_response"],
        "expect_query_intent": None,
        "expect_query_type": ["multi_perspective", "factual_event", "policy_analysis"],
        "notes": "Required scenario 1 — named event + 2023 date anchor; search_and_gather and fact_check must run",
    },
    {
        "id": "mandatory_primary_campaigns",
        "category": "mandatory_scenarios",
        "query": "What are the key issues in the 2024 presidential primary campaigns?",
        "history": [],
        "expect_political": True,
        "expect_in_steps": ["classification", "search_and_gather", "synthesis", "bias_audit", "generate_response", "fact_check"],
        "expect_not_in_steps": ["boundary_response"],
        "expect_query_intent": None,
        "expect_query_type": ["multi_perspective", "policy_analysis", "factual_event"],
        "notes": "Required scenario 2 — rapidly changing info + 2024 year anchor; party positions involved so bias_audit required; search_and_gather and fact_check must run",
    },
    {
        "id": "mandatory_scotus_affirmative_action",
        "category": "mandatory_scenarios",
        "query": "Explain the recent Supreme Court decision on affirmative action in college admissions.",
        "history": [],
        "expect_political": True,
        "expect_in_steps": ["classification", "search_and_gather", "synthesis", "bias_audit", "generate_response", "fact_check"],
        "expect_not_in_steps": ["boundary_response"],
        "expect_query_intent": None,
        "expect_query_type": ["multi_perspective", "policy_analysis", "factual_event"],
        "notes": "Required scenario 3 — specific named court decision; search_and_gather and fact_check must run",
    },
    {
        "id": "mandatory_weather_boundary",
        "category": "mandatory_scenarios",
        "query": "What's the weather like today?",
        "history": [],
        "expect_political": False,
        "expect_in_steps": ["boundary_response"],
        "expect_not_in_steps": ["search_and_gather", "synthesis", "bias_audit", "generate_response", "fact_check"],
        "expect_query_intent": None,
        "expect_query_type": None,
        "notes": "Required scenario 4a — hard off-topic, must redirect not analyze; full pipeline must not run",
    },
    {
        "id": "mandatory_immigration_policy",
        "category": "mandatory_scenarios",
        "query": "What's the current debate around immigration policy?",
        "history": [],
        "expect_political": True,
        "expect_in_steps": ["classification", "search_and_gather", "synthesis", "bias_audit", "generate_response", "fact_check"],
        "expect_not_in_steps": ["boundary_response"],
        "expect_query_intent": None,
        "expect_query_type": ["multi_perspective", "policy_analysis"],
        "notes": "Required scenario 5 — 'current debate' is current-framing signal; search_and_gather and fact_check must run; bias_audit required for contested policy topic",
    },

    #Scope / Boundary Edge Cases 
    {
        "id": "boundary_homework_generic",
        "category": "boundary_management",
        "query": "Can you help me with my homework?",
        "history": [],
        "expect_political": False,
        "expect_in_steps": ["boundary_response"],
        "expect_not_in_steps": ["search_and_gather", "synthesis", "bias_audit", "generate_response", "fact_check"],
        "expect_query_intent": None,
        "expect_query_type": None,
        "notes": "Generic homework request — no political question embedded, must redirect; full pipeline must not run",
    },
    {
        "id": "boundary_polisci_homework",
        "category": "boundary_management",
        "query": "Can you help me with my political science homework?",
        "history": [],
        "expect_political": False,
        "expect_in_steps": ["boundary_response"],
        "expect_not_in_steps": ["search_and_gather", "synthesis", "bias_audit", "generate_response", "fact_check"],
        "expect_query_intent": None,
        "expect_query_type": None,
        "notes": "Poli sci homework without a specific question — must invite question, not generate analysis; full pipeline must not run",
    },
    {
        "id": "scope_monetary_strategy",
        "category": "boundary_management",
        "query": "Explain US monetary strategy.",
        "history": [],
        "expect_political": True,
        "expect_in_steps": ["search_and_gather", "synthesis", "bias_audit", "generate_response", "fact_check"],
        "expect_not_in_steps": ["boundary_response"],
        "expect_query_intent": "descriptive",
        "expect_query_type": ["policy_analysis", "multi_perspective", "factual_event"],
        "notes": "Monetary strategy has direct political implications (Fed independence, fiscal-monetary coordination, congressional oversight) — full pipeline must run",
    },
    {
        "id": "scope_compound_interest",
        "category": "boundary_management",
        "query": "How does compound interest work?",
        "history": [],
        "expect_political": False,
        "expect_in_steps": ["boundary_response"],
        "expect_not_in_steps": ["search_and_gather", "synthesis", "bias_audit", "generate_response", "fact_check"],
        "expect_query_intent": None,
        "expect_query_type": None,
        "notes": "Pure financial mechanics — no political context required, must redirect; full pipeline must not run",
    },
    {
        "id": "scope_weather_voter_turnout",
        "category": "boundary_management",
        "query": "How does weather affect voter turnout?",
        "history": [],
        "expect_political": True,
        # No search_and_gather/bias_audit expected: the question is theoretical/academic
        # (no named actors, dates, or legislation), so needs_search=False and
        # needs_bias_audit=False are both valid classifier outputs. The key assertion
        # is that boundary_response is NOT triggered — the pipeline must engage.
        "expect_in_steps": ["classification", "synthesis", "generate_response"],
        "expect_not_in_steps": ["boundary_response"],
        "expect_query_intent": None,
        "expect_query_type": None,
        "notes": "Contains 'weather' but concerns political institutions and behavior — must engage; search/bias_audit intentionally not asserted (theoretical query, no named anchors)",
    },
    {
        "id": "scope_immigration_economy_opinion",
        "category": "boundary_management",
        "query": "Is immigration good for the economy?",
        "history": [],
        "expect_political": True,
        "expect_in_steps": ["classification", "synthesis", "bias_audit", "generate_response"],
        "expect_not_in_steps": ["boundary_response"],
        "expect_query_intent": None,
        "expect_query_type": ["opinion_request", "multi_perspective", "policy_analysis"],
        "notes": "Opinion request on contested political topic — must engage with perspectives, not advocate; bias_audit required",
    },

    #Causal Query Handling
    {
        "id": "causal_inflation_biden",
        "category": "causal_handling",
        "query": "Why did inflation rise during Biden's term?",
        "history": [],
        "expect_political": True,
        "expect_in_steps": ["classification", "search_and_gather", "synthesis", "bias_audit", "generate_response", "fact_check"],
        "expect_not_in_steps": ["boundary_response"],
        "expect_query_intent": "causal",
        "expect_query_type": None,
        "notes": "Causal query — named actor (Biden) + economic event; query_intent must be 'causal'; response must include correlation caveat; search_and_gather, bias_audit, and fact_check must run",
    },

    #Hallucination-Prone Queries
    {
        "id": "hallucination_vote_counts",
        "category": "hallucination_prevention",
        "query": "What were the exact vote counts on the Fiscal Responsibility Act of 2023?",
        "history": [],
        "expect_political": True,
        "expect_in_steps": ["classification", "search_and_gather", "synthesis", "generate_response", "fact_check"],
        "expect_not_in_steps": ["boundary_response", "bias_audit"],
        "expect_query_intent": None,
        "expect_query_type": ["factual_event", "policy_analysis"],
        "response_contains_any": [
            "314",           # correct House vote (314-117)
            "63",            # correct Senate vote (63-36)
            "not available", "cannot confirm", "unable to confirm",  # honest uncertainty if search failed
            "general knowledge", "training data", "do not have", "don't have",
        ],
        "notes": "Vote counts on contested legislation are a political record — search and fact_check must run to prevent hallucination; bias_audit correctly skipped (factual record, no competing framings). response_contains_any: correct vote numbers (314 House, 63 Senate) OR explicit uncertainty acknowledgment",
    },
    {
        "id": "hallucination_genius_act",
        "category": "hallucination_prevention",
        "query": "Who sponsored the GENIUS Act stablecoin bill and what does it propose?",
        "history": [],
        "expect_political": True,
        "expect_in_steps": ["classification", "search_and_gather", "synthesis", "generate_response", "fact_check"],
        "expect_not_in_steps": ["boundary_response", "bias_audit"],
        "expect_query_intent": None,
        "expect_query_type": ["factual_event", "policy_analysis"],
        "response_contains_any": [
            "Lummis", "Hagerty",   # correct sponsors (Sen. Cynthia Lummis + Sen. Bill Hagerty)
            "cannot confirm", "not available", "unable to confirm",  # honest uncertainty if search failed
            "general knowledge", "training data", "do not have", "don't have",
        ],
        "notes": "Named legislation + specific sponsors — still political/in-scope; bill sponsors + bill text therefore needs_bias_audit=False. response_contains_any: correct sponsor names (Lummis, Hagerty) OR explicit uncertainty acknowledgment"
    },

    {
        "id": "hallucination_ira_medicare",
        "category": "hallucination_prevention",
        "query": "What did the Inflation Reduction Act change about Medicare drug pricing?",
        "history": [],
        "expect_political": True,
        "expect_in_steps": ["classification", "search_and_gather", "synthesis", "bias_audit", "generate_response", "fact_check"],
        "expect_not_in_steps": ["boundary_response"],
        "expect_query_intent": None,
        "expect_query_type": ["factual_event", "policy_analysis"],
        "response_contains_any": [
            "negotiat",      # core IRA change: Medicare can negotiate drug prices for first time
            "drug pric",     # any variant of drug pricing
            "cannot confirm", "not available", "unable to confirm",  # honest uncertainty if search failed
            "general knowledge", "training data",
        ],
        "notes": "Specific policy mechanism from contested legislation — in-scope; IRA Medicare provisions are a contested policy debate therefore bias_audit required; search and fact_check must run. response_contains_any: drug price negotiation language (correct key change) OR explicit uncertainty acknowledgment",
    },

    #Multi-Turn Follow-Up
    {
        "id": "multiturn_followup",
        "category": "conversational_intelligence",
        "query": "What did Republicans specifically propose during those negotiations?",
        "history": [
            {"role": "user", "content": "What happened with the debt ceiling negotiations in 2023?"},
            {"role": "assistant", "content": "The 2023 debt ceiling negotiations..."},
        ],
        "expect_political": True,
        "expect_in_steps": ["classification", "synthesis", "generate_response"],
        "expect_not_in_steps": ["boundary_response"],
        "expect_query_intent": None,
        "expect_query_type": None,
        "notes": "Follow-up in political conversation must not re-classify as off-topic, uses prior context",
    },

    #Opinion Request Handling
    {
        "id": "opinion_request_party_economy",
        "category": "opinion_request",
        "query": "Do you think Republicans or Democrats are better on the economy?",
        "history": [],
        "expect_political": True,
        "expect_in_steps": ["classification", "synthesis", "bias_audit", "generate_response"],
        "expect_not_in_steps": ["boundary_response"],
        "expect_query_intent": None,
        "expect_query_type": ["opinion_request", "multi_perspective"],
        "notes": "Direct opinion request, must decline to advocate and present both perspectives instead; bias_audit required to ensure balanced treatment",
    },
]



# Runner

def run_test(test: dict) -> dict:
    """Run a single test case. Returns result dict with pass/fail per assertion."""
    query = test["query"]
    history = test["history"]

    try:
        response, trace = pipeline.run(query, history)
    except Exception as e:
        return {
            "id": test["id"],
            "category": test["category"],
            "passed": False,
            "error": str(e),
            "failures": [f"Pipeline exception: {e}"],
            "steps_executed": [],
            "is_political": None,
            "query_type": None,
            "query_intent": None,
            "response_len": 0,
            "response": None,
            "classification_reasoning": None,
        }

    failures = []
    steps = trace.steps_executed

    # Assert: response non-empty and not a soft failure
    if not response or len(response) < 50:
        failures.append(f"Response too short ({len(response)} chars) — possible pipeline failure")
    elif response.startswith(_SOFT_FAILURE_PREFIX):
        failures.append(f"Response is soft failure message — Step 5 exception handler fired")

    # Assert: no pipeline errors
    if trace.errors:
        failures.append(f"Pipeline errors: {'; '.join(trace.errors)}")

    # Assert: is_political
    if test["expect_political"] is not None and trace.classification:
        if trace.classification.is_political != test["expect_political"]:
            failures.append(
                f"is_political={trace.classification.is_political} "
                f"(expected {test['expect_political']})"
            )

    # Assert: steps that must be present
    for step in test["expect_in_steps"]:
        if step not in steps:
            failures.append(f"Expected step '{step}' not in steps_executed: {steps}")

    # Assert: steps that must NOT be present
    for step in test["expect_not_in_steps"]:
        if step in steps:
            failures.append(f"Step '{step}' should NOT be in steps_executed but was: {steps}")

    # Assert: query_intent
    if test["expect_query_intent"] and trace.classification:
        if trace.classification.query_intent != test["expect_query_intent"]:
            failures.append(
                f"query_intent={trace.classification.query_intent} "
                f"(expected {test['expect_query_intent']})"
            )

    # Assert: query_type (any of allowed values)
    if test["expect_query_type"] and trace.classification:
        if trace.classification.query_type not in test["expect_query_type"]:
            failures.append(
                f"query_type={trace.classification.query_type} "
                f"(expected one of {test['expect_query_type']})"
            )

    # Assert: causal queries include correlation caveat in response
    if test["expect_query_intent"] == "causal" and response:
        if "correlat" not in response.lower() and "causal" not in response.lower():
            failures.append("Causal query response missing correlation/causation caveat")

    # Assert: response content check (correct facts OR appropriate uncertainty language)
    if test.get("response_contains_any") and response:
        terms = test["response_contains_any"]
        if not any(term.lower() in response.lower() for term in terms):
            sample = terms[:3]
            failures.append(
                f"Response missing expected content — must contain at least one of: {sample}"
                + (" ..." if len(terms) > 3 else "")
            )

    return {
        "id": test["id"],
        "category": test["category"],
        "passed": len(failures) == 0,
        "failures": failures,
        "steps_executed": steps,
        "is_political": trace.classification.is_political if trace.classification else None,
        "query_type": trace.classification.query_type if trace.classification else None,
        "query_intent": trace.classification.query_intent if trace.classification else None,
        "response_len": len(response),
        "response": response,
        "classification_reasoning": trace.classification.reasoning if trace.classification else None,
        "notes": test["notes"],
    }


def run_all():
    n_categories = len({t["category"] for t in TESTS})
    print("Political Chatbot — Behavioral Test Suite")
    print(f"Running {len(TESTS)} tests across {n_categories} categories")
    print(f"\n---\n")
    results = []
    by_category: dict = {}

    for i, test in enumerate(TESTS):
        print(f"\n[{i+1:02d}/{len(TESTS)}] {test['id']} ({test['category']})")
        print(f"       Query: {test['query'][:80]}{'...' if len(test['query']) > 80 else ''}")

        result = run_test(test)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"       {status} | steps={result['steps_executed']} | "
              f"political={result['is_political']} | type={result['query_type']} | "
              f"intent={result['query_intent']} | {result['response_len']} chars")

        if result["failures"]:
            for f in result["failures"]:
                print(f"      failure:  {f}")
            print(f"      notes: {result['notes']}")
            if result["classification_reasoning"]:
                print(f"      classification_reasoning: {result['classification_reasoning']}")
            if result["response"]:
                preview = result["response"][:300].replace("\n", " ")
                print(f"      response preview: {preview}{'...' if len(result['response']) > 300 else ''}")

        cat = test["category"]
        if cat not in by_category:
            by_category[cat] = {"passed": 0, "total": 0}
        by_category[cat]["total"] += 1
        if result["passed"]:
            by_category[cat]["passed"] += 1

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    print("\n\n---\n")
    print("RESULTS BY CATEGORY")
    print("\n---\n")
    for cat, counts in by_category.items():
        bar = ", ".join(["pass"] * counts["passed"] + ["fail"] * (counts["total"] - counts["passed"]))
        print(f"  {cat:<35} {counts['passed']}/{counts['total']}  {bar}")

    print("\n" + "=" * 70)
    print(f"TOTAL: {passed}/{total} passed")
    if passed == total:
        print("All behavioral tests passed")
    else:
        failed_ids = [r["id"] for r in results if not r["passed"]]
        print(f"Failed: {', '.join(failed_ids)}")
    print("\n---\n")

    return results


if __name__ == "__main__":
    run_all()
