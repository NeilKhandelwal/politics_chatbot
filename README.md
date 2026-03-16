# Political Events AI Chatbot

LLM-powered agent that analyzes political questions with a multi-step reasoning pipeline, bias and fact checkers, and an evaluation suite. 

---

## Features and Functionality

-   **LLM-Reasoned Scope Classification**: Determines whether queries are political through language model. Off-topic queries short-circuit the pipeline with a reasoned redirect.
-   **6-Step Reasoning Pipeline**: Every political query flows through a sequential pipeline with conditional steps and a re-synthesis loop (see [Reasoning Pipeline](#reasoning-pipeline) below).
-   **Multi-Perspective Synthesis**: Builds balanced perspectives across partisan, institutional, and affected-population lenses. Claims not grounded in search results are tagged `[model knowledge]` for transparency.
-   **Bias Self-Audit**: 6-prompt swap-test style audit (slant, depth parity, tone, omission, pre-mortem, metacognition) runs before the final response. If imbalance is detected, Step 3 re-synthesizes with correction notes injected before proceeding.
-   **Hallucination Prevention**: Tavily search grounding with explicit gap tracking; a dedicated fact-check step (Step 6) compares the final prose against retrieved evidence and corrects unsupported claims with qualified language.
-   **Multi-Turn Conversational Intelligence**: Conversation history is threaded through all pipeline prompts so follow-up questions resolve correctly against prior context.
-   **Reasoning Trace**: Every response exposes a full step-by-step audit trail in the UI. Expand the **Reasoning Trace** panel after any response to inspect the pipeline.
-   **Three-Layer Evaluation Suite**: SBERT anchor-based stance scoring, LLM framing audit, and argument quality scoring that are compared for compound analysis. 

---

## Quickstart
```bash
git clone https://github.com/NeilKhandelwal/politics_chatbot.git
cd politics_chatbot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then add keys
```

Environment (.env):
- `GEMINI_API_KEY` (required) — https://aistudio.google.com/apikey
- `TAVILY_API_KEY` (required) — https://app.tavily.com/
- `GEMINI_MODEL` (optional) — override model (default: `gemini-2.5-flash`)

Run the chatbot UI:
```bash
python app.py
```
Open `http://localhost:7860`, ask a question, and expand **Reasoning Trace** to inspect the steps.

---

## Reasoning Pipeline (current implementation)
File: `src/reasoning.py`

1) **Classify & Scope** — Decide if political, set query type, and craft search queries.
2) **Search & Gather** *(skipped when classifier sets `needs_search=False` for data-lookup queries)* — Tavily search, organize into `verified_facts`, `contested_claims`, `gaps`, `temporal_note`. Falls back to model knowledge if Tavily is unavailable.
3) **Multi-Perspective Synthesis** — Build perspectives, agreements, disagreements, uncertainty, confidence level.
4) **Bias Self-Audit** *(skipped when `needs_bias_audit=False` for purely factual queries)* — 6-prompt swap-test checks; if imbalance detected, re-runs Step 3 with corrections (max 1 iteration).
5) **Generate Response** — Final answer incorporating synthesis + audit corrections.
6) **Fact-Check** *(when search results exist)* — Compares response prose against retrieved evidence; corrects unsupported claims in-place.

If a step fails to return valid JSON, the pipeline falls back to a simpler balanced-response prompt (logged in the trace). Boundary cases (non-political) short-circuit after Step 1 with a polite redirect.

Prompts are in `src/prompts.py`.

---

## Evaluation Suite
Directory: `eval/`

- `behavior_test.py` — Live pipeline mechanics tests: scope classification, steps executed, required scenarios, and response content assertions. Runs independently (no `results.json` needed).
- `run_eval.py` — Runs symmetric swap pairs from `eval/pairs.json` through the pipeline; saves raw responses to `eval/results.json`. Required before running the three analysis layers below.
- `score.py` — SBERT anchor-based cosine distance on response pairs (Bang et al. 2024); flags stance asymmetry by magnitude.
- `judge.py` — LLM framing audit using a decomposed 5-dimension rubric (attribution verbs, hedging, tone, lead framing, depth parity); cross-validates framing direction against SBERT.
- `quality.py` — LLM argument quality scoring (specificity, relevance, grounding); flags pairs with significant quality gaps between swap responses.
- `suite.py` — One-click orchestrator: regenerates `results.json` if stale, runs behavior tests, runs analysis layers silently, then prints quality summary and cross-layer rubric findings. Exits non-zero if behavior tests fail or any pair errors out.
- `common.py` — Shared constants and helpers used by the analysis layers.

Recommended (from repo root):
```bash
python eval/suite.py
```
Requires `GEMINI_API_KEY` and `TAVILY_API_KEY`. Run individual modules directly for full per-pair detail.

---

## Alignment with Project Spec (`docs/LLM_Project.md`)
- **No keyword classification**: classification is LLM-reasoned (Step 1).
- **Bias mitigation**: swap-test inspired audit (Step 4) plus symmetric prompts.
- **Hallucination control**: search grounding when available; explicit `gaps` surfaced; confidence level and uncertainty notes in synthesis; boundary guardrail for off-scope asks.
- **Conversational intelligence**: history is threaded through prompts; non-political queries get reasoned redirects.
- **Evaluation rigor**: swap-test scoring (SBERT), framing audit (LLM judge), quality check, and behavior tests for the required scenarios.

---


## Notes & Known Limitations
- The pipeline depends on the model returning strict JSON for Steps 2–4; malformed JSON triggers the fallback responder. If this happens often, lower temperature or tighten prompts in `src/reasoning.py`.
- Without `TAVILY_API_KEY`, the pipeline skips search and synthesizes from model knowledge only — responses will be less grounded in current information.
- Evaluation layers call external APIs; expect latency and token usage.

---

## Project Structure
```
app.py                # Gradio web interface
requirements.txt      # Python dependencies
src/
  reasoning.py        # 6-step pipeline orchestration and data classes
  prompts.py          # Prompt templates for all pipeline steps
eval/
  pairs.json          # Symmetric swap-test pair definitions
  behavior_test.py    # Live pipeline mechanics tests
  run_eval.py         # Run pairs through pipeline, write results.json
  score.py            # SBERT anchor-based stance scoring
  judge.py            # LLM framing audit (decomposed 5-dim rubric)
  quality.py          # Argument quality scoring
  suite.py            # One-click eval orchestrator
  common.py           # Shared constants and helpers
docs/
  LLM_Project.md      # Assignment spec and rubric
```
