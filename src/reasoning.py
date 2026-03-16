"""
Execution reasoning pipeline

Steps 2, 4, and 6 are conditional:
    - Step 2 runs only if classification sets needs_search=True
    - Step 4 runs only if classification sets needs_bias_audit=True
    - Step 3->4 loops if step 4 sets requires_resynthesis=True
      Step 3 re-reruns with corrections injected, then goes back to step 4
    - Step 6 runs only if search data was gathered

Steps 1-4 and 6 use Gemini's JSON mode (response_mime_type="application/json") 
structured output, no markdown fences, no parsing hacks.
Step 5 returns raw prose; all other steps return dataclasses.

Step 3 -> 4 is a conditional loop with a max of 1 iteration

Step 6 is non-linear and points back to step 5 if triggered
    - verdict="clean" then Step 5 response returned unchanged
    - verdict="corrected" then Step 6 replaces specific claims with qualified language
      (e.g. "based on general knowledge", "search results did not cover this")
Step 6 audits output text, catching potential hallucinations introduced during Step 5's JSON-to-prose conversion.


See prompts.py for prompt design, bias check methodology, and citations.
"""

import json
import logging
import os
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from google import genai
from google.genai import types

from .prompts import (
    STEP1_CLASSIFY_AND_SCOPE,
    STEP2_SEARCH_AND_GATHER,
    STEP3_MULTI_PERSPECTIVE_SYNTHESIS,
    STEP4_BIAS_SELF_AUDIT,
    STEP5_GENERATE_RESPONSE,
    STEP6_FACT_CHECK,
    STEP_BOUNDARY_RESPONSE,
)

logger = logging.getLogger(__name__)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SOFT_FAILURE_PREFIX = "I encountered an issue generating my analysis."

_MAX_HISTORY_TURNS = 10  # cap history passed to prompts to stay within prompt budget
_BIAS_AUDIT_SKIPPED = "skipped — data lookup"
_SEARCH_SKIPPED     = "skipped — definitional query"
_MAX_REVISIONS = 1  # max re-synthesis iterations before force-proceeding to Step 5


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)


# Data classes for pipeline steps

@dataclass
class ClassificationResult:
    is_political: bool
    confidence: float
    reasoning: str
    political_dimensions: list[str]
    needs_search: bool
    search_queries: list[str]
    query_type: str
    query_intent: str = "descriptive"
    needs_bias_audit: bool = True


@dataclass
class GatheredInfo:
    verified_facts: list[str]
    contested_claims: list[str]
    gaps: list[str]
    temporal_note: str
    source_reliability_notes: list[str] = field(default_factory=list)
    perspectives_signal: list[dict] = field(default_factory=list)  # pre-grouped stakeholder evidence from Step 2; drives Step 3 perspective construction


@dataclass
class PerspectiveAnalysis:
    actor_or_lens: str
    position: str
    key_arguments: list[str]
    evidence_cited: list[str]


@dataclass
class Synthesis:
    perspectives: list[PerspectiveAnalysis]
    areas_of_agreement: list[str]
    core_disagreements: list[str]
    uncertainty_notes: list[str]
    confidence_level: str


@dataclass
class BiasAudit:
    swap_test_result: str
    depth_parity_result: str
    tone_check_result: str
    omission_check_result: str
    corrections_needed: list[str]
    overall_assessment: str
    requires_resynthesis: bool = False  # True only for structural issues; triggers Step 3->4 loop


@dataclass
class FactCheckResult:
    hallucinations: list[dict]  # each: {claim, issue, evidence_note}
    revised_response: str
    verdict: str  # "clean" | "corrected"


@dataclass
class ReasoningTrace:
    classification: Optional[ClassificationResult] = None
    gathered_info: Optional[GatheredInfo] = None
    synthesis: Optional[Synthesis] = None
    bias_audit: Optional[BiasAudit] = None
    fact_check: Optional[FactCheckResult] = None
    final_response: str = ""
    steps_executed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    revision_count: int = 0  # number of Step 3->4 re-synthesis iterations performed

# LLM helper

def _call(client: genai.Client, prompt: str, *, temperature: float = 0.3,
          as_json: bool = False, max_output_tokens: int = 4096,
          no_thinking: bool = False) -> str:
    """Single Gemini call. as_json=True enforces valid JSON output through JSON mode.

    JSON steps and no_thinking=True both disable thinking (thinking_budget=0).
    Thinking tokens compete with output tokens for the same budget, causing
    truncated responses. JSON steps disable it for structured extraction;
    no_thinking=True is for simple prose steps that don't need deep reasoning
    (e.g. boundary redirect). All behavioral constraints are inline in each
    step's prompt.
    """
    # Conditionally merge JSON-mode keys into the config via dict unpacking.
    # thinking_budget=0 disables thinking tokens: they compete with output tokens
    # for the same budget, causing truncated JSON on longer structured responses.
    disable_thinking = as_json or no_thinking
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        **({"response_mime_type": "application/json"} if as_json else {}),
        **({"thinking_config": types.ThinkingConfig(thinking_budget=0)} if disable_thinking else {}),
    )
    text = client.models.generate_content(model=MODEL, contents=prompt, config=config).text
    if text is None:
        raise ValueError("Model returned empty response (text=None)")
    return text.strip()


_NON_ASCII_MAP = str.maketrans({
    "\u2013": "-",   # en-dash
    "\u2014": "-",   # em-dash
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote / apostrophe
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u2026": "...", # ellipsis
    "\u00a0": " ",   # non-breaking space
})


def _sanitize_json(raw: str) -> str:
    """Replace common non-ASCII punctuation that breaks json.loads.

    The Step 2 prompt instructs the model to use plain ASCII, but Gemini
    occasionally emits typographic characters (em-dashes, curly quotes) from
    search content it paraphrases. unicodedata.normalize then the translation
    table catches both composed and decomposed forms.
    """
    raw = unicodedata.normalize("NFKC", raw)
    return raw.translate(_NON_ASCII_MAP)


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(No prior conversation)"
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-_MAX_HISTORY_TURNS:])


def _serialize_gathered(g: GatheredInfo) -> str:
    return json.dumps({
        "verified_facts": g.verified_facts,
        "contested_claims": g.contested_claims,
        "gaps": g.gaps,
        "temporal_note": g.temporal_note,
        "source_reliability_notes": g.source_reliability_notes,
        "perspectives_signal": g.perspectives_signal,  # pre-grouped stakeholder evidence; Step 3 uses this to seed perspective construction
    }, indent=2)


def _serialize_synthesis(synthesis: Synthesis, *, include_evidence: bool = False) -> str:
    """Serialize synthesis for prompt injection.

    include_evidence=False (Step 4): omits evidence_cited because the bias audit
    only evaluates framing and language — [model knowledge] tags in key_arguments
    carry the signal Step 4 needs.
    include_evidence=True (Step 5): includes evidence_cited so the response generator
    can detect grounding disparity and append a source note.

    The `*` makes include_evidence keyword-only to prevent accidental positional use.
    The dict-unpack `**({"evidence": ...} if include_evidence else {})` conditionally
    adds the key inside the list comprehension without a separate branch.
    """
    return json.dumps({
        "perspectives": [
            {"actor": p.actor_or_lens, "position": p.position, "arguments": p.key_arguments,
             **({"evidence": p.evidence_cited} if include_evidence else {})}
            for p in synthesis.perspectives
        ],
        "agreements": synthesis.areas_of_agreement,
        "disagreements": synthesis.core_disagreements,
        "uncertainty": synthesis.uncertainty_notes,
    }, indent=2)


def _record_failure(trace: "ReasoningTrace", step: int, e: Exception) -> tuple[str, "ReasoningTrace"]:
    msg = f"Step {step} failed: {e}"
    trace.errors.append(msg)
    logger.error(msg)
    response = SOFT_FAILURE_PREFIX + " Please try again or rephrase your question."
    trace.final_response = response
    trace.steps_executed.append(f"step{step}_failure")
    return response, trace


# Search

def _search(queries: list[str]) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "(Search unavailable, no TAVILY_API_KEY. Using model knowledge only.)"
    try:
        # Lazy import: avoids requiring tavily at startup when TAVILY_API_KEY isn't set.
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
    except Exception as e:
        return f"(Search init failed: {e})"

    results = []
    domain_counts: dict[str, int] = {}  # cap per-domain results to avoid one weak source crowding out others
    for query in queries[:3]:  # cap at 3 queries to stay within Tavily free-tier rate limits
        try:
            for r in client.search(query=query, max_results=5).get("results", []):
                url = r.get('url', '')
                try:
                    domain = url.split('//')[1].split('/')[0]
                except IndexError:
                    domain = url
                if domain_counts.get(domain, 0) >= 2:
                    continue
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                content = ' '.join((r.get('content') or '').split())[:1500]  # collapse whitespace; 1500 chars — enough to capture legal reasoning, not just article ledes
                results.append(f"Source: {url}\nTitle: {r.get('title')}\nContent: {content}\n")
        except Exception as e:
            results.append(f"(Search failed for '{query}': {e})")
    # "---" markdown HR as separator so individual results are visually distinct when injected into the prompt
    return "\n---\n".join(results) if results else "(No search results found.)"


# Reasoning pipeline

class ReasoningPipeline:
    def __init__(self):
        self.client = _get_client()
        logger.info(f"Pipeline initialized: model={MODEL}")

    def run(self, user_message: str, history: list[dict]) -> tuple[str, ReasoningTrace]:
        trace = ReasoningTrace()
        hist = _format_history(history)

        # Step 1: Classify
        try:
            classification = self._step1_classify(user_message, hist)
            trace.classification = classification
            trace.steps_executed.append("classification")
            logger.info(f"Step 1: {classification.query_type}, political={classification.is_political}")
        except Exception as e:
            trace.errors.append(f"Step 1 failed: {e}")
            logger.error(f"Step 1 failed: {e}")
            # Fail open: treat as political so the query isn't silently dropped.
            # Field rationale: confidence=0.5 (uncertain), needs_search=False (skip
            # search to avoid compounding the failure), query_type="factual_event"
            # (safest default — triggers the bias audit path). Steps 3 and 5 will
            # still produce a reasonable response even on a misclassified query.
            classification = ClassificationResult(
                is_political=True, confidence=0.5, reasoning="classification failed",
                political_dimensions=[], needs_search=False, search_queries=[], query_type="factual_event",
            )
            trace.classification = classification

        # Off-topic short-circuit redirection
        if not classification.is_political:
            response = _call(self.client, STEP_BOUNDARY_RESPONSE.format(
                user_message=user_message, reasoning=classification.reasoning, history=hist
            ), temperature=0.5, max_output_tokens=512, no_thinking=True)
            trace.final_response = response
            trace.steps_executed.append("boundary_response")
            return response, trace

        # Step 2: Search & Gather
        # gathered stays None on failure or when needs_search=False — Step 3 detects this
        # and falls back to model knowledge only; Step 6 is skipped entirely (nothing to
        # fact-check against). Skipped queries record a sentinel in steps_executed so the
        # trace is not silently empty for this step.
        gathered = None
        if not classification.needs_search:
            trace.steps_executed.append(_SEARCH_SKIPPED)
        else:
            try:
                # Fall back to the raw message if the classifier produced no search queries.
                queries = classification.search_queries if classification.search_queries else [user_message]
                gathered = self._step2_gather(user_message, _search(queries), hist)
                trace.gathered_info = gathered
                trace.steps_executed.append("search_and_gather")
            except Exception as e:
                trace.errors.append(f"Step 2 failed: {e}")
                logger.error(f"Step 2 failed: {e}")
                trace.steps_executed.append("search_failed")

        # Steps 3->4: Synthesize + Bias Audit (with conditional re-synthesis loop)
        try:
            synthesis = self._step3_synthesize(user_message, gathered, hist,
                                               query_type=classification.query_type)
            trace.synthesis = synthesis
            trace.steps_executed.append("synthesis")
        except Exception as e:
            return _record_failure(trace, 3, e)

        if classification.needs_bias_audit:
            try:
                audit = self._step4_bias_audit(user_message, synthesis)
                trace.bias_audit = audit
                trace.steps_executed.append("bias_audit")
            except Exception as e:
                return _record_failure(trace, 4, e)

            # Re-synthesis loop: if audit flags structural issues, re-run Step 3 with
            # corrections injected, then re-audit. Capped at _MAX_REVISIONS=1 iteration.
            if audit.requires_resynthesis and trace.revision_count < _MAX_REVISIONS:
                trace.revision_count += 1
                logger.info(f"Step 4: structural issues detected — re-synthesizing (revision {trace.revision_count})")
                try:
                    synthesis = self._step3_synthesize(user_message, gathered, hist,
                                                       corrections=audit.corrections_needed,
                                                       query_type=classification.query_type)
                    trace.synthesis = synthesis
                    trace.steps_executed.append(f"synthesis_revision_{trace.revision_count}")
                except Exception as e:
                    return _record_failure(trace, 3, e)
                try:
                    audit = self._step4_bias_audit(user_message, synthesis)
                    trace.bias_audit = audit
                    trace.steps_executed.append(f"bias_audit_revision_{trace.revision_count}")
                except Exception as e:
                    return _record_failure(trace, 4, e)
                if audit.requires_resynthesis:
                    logger.info("Step 4: still flagging after re-synthesis — force-proceeding to Step 5")
        else:
            # Intentionally skipped (data-lookup query, needs_bias_audit=False).
            # overall_assessment="balanced" prevents Step 5 from adding unnecessary reasoning.
            audit = BiasAudit(_BIAS_AUDIT_SKIPPED, _BIAS_AUDIT_SKIPPED,
                              _BIAS_AUDIT_SKIPPED, _BIAS_AUDIT_SKIPPED, [], "balanced",
                              requires_resynthesis=False)
            trace.bias_audit = audit

        # Step 5: Generate
        try:
            response = self._step5_generate(user_message, hist, synthesis, audit, classification.query_intent)
            trace.final_response = response
            trace.steps_executed.append("generate_response")
        except Exception as e:
            return _record_failure(trace, 5, e)

        # Step 6: Fact Check (requires search results)
        if gathered:
            try:
                fact_check = self._step6_fact_check(user_message, gathered, response)
                trace.fact_check = fact_check
                trace.steps_executed.append("fact_check")
                if fact_check.verdict == "corrected":
                    response = fact_check.revised_response
                    trace.final_response = response
                    logger.info(f"Step 6: {len(fact_check.hallucinations)} issue(s) corrected")
                else:
                    logger.info("Step 6: response is clean")
            except Exception as e:
                trace.errors.append(f"Step 6 failed: {e}")
                logger.error(f"Step 6 failed: {e}")

        return response, trace

    # Steps

    def _step1_classify(self, user_message: str, history: str) -> ClassificationResult:
        data = json.loads(_call(self.client,
            STEP1_CLASSIFY_AND_SCOPE.format(history=history, user_message=user_message),
            temperature=0.1, as_json=True, max_output_tokens=1024))  # 0.1: classification must be deterministic
        return ClassificationResult(
            is_political=data.get("is_political", True),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            political_dimensions=data.get("political_dimensions", []),
            needs_search=data.get("needs_search", True),
            search_queries=data.get("search_queries", []),
            query_type=data.get("query_type", "factual_event"),
            query_intent=data.get("query_intent", "descriptive"),
            needs_bias_audit=data.get("needs_bias_audit", True),
        )

    def _step2_gather(self, user_message: str, search_results: str, history: str = "") -> GatheredInfo:
        prompt = STEP2_SEARCH_AND_GATHER.format(
            user_message=user_message, search_results=search_results, history=history)
        raw = _call(self.client, prompt, temperature=0.2, as_json=True)  # 0.2: extraction — low creativity needed
        try:
            data = json.loads(_sanitize_json(raw))
        except json.JSONDecodeError:
            # Retry once at higher temperature — ensures a genuinely different output rather
            # than near-identical JSON that reproduces the same structural mistake.
            raw = _call(self.client, prompt, temperature=0.5, as_json=True)
            data = json.loads(_sanitize_json(raw))  # propagates if still malformed
        return GatheredInfo(
            verified_facts=data.get("verified_facts", []),
            contested_claims=data.get("contested_claims", []),
            gaps=data.get("gaps", []),
            temporal_note=data.get("temporal_note", ""),
            source_reliability_notes=data.get("source_reliability_notes", []),
            perspectives_signal=data.get("perspectives_signal", []),
        )

    def _step3_synthesize(self, user_message: str, gathered: Optional[GatheredInfo], history: str,
                          corrections: list[str] | None = None,
                          query_type: str = "factual_event") -> Synthesis:
        info_str = _serialize_gathered(gathered) if gathered else "(No search results — using model knowledge only)"
        if corrections:
            lines = "\n".join(f"- {c}" for c in corrections)
            corrections_block = f"\nREQUIRED CORRECTIONS from prior bias audit — apply before producing output:\n{lines}\n"
        else:
            corrections_block = ""
        data = json.loads(_call(self.client,
            STEP3_MULTI_PERSPECTIVE_SYNTHESIS.format(
                user_message=user_message, gathered_info=info_str,
                history=history, corrections_block=corrections_block,
                query_type=query_type),
            temperature=0.4, as_json=True, max_output_tokens=4096))  # 0.4: synthesis — needs breadth across perspectives
        return Synthesis(
            perspectives=[PerspectiveAnalysis(
                actor_or_lens=p.get("actor_or_lens", ""),
                position=p.get("position", ""),
                key_arguments=p.get("key_arguments", []),
                evidence_cited=p.get("evidence_cited", []),
            ) for p in data.get("perspectives", [])],
            areas_of_agreement=data.get("areas_of_agreement", []),
            core_disagreements=data.get("core_disagreements", []),
            uncertainty_notes=data.get("uncertainty_notes", []),
            confidence_level=data.get("confidence_level", "moderate"),
        )

    def _step4_bias_audit(self, user_message: str, synthesis: Synthesis) -> BiasAudit:
        """
        evidence_cited is excluded because only Step 4 audits framing and language,
        not source grounding. The [model knowledge] tags in key_arguments carry the
        search/model-knowledge signal Step 4 needs. evidence_cited is passed to Step 5,
        which uses it to detect grounding disparity and append a source note to the response.
        """
        synthesis_str = _serialize_synthesis(synthesis)
        data = json.loads(_call(self.client,
            STEP4_BIAS_SELF_AUDIT.format(user_message=user_message, synthesis=synthesis_str),
            temperature=0.2, as_json=True, max_output_tokens=4096))  # 0.2: audit must be consistent and rule-driven
        return BiasAudit(
            swap_test_result=data.get("swap_test_result", "skipped"),
            depth_parity_result=data.get("depth_parity_result", "skipped"),
            tone_check_result=data.get("tone_check_result", "skipped"),
            omission_check_result=data.get("omission_check_result", "skipped"),
            corrections_needed=data.get("corrections_needed", []),
            overall_assessment=data.get("overall_assessment", "skipped"),
            requires_resynthesis=bool(data.get("requires_resynthesis", False)),  # explicit bool() guards against model returning 0/1 or string "true"
        )

    def _step5_generate(self, user_message: str, history: str,
                        synthesis: Synthesis, audit: BiasAudit, query_intent: str = "descriptive") -> str:
        response = _call(self.client, STEP5_GENERATE_RESPONSE.format(
            user_message=user_message,
            history=history,
            synthesis=_serialize_synthesis(synthesis, include_evidence=True),
            bias_audit=json.dumps({"swap_test": audit.swap_test_result, "depth_parity": audit.depth_parity_result,
                                   "tone_check": audit.tone_check_result, "omission_check": audit.omission_check_result}, indent=2),
            corrections=json.dumps(audit.corrections_needed or ["None — analysis passed bias checks"]),  # placeholder keeps prompt context consistent when no corrections exist
            confidence=synthesis.confidence_level,
            query_intent=query_intent,
        ), temperature=0.5, max_output_tokens=3072)  # 0.5: prose generation — most variation desired for natural language

        # Programmatic enforcement: STEP5 prompt instructs the model to append the causal, but isn't 100% reliable
        if query_intent == "causal":
            causal_note = "Note: This analysis describes patterns and correlations. It does not establish direct causal relationships."
            # "correlat" matches both "correlation" and "correlates" — avoids appending
            # a duplicate note if the model already included causal/correlation language.
            if "correlat" not in response.lower() and "causal" not in response.lower():
                response = response + "\n\n" + causal_note
        return response

    def _step6_fact_check(self, user_message: str, gathered: GatheredInfo, response: str) -> FactCheckResult:
        gathered_str = _serialize_gathered(gathered)
        data = json.loads(_call(self.client,
            STEP6_FACT_CHECK.format(
                user_message=user_message,
                gathered_info=gathered_str,
                response=response,
            ),
            temperature=0.1, as_json=True, max_output_tokens=3072))  # 0.1: fact-check must be precise, not creative
        return FactCheckResult(
            hallucinations=data.get("hallucinations", []),
            revised_response=data.get("revised_response", response),  # falls back to original if model omits the field
            verdict=data.get("verdict", "clean"),
        )
