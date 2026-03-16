"""
Gradio web interface for chatbot

Required APIs
export GEMINI_API_KEY=your-key-here     
export TAVILY_API_KEY=your-key-here     
"""

import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv(override=True)

from src.reasoning import ReasoningPipeline, ReasoningTrace

pipeline = ReasoningPipeline()


def _section(title: str) -> str:
    return f"\n---\n### {title}"


def _fmt_classification(c) -> list[str]:
    parts = [
        _section("Step 1 — Classification"),
        f"- **Type:** `{c.query_type}` | **Intent:** `{c.query_intent}` | **Political:** {c.is_political} ({c.confidence:.0%} confidence)",
        f"- **Reasoning:** {c.reasoning}",
    ]
    if c.search_queries:
        parts.append("- **Search queries generated:**")
        for q in c.search_queries:
            parts.append(f"  - *{q}*")
    return parts


def _fmt_gathered(g) -> list[str]:
    parts = [_section("Step 2 — Search + Gather"), f"- **Temporal note:** {g.temporal_note}"]
    if g.verified_facts:
        parts.append(f"- **Verified facts ({len(g.verified_facts)}):**")
        for f_ in g.verified_facts[:4]:  # show first 4 only — full list in trace.gathered_info
            parts.append(f"  - {f_}")
    if g.contested_claims:
        parts.append(f"- **Contested claims ({len(g.contested_claims)}):**")
        for claim in g.contested_claims[:2]:
            parts.append(f"  - {claim}")
    if g.gaps:
        parts.append("- **Gaps in search:**")
        for gap in g.gaps:
            parts.append(f"  - {gap}")
    return parts


def _fmt_synthesis(s) -> list[str]:
    parts = [_section(f"Step 3 — Multi-Perspective Synthesis  *(confidence: {s.confidence_level})*")]
    for p in s.perspectives:
        parts.append(f"\n**{p.actor_or_lens}** — *{p.position[:120]}*")  # truncate position to 120 chars for display
        for arg in p.key_arguments:
            parts.append(f"  - {arg}")
    if s.areas_of_agreement:
        parts.append(f"\n**Areas of agreement:** {' / '.join(s.areas_of_agreement[:2])}")
    if s.uncertainty_notes:
        parts.append(f"\n**Uncertainty:** {' / '.join(s.uncertainty_notes[:2])}")
    return parts


def _fmt_bias_audit(b, revision_count: int = 0) -> list[str]:
    overall_icon = "pass" if b.overall_assessment == "balanced" else "potentially unbalanced"
    resynthesis_tag = f"  *(re-synthesized {revision_count}×)*" if revision_count > 0 else ""
    parts = [
        _section(f"Step 4 — Bias Audit  {overall_icon} *{b.overall_assessment}*{resynthesis_tag}"),
        f"- **Swap test:** {b.swap_test_result}",
        f"- **Depth parity:** {b.depth_parity_result}",
        f"- **Tone check:** {b.tone_check_result}",
        f"- **Omission check:** {b.omission_check_result}",
    ]
    if b.corrections_needed:
        parts.append(f"- **Corrections applied ({len(b.corrections_needed)}):**")
        for corr in b.corrections_needed:
            parts.append(f"  - corrected {corr}")
    else:
        parts.append("- *No corrections needed.*")
    return parts


def _fmt_fact_check(fc) -> list[str]:
    verdict_icon = "pass" if fc.verdict == "clean" else "reworked"
    parts = [_section(f"Step 6 — Fact Check  {verdict_icon} *{fc.verdict}*")]
    if fc.hallucinations:
        parts.append(f"- **Issues found ({len(fc.hallucinations)}) — response was corrected:**")
        for h in fc.hallucinations:
            parts.append(f"  - `{h.get('issue', 'unsupported')}`: *\"{h.get('claim', '')}\"*")
            parts.append(f"    - {h.get('evidence_note', '')}")
    else:
        parts.append("- *All factual claims supported by retrieved evidence.*")
    return parts


def _format_trace(trace: ReasoningTrace) -> str:
    parts = [f"**Steps executed:** `{' -> '.join(trace.steps_executed) if trace.steps_executed else 'none'}`"]
    if trace.classification:
        parts.extend(_fmt_classification(trace.classification))
    if trace.gathered_info:
        parts.extend(_fmt_gathered(trace.gathered_info))
    if trace.synthesis:
        parts.extend(_fmt_synthesis(trace.synthesis))
    if trace.bias_audit:
        parts.extend(_fmt_bias_audit(trace.bias_audit, trace.revision_count))
    if trace.fact_check:
        parts.extend(_fmt_fact_check(trace.fact_check))
    if trace.errors:
        parts.append(f"\n---\n **Pipeline errors:** {'; '.join(trace.errors)}")
    return "\n".join(parts)


def respond(message: str, history: list):
    # Show the user message immediately, then process.
    # Re-dict the history to strip any extra Gradio UI keys — pipeline expects only role/content.
    pipeline_history = [{"role": m["role"], "content": m["content"]} for m in history]
    pending_history = history + [{"role": "user", "content": message}]
    yield pending_history, "*Processing...*", ""

    try:
        response, trace = pipeline.run(message, pipeline_history)
    except Exception as e:
        response = f"I encountered an issue processing your question. Please try again. ({e})"
        trace = ReasoningTrace(errors=[str(e)])

    final_history = pending_history + [{"role": "assistant", "content": response}]
    yield final_history, _format_trace(trace), ""  # empty string clears the input textbox after submission


examples = [
    "What happened with the debt ceiling negotiations in 2023? What were the key positions of both parties?",
    "What are the key issues in the 2024 presidential primary campaigns?",
    "Explain the recent Supreme Court decision on affirmative action in college admissions.",
    "What's the weather like today?",
    "What's the current debate around immigration policy?",
]

with gr.Blocks(title="Political AI Analyst") as demo:
    gr.Markdown(
        "# Political AI Analyst\n"
        "Ask anything political! "
        "*Non-political queries will be redirected. Expand **Reasoning Trace** below to inspect the pipeline.*"
    )

    chatbot = gr.Chatbot(height=480, show_label=False)

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a political question...",
            show_label=False,
            scale=9,
            autofocus=True,
        )

        send_btn = gr.Button("Send", scale=1, variant="primary")

    with gr.Accordion("Reasoning Trace", open=False):
        trace_display = gr.Markdown("*Ask a question to see the step-by-step reasoning trace.*\n\n[model knowledge] = claim not grounded in search results  |  corrected = fact-check corrected a claim")

    gr.Examples(examples=examples, inputs=msg, label="Example questions")

    # Wire up interactions
    submit_args = dict(fn=respond, inputs=[msg, chatbot], outputs=[chatbot, trace_display, msg])
    msg.submit(**submit_args)
    send_btn.click(**submit_args)


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set. Get a free key at: https://aistudio.google.com/apikey")
    tavily = "enabled" if os.getenv("TAVILY_API_KEY") else "disabled (set TAVILY_API_KEY to enable search)"
    print(f"Search grounding: {tavily}\n")
    demo.launch()
