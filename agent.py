"""
AI Research Agent: Utilizing LangGraph + free-tier LLMs + free-tier search
LLM:    Gemini 2.0 Flash (failed for now) and Ollama/Llama 3.2 (local, free, no limits)
Search: Tavily , Jina 
"""

# Imports
import os
import json
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from llm_provider import get_llm
from search_provider import search_web
import operator


# State
class ResearchState(TypedDict):
    question: str
    search_queries: List[str]
    search_results: Annotated[List[dict], operator.add]
    sources: List[dict]
    final_report: str
    quality_score: int
    iterations: int
    error: str


# Configuration for the agent (specifying iterations, quality, LLM and search provider)
MAX_ITERATIONS = 3
QUALITY_THRESHOLD = 7
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto") # gemini / ollama / auto
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "auto")  # tavily / jina / auto


# Nodes
# This node generates about 4 specific research questions, 
# this will be the basis on which the search is carried out. 
def plan_node(state: ResearchState) -> dict:
    """Decompose the question into 3–4 focused search queries."""
    llm = get_llm(provider=LLM_PROVIDER)
    prompt = f"""You are a research planner. Given a research question, produce 3-4 specific 
web search queries that together will answer the question comprehensively.

Research question: {state['question']}

Respond with ONLY a JSON array of strings, e.g.:
["query one", "query two", "query three"]

No preamble, no markdown fences."""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    try:
        # Strip markdown fences if the model added them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        queries = json.loads(raw.strip())
        if not isinstance(queries, list):
            raise ValueError("Not a list")
        queries = [str(q) for q in queries[:4]]
    except Exception:
        queries = [state["question"]]

    print(f"\n📋 Planned {len(queries)} queries:")
    for q in queries:
        print(f"   • {q}")

    return {"search_queries": queries, "iterations": state.get("iterations", 0)}

# Run the queries generated in the plan node and search the web
def search_node(state: ResearchState) -> dict:
    """Run queries through the configured search provider."""
    all_results = []
    sources = []

    for query in state["search_queries"]:
        print(f"\n🔍 Searching: {query}")
        try:
            results = search_web(query, max_results=3, provider=SEARCH_PROVIDER)
            for r in results:
                all_results.append({**r, "query": query})
                sources.append({
                    "url": r["url"],
                    "title": r["title"],
                    "snippet": r["content"][:200],
                })
            print(f"   ✓ {len(results)} results")
        except Exception as e:
            print(f"   ✗ Search failed: {e}")
            all_results.append({"query": query, "error": str(e)})

    # Deduplicate sources by URL
    seen = set()
    unique_sources = []
    for s in sources:
        if s["url"] not in seen:
            seen.add(s["url"])
            unique_sources.append(s)

    return {"search_results": all_results, "sources": unique_sources}

# Synthesising the results into a structured report format for ease of viewing
def synthesise_node(state: ResearchState) -> dict:
    """Synthesise search results into a structured markdown report."""
    llm = get_llm(provider=LLM_PROVIDER)

    context_parts = []
    for r in state["search_results"]:
        if "error" in r:
            continue
        context_parts.append(
            f"Source: {r.get('title', 'Unknown')} ({r.get('url', '')})\n"
            f"Content: {r.get('content', '')}\n"
        )
    context = "\n---\n".join(context_parts)

    sources_md = "\n".join(
        f"{i+1}. [{s['title']}]({s['url']})"
        for i, s in enumerate(state["sources"])
    )

    prompt = f"""You are a research analyst. Synthesise the search results below into a 
comprehensive, well-structured markdown report that answers the research question.

Research question: {state['question']}

Search results:
{context}

Requirements:
- Start with a brief executive summary (2-3 sentences)
- Use clear ## headings to organise findings
- Include specific facts, numbers, and examples from the sources
- Note any contradictions or uncertainty between sources
- End with a "Key Takeaways" section (3-5 bullet points)
- Be objective and cite sources inline as (Source: title)

Write the report now:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    report = response.content.strip()
    full_report = f"{report}\n\n## Sources\n{sources_md}"

    print(f"\n✅ Report synthesised ({len(report)} chars)")
    return {"final_report": full_report}

# Evaluate the quality of the generated report and rate it in the scale of 1-10
# Utilizes source, comphrension and overall structure to evaluate the result
def evaluate_node(state: ResearchState) -> dict:
    """Self-evaluate report quality, return score 1–10."""
    llm = get_llm(provider=LLM_PROVIDER)
    prompt = f"""Rate the quality of this research report on a scale of 1-10.

Question: {state['question']}

Report (first 1500 chars):
{state['final_report'][:1500]}

Score criteria:
- 9-10: Comprehensive, well-sourced, specific facts, clear structure
- 7-8: Good coverage, mostly specific, minor gaps
- 5-6: Partial coverage, some vague claims
- 1-4: Thin, vague, or off-topic

Respond with ONLY a single integer (1-10). Nothing else."""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        score = int(response.content.strip())
        score = max(1, min(10, score))
    except Exception:
        score = 7

    iterations = state.get("iterations", 0) + 1
    print(f"\n📊 Quality score: {score}/10 (iteration {iterations}/{MAX_ITERATIONS})")
    return {"quality_score": score, "iterations": iterations}


# Routing
# Re-searching and increasing the iterations based on the quality score
def should_continue(state: ResearchState) -> str:
    if state["iterations"] >= MAX_ITERATIONS:
        print("   → Max iterations reached, finishing")
        return "finish"
    if state["quality_score"] < QUALITY_THRESHOLD:
        print(f"   → Score {state['quality_score']} < {QUALITY_THRESHOLD}, re-searching")
        return "re_search"
    print(f"   → Score {state['quality_score']} ≥ {QUALITY_THRESHOLD}, finishing")
    return "finish"


# Graph 
def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("synthesise", synthesise_node)
    graph.add_node("evaluate", evaluate_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "synthesise")
    graph.add_edge("synthesise", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        should_continue,
        {"re_search": "search", "finish": END},
    )
    return graph.compile()


# Public API 
# Run the pipeline and get the results for the question 
# Well structured result, sources, report quality score
def run_research(question: str) -> dict:
    """Run the full research pipeline and return results."""
    from llm_provider import get_available_providers
    providers = get_available_providers()
    print(f"\n{'='*60}")
    print(f"🔬 Research question: {question}")
    print(f"🤖 LLM: {LLM_PROVIDER} | 🔍 Search: {SEARCH_PROVIDER}")
    if providers:
        print(f"Available LLMs: {', '.join(providers)}")
    print(f"{'='*60}")

    app = build_graph()
    initial_state = ResearchState(
        question=question,
        search_queries=[],
        search_results=[],
        sources=[],
        final_report="",
        quality_score=0,
        iterations=0,
        error="",
    )
    final_state = app.invoke(initial_state)
    return {
        "question": question,
        "report": final_state["final_report"],
        "sources": final_state["sources"],
        "quality_score": final_state["quality_score"],
        "iterations": final_state["iterations"],
    }


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What are the most promising techniques for AI safety in 2026?"
    result = run_research(question)
    print(f"\n{'='*60}")
    print(result["report"])
    print(f"\n✅ Done | Quality: {result['quality_score']}/10 | Iterations: {result['iterations']}")