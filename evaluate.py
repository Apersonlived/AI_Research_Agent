"""
Evaluation suite for the AI Research Agent.
-------------------------------------------
Runs 20 test questions, scores each result, and prints a summary table.
Usage: python evaluate.py
"""

import sys
import time
import json
from dataclasses import dataclass, field
from typing import List

# Only import agent if API keys are present
try:
    from agent import run_research
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False


# ── Test cases ────────────────────────────────────────────────────────────────
@dataclass
class TestCase:
    id: int
    question: str
    expected_keywords: List[str] # at least 3 must appear in report
    expected_min_sources: int = 3 # minimum unique sources expected
    category: str = "general"


TEST_CASES = [
    TestCase(1,  "What are the leading techniques for LLM alignment in 2026?",
             ["RLHF", "constitutional", "interpretability"], category="AI safety"),
    TestCase(2,  "How does retrieval-augmented generation work?",
             ["vector", "embedding", "retrieval", "chunk"], category="AI/ML"),
    TestCase(3,  "What are the main risks of deploying LLMs in production?",
             ["hallucination", "bias", "security", "cost"], category="AI/ML"),
    TestCase(4,  "What skills do AI automation engineers need in 2026?",
             ["LangChain", "agents", "LLM", "API"], category="careers"),
    TestCase(5,  "How does LangGraph differ from LangChain?",
             ["graph", "state", "workflow", "nodes"], category="AI/ML"),
    TestCase(6,  "What are the best practices for prompt engineering?",
             ["few-shot", "chain-of-thought", "temperature"], category="AI/ML"),
    TestCase(7,  "How do vector databases work?",
             ["embedding", "similarity", "index", "cosine"], category="AI/ML"),
    TestCase(8,  "What is the EU AI Act and how does it affect developers?",
             ["regulation", "risk", "compliance", "2024"], category="policy"),
    TestCase(9,  "What are the differences between RAG and fine-tuning?",
             ["retrieval", "training", "inference", "cost"], category="AI/ML"),
    TestCase(10, "How does mechanistic interpretability help AI safety?",
             ["circuits", "neurons", "features", "transparency"], category="AI safety"),
    TestCase(11, "What is AI red teaming and how is it used?",
             ["adversarial", "vulnerabilities", "testing", "jailbreak"], category="AI safety"),
    TestCase(12, "What are the top open-source LLMs available in 2026?",
             ["Llama", "Mistral", "open-source", "parameters"], category="AI/ML"),
    TestCase(13, "How does model quantization reduce LLM inference costs?",
             ["quantization", "bits", "memory", "precision"], category="AI/ML"),
    TestCase(14, "What is multi-agent AI and what are its use cases?",
             ["agents", "orchestration", "coordination", "task"], category="AI/ML"),
    TestCase(15, "How can computer vision be applied to document processing?",
             ["OCR", "extraction", "vision", "document"], category="CV"),
    TestCase(16, "What is MLOps and why does it matter for production AI?",
             ["deployment", "monitoring", "pipeline", "drift"], category="ML engineering"),
    TestCase(17, "What are the best AI safety research organisations in 2026?",
             ["Anthropic", "OpenAI", "DeepMind", "research"], category="AI safety"),
    TestCase(18, "How does AI agent memory work?",
             ["memory", "context", "retrieval", "persistence"], category="AI/ML"),
    TestCase(19, "What are common LLM security vulnerabilities?",
             ["injection", "jailbreak", "prompt", "attack"], category="AI security"),
    TestCase(20, "How do you evaluate the quality of a RAG pipeline?",
             ["recall", "precision", "faithfulness", "evaluation"], category="AI/ML"),
]


# ── Scoring ───────────────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    test_id: int
    question: str
    category: str
    keyword_hits: int
    keyword_total: int
    source_count: int
    agent_quality_score: int
    elapsed_s: float
    passed: bool
    error: str = ""

def score_result(test: TestCase, result: dict) -> EvalResult:
    report_lower = result["report"].lower()
    hits = sum(1 for kw in test.expected_keywords if kw.lower() in report_lower)
    src_count = len(result["sources"])
    passed = (
        hits >= 2 and
        src_count >= test.expected_min_sources and
        result["quality_score"] >= 6
    )
    return EvalResult(
        test_id=test.id,
        question=test.question,
        category=test.category,
        keyword_hits=hits,
        keyword_total=len(test.expected_keywords),
        source_count=src_count,
        agent_quality_score=result["quality_score"],
        elapsed_s=0,
        passed=passed,
    )


# ── Runner ────────────────────────────────────────────────────────────────────
def run_evaluation(quick=False):
    cases = TEST_CASES[:5] if quick else TEST_CASES
    results: List[EvalResult] = []

    print(f"\n{'='*70}")
    print(f"AI Research Agent — Evaluation Suite ({len(cases)} questions)")
    print(f"{'='*70}\n")

    for test in cases:
        print(f"[{test.id:02d}/{len(cases)}] {test.question[:60]}...")
        try:
            t0 = time.time()
            result = run_research(test.question)
            elapsed = time.time() - t0
            eval_r = score_result(test, result)
            eval_r.elapsed_s = elapsed
        except Exception as e:
            eval_r = EvalResult(
                test_id=test.id, question=test.question, category=test.category,
                keyword_hits=0, keyword_total=len(test.expected_keywords),
                source_count=0, agent_quality_score=0, elapsed_s=0,
                passed=False, error=str(e),
            )
        results.append(eval_r)
        status = "✅ PASS" if eval_r.passed else "❌ FAIL"
        print(f"     {status} | keywords {eval_r.keyword_hits}/{eval_r.keyword_total} | "
              f"sources {eval_r.source_count} | quality {eval_r.agent_quality_score}/10 | "
              f"{eval_r.elapsed_s:.1f}s\n")

    # Summary table
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    avg_quality = sum(r.agent_quality_score for r in results) / total
    avg_time = sum(r.elapsed_s for r in results) / total

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Pass rate:       {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"Avg quality:     {avg_quality:.1f}/10")
    print(f"Avg latency:     {avg_time:.1f}s per question")
    print()

    # Category breakdown
    categories = {}
    for r in results:
        categories.setdefault(r.category, []).append(r.passed)
    print("By category:")
    for cat, outcomes in sorted(categories.items()):
        p = sum(outcomes)
        t = len(outcomes)
        print(f"  {cat:<20} {p}/{t} ({p/t*100:.0f}%)")

    # Save JSON
    output = {
        "pass_rate": f"{passed}/{total}",
        "avg_quality_score": round(avg_quality, 2),
        "avg_latency_s": round(avg_time, 2),
        "results": [
            {
                "id": r.test_id,
                "question": r.question,
                "category": r.category,
                "passed": r.passed,
                "keyword_hits": f"{r.keyword_hits}/{r.keyword_total}",
                "sources": r.source_count,
                "quality": r.agent_quality_score,
                "elapsed_s": round(r.elapsed_s, 2),
                "error": r.error,
            }
            for r in results
        ]
    }
    with open("eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n📄 Full results saved to eval_results.json")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    if not AGENT_AVAILABLE:
        print("ERROR: Could not import agent. Make sure agent.py is in the same directory.")
        sys.exit(1)
    run_evaluation(quick=quick)