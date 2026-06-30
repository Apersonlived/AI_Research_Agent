"""
Streamlit frontend for the AI Research Agent.
Run with: streamlit run app.py
"""

import streamlit as st
import time
from agent import run_research

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 AI Research Agent")
st.caption("Powered by LangGraph + Tavily")

with st.sidebar:
    st.header("How it works")
    st.markdown("""
    **Graph flow:**
    1. **Plan** — Claude decomposes your question into 3–4 search queries
    2. **Search** — Tavily fetches fresh web results for each query
    3. **Synthesise** — Claude writes a structured markdown report
    4. **Evaluate** — Claude scores quality 1–10
    5. **Loop** — If score < 7, searches again (max 3 iterations)

    **Tips:**
    - Specific questions get better reports
    - Phrase as a proper research question
    - Works best for factual / analytical topics
    """)
    st.divider()

# Main input
question = st.text_area(
    "Research question",
    value=st.session_state.get("question_input", ""),
    placeholder="e.g. What are the best practices for building production AI agents in 2026?",
    height=80,
    key="question_area",
)

col1, col2 = st.columns([1, 5])
with col1:
    run_btn = st.button("🔍 Research", type="primary", use_container_width=True)

if run_btn and question.strip():
    with st.status("Running research agent...", expanded=True) as status:
        st.write("📋 Planning search queries...")
        start = time.time()

        # Live log container
        log_container = st.empty()

        try:
            result = run_research(question.strip())
            elapsed = time.time() - start
            status.update(
                label=f"✅ Complete — {elapsed:.1f}s | Quality: {result['quality_score']}/10 | {result['iterations']} iteration(s)",
                state="complete",
            )
        except EnvironmentError as e:
            st.error(f"⚠️ Configuration error: {e}\n\nSet your Google_API_KEY and TAVILY_API_KEY environment variables.")
            st.stop()
        except Exception as e:
            st.error(f"Agent error: {e}")
            st.stop()

    # Results
    tab1, tab2, tab3 = st.tabs(["📄 Report", "🔗 Sources", "📊 Metadata"])

    with tab1:
        st.markdown(result["report"])
        st.download_button(
            "⬇ Download report",
            data=result["report"],
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab2:
        if result["sources"]:
            for i, src in enumerate(result["sources"], 1):
                with st.expander(f"{i}. {src['title']}", expanded=False):
                    st.markdown(f"**URL:** {src['url']}")
                    st.caption(src.get("snippet", ""))
        else:
            st.info("No sources collected.")

    with tab3:
        st.json({
            "question": result["question"],
            "quality_score": f"{result['quality_score']}/10",
            "iterations": result["iterations"],
            "sources_found": len(result["sources"]),
        })

elif run_btn:
    st.warning("Please enter a research question.")