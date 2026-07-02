"""Streamlit chat UI for the Agentic RAG Analyst.

Run: streamlit run app.py
"""
import streamlit as st

from src.rag import RagAnalyst

st.set_page_config(page_title="Agentic RAG Analyst", page_icon="🔎")
st.title("Agentic RAG Analyst")
st.caption("Ask questions over the ingested corpus. Answers are grounded with citations.")


@st.cache_resource
def get_analyst():
    return RagAnalyst()


question = st.text_input("Your question", placeholder="e.g. What features drive rebound predictions?")

if question:
    with st.spinner("Retrieving and reasoning..."):
        try:
            ans = get_analyst().ask(question)
            st.markdown(ans.text)
            with st.expander("Sources"):
                for s in ans.sources:
                    st.write(f"- {s}")
        except Exception as e:
            st.error(f"Error: {e}. Did you run `python -m src.ingest data/` and set OPENAI_API_KEY?")
