import os
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.rag_faiss import answer_question

LOG_CSV_PATH = os.path.join("data", "observability", "rag_logs.csv")

st.set_page_config(page_title="RAG Chatbot (FAISS + LangChain)", page_icon="🤖")
st.title("RAG Chatbot (FAISS + LangChain)")
st.caption("Put docs into data/raw/pubmed, run ingestion, then ask questions with citations.")

tab_chat, tab_obs = st.tabs(["Chat", "Observability"])

with st.sidebar:
    st.header("Controls")

    mode = st.selectbox("Search mode", ["Hybrid", "Vector"], index=0)
    rerank = st.checkbox("Enable reranking", value=True)

    filetype_filter = st.selectbox("Doc type filter", ["All", "PDF", "TXT", "MD"], index=0)
    source_filter = st.text_input(
        "Metadata filter (source contains…)",
        help="Example: type part of a filename like 'CFIR' or 'bmjopen'."
    ).strip() or None

    memory_turns = st.slider(
        "Conversation memory turns",
        min_value=0, max_value=3, value=2,
        help="Keeps last N Q/A turns for follow-ups (still retrieves from docs every time)."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
if "turns" not in st.session_state:
    st.session_state.turns = []

with tab_chat:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("citations"):
                st.markdown("**Citations:**")
                for c in m["citations"]:
                    st.write(c)

    q = st.chat_input("Ask a question...")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        history_for_model = st.session_state.turns[-memory_turns:] if memory_turns > 0 else []

        result = answer_question(
            q,
            history=history_for_model,
            memory_turns=memory_turns,
            source_filter=source_filter,
            filetype_filter=filetype_filter,
            mode=mode,
            rerank=rerank,
        )

        answer = result["answer"]
        citations = result.get("citations", [])

        st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})
        st.session_state.turns.append({"user": q, "assistant": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)
            if citations:
                st.markdown("**Citations:**")
                for c in citations:
                    st.write(c)

with tab_obs:
    st.caption("Observability logs are written locally to CSV as you use the chatbot.")

    if os.path.exists(LOG_CSV_PATH):
        df = pd.read_csv(
            LOG_CSV_PATH,
            engine="python",
            on_bad_lines="skip",
            quotechar='"',
            escapechar="\\",
        )

        st.subheader("Recent queries")
        st.dataframe(df.tail(50), use_container_width=True)

        st.subheader("Latency (ms)")
        fig = plt.figure()
        plt.plot(df["total_ms"].tail(100).values)
        plt.ylabel("total_ms")
        plt.xlabel("last 100 queries")
        st.pyplot(fig)

        st.subheader("Modes (count)")
        fig2 = plt.figure()
        counts = df["mode"].fillna("unknown").value_counts()
        plt.bar(range(len(counts)), counts.values)
        plt.xticks(range(len(counts)), counts.index, rotation=0)
        plt.ylabel("count")
        st.pyplot(fig2)

    else:
        st.info("No logs yet. Ask a few questions in the Chat tab first.")
