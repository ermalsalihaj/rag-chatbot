import streamlit as st
from rag.rag_faiss import answer_question

st.set_page_config(page_title="RAG Chatbot (FAISS + LangChain)", page_icon="🤖")
st.title("RAG Chatbot (FAISS + LangChain)")
st.caption("Put PDFs into data/raw/pubmed, run ingestion, then ask questions with citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

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

    result = answer_question(q)
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "citations": result.get("citations", [])
    })

    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        if result.get("citations"):
            st.markdown("**Citations:**")
            for c in result["citations"]:
                st.write(c)
