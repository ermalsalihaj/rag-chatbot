import streamlit as st

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("RAG Chatbot")
st.caption("Project #4 — Retrieval-Augmented Generation chatbot with citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("Ask a question...")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # Placeholder response (Day 3+ will replace this with retrieval + citations)
    answer = "I’m set up. Next steps: ingestion → embeddings → index → retrieval → grounded answer + citations."
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
