# app_streamlit.py

import streamlit as st
from complete_legal_rag_chatbot import LegalRAGChatbot

st.set_page_config(page_title="Legal RAG Chatbot", layout="wide")
st.title("📚 Legal Document RAG Chatbot (Local Only)")

# Upload and process PDF
chatbot = LegalRAGChatbot(local=True)
uploaded = st.file_uploader("Upload your legal PDF", type="pdf")
if uploaded:
    status = chatbot.process_pdf(uploaded)
    st.success(status)

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

# User question input
question = st.text_input("Ask a question about the document:")
if st.button("Send"):
    if question.strip():
        answer = chatbot.chat(question)
        st.session_state.history.append(("You", question))
        st.session_state.history.append(("Bot", answer))
    else:
        st.warning("Please enter a question.")

# Display chat history
for speaker, msg in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")

st.markdown("---")
st.caption("Runs entirely locally using DialoGPT-medium for generation.")
