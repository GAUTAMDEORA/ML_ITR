# app_streamlit.py

import sys
import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


# SQLite fix for chromadb compatibility
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from complete_legal_rag_chatbot import LegalRAGChatbot

st.set_page_config(page_title="Legal RAG Chatbot", layout="wide")
st.title("📚 Legal Document RAG Chatbot (Local - No API Key)")

chatbot = LegalRAGChatbot()

uploaded_file = st.file_uploader("Upload your legal PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF, please wait..."):
        msg = chatbot.process_pdf(uploaded_file)
        st.success(msg)

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("---")
st.write("### Ask questions about the uploaded document")

query = st.text_input("Enter your question here:", key="input_query")

if st.button("Send") and query.strip():
    with st.spinner("Generating answer..."):
        answer = chatbot.chat(query.strip())
        st.session_state.history.append(("You", query.strip()))
        st.session_state.history.append(("Bot", answer))
        st.experimental_rerun()

# Display chat history
for speaker, text in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")

st.markdown("---")
st.caption("Built with ChromaDB + Sentence Transformers + DialoGPT local model running entirely on Streamlit.")
