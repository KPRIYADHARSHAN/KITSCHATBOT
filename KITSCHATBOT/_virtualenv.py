import streamlit as st
import os
from langchain_helper import create_vector_db, get_qa_chain

st.set_page_config(page_title="KITS ChatBot", page_icon="🤖", layout="centered")
st.title("KITS ChatBot")
st.markdown(
    "Ask me anything about Karunya Institute of Technology and Sciences - admissions, programs, campus life, and more!")

# Sidebar for knowledge base creation
st.sidebar.title("Knowledge Base")
if st.sidebar.button("Create Knowledge Base"):
    with st.spinner("Creating knowledge base... This may take a few minutes."):
        create_vector_db()
    st.sidebar.success("Knowledge base created successfully! You can now ask questions.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input field for user question
question = st.text_input("Your Question:", key="input", placeholder="Type your question here...")

# Process question and display response
if question:
    # Check if faiss_index exists
    if os.path.exists("faiss_index") and os.path.exists("faiss_index/index.faiss"):
        qa_chain = get_qa_chain()
        response = qa_chain(question)

        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "message": question})
        st.session_state.chat_history.append({"role": "bot", "message": response["result"]})
    else:
        st.error("Please create the knowledge base first by clicking 'Create Knowledge Base' in the sidebar.")

# Display chat history
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**You**: {chat['message']}")
    else:
        st.markdown(f"**KITSChatBot**: {chat['message']}")

# Clear chat history
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

st.markdown("---")
st.markdown("For further assistance, email admissions@karunya.edu.")