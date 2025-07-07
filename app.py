import os
import shutil
import fitz
import streamlit as st
from pathlib import Path
from ollama import Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize Ollama client and embeddings
client = Client()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Extract text from PDF ---
def extract_text(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

# --- Chunk text ---
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

# --- Rebuild vector DB from uploaded PDFs ---
def load_vectorstore_from_uploaded_pdfs(pdf_paths):
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
    os.makedirs("faiss_index", exist_ok=True)

    all_text = ""
    for path in pdf_paths:
        all_text += extract_text(path)
    chunks = chunk_text(all_text)

    vs = FAISS.from_texts(chunks, embedding=embeddings)
    vs.save_local("faiss_index")
    return vs

# --- Prompt format ---
def format_prompt(context, question):
    return f"""
You are a helpful AI like ChatGPT, trained to answer clearly and accurately using ONLY the context below.

---
Context:
{context}
---

Question:
{question}

Guidelines:
- Be structured, detailed, and contextual
- Say "Not found in context" if unsure
"""

# --- Stream response from Ollama ---
def stream_ollama_response(prompt):
    stream = client.chat(
        model="deepseek-r1:7b",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    full_response = ""
    for chunk in stream:
        token = chunk.get("message", {}).get("content", "")
        full_response += token
        yield token
    return full_response

# --- Streamlit App Setup ---
st.set_page_config(
    page_title="📄 AI PDF Research Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 1.5rem;
    }

    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f9fafb;
        color: #333333;
    }

    .main > div {
        padding: 1.5rem 2rem 2rem 2rem;
        max-width: 1200px;
        margin: auto;
    }

    .chat-bubble {
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        border-radius: 1rem;
        font-size: 1.1rem;
        line-height: 1.5;
        max-width: 90%;
        white-space: pre-wrap;
        box-shadow: 0 2px 6px rgb(0 0 0 / 0.1);
        transition: background-color 0.3s ease;
    }

    .user-bubble {
        background-color: #dcf8c6;
        text-align: right;
        margin-left: auto;
        color: #2f4f2f;
        font-weight: 600;
    }

    .ai-bubble {
        background-color: #e6e6e6;
        text-align: left;
        margin-right: auto;
        color: #333333;
    }

    .uploaded-pdf-list {
        list-style: none;
        padding-left: 0;
        margin-top: 0.5rem;
    }
    .uploaded-pdf-list li {
        margin-bottom: 0.5rem;
        font-size: 1rem;
        color: #555555;
    }

    div[data-testid="stChatInput"] {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    div[data-testid="stChatInput"] > div > div > textarea {
        font-size: 1.1rem;
        padding: 0.9rem 1.2rem;
        border-radius: 0.6rem;
        border: 1px solid #ccc;
        min-height: 55px;
    }

    .stSpinner {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 AI PDF Research Assistant")

# Layout columns with gap
left_col, right_col = st.columns([3, 1], gap="large")

# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# PDF Upload section
with right_col:
    st.subheader("📎 Uploaded PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        pdf_paths = []
        os.makedirs("data", exist_ok=True)
        for file in uploaded_files:
            file_path = f"data/{file.name}"
            with open(file_path, "wb") as f:
                f.write(file.read())
            pdf_paths.append(file_path)

        st.markdown(
            "<ul class='uploaded-pdf-list'>" +
            "".join([f"<li>📄 {file.name}</li>" for file in uploaded_files]) +
            "</ul>",
            unsafe_allow_html=True,
        )

        with st.spinner("Processing PDFs..."):
            vectorstore = load_vectorstore_from_uploaded_pdfs(pdf_paths)
        st.success("✅ Ready. Ask any question below.")

# Chat history and input
with left_col:
    for msg in st.session_state.history:
        role_class = "user-bubble" if msg["role"] == "user" else "ai-bubble"
        st.markdown(
            f"<div class='chat-bubble {role_class}'>{msg['content']}</div>",
            unsafe_allow_html=True,
        )

    if uploaded_files:
        question = st.chat_input("Ask a question about the PDFs...")

        if question:
            # Immediately show user's question
            st.session_state.history.append({"role": "user", "content": question})

            # Search vector DB
            results = vectorstore.similarity_search(question, k=5)
            context = "\n".join([r.page_content for r in results])
            prompt = format_prompt(context, question)

            # Stream and collect response
            response = ""
            with st.spinner("AI is thinking..."):
                for chunk in stream_ollama_response(prompt):
                    response += chunk

            # Save AI answer
            st.session_state.history.append({"role": "ai", "content": response})

            # Rerun to refresh UI
            st.rerun()

# Save logs
if "history" in st.session_state and len(st.session_state.history) > 0:
    os.makedirs("logs", exist_ok=True)
    with open("logs/chat_log.txt", "a", encoding="utf-8") as f:
        for msg in st.session_state.history:
            role = "Q" if msg["role"] == "user" else "A"
            f.write(f"{role}: {msg['content'].strip()}\n")
        f.write("\n---\n")
