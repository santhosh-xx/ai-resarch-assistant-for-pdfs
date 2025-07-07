# 📄 AI Research Assistant for PDFs

Welcome to the **AI Research Assistant for PDFs**! This project enables you to upload PDFs and interactively ask questions about their content using state-of-the-art language models and vector search. Whether you're a researcher, student, or data enthusiast, this tool streamlines document analysis and helps you extract answers from large collections of PDFs—instantly.

---

## 🚀 Features

- **PDF Upload & Parsing**: Seamlessly upload one or more PDF files and extract their text content.
- **Text Chunking**: Automatically splits extracted text into manageable, context-preserving chunks for better search and retrieval.
- **Semantic Search**: Utilizes FAISS and HuggingFace embeddings for fast, accurate similarity search over PDF content.
- **Conversational QA**: Ask questions in natural language and receive context-aware answers powered by a local large language model (LLM) via Ollama and DeepSeek-R1.
- **Interactive UI**: Clean, modern interface built with Streamlit for real-time chat and visualization.
- **Persistent Logging**: All Q&A interactions are logged for future reference and reproducibility.

---

## 🏗️ How It Works

### 1. PDF Extraction
- The system reads each uploaded PDF using PyMuPDF (`fitz`) and extracts the text from every page.

### 2. Text Chunking
- The extracted text is split into overlapping chunks (default: 500 characters with 100 overlap) using `RecursiveCharacterTextSplitter` from LangChain. This improves the accuracy of downstream retrieval.

### 3. Vector Database Creation
- All text chunks are embedded using the `all-MiniLM-L6-v2` model from HuggingFace.
- Embeddings are stored in a FAISS vector database for rapid similarity search.

### 4. Question Answering
- When you ask a question, the app retrieves the most relevant chunks from the database.
- The context and your question are formatted into a prompt and sent to a local LLM (DeepSeek-R1 via Ollama).
- The AI model generates a concise, context-aware answer, which is streamed back to the interface.

### 5. Logging
- Every Q&A pair is logged to `logs/answers_log.txt` or `logs/chat_log.txt` for later auditing or analysis.

---

## 🖥️ Application Structure

### `main.py`
A command-line pipeline for:
- Extracting text from PDFs in the `data/` folder.
- Chunking and embedding the text.
- Creating a vector database.
- Interacting in a loop: You can ask questions, get answers via DeepSeek-R1, and have all Q&A saved to logs.

### `app.py`
A modern, interactive web app utilizing Streamlit:
- Upload PDFs and view a list of uploaded files.
- Automatic vectorstore rebuilding on new uploads.
- Real-time conversational interface with chat bubbles.
- PDF context retrieval and answer streaming via Ollama.
- Persistent chat history and log saving.

### `requirements.txt`
All Python dependencies for environment setup:
```
streamlit
pymupdf
langchain
langchain-community
langchain-huggingface
sentence-transformers
faiss-cpu
ollama
```

### `.gitignore`
Standard ignores for:
- Virtual environments
- Python cache files
- Environment secrets
- Log files

---

## 🛠️ Getting Started

### 1. Clone the repository:
```bash
git clone https://github.com/santhosh-xx/ai-resarch-assistant-for-pdfs.git
cd ai-resarch-assistant-for-pdfs
```

### 2. Install dependencies:
It's recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Set up Ollama and DeepSeek-R1:
- Install and run [Ollama](https://ollama.com/).
- Download the `deepseek-r1:7b` model with Ollama:
  ```bash
  ollama pull deepseek-r1:7b
  ```

### 4. Launch the web app:
```bash
streamlit run app.py
```
Upload your PDFs and start asking questions!

Alternatively, use the command-line tool:
```bash
python main.py
```

---

## 📦 Data & Index Directories

- `data/`: Store your PDF files here for CLI mode or upload via the UI.
- `faiss_index/`: Stores the vector database for efficient retrieval.
- `logs/`: Q&A logs and chat history.

---

## 🤖 Under the Hood

- **PDF Parsing**: Fast, robust extraction using PyMuPDF.
- **Embeddings**: Uses HuggingFace's MiniLM for semantic understanding.
- **Vector Search**: FAISS for lightning-fast similarity queries.
- **LLM**: DeepSeek-R1 running locally, ensuring privacy and low latency.
- **Frontend**: Streamlit with a custom-styled chat UI, supporting interactive, multi-turn conversations.

---

## 💡 Example Usage

- Upload academic papers, reports, or research documents as PDFs.
- Ask questions like “What is the main conclusion of the first document?” or “Summarize the methodology used.”
- Instantly receive context-backed answers, with all sources and interactions logged.

---

## 📝 Customization Tips

- You can tweak the chunk size or embedding model in `main.py` or `app.py` for different document types.
- Change the number of retrieved chunks (`k` in similarity search) to balance accuracy and performance.
- The prompt format is easily adaptable for more creative or constrained AI behavior.

---

## 🛡️ Security & Privacy

- All data remains local; no document or question is sent to a third-party server.
- Logs and vector indexes are stored in your project directory for full transparency.

---

## 📧 Contributions & Issues

Feel free to open an issue or pull request for improvements, bug fixes, or new features!



---

Made with ❤️ by [santhosh-xx](https://github.com/santhosh-xx)
