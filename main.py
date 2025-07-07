import os
import fitz  # PyMuPDF
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Use updated import if you've run: pip install -U langchain-huggingface
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load .env variables (optional)
load_dotenv()

# Step 1: Extract text from all PDFs in 'data/' folder
def extract_text_from_all_pdfs(folder_path="data"):
    text = ""
    for file in Path(folder_path).glob("*.pdf"):
        print(f"📄 Reading {file.name}")
        doc = fitz.open(file)
        for page in doc:
            text += page.get_text()
    print(f"📄 Extracted text: {len(text)} characters")
    return text

# Step 2: Chunk the text
def chunk_text(text):
    print("🛠️ Chunking text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    print(f"🔢 Created {len(chunks)} chunks")
    return chunks

# Step 3: Create vector DB using HuggingFace + FAISS
def create_vector_db(chunks):
    print("🛠️ Creating vector DB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    print("🔢 Vector DB created")
    return vectorstore

# Step 4: Ask DeepSeek-R1 locally via Ollama subprocess (UTF-8 safe)
def ask_deepseek(question, context):
    prompt = f"""Context:\n{context}\n\nQuestion: {question}\n\nAnswer briefly and clearly:"""
    result = subprocess.run(
        ["ollama", "run", "deepseek-r1:7b"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8").strip()

# Step 5: Save Q&A to log file
def save_answer_to_file(question, answer, file_path="logs/answers_log.txt"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n\n---\n")
        f.write(f"Q: {question.strip()}\n")
        f.write(f"A: {answer.strip()}\n")

# Main Execution
if __name__ == "__main__":
    print("🚀 Starting execution...")
    text = extract_text_from_all_pdfs()

    if text:
        print("✅ PDF text extracted.")
        chunks = chunk_text(text)
        print(f"✅ Total chunks created: {len(chunks)}")
        vector_db = create_vector_db(chunks)
        print("✅ Vector database created.")

        while True:
            query = input("\nAsk a question (or type 'exit' to quit): ")
            if query.lower() == "exit":
                break

            results = vector_db.similarity_search(query, k=2)
            combined_context = "\n".join([r.page_content for r in results])

            print("\n🤖 DeepSeek-R1 says:\n")
            print("🤖 Asking DeepSeek-R1:", query)

            answer = ask_deepseek(query, combined_context)
            print(answer)

            save_answer_to_file(query, answer)
            print("💾 Answer saved to logs/answers_log.txt")
    else:
        print("⚠ No text extracted.")
