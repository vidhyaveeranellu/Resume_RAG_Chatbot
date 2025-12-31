# Resume RAG Chatbot 🧠📄

A Retrieval-Augmented Generation (RAG) chatbot that allows HRs to upload bulk resume PDFs and ask questions about candidates.

## 🔧 Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- Ollama (Local LLM)
- nomic-embed-text
- llama3

## 🚀 Features
- Upload multiple resumes (PDF)
- Semantic search on resumes
- Ask questions about candidates
- No OpenAI API key required
- Fully local LLM

## 🖥️ Setup Instructions

### 1️⃣ Install Python
Python 3.10 or above

### 2️⃣ Install Ollama
https://ollama.com

Pull models:
```bash
ollama pull nomic-embed-text
ollama pull llama3
