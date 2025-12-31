import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM

st.set_page_config(page_title="Resume RAG Bot")
st.title("📄 Resume RAG Chatbot")

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type="pdf",
    accept_multiple_files=True
)

all_text = ""

if uploaded_files:
    for pdf in uploaded_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            all_text += page.extract_text() + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(all_text)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    st.success("✅ Resumes indexed successfully")

    query = st.text_input("Ask about a candidate (skills, experience, name)")

    if query:
        docs = vectorstore.similarity_search(query, k=4)

        context = "\n\n".join([doc.page_content for doc in docs])

        llm = OllamaLLM(model="llama3")

        prompt = f"""
You are an HR assistant.
Answer the question using ONLY the resume content below.
If not found, say "Information not available".

RESUME DATA:
{context}

QUESTION:
{query}
"""

        answer = llm.invoke(prompt)

        st.subheader("🤖 Answer")
        st.write(answer)
