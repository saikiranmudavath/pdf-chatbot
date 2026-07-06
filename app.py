import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import ollama

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")

# -------------------------------
# Load Embedding Model
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------------------
# Extract PDF Text
# -------------------------------
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -------------------------------
# Better Chunking
# -------------------------------
def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_text(text)


# -------------------------------
# Create Vector Index
# -------------------------------
def create_index(chunks):

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(np.array(embeddings))

    return index


# -------------------------------
# Retrieve Context
# -------------------------------
def retrieve(query, index, chunks, k=3):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        np.array(query_embedding),
        k
    )

    retrieved_chunks = []

    for idx in indices[0]:
        retrieved_chunks.append(chunks[idx])

    return "\n\n".join(retrieved_chunks)


# -------------------------------
# Ask Ollama
# -------------------------------
def ask_llm(question, context):

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the information given below.

If the answer is not present in the context, reply:

"I couldn't find that information in the uploaded PDF."

Keep your answer short and precise.

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# -------------------------------
# UI
# -------------------------------
st.title("📄 PDF Chatbot (Offline - Ollama)")
st.write("Upload a PDF and ask questions.")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    with st.spinner("Processing PDF..."):

        text = extract_text(uploaded_file)

        chunks = chunk_text(text)

        index = create_index(chunks)

    st.success("✅ PDF Ready!")

    question = st.text_input("Ask your question")

    if question:

        with st.spinner("Thinking..."):

            context = retrieve(
                question,
                index,
                chunks
            )

            answer = ask_llm(
                question,
                context
            )

        st.subheader("📌 Answer")

        st.success(answer)

        with st.expander("Retrieved Context"):

            st.write(context)
