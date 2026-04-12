import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")

# Load model (cached)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Extract text
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# Chunk text
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Create FAISS index
def create_index(chunks):
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index

# Search
def search(query, index, chunks, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    return [chunks[i] for i in indices[0]]

# UI
st.title("📄 PDF Chatbot (No API Key)")
st.write("Upload a PDF and ask questions from it.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text(uploaded_file)
        chunks = chunk_text(text)
        index = create_index(chunks)

    st.success("✅ PDF Ready!")

    query = st.text_input("Ask your question:")

    if query:
        results = search(query, index, chunks)

        st.subheader("📌 Answer:")
        for res in results:
            st.write(res)
