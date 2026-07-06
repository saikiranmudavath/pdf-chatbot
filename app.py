import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")

# -----------------------
# Load Embedding Model
# -----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------
# Extract PDF Text
# -----------------------
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


# -----------------------
# Better Chunking
# -----------------------
def chunk_text(text, max_chars=500):

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:

        if len(current) + len(para) < max_chars:
            current += para + "\n"

        else:
            chunks.append(current.strip())
            current = para + "\n"

    if current:
        chunks.append(current.strip())

    return chunks


# -----------------------
# Create Vector Index
# -----------------------
def create_index(chunks):

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(np.array(embeddings).astype("float32"))

    return index


# -----------------------
# Search
# -----------------------
def search(query, index, chunks, k=3):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx != -1:
            results.append({
                "score": float(score),
                "text": chunks[idx]
            })

    return results


# -----------------------
# UI
# -----------------------
st.title("📄 PDF Semantic Search")
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

    st.success(f"PDF Ready! ({len(chunks)} chunks indexed)")

    query = st.text_input("Ask a question")

    if query:

        results = search(query, index, chunks)

        if results:

            st.subheader("📌 Best Match")

            st.success(results[0]["text"])

            st.caption(f"Similarity Score: {results[0]['score']:.3f}")

            if len(results) > 1:

                with st.expander("Other Relevant Sections"):

                    for i, res in enumerate(results[1:], start=2):

                        st.markdown(f"### Match {i}")
                        st.caption(f"Score: {res['score']:.3f}")
                        st.write(res["text"])

        else:

            st.warning("No relevant information found.")
