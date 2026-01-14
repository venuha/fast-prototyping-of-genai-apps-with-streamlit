import streamlit as st
import numpy as np
import openai
from dotenv import load_dotenv

# Safe import for PDF
try:
    import PyPDF2
except Exception as e:
    PyPDF2 = None

load_dotenv()

st.set_page_config(page_title="RAG PDF Assistant", layout="wide")
st.title("📄🔎 RAG PDF Assistant")

# Always show environment/debug info
st.sidebar.header("Debug")
st.sidebar.write("Python OK ✅")
st.sidebar.write("PyPDF2 available:", PyPDF2 is not None)

# Create client safely
try:
    client = openai.OpenAI()
    st.sidebar.write("OpenAI client OK ✅")
except Exception as e:
    st.sidebar.error(f"OpenAI client init failed: {e}")
    st.stop()

def extract_text_from_pdf(uploaded_file) -> str:
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 not installed. Run: python -m pip install PyPDF2")

    reader = PyPDF2.PdfReader(uploaded_file)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()

def chunk_text(text: str, chunk_chars: int = 1800, overlap: int = 200):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == n:
            break
    return chunks

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

def embed_texts(texts, model="text-embedding-3-small"):
    resp = client.embeddings.create(model=model, input=texts)
    vecs = [normalize(np.array(d.embedding, dtype=np.float32)) for d in resp.data]
    return vecs

def retrieve_top_k(query, chunks, vecs, k=5):
    q = embed_texts([query])[0]
    sims = np.dot(np.vstack(vecs), q)
    idxs = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i]), chunks[int(i)]) for i in idxs]

def grounded_answer(question, retrieved):
    context = "\n\n---\n\n".join([f"[Chunk {i}]\n{txt}" for i, _, txt in retrieved])

    resp = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content":
             "Answer ONLY from the provided context. If not found, say you don't know. "
             "Cite sources like [Chunk 2]."},
            {"role": "user", "content": f"QUESTION: {question}\n\nCONTEXT:\n{context}"}
        ],
        temperature=0.2,
        max_output_tokens=700
    )
    return resp.output[0].content[0].text.strip()

st.divider()
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
k = st.slider("Top-K chunks", 2, 10, 5)

if uploaded:
    try:
        text = extract_text_from_pdf(uploaded)
        st.write("Extracted characters:", len(text))

        if len(text) == 0:
            st.warning("No text extracted. This PDF may be scanned images (needs OCR).")
            st.stop()

        chunks = chunk_text(text)
        st.write("Chunks:", len(chunks))

        if st.button("🧱 Build Index"):
            with st.spinner("Embedding chunks..."):
                vecs = []
                for i in range(0, len(chunks), 64):
                    vecs.extend(embed_texts(chunks[i:i+64]))
            st.session_state["chunks"] = chunks
            st.session_state["vecs"] = vecs
            st.success("Index built ✅")

    except Exception as e:
        st.error(f"PDF processing failed: {e}")
        st.stop()

st.divider()
question = st.text_input("Ask a question about the PDF")

if st.button("💬 Ask"):
    if "chunks" not in st.session_state:
        st.warning("Upload a PDF and click Build Index first.")
    elif not question.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Retrieving..."):
            retrieved = retrieve_top_k(question, st.session_state["chunks"], st.session_state["vecs"], k=k)
        st.subheader("Retrieved chunks")
        for i, score, txt in retrieved:
            with st.expander(f"Chunk {i} (score={score:.3f})"):
                st.write(txt)

        with st.spinner("Answering..."):
            ans = grounded_answer(question, retrieved)
        st.subheader("Answer")
        st.write(ans)
