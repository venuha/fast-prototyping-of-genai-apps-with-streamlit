import streamlit as st
import openai
from dotenv import load_dotenv
import PyPDF2

load_dotenv()
client = openai.OpenAI()

st.set_page_config(page_title="PDF Summarizer", layout="wide")
st.title("📄 GenAI PDF Summarizer")

st.write("Upload a PDF from your computer, extract its text, and summarize it using GenAI.")

# ----------------------------
# Helpers
# ----------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract text from a text-based PDF using PyPDF2.
    Note: If the PDF is scanned images, this will return little/no text.
    """
    reader = PyPDF2.PdfReader(uploaded_file)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts).strip()

def chunk_text(text: str, chunk_size: int = 8000):
    """Simple chunking to handle long documents."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def summarize_with_openai(text: str, style: str = "Bullets") -> str:
    system_msg = (
        "You are a careful assistant that summarizes documents. "
        "Be accurate and do not invent details. If content is unclear, say so."
    )

    style_map = {
        "Short (5-7 lines)": "Summarize in 5-7 lines.",
        "Bullets": "Summarize in bullet points with key themes.",
        "Detailed": "Provide a detailed summary with headings and key points.",
        "Action Items": "List action items, decisions, risks, and open questions if any."
    }

    user_msg = (
        f"{style_map.get(style, 'Summarize clearly.')}\n\n"
        f"DOCUMENT:\n{text}"
    )

    resp = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_output_tokens=800,
    )
    return resp.output[0].content[0].text.strip()

def summarize_large_pdf(full_text: str, style: str) -> str:
    """
    Two-pass approach:
    1) summarize each chunk
    2) summarize all chunk summaries into final output
    """
    chunks = chunk_text(full_text, chunk_size=8000)
    chunk_summaries = []

    for i, ch in enumerate(chunks, start=1):
        with st.spinner(f"Summarizing chunk {i}/{len(chunks)}..."):
            chunk_summaries.append(summarize_with_openai(ch, style="Bullets"))

    combined = "\n\n".join([f"Chunk {i} summary:\n{s}" for i, s in enumerate(chunk_summaries, start=1)])

    with st.spinner("Creating final summary..."):
        final = summarize_with_openai(combined, style=style)

    return final

# ----------------------------
# UI
# ----------------------------
uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

style = st.selectbox(
    "Summary style",
    ["Short (5-7 lines)", "Bullets", "Detailed", "Action Items"]
)

col1, col2 = st.columns([1, 1])
summarize_btn = col1.button("🧠 Summarize PDF", type="primary")
clear_btn = col2.button("🧹 Clear")

if clear_btn:
    st.session_state.pop("pdf_text", None)
    st.session_state.pop("summary", None)
    st.success("Cleared!")

if uploaded_pdf is not None:
    try:
        pdf_text = extract_text_from_pdf(uploaded_pdf)

        if not pdf_text:
            st.warning(
                "No text could be extracted. This PDF might be scanned images. "
                "If you want, I can give you OCR-based code next."
            )
        else:
            st.session_state["pdf_text"] = pdf_text

            with st.expander("🔎 Preview extracted text"):
                st.text_area("Extracted text (preview)", pdf_text[:12000], height=250)

            st.info(f"Extracted characters: {len(pdf_text):,}")

    except Exception as e:
        st.error(f"Failed to read PDF: {e}")

if summarize_btn:
    if "pdf_text" not in st.session_state or not st.session_state["pdf_text"].strip():
        st.warning("Upload a PDF with extractable text first.")
    else:
        text = st.session_state["pdf_text"]

        if len(text) > 12000:
            summary = summarize_large_pdf(text, style=style)
        else:
            with st.spinner("Summarizing..."):
                summary = summarize_with_openai(text, style=style)

        st.session_state["summary"] = summary

if "summary" in st.session_state:
    st.subheader("✅ Summary")
    st.write(st.session_state["summary"])

    st.download_button(
        "⬇️ Download Summary",
        data=st.session_state["summary"].encode("utf-8"),
        file_name="pdf_summary.txt",
        mime="text/plain"
    )
