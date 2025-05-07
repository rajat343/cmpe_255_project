import os
import hashlib
from pathlib import Path
from typing import List, Tuple
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks.manager import Callbacks
from langchain_core.caches import BaseCache  
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
import fitz  # PyMuPDF for image extraction
import sqlite3  # For storing extracted text and images
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
from PyPDF2 import PdfReader

ChatOpenAI.model_rebuild()

def get_pdf_hash(pdf_content: bytes) -> str:
    return hashlib.md5(pdf_content).hexdigest()

def process_pdf(pdf_file) -> Tuple[str, Path]:
    content = pdf_file.read()
    h = get_pdf_hash(content)
    dest = UPLOAD_DIR / f"{h}.pdf"
    if not dest.exists():
        dest.write_bytes(content)
    return h, dest

def is_pdf_processed(pdf_hash: str, db_path: Path) -> bool:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pdf_text WHERE hash = ?", (pdf_hash,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists

def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    text_pages = [page.extract_text() for page in reader.pages if page.extract_text()]
    return "\n".join(text_pages)

def extract_images_from_pdf(pdf_path: Path, save_dir: Path, pdf_hash: str, db_path: Path) -> List[str]:
    doc = fitz.open(str(pdf_path))
    image_paths = []
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Remove old images for this PDF
    cur.execute("DELETE FROM pdf_images WHERE pdf_hash = ?", (pdf_hash,))

    for page_index in range(len(doc)):
        for img_index, img in enumerate(doc[page_index].get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            image_path = save_dir / f"{pdf_path.stem}_page{page_index+1}_img{img_index+1}.{ext}"
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            image_paths.append(str(image_path))
            # store in DB
            cur.execute("""
                INSERT INTO pdf_images (pdf_hash, filename, image_data, page_number, img_index, ext)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (pdf_hash, pdf_path.name, image_bytes, page_index + 1, img_index + 1, ext))

    conn.commit()
    conn.close()
    return image_paths

def init_text_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pdf_text (
            hash TEXT PRIMARY KEY,
            filename TEXT,
            content TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pdf_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_hash TEXT,
            filename TEXT,
            image_data BLOB,
            page_number INTEGER,
            img_index INTEGER,
            ext TEXT
        )
    """)
    conn.commit()
    conn.close()

def extract_and_store_text(pdf_path: Path, pdf_hash: str, db_path: Path) -> str:
    full_text = extract_text_from_pdf(pdf_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM pdf_text WHERE hash = ?", (pdf_hash,))
    cur.execute(
        "INSERT INTO pdf_text (hash, filename, content) VALUES (?, ?, ?)",
        (pdf_hash, pdf_path.name, full_text)
    )
    conn.commit()
    conn.close()
    return full_text

def get_or_create_vectorstore(pdf_paths: List[Path]) -> FAISS:
    combined = hashlib.md5("".join(sorted(map(str, pdf_paths))).encode()).hexdigest()
    vs_path = EMBEDDINGS_DIR / combined
    embeddings = OpenAIEmbeddings()
    if vs_path.exists():
        return FAISS.load_local(str(vs_path), embeddings, allow_dangerous_deserialization=True)
    docs: List[Document] = []
    for p in pdf_paths:
        docs.extend(PyPDFLoader(str(p)).load())
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(str(vs_path))
    total_chars = sum(len(d.page_content) for d in docs)
    est_tokens = total_chars / 4
    st.sidebar.write(f"Embedding cost (est): ${est_tokens/1000*0.0001:.4f}")
    return vs

def setup_chain(vs: FAISS) -> ConversationalRetrievalChain:
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vs.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        chain_type="stuff",
        verbose=True
    )

def display_stored_images(db_path: Path):
    with st.sidebar.expander("📷 View Extracted Images", expanded=False):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT filename FROM pdf_images")
        files = [row[0] for row in cur.fetchall()]
        selected_file = st.selectbox("Select a PDF", files, key="image_viewer_select")
        if selected_file:
            cur.execute("SELECT image_data, page_number, img_index, ext FROM pdf_images WHERE filename = ?", (selected_file,))
            for img_data, page_num, img_idx, ext in cur.fetchall():
                image = Image.open(io.BytesIO(img_data))
                st.image(
                    image,
                    caption=f"{selected_file} – Page {page_num}, Image {img_idx}",
                    use_container_width=True
                )
        conn.close()

def display_stored_text(db_path: Path):
    with st.sidebar.expander("📝 View Extracted Text", expanded=False):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT filename FROM pdf_text")
        files = [row[0] for row in cur.fetchall()]
        selected_file = st.selectbox("Select a PDF to view its text", files, key="text_viewer_select")
        if selected_file:
            cur.execute("SELECT content FROM pdf_text WHERE filename = ?", (selected_file,))
            result = cur.fetchone()
            if result:
                preview = result[0][:50000]
                st.text_area("Preview:", value=preview, height=400)
        conn.close()

def display_stored_eda(db_path: Path):
    with st.sidebar.expander("📈 View EDA (Word & Sentence Analysis)", expanded=False):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT filename FROM pdf_text")
        files = [row[0] for row in cur.fetchall()]
        selected_file = st.selectbox("Select a PDF for EDA", files, key="eda_viewer_select")
        if selected_file:
            cur.execute("SELECT content FROM pdf_text WHERE filename = ?", (selected_file,))
            text = cur.fetchone()[0]
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts = Counter(words)
            df = pd.DataFrame(word_counts.most_common(10), columns=["Word", "Frequency"])
            st.dataframe(df)
            fig, ax = plt.subplots()
            df.plot(kind='bar', x='Word', y='Frequency', ax=ax, legend=False)
            ax.set_title("Top 10 Words")
            st.pyplot(fig)

            sentences = re.split(r'[.!?]', text)
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths:
                avg_len = round(sum(lengths) / len(lengths), 2)
                st.markdown(f"Average sentence length: **{avg_len}** words")
                fig2, ax2 = plt.subplots()
                ax2.hist(lengths, bins=20, edgecolor='black')
                ax2.set_title("Sentence Length Distribution")
                ax2.set_xlabel("Words per sentence")
                ax2.set_ylabel("Frequency")
                st.pyplot(fig2)
        conn.close()

# ──────────────────────────────────────────────────────────

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set OPENAI_API_KEY in your .env file")

HERE = Path(__file__).parent
UPLOAD_DIR = HERE / "uploaded_pdfs"
EMBEDDINGS_DIR = HERE / "embeddings"
IMAGES_DIR = HERE / "extracted_images"
TEXT_DB_PATH = HERE / "pdf_text.db"
UPLOAD_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

init_text_db(TEXT_DB_PATH)

def main():
    st.set_page_config(page_title="PDF Chat Assistant", page_icon="📚", layout="centered")
    st.title("📚 PDF Chat Assistant")

    # Initialize chat/session state
    if "qa_cache" not in st.session_state:
        st.session_state.qa_cache = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.qa_cache = {}
        st.session_state.pop("vectorstore", None)
        st.session_state.pop("conversation_chain", None)

    # --- File upload & processing ---
    uploaded = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    new_uploads = False
    pdf_paths: List[Path] = []

    if uploaded:
        for f in uploaded:
            h, p = process_pdf(f)
            pdf_paths.append(p)

            if not is_pdf_processed(h, TEXT_DB_PATH):
                # first‐time extraction
                imgs = extract_images_from_pdf(p, IMAGES_DIR, h, TEXT_DB_PATH)
                st.sidebar.write(f"Extracted {len(imgs)} images from {p.name}")
                txt = extract_and_store_text(p, h, TEXT_DB_PATH)
                st.sidebar.write(f"Stored text from {p.name}")
                new_uploads = True

        if new_uploads:
            # force rebuild vectorstore & chain to include new docs
            st.session_state.pop("vectorstore", None)
            st.session_state.pop("conversation_chain", None)

    # --- Display stored content dropdowns dynamically ---
    display_stored_images(TEXT_DB_PATH)
    display_stored_text(TEXT_DB_PATH)
    display_stored_eda(TEXT_DB_PATH)

    # if no files, stop here
    if not uploaded:
        return

    # --- Embeddings & Chat chain setup (cached across runs) ---
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = get_or_create_vectorstore(pdf_paths)

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = setup_chain(st.session_state.vectorstore)

    # --- Render prior chat history ---
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- New user query ---
    q = st.chat_input("Ask about your PDFs…")
    if not q:
        return

    st.session_state.chat_history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.spinner("Processing your query…"):
        if q in st.session_state.qa_cache:
            answer = "(From cache) " + st.session_state.qa_cache[q]
        else:
            resp = st.session_state.conversation_chain(
                {"question": q, "chat_history": st.session_state.chat_history}
            )
            docs = resp.get("source_documents", [])
            if docs:
                answer = resp["answer"]
                st.session_state.qa_cache[q] = answer
            else:
                answer = (
                    "I’m sorry, the information you requested is not in the uploaded PDFs. "
                    "Try a different query or upload more documents."
                )

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

if __name__ == "__main__":
    main()
