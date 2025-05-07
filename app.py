import os
import hashlib
from pathlib import Path
from typing import List, Tuple
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_core.callbacks.manager import Callbacks
from langchain_core.caches import BaseCache
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document

import collections
import sqlite3
import fitz  # PyMuPDF
import matplotlib.pyplot as plt

ChatOpenAI.model_rebuild()

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set OPENAI_API_KEY in your .env file")

HERE = Path(__file__).parent
UPLOAD_DIR = HERE / "uploaded_pdfs"
EMBEDDINGS_DIR = HERE / "embeddings"
UPLOAD_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)

DB_PATH = HERE / "pdf_data.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS extracted_text (
        pdf_hash TEXT,
        page_num INTEGER,
        content TEXT
    )
    """
)
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS extracted_images (
        pdf_hash TEXT,
        page_num INTEGER,
        img_index INTEGER,
        image_path TEXT
    )
    """
)
conn.commit()

def get_pdf_hash(pdf_content: bytes) -> str:
    return hashlib.md5(pdf_content).hexdigest()

def process_pdf(pdf_file) -> Tuple[str, Path]:
    content = pdf_file.read()
    h = get_pdf_hash(content)
    dest = UPLOAD_DIR / f"{h}.pdf"
    if not dest.exists():
        dest.write_bytes(content)
    return h, dest

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
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
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


def extract_and_store(pdf_paths: List[Path]) -> None:
    """Extract text & images and insert into SQLite tables"""
    for p in pdf_paths:
        h = p.stem
        doc = fitz.open(p)
        for i, page in enumerate(doc):
            text = page.get_text()
            cur.execute(
                "INSERT INTO extracted_text VALUES (?,?,?)",
                (h, i, text),
            )
            for img_idx, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                img_path = UPLOAD_DIR / f"{h}_page{i}_img{img_idx}.png"
                pix.save(img_path)
                cur.execute(
                    "INSERT INTO extracted_images VALUES (?,?,?,?)",
                    (h, i, img_idx, str(img_path)),
                )
        conn.commit()


def run_text_eda(pdf_hash: str):
    cur.execute("SELECT content FROM extracted_text WHERE pdf_hash=?", (pdf_hash,))
    rows = cur.fetchall()
    if not rows:
        return None
    all_text = " ".join(r[0] for r in rows)
    words = all_text.split()
    freq = collections.Counter(words)
    top10 = freq.most_common(10)
    if not top10:
        return None
    labels, counts = zip(*top10)
    fig, ax = plt.subplots()
    ax.bar(labels, counts)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(f"Top 10 words for {pdf_hash}")
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="PDF Chat Assistant", page_icon="📚", layout="centered")
    st.title("📚 PDF Chat Assistant")

    if "qa_cache" not in st.session_state:
        st.session_state.qa_cache = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    uploaded = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    if not uploaded:
        return

    pdf_paths: List[Path] = []
    for f in uploaded:
        _, p = process_pdf(f)
        pdf_paths.append(p)

    # extract & store data
    extract_and_store(pdf_paths)

    if st.sidebar.button("Run EDA"):
        for p in pdf_paths:
            fig = run_text_eda(p.stem)
            if fig:
                st.pyplot(fig)

    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.qa_cache = {}
        st.session_state.pop("vectorstore", None)
        st.session_state.pop("conversation_chain", None)

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = get_or_create_vectorstore(pdf_paths)

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = setup_chain(st.session_state.vectorstore)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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
