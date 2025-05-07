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


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set OPENAI_API_KEY in your .env file")

HERE = Path(__file__).parent
UPLOAD_DIR = HERE / "uploaded_pdfs"
EMBEDDINGS_DIR = HERE / "embeddings"
UPLOAD_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)

def main():
    st.set_page_config(page_title="PDF Chat Assistant", page_icon="📚", layout="centered")
    st.title("📚 PDF Chat Assistant")

    if "qa_cache" not in st.session_state:
        st.session_state.qa_cache = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.qa_cache = {}
        st.session_state.pop("vectorstore", None)
        st.session_state.pop("conversation_chain", None)

    uploaded = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    if not uploaded:
        return

    pdf_paths: List[Path] = []
    for f in uploaded:
        _, p = process_pdf(f)
        pdf_paths.append(p)

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
