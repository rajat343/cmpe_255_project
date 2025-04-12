import os
from typing import List
import streamlit as st
from pathlib import Path
import hashlib
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document

def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        st.error("Please set OPENAI_API_KEY in your .env file")
        return

    working_dir = os.path.dirname(os.path.abspath(__file__))
    uploaded_pdfs_dir = Path(working_dir) / "uploaded_pdfs"
    embeddings_dir = Path(working_dir) / "embeddings"
    uploaded_pdfs_dir.mkdir(exist_ok=True)
    embeddings_dir.mkdir(exist_ok=True)

    if "qa_cache" not in st.session_state:
        st.session_state.qa_cache = {}

    def get_pdf_hash(pdf_content: bytes) -> str:
        return hashlib.md5(pdf_content).hexdigest()

    def process_pdf(pdf_file) -> tuple[str, Path]:
        pdf_content = pdf_file.read()
        pdf_hash = get_pdf_hash(pdf_content)
        pdf_path = uploaded_pdfs_dir / f"{pdf_hash}.pdf"
        if not pdf_path.exists():
            with open(pdf_path, "wb") as f:
                f.write(pdf_content)
        return pdf_hash, pdf_path

    def get_or_create_vectorstore(pdf_paths: List[Path]) -> FAISS:
        combined_hash = hashlib.md5("".join(sorted([str(p) for p in pdf_paths])).encode()).hexdigest()
        vectorstore_path = embeddings_dir / combined_hash
        embeddings = OpenAIEmbeddings()
        if vectorstore_path.exists():
            return FAISS.load_local(
                str(vectorstore_path), 
                embeddings,
                allow_dangerous_deserialization=True
            )
        documents = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(str(pdf_path))
            documents.extend(loader.load())
        vectorstore = FAISS.from_documents(documents, embeddings)
        total_chars = sum(len(doc.page_content) for doc in documents)
        estimated_tokens = total_chars / 4  # rough estimate
        embedding_cost = (estimated_tokens / 1000) * 0.0001
        st.sidebar.write(f"Embedding cost (estimate): ${embedding_cost:.4f}")
        vectorstore.save_local(str(vectorstore_path))
        return vectorstore

    def setup_chain(vectorstore):
        llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            chain_type="stuff",
            verbose=True
        )
        return chain

    st.set_page_config(page_title="PDF Chat Assistant", page_icon="📚", layout="centered")
    st.title("📚 PDF Chat Assistant")

    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.qa_cache = {}
        st.session_state.pop('conversation_chain', None)

    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        pdf_paths = []
        for pdf_file in uploaded_files:
            pdf_hash, pdf_path = process_pdf(pdf_file)
            pdf_paths.append(pdf_path)

        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = get_or_create_vectorstore(pdf_paths)

        if "conversation_chain" not in st.session_state:
            st.session_state.conversation_chain = setup_chain(st.session_state.vectorstore)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Ask about your PDFs...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.spinner("Processing your query..."):
                if user_input in st.session_state.qa_cache:
                    assistant_response = f"(From cache) {st.session_state.qa_cache[user_input]}"
                    query_cost = 0
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                else:
                    response = st.session_state.conversation_chain({"question": user_input, "chat_history": st.session_state.chat_history})
                    source_documents = response.get("source_documents", [])
                    
                    if source_documents:
                        assistant_response = response["answer"]
                        st.session_state.qa_cache[user_input] = assistant_response
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                    else:
                        assistant_response = "I'm sorry, the information you requested is not available in the uploaded PDF documents. Please try a different query or upload additional documents."
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

                    input_chars = len(user_input) + len(str(source_documents))
                    output_chars = len(assistant_response)
                    input_tokens = input_chars / 4
                    output_tokens = output_chars / 4
                    query_cost = (input_tokens / 1000 * 0.0015) + (output_tokens / 1000 * 0.002)

            with st.chat_message("assistant"):
                st.markdown(assistant_response)
                if query_cost > 0:
                    st.sidebar.write(f"Query cost: ${query_cost:.4f}")

if __name__ == "__main__":
    main()
