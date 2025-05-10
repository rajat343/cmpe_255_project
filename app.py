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
import fitz  # PyMuPDF for image extraction and PDF translation
import sqlite3  # For storing extracted text and images
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
from PyPDF2 import PdfReader
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
import pytesseract

ChatOpenAI.model_rebuild()

# Retry logic for transient errors (from layour.py)
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def translate_with_gpt(text, target_language):
    prompt = f"""Translate the following text to {target_language}. Only return the translated text without commentary.

{text}
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response["choices"][0]["message"]["content"].strip()

def split_text(text, max_chars=1500):
    # Split large blocks into smaller paragraphs
    paragraphs = text.split("\n")
    batches, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += para + "\n"
        else:
            batches.append(current.strip())
            current = para + "\n"
    if current.strip():
        batches.append(current.strip())
    return batches

def translate_pdf_with_gpt(input_pdf, output_pdf, target_language="French"):
    doc = fitz.open(input_pdf)
    
    progress_bar = st.progress(0)
    total_pages = len(doc)
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")
        for i, block in enumerate(blocks):
            x0, y0, x1, y1, text, *_ = block
            if text.strip():
                try:
                    translated = ""
                    if len(text) < 1500:
                        translated = translate_with_gpt(text, target_language)
                    else:
                        parts = split_text(text)
                        translated_parts = [translate_with_gpt(p, target_language) for p in parts]
                        translated = "\n".join(translated_parts)

                    # Overwrite original text
                    rect = fitz.Rect(x0, y0, x1, y1)
                    page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
                    page.insert_text((x0, y0), translated, fontsize=10)
                except Exception as e:
                    st.error(f"[Page {page_num+1}] Translation error: {e}")
        
        # Update progress bar
        progress_bar.progress((page_num + 1) / total_pages)
        
    doc.save(output_pdf)
    return output_pdf

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

def ocr_contains_caption(image: Image.Image) -> str:
    try:
        text = pytesseract.image_to_string(image)
        for pattern in [r"(Figure|Table|Chart)\s*\d+[:\.]?", r"(Exhibit|Diagram)\s*\d+[:\.]?"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group()
    except Exception:
        pass
    return None

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

            caption_text = None
            try:
                pil_img = Image.open(image_path)
                caption_text = ocr_contains_caption(pil_img)
            except Exception:
                pass
            # store in DB
            cur.execute("""
                INSERT INTO pdf_images (pdf_hash, filename, image_data, page_number, img_index, ext, caption)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (pdf_hash, pdf_path.name, image_bytes, page_index + 1, img_index + 1, ext, caption_text))

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
            ext TEXT,
            caption TEXT
        )
    """)
    # Add a new table for translated PDFs
    cur.execute("""
        CREATE TABLE IF NOT EXISTS translated_pdfs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_hash TEXT,
            original_filename TEXT,
            translated_hash TEXT,
            translated_filename TEXT,
            target_language TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
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
    print(f"Embedding cost (est): ${est_tokens/1000*0.0001:.4f}")
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

def display_figures_tables(db_path: Path):
    with st.sidebar.expander("📊 View Extracted Figures/Tables", expanded=False):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Pull filenames that have OCR-detected captions
        cur.execute("""
            SELECT DISTINCT filename FROM pdf_images WHERE caption IS NOT NULL
        """)
        files = [row[0] for row in cur.fetchall()]
        if not files:
            st.info("No OCR-detected figures/tables found. Try uploading a PDF with diagrams or tables.")
            conn.close()
            return
        
        selected_file = st.selectbox("Select a PDF", files, key="fig_table_select")

        if selected_file:
            st.markdown("**📌 Figures & Tables Detected via OCR**")
            cur.execute("""
                SELECT page_number, img_index, caption, image_data FROM pdf_images
                WHERE filename = ? AND caption IS NOT NULL ORDER BY page_number
            """, (selected_file,))
            
            results = cur.fetchall()

            if results:
                for page, idx, caption, img_data in results:
                    image = Image.open(io.BytesIO(img_data))
                    st.image(image, caption=f"{caption} (Page {page})", use_container_width=True)
            else:
                st.warning("No OCR-detected captions found in this file.")

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

def display_translation_ui(db_path: Path):
    with st.sidebar.expander("🌍 Translate PDFs", expanded=False):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT filename FROM pdf_text")
        files = [row[0] for row in cur.fetchall()]
        
        if not files:
            st.info("Upload PDFs to enable translation")
            conn.close()
            return
            
        selected_file = st.selectbox("Select a PDF to translate", files, key="translation_select")
        
        languages = [
            "French", "Spanish", "German", "Italian", "Portuguese", 
            "Russian", "Japanese", "Chinese", "Arabic", "Hindi"
        ]
        target_language = st.selectbox("Select target language", languages, key="language_select")
        
        if st.button("Translate PDF", key="translate_button"):
            if selected_file:
                with st.spinner(f"Translating {selected_file} to {target_language}..."):
                    # Get the original file path
                    cur.execute("SELECT hash FROM pdf_text WHERE filename = ?", (selected_file,))
                    original_hash = cur.fetchone()[0]
                    original_path = UPLOAD_DIR / f"{original_hash}.pdf"
                    
                    # Create a new filename for the translated version
                    translated_filename = f"{selected_file.split('.')[0]}_{target_language}.pdf"
                    translated_path = TRANSLATED_DIR / translated_filename
                    
                    # Perform translation
                    try:
                        result_path = translate_pdf_with_gpt(str(original_path), str(translated_path), target_language)
                        
                        # Store the translated file info in the DB
                        translated_hash = get_pdf_hash(Path(result_path).read_bytes())
                        cur.execute("""
                            INSERT INTO translated_pdfs 
                            (original_hash, original_filename, translated_hash, translated_filename, target_language)
                            VALUES (?, ?, ?, ?, ?)
                        """, (original_hash, selected_file, translated_hash, translated_filename, target_language))
                        conn.commit()
                        
                        # Provide download link
                        with open(result_path, "rb") as file:
                            st.download_button(
                                label=f"Download {target_language} PDF",
                                data=file,
                                file_name=translated_filename,
                                mime="application/pdf"
                            )
                        st.success(f"Translation to {target_language} completed!")
                    except Exception as e:
                        st.error(f"Translation failed: {e}")
        
        # Show previously translated PDFs
        st.subheader("Previously Translated PDFs")
        cur.execute("""
            SELECT translated_filename, target_language, original_filename 
            FROM translated_pdfs 
            ORDER BY timestamp DESC
        """)
        translated_files = cur.fetchall()
        
        if translated_files:
            for t_filename, t_language, o_filename in translated_files:
                st.text(f"{o_filename} → {t_language}")
                translated_path = TRANSLATED_DIR / t_filename
                if translated_path.exists():
                    with open(translated_path, "rb") as file:
                        st.download_button(
                            label=f"Download {t_filename}",
                            data=file,
                            file_name=t_filename,
                            mime="application/pdf",
                            key=f"dl_{t_filename}"
                        )
        else:
            st.info("No translated PDFs yet")
        
        conn.close()

def run_user_code(code: str):
    before = set(plt.get_fignums())
    user_ns = {
        "plt": plt,
        "pd": pd,
        "io": io,
        "Image": Image,
        "st": st,
    }
    try:
        exec(code, user_ns)
    except Exception as e:
        return f"❌ Error running code: {e}"
    after = set(plt.get_fignums())
    new_figs = after - before
    if not new_figs:
        return "⚠️ No new figures were created by your code."
    for fig_num in sorted(new_figs):
        fig = plt.figure(fig_num)
        st.pyplot(fig)
        plt.close(fig)
    return "✅ Executed your code and displayed the new figure(s)."

# ──────────────────────────────────────────────────────────

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set OPENAI_API_KEY in your .env file")
openai.api_key = OPENAI_API_KEY  # Set for the openai module

HERE = Path(__file__).parent
UPLOAD_DIR = HERE / "uploaded_pdfs"
EMBEDDINGS_DIR = HERE / "embeddings"
IMAGES_DIR = HERE / "extracted_images"
TRANSLATED_DIR = HERE / "translated_pdfs"  # New directory for translated PDFs
TEXT_DB_PATH = HERE / "pdf_text.db"
UPLOAD_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
TRANSLATED_DIR.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

init_text_db(TEXT_DB_PATH)

def main():
    st.set_page_config(page_title="PDF Chat Assistant", page_icon="📚", layout="wide")
    st.title("📚 PDF Chat Assistant")

    # Session state
    st.session_state.setdefault("qa_cache", {})
    st.session_state.setdefault("chat_history", [])

    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history.clear()
        st.session_state.qa_cache.clear()
        for k in ["vectorstore", "conversation_chain"]:
            st.session_state.pop(k, None)

    # PDF upload & processing
    uploaded = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    new_uploads = False
    pdf_paths: List[Path] = []

    if uploaded:
        for f in uploaded:
            h, p = process_pdf(f)
            pdf_paths.append(p)

            if not is_pdf_processed(h, TEXT_DB_PATH):
                extract_images_from_pdf(p, IMAGES_DIR, h, TEXT_DB_PATH)
                extract_and_store_text(p, h, TEXT_DB_PATH)
                new_uploads = True
        if new_uploads:
            for k in ["vectorstore", "conversation_chain"]:
                st.session_state.pop(k, None)

    # Sidebars: images, text, EDA, translation (new)
    display_stored_images(TEXT_DB_PATH)
    display_stored_text(TEXT_DB_PATH)
    display_figures_tables(TEXT_DB_PATH)
    display_stored_eda(TEXT_DB_PATH)
    display_translation_ui(TEXT_DB_PATH)  # Add the translation UI

    if not uploaded:
        return

    # Initialize or reuse vectorstore & chain
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = get_or_create_vectorstore(pdf_paths)

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = setup_chain(st.session_state.vectorstore)

    # Render prior chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New user input
    q = st.chat_input("Ask about PDFs, paste Python code in ```python…``` to run it.")
    if not q:
        return

    # Echo user
    st.session_state.chat_history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    # 1) Check for Python code blocks
    code_blocks = re.findall(r"```python(.*?)```", q, flags=re.DOTALL)
    if code_blocks:
        # Run each block
        for block in code_blocks:
            result = run_user_code(block)
            st.markdown(result)
        answer = "Executed your Python code above."
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
        return

    # 2) Otherwise fallback to PDF Q&A
    with st.spinner("Processing your query…"):
        if q in st.session_state.qa_cache:
            answer = "(From cache) " + st.session_state.qa_cache[q]
        else:
            resp = st.session_state.conversation_chain({
                "question": q,
                "chat_history": st.session_state.chat_history
            })
            docs = resp.get("source_documents", [])
            if docs:
                answer = resp["answer"]
                st.session_state.qa_cache[q] = answer
            else:
                answer = (
                    "I'm sorry, the information you requested is not in the uploaded PDFs. "
                    "Try a different query or upload more documents."
                )

    # Display answer
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

if __name__ == "__main__":
    main()
