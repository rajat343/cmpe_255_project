# PDF Chat Assistant 📚

A Streamlit-based application that lets you upload PDF documents and interact with their content—both text and images—using natural language. It combines OCR, semantic search, embedded Python execution, exploratory data analysis, and GPT-powered translation in one unified dashboard, all backed by a lightweight SQLite database.

## Key Features

-   **PDF Upload & Storage**  
    Upload one or more PDFs; files are hashed and stored locally under `uploaded_pdfs/`.

-   **SQLite Persistence**  
    All extracted text, images, captions and translation metadata are stored in a local SQLite database (`pdf_text.db`), ensuring fast lookups and schema migrations.

-   **Text Extraction**  
    Extracts raw page text with PyPDF2 and stores it in the `pdf_text` table.

-   **Image Extraction & OCR**  
    Pulls out every embedded image via PyMuPDF, runs PaddleOCR (and optional Tesseract) on each, and saves both the bytes and any detected captions in the `pdf_images` table.

-   **BLIP Image Captioning**  
    Generates high-level captions for each image using Hugging Face’s BLIP model and stores them alongside OCR text for richer retrieval.

-   **Semantic Search (Text + Images)**  
    Vectorizes all text and OCR/caption snippets via OpenAI Embeddings, indexes them with FAISS, and at query time retrieves the top-k semantically closest passages for GPT-4 to answer.

-   **Conversational QA**  
    Uses LangChain’s `ConversationalRetrievalChain` (GPT-4-turbo) to maintain chat history and provide context-aware, source-grounded answers.

-   **Response Caching & Cost Estimation**  
    Caches Q&A pairs for faster follow-ups and shows estimated embedding/query costs in the sidebar.

-   **Python Code Execution**  
    Paste `python …` blocks in chat to run them in a sandboxed namespace; any new Matplotlib figures appear inline.

-   **Exploratory Data Analysis**  
    In the “View EDA” sidebar you’ll find:

    -   Top-10 word frequency bar chart
    -   Sentence-length distribution histogram
    -   Sentiment polarity histogram (via TextBlob)
    -   Part-of-speech distribution bar chart (via spaCy)
    -   Named-entity counts table

-   **Figures & Tables Browser**  
    Automatically detects “Figure N:” or “Table N:” captions in images and lets you browse all such images in a dedicated sidebar.

-   **GPT-Powered PDF Translation**  
    Pick any uploaded PDF and target language (French, Spanish, German, etc.); the app streams page-by-page text through GPT-3.5-turbo, overlays translations back into a new PDF, and lets you download it. All translations are recorded in SQLite for easy retrieval.

-   **Conversation Management**

    -   Clear chat history button
    -   Persistent FAISS index for faster reloads
    -   Memory buffer for multi-turn context

-   **Robustness & Performance**
    -   Automatic schema migrations for new columns
    -   Safe OpenMP & PyTorch workarounds baked into `app.py`
    -   Retry logic (tenacity) for transient translation errors

## Team Members & Responsibilities

-   **Rajat Mishra**: PDF upload & QA pipeline  
    Designed the file-upload/validation flow, FAISS vector store, LangChain document loader, OpenAI embeddings integration, and the ConversationalRetrievalChain for text-based Q&A.

-   **Kunal Goel**: Image/table/chart extraction & EDA  
    Implemented PyMuPDF image extraction, Tesseract OCR, regex-based caption detection, image preprocessing, metadata storage, and the Exploratory Data Analysis dashboards.

-   **Reet Khanchandani**: UI & PDF translation  
    Built the Streamlit interface, GPT-3.5-turbo translation pipeline (segmentation, batching, error handling, progress tracking), format preservation, multi-language support, and translation history in SQLite.

-   **Ritika Khandelwal**: Image/table Q&A & analytics  
    Extended the retrieval chain to include OCR, enabling semantic Q&A over figures and tables, and added custom analysis features (POS tagging, NER, style metrics) via spaCy/TextBlob.

## Prerequisites

-   Python 3.8 or higher
-   Tesseract OCR (if you want Tesseract support)
-   OpenAI API key
-   (Optional) GPU/CUDA for faster BLIP captioning

## Installation

```bash
git clone https://github.com/rajat343/cmpe_255_project.git
cd cmpe_255_project

# Create & activate a venv or conda env
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
# .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```
