# PDF Chat Assistant 📚

A Streamlit-based application that allows users to upload PDF documents and interact with their content using natural language queries. The app uses OpenAI's GPT-4 model and FAISS vector storage for efficient document retrieval and question answering.

## Features

-   PDF document upload and processing
-   Natural language querying of PDF content
-   Conversation memory for context-aware responses
-   Response caching for improved performance
-   Cost estimation for embeddings and queries
-   Clear chat history functionality
-   Vector store persistence for faster subsequent loads

## Prerequisites

-   Python 3.8 or higher
-   OpenAI API key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/rajat343/cmpe_255_project.git
cd cmpe_255_project
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root directory and add your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the Streamlit app:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`).

3. Upload one or more PDF documents using the file uploader.

4. Start asking questions about the content of your PDFs in the chat interface.

## Project Structure

```
cmpe_255_project/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
├── uploaded_pdfs/     # Directory for stored PDFs
└── embeddings/        # Directory for stored vector embeddings
```

## Features in Detail

### Document Processing

-   PDFs are processed once and stored locally
-   Document embeddings are cached for faster subsequent loads
-   Multiple PDFs can be processed simultaneously

### Conversation Management

-   Maintains conversation history for context-aware responses
-   Allows clearing of chat history
-   Caches question-answer pairs for improved performance

### Cost Management

-   Displays estimated costs for embeddings generation
-   Shows per-query costs for API usage
-   Uses efficient retrieval methods to minimize API calls

## Technical Implementation

The application uses:

-   `langchain` for document processing and chat chain management
-   `FAISS` for efficient vector similarity search
-   `OpenAI's GPT-4` for generating responses
-   `Streamlit` for the web interface
-   Document hashing for efficient storage and retrieval

## Limitations

-   PDF processing may take longer for large documents
-   API costs can accumulate with heavy usage
-   Requires stable internet connection for API calls
-   Maximum token limit applies based on GPT-4 model constraints

## Cost Considerations

The application uses OpenAI's API which has associated costs:

-   Embedding generation: $0.0001 per 1K tokens
-   Query processing:
    -   Input: $0.0015 per 1K tokens
    -   Output: $0.002 per 1K tokens

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Acknowledgments

-   Built with [Streamlit](https://streamlit.io/)
-   Uses [LangChain](https://python.langchain.com/) for document processing
-   Powered by [OpenAI](https://openai.com/) GPT-4
