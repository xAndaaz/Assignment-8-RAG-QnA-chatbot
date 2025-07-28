# RAG Q&A Chatbot

This project is a sophisticated question-answering chatbot that leverages the Retrieval-Augmented Generation (RAG) architecture. It uses document retrieval from a local vector store and a powerful Large Language Model (LLM) to provide intelligent, context-aware responses based on a given set of documents.

## Features

- **Document Processing:** Ingests and processes PDF documents from a local directory.
- **Vector Embeddings:** Uses Hugging Face `sentence-transformers` to create high-quality vector embeddings locally and for free.
- **Vector Store:** Stores and retrieves document embeddings efficiently using `FAISS` (Facebook AI Similarity Search).
- **Generative AI:** Integrates with Google's Gemini Pro for intelligent response generation.
- **Contextual Answers:** Ensures answers are based on the information present in the provided documents, minimizing hallucinations.
- **Secure API Key Management:** Uses `.env` files to keep credentials safe and out of source code.

## Tech Stack

- **Python**
- **LangChain**: The core framework for building the RAG pipeline.
- **Google Generative AI (Gemini Pro)**: The LLM for generating answers.
- **Hugging Face Sentence Transformers**: The model for creating text embeddings.
- **FAISS**: The library for efficient similarity search and vector storage.
- **PyPDF**: For loading and parsing PDF documents.

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Assignment-8-RAG-QnA-chatbot.git
cd Assignment-8-RAG-QnA-chatbot
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required Python libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key

The chatbot requires a Google API key to use the Gemini model.

1.  Create a file named `.env` in the root directory of the project.
2.  Add your Google API key to the `.env` file as follows:

    ```
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    ```

## Usage

### 1. Add Your Documents

Place all the PDF files you want the chatbot to learn from into the `documents` directory. If the directory does not exist, the ingestion script will create it for you.

### 2. Create the Vector Store

Before running the chatbot, you must process your documents and create the vector store. Run the ingestion script:

```bash
python ingest.py
```

This script will:
- Load the PDFs from the `documents` folder.
- Split them into smaller, manageable chunks.
- Generate embeddings using a Hugging Face model.
- Create a `FAISS` index and save it to the `vectorstore` directory.

### 3. Run the Chatbot

Once the vector store is created, you can start the chatbot.

```bash
python main.py
```

The application will load the vector store and the LLM, and you can begin asking questions in the terminal. To quit the application, type `exit`.

## Project Structure

```
.
├── documents/          # Add your source PDF documents here
├── vectorstore/        # Stores the FAISS vector index (auto-generated)
├── .env                # For storing your API keys (create this yourself)
├── .gitignore          # Specifies files to be ignored by Git
├── ingest.py           # Script to process documents and create the vector store
├── main.py             # Main application script to run the chatbot
├── README.md           # Project documentation
└── requirements.txt    # Lists the Python dependencies
```
