# PDF Question Answering System using RAG

## Project Overview
This project is a PDF Question Answering System built using Retrieval-Augmented Generation (RAG). Users can upload a PDF and ask questions about its content. The system retrieves relevant information from the PDF and uses a local language model to generate answers.

## Technologies Used
- Python
- Streamlit
- LangChain
- ChromaDB (Vector Database)
- Hugging Face Sentence Transformers
- Ollama
- Llama 3.2
- PyPDF

## How It Works
1. The user uploads a PDF.
2. The system extracts text from the PDF.
3. The extracted text is divided into smaller chunks.
4. The chunks are converted into embeddings.
5. The embeddings are stored in ChromaDB.
6. When the user asks a question, the system retrieves similar chunks using similarity search.
7. The retrieved context is passed to the Llama 3.2 model through Ollama.
8. The model generates an answer based on the retrieved PDF content.

## Prerequisites
- Python installed
- Ollama installed
- The `llama3.2` model downloaded in Ollama

## Installation

Clone the repository and open the project folder.

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Make sure Ollama is running and the model is available:

```bash
ollama run llama3.2
```

In another terminal, start the application:

```bash
streamlit run streamlit_app.py
```

Open the local URL displayed in the terminal.

## Main Files
- `streamlit_app.py` — Streamlit user interface and PDF question-answering workflow.
- `app.py` — Additional Python application file.
- `requirements.txt` — Python package dependencies.
- `documents/` — Folder for sample PDF documents.

## Features
- Upload PDF documents.
- Ask questions about PDF content.
- Retrieve relevant text using vector similarity search.
- Generate answers using a locally running language model.
- Display relevant source pages.

## Note
The language model and embedding model may need to be downloaded the first time the application runs. Answer quality depends on the PDF content and the retrieved context.
