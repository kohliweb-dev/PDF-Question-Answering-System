import streamlit as st
import hashlib
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


st.set_page_config(
    page_title="PDF Question Answering System Using RAG"
)

st.title("PDF Question Answering System")
st.write("Upload a PDF and ask questions about its content.")


# 1. Load the PDF and split it into chunks
@st.cache_data
def process_pdf(pdf_bytes):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as file:
            file.write(pdf_bytes)
            temp_path = file.name

        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(documents)

        return documents, chunks

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# 2. Load the embedding model once
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# 3. Create a vector store for the uploaded PDF
@st.cache_resource
def create_vectorstore(pdf_hash, pdf_bytes):
    _, chunks = process_pdf(pdf_bytes)
# The underscore _ means we don't need to use the first returned value here. We only need chunks.
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        collection_name=f"pdf_{pdf_hash[:20]}"
    )


# 4. Set up the local LLM
@st.cache_resource
def get_llm():
    return ChatOllama(
        model="llama3.2",
        temperature=0
    )


# 5. Create the RAG prompt
prompt = ChatPromptTemplate.from_template("""
You are a PDF question-answering assistant.

Follow these rules:
1. Answer using only the provided context.
2. Do not make up facts or guess.
3. If the context does not contain the answer, say:
   "I could not find the answer in the provided PDF content."
4. Give a simple and clear answer.

Context:
{context}

Question:
{question}

Answer:
""")


# 6. Upload the PDF
pdf_uploader = st.file_uploader(
    "Upload your PDF file",
    type=["pdf"]
)


if pdf_uploader is not None:
    pdf_bytes = pdf_uploader.getvalue()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    try:
        with st.spinner("Reading and splitting your PDF..."):
            documents, chunks = process_pdf(pdf_bytes)

        st.success("PDF processed successfully!")

        st.write("Total pages:", len(documents))
        st.write("Total chunks:", len(chunks))

        with st.spinner("Creating embeddings and preparing Chroma..."):
            vectorstore = create_vectorstore(
                pdf_hash, pdf_bytes
            )

        st.success("Your PDF is ready for questions!")

        # 7. Ask a question
        with st.form("question_form"):
            question = st.text_input(
                "Ask a question about your PDF"
            )
            submitted = st.form_submit_button("Get Answer")

        if submitted:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                # 8. Retrieve relevant chunks
                with st.spinner("Searching your PDF..."):
                    results = vectorstore.similarity_search(
                        question,
                        k=5
                    )

                context = ""
                for doc in results:
                    context = context + doc.page_content + "\n\n"

                # 9. Generate an answer using the LLM
                with st.spinner("Generating your answer..."):
                    chain = prompt | get_llm()

                    response = chain.invoke({
                        "context": context,
                        "question": question
                    })

                st.subheader("Answer")
                st.write(response.content)

                # 10. Show source pages
                st.subheader("Source pages")

                pages = []

                for doc in results:
                     page = doc.metadata.get("page")

                if isinstance(page, int):
                    page_number = page + 1

                if page_number not in pages:
                    pages.append(page_number)

                if pages:
                    pages.sort()

                    for page_number in pages:
                        st.write("Page:", page_number)
                else:
                    st.write("Source page numbers unavailable.")

    except Exception as e:
        st.error(
            "Something went wrong while processing your PDF "
            "or generating the answer."
        )
        st.write("Error details:", str(e))
