
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import os
# 1. Load the PDF
pdf_path = "documents/company.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("Total pages:", len(documents))

# 2. Split the PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# 3. Create embeddings
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store the chunks in Chroma


if os.path.exists("./chroma_db"):
    vector_store = Chroma(
        collection_name="microsoft_annual_report",
        embedding_function=embeddings_model,
        persist_directory="./chroma_db"
    )
    print("Existing Chroma database loaded.")
else:
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        collection_name="microsoft_annual_report",
        persist_directory="./chroma_db"
    )
    print("New Chroma database created.")

# 5. Retrieve relevant chunks
question = input("ask your question related to microsoft")

results = vector_store.similarity_search(question, k=3)

context = ""

for doc in results:
    context = context + doc.page_content + "\n\n"

# 6. Create the prompt
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

# 7. Connect to the local LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# 8. Generate the answer
chain = prompt | llm

response = chain.invoke({
    "context": context,
    "question": question
})

print("\nQuestion:", question)
print("\nAnswer:")
print(response.content)

print("\nSources:")
for doc in results:
    print("Page:", doc.metadata.get("page", "Unknown"))
