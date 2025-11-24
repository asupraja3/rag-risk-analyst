import os
from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

pdf_path = "dataset/tesla_10K.pdf"
# pdf_path = "D:\Work_USA\AIML\Projects\rag-risk-analyst\dataset\tesla_10K.pdf"

def create_knowledge_base():
    # ### [REAL-WORLD SCENARIO]:
    # In a real firm, we wouldn't load a static PDF.
    # We would connect to a Data Lake (like Snowflake or S3)
    # where all new filings are automatically dropped.

    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # ### [REAL-WORLD SCENARIO]:
    # Text Splitting is complex in finance.
    # We would use a "Semantic Chunker" that keeps tables intact.
    # Breaking a Balance Sheet table into 2 chunks ruins the data.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from the document.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ### [REAL-WORLD SCENARIO]:
    # Instead of FAISS local save, we would push these vectors
    # to a distributed cluster like Milvus or Pinecone.
    # Code: pinecone_index.upsert(vectors=chunks)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_financial_index")


if __name__ == "__main__":
    create_knowledge_base()