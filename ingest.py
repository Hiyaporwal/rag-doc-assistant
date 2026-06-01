from pathlib import Path
import shutil

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


def ingest_documents(data_path="./data", db_path="./chroma_db", recreate_db=True):
    pdf_files = list(Path(data_path).glob("**/*.pdf"))
    if not pdf_files:
        raise ValueError("No PDF files found. Upload at least one PDF before processing.")

    if recreate_db and Path(db_path).exists():
        shutil.rmtree(db_path)

    loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    if not documents:
        raise ValueError("No pages could be loaded from the uploaded PDFs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    print(f"Created {len(chunks)} chunks")
    if not chunks:
        raise ValueError(
            "No text could be extracted from the uploaded PDFs. Try a PDF with selectable text."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True},
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
    )
    print("Documents embedded and stored in ChromaDB")
    return vectorstore


if __name__ == "__main__":
    ingest_documents()
