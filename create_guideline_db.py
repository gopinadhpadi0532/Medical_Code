from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# --- Configuration ---
GUIDELINE_PDF_PATH = "icd-10-cm-guidelines-2024.pdf"
PERSISTENT_DB_PATH = "db/guideline_db"

def main():
    """
    This script loads the official ICD-10 guidelines, splits them into chunks,
    creates embeddings, and saves them to a persistent ChromaDB database on disk.
    """
    print("--- Starting Guideline Database Creation ---")
    
    # 1. Check if the database already exists
    if os.path.exists(PERSISTENT_DB_PATH):
        print(f"Database already exists at {PERSISTENT_DB_PATH}. Aborting.")
        return

    # 2. Check if the PDF exists
    if not os.path.exists(GUIDELINE_PDF_PATH):
        print(f"Error: Guideline PDF not found at {GUIDELINE_PDF_PATH}.")
        print("Please download it from the CDC website and place it in the project folder.")
        return

    print(f"Loading document: {GUIDELINE_PDF_PATH}")
    loader = PyPDFLoader(GUIDELINE_PDF_PATH)
    documents = loader.load()

    print("Splitting document into text chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print("Creating embeddings and vector store. This may take a few minutes...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create the database and persist it to disk
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSISTENT_DB_PATH
    )
    print("--- Guideline Database created successfully! ---")
    print(f"Database saved at: {PERSISTENT_DB_PATH}")

if __name__ == "__main__":
    main()