from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_google_genai import GoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from docx import Document
import requests
from io import BytesIO
import re
import os

# Constants
DATA_PATH = "data"
VECTOR_DB_PATH = "vectorstores/db_faiss"
MODEL_PATH = "./models/all-MiniLM-L6-v2-f16.gguf"
# Extract Google Drive links from the first DOCX file found


def extract_google_drive_links_from_docx_file(docx_file_path):
    """Extracts Google Drive sharable file links from a DOCX file.

    Args:
        docx_file_path (str): The file path to the DOCX file.

    Returns:
        list: List of extracted Google Drive links.
    """
    google_drive_links = []
    doc = Document(docx_file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    docx_content = '\n'.join(full_text)
    links = re.findall(r"https://drive\.google\.com/file/d/([^\/]+)/view", docx_content)
    print("DOCX content:", docx_content)  # Debugging output
    
    for link in links:
        google_drive_links.append(f"https://drive.google.com/file/d/{link}/view")
    return google_drive_links


def create_db_from_files():
    # Load documents from directory using DirectoryLoader
    docx_loader = DirectoryLoader(DATA_PATH, glob="*.docx", loader_cls=Docx2txtLoader)
    docx_files = docx_loader.load()
    
    print("Document files:", docx_files)  # Debugging output

    # Initialize list to store Google Drive links
    google_drive_links = []
    
    # Loop through the document paths and extract Google Drive links
    for docx_path in docx_files:
        # Check if 'docx_path' is actually a string path before using it
        if isinstance(docx_path, str):
            google_drive_links.extend(extract_google_drive_links_from_docx_file(docx_path))
        else:
            # If 'docx_path' is not a string, you need to find out how to handle it
            pass
    
    print("Google Drive links:", google_drive_links)  # Debugging output

    # Process PDF data from links only if there are links found
    pdf_texts = []
    for link in google_drive_links:
        pdf_id = link.split('/')[-2]
        pdf_url = f"https://drive.google.com/uc?id={pdf_id}&export=download"
        response = requests.get(pdf_url)
        if response.status_code == 200:
            # Create BytesIO object from response content
            pdf_data = BytesIO(response.content)
            pdf_texts.append(pdf_data)
    
    # If no Google Drive links are found, print a message
    if not google_drive_links:
        print("No Google Drive links found.")
    
    # Load PDF documents from extracted texts
    # pdf_loader = PyPDFLoader(pdf_texts)
    # pdf_documents = pdf_loader.load()
    
    # Split PDF documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=200)
    docx_chunks = text_splitter.split_documents(docx_files)
    
    # Load embedding model
    embedding_model = GPT4AllEmbeddings(model_file=MODEL_PATH)
    
    # Create FAISS Vector DB
    db = FAISS.from_documents(docx_chunks, embedding_model)
    db.save_local(VECTOR_DB_PATH)
    
    
    return db

def create_db_from_files_PDF():
    # Load documents from directory using DirectoryLoader
    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_files = pdf_loader.load()
    
    print("Document files:", pdf_files)  # Debugging output

    
    # Split PDF documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=200)
    pdf_chunks = text_splitter.split_documents(pdf_files)
    
    # Load embedding model
    embedding_model = GPT4AllEmbeddings(model_file=MODEL_PATH)
    
    # Create FAISS Vector DB
    db = FAISS.from_documents(pdf_chunks, embedding_model)
    db.save_local(VECTOR_DB_PATH)
    
    
    return db


# Usage
db_from_files = create_db_from_files()
create_db_from_files_PDF()