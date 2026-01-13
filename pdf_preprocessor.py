"""
PDF Preprocessor for RAG System
Uses PyPDFLoader and RecursiveCharacterTextSplitter to preprocess PDF documents
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional
import os


class PDFPreprocessor:
    """
    PDF Preprocessor that loads PDF documents and splits them into chunks
    for embedding and retrieval.
    
    Default Baseline:
    - RecursiveCharacterTextSplitter
    - chunk_size = 1024 (tokens)
    - chunk_overlap = 100
    """
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the PDF preprocessor.
        
        Args:
            chunk_size: Size of each text chunk (default: 1024 tokens)
            chunk_overlap: Overlap between chunks (default: 100 tokens)
            separators: Custom separators for text splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize the text splitter with default baseline configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> List:
        """
        Load PDF pages as structured documents with metadata using PyPDFLoader.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects with page content and metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        return documents
    
    def split_documents(self, documents: List) -> List:
        """
        Split documents into chunks that preserve context, for embedding.
        
        Args:
            documents: List of Document objects from PDF loader
            
        Returns:
            List of Document chunks
        """
        if not documents:
            return []
        
        # Split documents into chunks using RecursiveCharacterTextSplitter
        chunks = self.text_splitter.split_documents(documents)
        
        return chunks
    
    def preprocess(self, pdf_path: str) -> List:
        """
        Complete preprocessing pipeline: load PDF and split into chunks.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document chunks ready for embedding
        """
        # Step 1: Load PDF pages as structured documents with metadata
        documents = self.load_pdf(pdf_path)
        
        # Step 2: Split documents into chunks that preserve context
        chunks = self.split_documents(documents)
        
        return chunks
    
    def preprocess_multiple(self, pdf_paths: List[str]) -> List:
        """
        Preprocess multiple PDF files.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of Document chunks from all PDFs
        """
        all_chunks = []
        
        for pdf_path in pdf_paths:
            try:
                chunks = self.preprocess(pdf_path)
                all_chunks.extend(chunks)
                print(f"✓ Processed {pdf_path}: {len(chunks)} chunks")
            except Exception as e:
                print(f"✗ Error processing {pdf_path}: {str(e)}")
        
        return all_chunks


if __name__ == '__main__':
    # Example usage
    preprocessor = PDFPreprocessor(
        chunk_size=1024,
        chunk_overlap=100
    )
    
    # Example: Preprocess a single PDF
    # pdf_path = "example.pdf"
    # chunks = preprocessor.preprocess(pdf_path)
    # print(f"Total chunks: {len(chunks)}")
    # print(f"First chunk: {chunks[0].page_content[:200]}...")
    # print(f"Metadata: {chunks[0].metadata}")
    
    print("PDF Preprocessor initialized with default baseline:")
    print(f"  - RecursiveCharacterTextSplitter")
    print(f"  - chunk_size = {preprocessor.chunk_size} tokens")
    print(f"  - chunk_overlap = {preprocessor.chunk_overlap} tokens")
    pdf_path = "CV_Hengze_Ye - Lenovo2.pdf"
    chunks = preprocessor.preprocess(pdf_path)
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk: {chunks[0].page_content[:200]}...")
    print(f"Metadata: {chunks[0].metadata}")

