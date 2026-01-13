"""
RAG System - Main program integrating PDF preprocessing, embedding, and FAISS search
"""

import faiss
import numpy as np
from typing import List, Tuple, Optional
import os
import pickle

from pdf_preprocessor import PDFPreprocessor
from embedding import local_embedding

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not installed. LLM response generation will be disabled.")


class RAGSystem:
    """
    Complete RAG (Retrieval-Augmented Generation) system that:
    1. Preprocesses PDF documents
    2. Generates embeddings
    3. Builds FAISS index for similarity search
    4. Retrieves relevant chunks based on user queries
    """
    
    def __init__(
        self, 
        embedding_dim: Optional[int] = None,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the RAG system.
        
        Args:
            embedding_dim: Dimension of embeddings. If None, will be detected automatically.
            openai_api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY.
            openai_model: OpenAI model to use for generation (default: gpt-3.5-turbo)
        """
        self.preprocessor = PDFPreprocessor()
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.embedding_dim = embedding_dim
        self.chunk_texts = []  # Store original chunk texts for retrieval
        
        # Initialize OpenAI client if available
        self.openai_client = None
        self.openai_model = openai_model
        if OPENAI_AVAILABLE:
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    print("✓ OpenAI client initialized")
                except Exception as e:
                    print(f"⚠️  Error initializing OpenAI client: {str(e)}")
            # Don't print warning on init - user can configure later via menu
        
    def _get_embedding_dim(self) -> int:
        """Get embedding dimension by testing with a sample text."""
        if self.embedding_dim:
            return self.embedding_dim
        
        # Test with a sample text to get dimension
        test_embedding = local_embedding(["test"])
        dim = len(test_embedding[0])
        self.embedding_dim = dim
        return dim
    
    def add_pdf(self, pdf_path: str) -> int:
        """
        Add a PDF file to the system.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Number of chunks added
        """
        print(f"\n📄 Processing PDF: {pdf_path}")
        
        # Preprocess PDF
        new_chunks = self.preprocessor.preprocess(pdf_path)
        num_chunks = len(new_chunks)
        
        if num_chunks == 0:
            print(f"⚠️  No chunks extracted from {pdf_path}")
            return 0
        
        # Store chunk texts
        chunk_texts = [chunk.page_content for chunk in new_chunks]
        self.chunks.extend(new_chunks)
        self.chunk_texts.extend(chunk_texts)
        
        print(f"✓ Added {num_chunks} chunks from {pdf_path}")
        print(f"  Total chunks in system: {len(self.chunks)}")
        
        return num_chunks
    
    def add_pdfs(self, pdf_paths: List[str]) -> int:
        """
        Add multiple PDF files to the system.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            Total number of chunks added
        """
        total_chunks = 0
        for pdf_path in pdf_paths:
            total_chunks += self.add_pdf(pdf_path)
        return total_chunks
    
    def build_index(self):
        """
        Generate embeddings for all chunks and build FAISS index.
        """
        if not self.chunks:
            raise ValueError("No chunks available. Please add PDF files first.")
        
        print(f"\n🔄 Generating embeddings for {len(self.chunks)} chunks...")
        
        # Get embedding dimension
        dim = self._get_embedding_dim()
        
        # Generate embeddings in batches (to avoid overwhelming the API)
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(self.chunk_texts), batch_size):
            batch = self.chunk_texts[i:i + batch_size]
            batch_embeddings = local_embedding(batch)
            all_embeddings.extend(batch_embeddings)
            print(f"  Processed {min(i + batch_size, len(self.chunk_texts))}/{len(self.chunk_texts)} chunks...")
        
        # Convert to numpy array
        self.embeddings = np.array(all_embeddings).astype("float32")
        
        print(f"✓ Generated embeddings with dimension {dim}")
        print(f"🔄 Building FAISS index...")
        
        # Create FAISS index (Inner Product for cosine similarity)
        self.index = faiss.IndexFlatIP(dim)
        
        # Normalize embeddings (L2 normalization for cosine similarity)
        faiss.normalize_L2(self.embeddings)
        
        # Add embeddings to index
        self.index.add(self.embeddings)
        
        print(f"✓ FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Search for relevant chunks based on user query.
        
        Args:
            query: User's search query
            k: Number of results to return
            
        Returns:
            List of tuples: (chunk_text, similarity_score, metadata)
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index not built. Please call build_index() first.")
        
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        
        print(f"\n🔍 Searching for: '{query}'")
        
        # Generate embedding for query
        query_embedding = local_embedding([query])[0]
        query_vector = np.array([query_embedding]).astype("float32")
        
        # Normalize query vector
        faiss.normalize_L2(query_vector)
        
        # Search
        k = min(k, self.index.ntotal)  # Don't request more than available
        scores, indices = self.index.search(query_vector, k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append((
                    chunk.page_content,
                    float(score),
                    chunk.metadata
                ))
        
        return results
    
    def generate_response(
        self, 
        query: str, 
        retrieved_chunks: List[Tuple[str, float, dict]],
        max_context_length: int = 3000
    ) -> str:
        """
        Generate LLM response based on user query and retrieved context.
        
        Args:
            query: User's query
            retrieved_chunks: List of retrieved chunks from search
            max_context_length: Maximum length of context to include (in characters)
            
        Returns:
            Generated response from LLM
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Please provide API key.")
        
        # Build context from retrieved chunks
        context_parts = []
        current_length = 0
        
        for i, (text, score, metadata) in enumerate(retrieved_chunks):
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', 'N/A')
            
            chunk_info = f"[Chunk {i+1} from {source}, page {page}]\n{text}\n"
            
            if current_length + len(chunk_info) > max_context_length:
                break
            
            context_parts.append(chunk_info)
            current_length += len(chunk_info)
        
        context = "\n".join(context_parts)
        
        # Create prompt
        system_prompt = """You are a helpful assistant that answers questions based on the provided context from PDF documents. 
Use only the information from the context to answer the question. If the context doesn't contain enough information to answer the question, 
say so clearly. Be concise and accurate in your response."""
        
        user_prompt = f"""Based on the following context from PDF documents, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            print(f"\n🤖 Generating response using {self.openai_model}...")
            
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            error_msg = str(e)
            
            # Handle specific OpenAI API errors
            if "insufficient_quota" in error_msg or "quota" in error_msg.lower():
                raise ValueError(
                    "OpenAI API Quota Exceeded:\n"
                    "You have exceeded your current OpenAI API quota. Please:\n"
                    "1. Check your billing and usage at https://platform.openai.com/usage\n"
                    "2. Add payment method or upgrade your plan\n"
                    "3. Wait for your quota to reset\n\n"
                    "You can still use the search functionality without LLM responses."
                )
            elif "401" in error_msg or "authentication" in error_msg.lower() or "invalid_api_key" in error_msg.lower():
                raise ValueError(
                    "OpenAI API Authentication Error:\n"
                    "Your API key is invalid or expired. Please:\n"
                    "1. Check your API key at https://platform.openai.com/api-keys\n"
                    "2. Generate a new API key if needed\n"
                    "3. Update it using menu option 7"
                )
            elif "429" in error_msg:
                if "rate_limit" in error_msg.lower():
                    raise ValueError(
                        "OpenAI API Rate Limit Exceeded:\n"
                        "You're making too many requests too quickly. Please wait a moment and try again."
                    )
                else:
                    raise ValueError(
                        "OpenAI API Error (429):\n"
                        "Request was rate limited. This could be due to:\n"
                        "- Too many requests per minute\n"
                        "- Quota exceeded\n"
                        "Please wait and try again, or check your usage limits."
                    )
            else:
                raise ValueError(f"Error generating response: {error_msg}")
    
    def search_and_generate(
        self, 
        query: str, 
        k: int = 5,
        generate_response: bool = True
    ) -> Tuple[List[Tuple[str, float, dict]], Optional[str]]:
        """
        Search for relevant chunks and optionally generate LLM response.
        
        Args:
            query: User's search query
            k: Number of results to return
            generate_response: Whether to generate LLM response
            
        Returns:
            Tuple of (search_results, llm_response)
        """
        # Search for relevant chunks
        results = self.search(query, k)
        
        # Generate LLM response if requested and OpenAI is available
        llm_response = None
        if generate_response and self.openai_client:
            try:
                llm_response = self.generate_response(query, results)
            except Exception as e:
                print(f"⚠️  Error generating LLM response: {str(e)}")
        elif generate_response and not self.openai_client:
            print("⚠️  OpenAI not available. Skipping LLM response generation.")
        
        return results, llm_response
    
    def save_index(self, filepath: str):
        """
        Save the FAISS index and chunks to disk.
        
        Args:
            filepath: Path to save the index (without extension)
        """
        if self.index is None:
            raise ValueError("No index to save. Please build index first.")
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save chunks and embeddings
        with open(f"{filepath}.pkl", "wb") as f:
            pickle.dump({
                'chunks': self.chunks,
                'chunk_texts': self.chunk_texts,
                'embeddings': self.embeddings,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"✓ Saved index to {filepath}.index and data to {filepath}.pkl")
    
    def load_index(self, filepath: str):
        """
        Load the FAISS index and chunks from disk.
        
        Args:
            filepath: Path to the saved index (without extension)
        """
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.index")
        
        # Load chunks and embeddings
        with open(f"{filepath}.pkl", "rb") as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.chunk_texts = data['chunk_texts']
            self.embeddings = data['embeddings']
            self.embedding_dim = data['embedding_dim']
        
        print(f"✓ Loaded index from {filepath}.index")
        print(f"  Total chunks: {len(self.chunks)}")
        print(f"  Index size: {self.index.ntotal} vectors")


def main():
    """Main interactive program for RAG system."""
    print("=" * 60)
    print("🚀 BeeGalaxy RAG System")
    print("=" * 60)
    
    rag = RAGSystem()
    
    while True:
        print("\n" + "-" * 60)
        print("Options:")
        print("  1. Add PDF file(s)")
        print("  2. Build/rebuild index")
        print("  3. Search/Query (with LLM response)")
        print("  4. Save index")
        print("  5. Load index")
        print("  6. Show statistics")
        print("  7. Configure OpenAI API key")
        print("  8. Exit")
        print("-" * 60)
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == "1":
            # Add PDF files
            pdf_input = input("Enter PDF file path(s), separated by commas: ").strip()
            pdf_paths = [path.strip() for path in pdf_input.split(",")]
            
            try:
                total = rag.add_pdfs(pdf_paths)
                print(f"\n✓ Successfully added {total} total chunks")
            except Exception as e:
                print(f"\n✗ Error: {str(e)}")
        
        elif choice == "2":
            # Build index
            try:
                rag.build_index()
            except Exception as e:
                print(f"\n✗ Error: {str(e)}")
        
        elif choice == "3":
            # Search and Generate
            if rag.index is None:
                print("\n⚠️  Index not built. Please build index first (option 2).")
                continue
            
            query = input("\nEnter your search query: ").strip()
            if not query:
                print("⚠️  Query cannot be empty.")
                continue
            
            try:
                k = input("Number of results to return (default 5): ").strip()
                k = int(k) if k else 5
                
                # Ask if user wants LLM response
                use_llm = False
                if rag.openai_client:
                    llm_choice = input("Generate LLM response? (y/n, default y): ").strip().lower()
                    use_llm = llm_choice != 'n'
                
                # Search and optionally generate response
                results, llm_response = rag.search_and_generate(query, k, generate_response=use_llm)
                
                # Display LLM response first if available
                if llm_response:
                    print("\n" + "=" * 60)
                    print("🤖 LLM Response:")
                    print("=" * 60)
                    print(llm_response)
                    print("=" * 60)
                elif use_llm:
                    print("\n⚠️  LLM response generation was skipped due to an error.")
                    print("   You can still view the retrieved context chunks below.")
                
                # Display search results
                print("\n" + "=" * 60)
                print(f"📊 Retrieved Context (Top {len(results)}):")
                print("=" * 60)
                
                for i, (text, score, metadata) in enumerate(results, 1):
                    print(f"\n[Result {i}] (Similarity: {score:.4f})")
                    print(f"Source: {metadata.get('source', 'Unknown')}")
                    print(f"Page: {metadata.get('page', 'N/A')}")
                    print("-" * 60)
                    # Show entire chunk content
                    print(text)
                    print()
                
            except Exception as e:
                print(f"\n✗ Error: {str(e)}")
        
        elif choice == "4":
            # Save index
            filepath = input("Enter filepath to save (without extension): ").strip()
            if not filepath:
                print("⚠️  Filepath cannot be empty.")
                continue
            
            try:
                rag.save_index(filepath)
            except Exception as e:
                print(f"\n✗ Error: {str(e)}")
        
        elif choice == "5":
            # Load index
            filepath = input("Enter filepath to load (without extension): ").strip()
            if not filepath:
                print("⚠️  Filepath cannot be empty.")
                continue
            
            try:
                rag.load_index(filepath)
            except Exception as e:
                print(f"\n✗ Error: {str(e)}")
        
        elif choice == "6":
            # Show statistics
            print("\n" + "=" * 60)
            print("📊 System Statistics:")
            print("=" * 60)
            print(f"Total chunks: {len(rag.chunks)}")
            if rag.index:
                print(f"Index size: {rag.index.ntotal} vectors")
                print(f"Embedding dimension: {rag.embedding_dim}")
            else:
                print("Index: Not built")
            if rag.openai_client:
                print(f"OpenAI: ✓ Configured (Model: {rag.openai_model})")
            else:
                print("OpenAI: ✗ Not configured")
            print("=" * 60)
        
        elif choice == "7":
            # Configure OpenAI API key
            if not OPENAI_AVAILABLE:
                print("\n⚠️  OpenAI package not installed. Please install it with: pip install openai")
                continue
            
            api_key = input("Enter OpenAI API key (or press Enter to use environment variable): ").strip()
            if api_key:
                try:
                    rag.openai_client = OpenAI(api_key=api_key)
                    rag.openai_model = input(f"Enter model name (default {rag.openai_model}): ").strip() or rag.openai_model
                    print(f"✓ OpenAI client configured with model: {rag.openai_model}")
                except Exception as e:
                    print(f"\n✗ Error: {str(e)}")
            else:
                # Try to use environment variable
                env_key = os.getenv("OPENAI_API_KEY")
                if env_key:
                    try:
                        rag.openai_client = OpenAI(api_key=env_key)
                        print(f"✓ Using OpenAI API key from environment variable")
                    except Exception as e:
                        print(f"\n✗ Error: {str(e)}")
                else:
                    print("⚠️  No API key provided and OPENAI_API_KEY environment variable not set.")
        
        elif choice == "8":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("\n⚠️  Invalid choice. Please enter 1-8.")


if __name__ == "__main__":
    main()

