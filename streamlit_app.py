"""
Streamlit Web App for BeeGalaxy RAG System
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
import pickle
import faiss
import numpy as np

from rag_system import RAGSystem
from pdf_preprocessor import PDFPreprocessor
from embedding import local_embedding

# Page configuration
st.set_page_config(
    page_title="BeeGalaxy RAG System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.index_built = False
    st.session_state.chunks_count = 0
    st.session_state.uploaded_files = []


def initialize_rag_system():
    """Initialize RAG system with OpenAI API key from environment or session state"""
    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
    if api_key:
        return RAGSystem(openai_api_key=api_key)
    else:
        return RAGSystem()


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory"""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 BeeGalaxy RAG System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key input
        st.subheader("OpenAI API Key")
        api_key_input = st.text_input(
            "Enter your OpenAI API key",
            type="password",
            value=st.session_state.get("openai_api_key", ""),
            help="Required for LLM response generation. You can also set OPENAI_API_KEY environment variable."
        )
        if api_key_input != st.session_state.get("openai_api_key", ""):
            st.session_state.openai_api_key = api_key_input
            # Reinitialize RAG system if it exists
            if st.session_state.rag_system:
                try:
                    from openai import OpenAI
                    st.session_state.rag_system.openai_client = OpenAI(api_key=api_key_input)
                    st.session_state.rag_system.openai_model = "gpt-3.5-turbo"
                    st.success("✓ API key updated")
                except Exception as e:
                    st.error(f"Error updating API key: {str(e)}")
        
        st.markdown("---")
        
        # System status
        st.subheader("📊 System Status")
        if st.session_state.rag_system:
            st.success("✓ RAG System Initialized")
            st.info(f"📄 Documents: {st.session_state.chunks_count} chunks")
            if st.session_state.index_built:
                st.success("✓ Index Built")
            else:
                st.warning("⚠️ Index Not Built")
        else:
            st.warning("⚠️ System Not Initialized")
        
        st.markdown("---")
        
        # Load saved index
        st.subheader("💾 Load Saved Index")
        saved_index_path = st.text_input("Enter saved index path (without extension)")
        if st.button("Load Index"):
            if saved_index_path:
                try:
                    if st.session_state.rag_system is None:
                        st.session_state.rag_system = initialize_rag_system()
                    st.session_state.rag_system.load_index(saved_index_path)
                    st.session_state.index_built = True
                    st.session_state.chunks_count = len(st.session_state.rag_system.chunks)
                    st.success(f"✓ Loaded index with {st.session_state.chunks_count} chunks")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading index: {str(e)}")
            else:
                st.warning("Please enter an index path")
        
        st.markdown("---")
        
        # Save index
        st.subheader("💾 Save Index")
        if st.session_state.index_built:
            save_index_path = st.text_input("Enter path to save (without extension)")
            if st.button("Save Index"):
                if save_index_path:
                    try:
                        st.session_state.rag_system.save_index(save_index_path)
                        st.success("✓ Index saved successfully")
                    except Exception as e:
                        st.error(f"Error saving index: {str(e)}")
                else:
                    st.warning("Please enter a save path")
        else:
            st.info("Build index first to save it")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Upload PDF Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to process"
        )
        
        if uploaded_files:
            # Initialize RAG system if not already done
            if st.session_state.rag_system is None:
                st.session_state.rag_system = initialize_rag_system()
            
            # Display uploaded files
            st.subheader("Uploaded Files:")
            for uploaded_file in uploaded_files:
                st.write(f"📄 {uploaded_file.name}")
            
            # Process button
            if st.button("📥 Process PDFs", type="primary"):
                with st.spinner("Processing PDFs..."):
                    try:
                        for uploaded_file in uploaded_files:
                            # Save uploaded file temporarily
                            file_path = save_uploaded_file(uploaded_file)
                            
                            # Add to RAG system
                            num_chunks = st.session_state.rag_system.add_pdf(file_path)
                            st.session_state.chunks_count += num_chunks
                            
                            # Clean up temp file
                            os.remove(file_path)
                        
                        st.success(f"✓ Processed {len(uploaded_files)} file(s). Total chunks: {st.session_state.chunks_count}")
                        st.session_state.uploaded_files = [f.name for f in uploaded_files]
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing PDFs: {str(e)}")
        
        st.markdown("---")
        
        # Build index section
        st.header("🔨 Build Search Index")
        
        if st.session_state.chunks_count > 0:
            st.info(f"Ready to build index with {st.session_state.chunks_count} chunks")
            
            if st.button("🔨 Build Index", type="primary"):
                with st.spinner("Building index and generating embeddings... This may take a while."):
                    try:
                        st.session_state.rag_system.build_index()
                        st.session_state.index_built = True
                        st.success("✓ Index built successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error building index: {str(e)}")
        else:
            st.warning("⚠️ Please upload and process PDFs first")
    
    with col2:
        st.header("❓ Ask Questions")
        
        if not st.session_state.index_built:
            st.warning("⚠️ Please upload PDFs and build the index first before asking questions.")
        else:
            # Query input
            query = st.text_area(
                "Enter your question:",
                height=100,
                placeholder="e.g., What is the main topic of the document?",
                help="Ask any question about the uploaded PDF documents"
            )
            
            # Options
            col_a, col_b = st.columns(2)
            with col_a:
                num_results = st.number_input("Number of results", min_value=1, max_value=20, value=5)
            with col_b:
                use_llm = st.checkbox("Generate LLM response", value=True, 
                                     disabled=not st.session_state.rag_system.openai_client)
            
            # Search button
            if st.button("🔍 Search", type="primary"):
                if query:
                    with st.spinner("Searching..."):
                        try:
                            # Perform search
                            results, llm_response = st.session_state.rag_system.search_and_generate(
                                query, 
                                k=num_results,
                                generate_response=use_llm and st.session_state.rag_system.openai_client
                            )
                            
                            # Display LLM response if available
                            if llm_response:
                                st.markdown("---")
                                st.subheader("🤖 AI Response")
                                st.markdown(f"<div class='info-box'>{llm_response}</div>", unsafe_allow_html=True)
                            
                            # Display search results
                            st.markdown("---")
                            st.subheader(f"📊 Retrieved Context ({len(results)} results)")
                            
                            for i, (text, score, metadata) in enumerate(results, 1):
                                with st.expander(f"Result {i} (Similarity: {score:.4f}) - {metadata.get('source', 'Unknown')}"):
                                    st.write(f"**Source:** {metadata.get('source', 'Unknown')}")
                                    st.write(f"**Page:** {metadata.get('page', 'N/A')}")
                                    st.markdown("---")
                                    st.write(text)
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please enter a question")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 2rem;'>"
        "BeeGalaxy RAG System | Powered by FAISS & OpenAI"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

