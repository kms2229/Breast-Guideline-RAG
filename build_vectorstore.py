"""
Build Vector Store from Breast Guideline PDFs
This script processes all PDFs and uploads to Pinecone cloud
"""

import os
import json
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from AnyFile_Loader import load_documents

load_dotenv()

# Configuration
PDF_SOURCE_PATH = "/Users/kavachshah/Downloads/degrees/PROJ__NLP/RA/Breast Guideline PDFs"
DATASET_PATH = "./Dataset"
PINECONE_INDEX_NAME = "breast-guidelines"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def main():
    print("=" * 80)
    print("BREAST GUIDELINE RAG - PINECONE VECTOR STORE BUILDER")
    print("=" * 80)
    
    # Get API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        openai_key = input("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = openai_key
    
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        pinecone_key = input("Enter your Pinecone API key: ")
        os.environ["PINECONE_API_KEY"] = pinecone_key
    
    print(f"\n1. Loading documents from: {PDF_SOURCE_PATH}")
    documents = load_documents(PDF_SOURCE_PATH)
    print(f"   Loaded {len(documents)} document chunks")
    
    # Split documents into smaller chunks
    print(f"\n2. Splitting documents (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"   Created {len(split_docs)} text chunks")
    
    # Create embeddings
    print("\n3. Creating embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Initialize Pinecone
    print(f"\n4. Initializing Pinecone...")
    pc = Pinecone(api_key=pinecone_key)
    
    # Check if index exists, create if not
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"   Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # text-embedding-3-small dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("   Waiting for index to be ready...")
        time.sleep(60)  # Wait for index initialization
    else:
        print(f"   Using existing index: {PINECONE_INDEX_NAME}")
    
    # Upload to Pinecone
    print(f"\n5. Uploading {len(split_docs)} documents to Pinecone...")
    print("   This may take 10-20 minutes depending on document count...")
    
    vectorstore = PineconeVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    print("   ✅ Upload complete!")
    
    # Create full-text JSON for complete paper retrieval
    print(f"\n6. Creating full-text JSON at: {DATASET_PATH}")
    os.makedirs(DATASET_PATH, exist_ok=True)
    
    full_text_data = {}
    for doc in documents:
        source = doc.metadata.get('source', 'unknown')
        if source not in full_text_data:
            full_text_data[source] = doc.page_content
        else:
            full_text_data[source] += "\n\n" + doc.page_content
    
    json_path = os.path.join(DATASET_PATH, "breast_guidelines_full_text.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_text_data, f, indent=2, ensure_ascii=False)
    
    print(f"   Saved {len(full_text_data)} full documents to JSON")
    
    # Test the vector store
    print("\n7. Testing Pinecone vector store...")
    test_query = "What are the guidelines for breast reconstruction?"
    results = vectorstore.similarity_search(test_query, k=3)
    print(f"   Test query: '{test_query}'")
    print(f"   Retrieved {len(results)} documents")
    if results:
        print(f"   Sample result: {results[0].page_content[:200]}...")
    
    print("\n" + "=" * 80)
    print("PINECONE VECTOR STORE BUILD COMPLETE!")
    print("=" * 80)
    print(f"\nPinecone Index: {PINECONE_INDEX_NAME}")
    print(f"Full-Text JSON: {json_path}")
    print(f"Total Documents: {len(full_text_data)}")
    print(f"Total Chunks: {len(split_docs)}")
    print("\nNext steps:")
    print("1. Add your Pinecone API key to .streamlit/secrets.toml")
    print("2. Run: streamlit run Chat_UI.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
