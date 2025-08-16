import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os

def main():
    """
    Create and return a vector store retriever for Indian Labour Law documents.
    """
    # Load the CSV data
    df = pd.read_csv('labour_chunks_converted.csv')
    
    # Create documents from the CSV data
    documents = []
    for _, row in df.iterrows():
        # Create a document with the chunk text as content
        doc = Document(
            page_content=row['chunk_text'],
            metadata={
                'id': row['id'],
                'source_document': row['source_document'],
                'source_page_number': row['source_page_number'],
                'primary_law_act': row['primary_law_act'],
                'topic': row['topic'],
                'keywords': row['keywords']
            }
        )
        documents.append(doc)
    
    # Initialize embeddings
    # Using a lightweight model for local use
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create vector store
    # Use a persistent directory for the vector store
    persist_directory = "./chroma_db"
    
    # Check if vector store already exists
    if os.path.exists(persist_directory):
        print("Loading existing vector store...")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        print("Creating new vector store...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"Vector store created with {len(documents)} documents")
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve top 5 most relevant documents
    )
    
    return retriever

if __name__ == "__main__":
    retriever = main()
    print("Vector store initialized successfully!")
