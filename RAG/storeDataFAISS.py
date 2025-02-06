from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import json
import os
from typing import List, Dict
from dotenv import load_dotenv

class FAISSDocumentStore:
    def __init__(self, api_key: str, db_path: str = "faiss_index"):
        """
        Initialize the FAISS document store
        
        Args:
            api_key: Google API key for Gemini
            db_path: Path to save/load the FAISS index
        """
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        self.db_path = db_path
        self.vectorstore = None

    def process_csv(self, file_path: str) -> List[Dict]:
        """Process CSV file into documents"""
        df = pd.read_csv(file_path)
        documents = []
        
        # Convert each row to a document
        for _, row in df.iterrows():
            # Combine all fields into a single text
            content = " | ".join([f"{col}: {str(val)}" for col, val in row.items()])
            documents.append({
                "content": content,
                "metadata": {
                    "source": file_path,
                    "type": "csv",
                    "row": _
                }
            })
        return documents

    def process_json(self, file_path: str) -> List[Dict]:
        """Process JSON file into documents"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        def process_json_item(item, path=""):
            if isinstance(item, dict):
                # Convert dictionary to string representation
                content = " | ".join([f"{k}: {str(v)}" for k, v in item.items()])
                documents.append({
                    "content": content,
                    "metadata": {
                        "source": file_path,
                        "type": "json",
                        "path": path
                    }
                })
            elif isinstance(item, list):
                for i, subitem in enumerate(item):
                    process_json_item(subitem, f"{path}[{i}]")

        process_json_item(data)
        return documents

    def create_index(self, directory: str):
        """
        Create FAISS index from all CSV and JSON files in directory
        
        Args:
            directory: Directory containing CSV and JSON files
        """
        all_documents = []
        
        # Process all CSV and JSON files in directory
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.csv'):
                    print(f"Processing CSV file: {file}")
                    documents = self.process_csv(file_path)
                    all_documents.extend(documents)
                elif file.endswith('.json'):
                    print(f"Processing JSON file: {file}")
                    documents = self.process_json(file_path)
                    all_documents.extend(documents)

        if not all_documents:
            raise ValueError("No documents found to process")

        # Create FAISS index
        texts = [doc["content"] for doc in all_documents]
        metadatas = [doc["metadata"] for doc in all_documents]
        
        print(f"Creating FAISS index with {len(texts)} documents...")
        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        # Save the index
        self.save_index()
        print(f"Index saved to {self.db_path}")

    def save_index(self):
        """Save the FAISS index"""
        if self.vectorstore:
            self.vectorstore.save_local(self.db_path)

    def load_index(self):
        """Load the FAISS index"""
        if os.path.exists(self.db_path):
            self.vectorstore = FAISS.load_local(self.db_path, self.embeddings)
            return True
        return False

    def search(self, query: str, k: int = 5):
        """
        Search the index
        
        Args:
            query: Search query
            k: Number of results to return
        """
        if not self.vectorstore:
            if not self.load_index():
                raise ValueError("No index found. Create index first.")
        
        return self.vectorstore.similarity_search(query, k=k)

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    # Initialize document store
    doc_store = FAISSDocumentStore(api_key=api_key)
    
    # Create index from files in current directory
    try:
        doc_store.create_index(".")
        print("Index created successfully!")
        
        # Test search
        query = "What is GEER?"
        results = doc_store.search(query)
        print(f"\nTest search for '{query}':")
        for doc in results:
            print(f"- {doc.page_content[:200]}...")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()