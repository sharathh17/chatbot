"""RAG (Retrieval-Augmented Generation) pipeline for semantic search and retrieval."""

import os
from typing import List, Dict, Any, Optional
import json


class VectorStore:
    """Simple in-memory vector store for document retrieval."""
    
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None):
        """Add a document to the store."""
        doc = {
            "content": content,
            "metadata": metadata or {},
            "doc_id": len(self.documents)
        }
        self.documents.append(doc)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Simple BM25-style search (mock implementation)."""
        # In production, this would use proper vector similarity
        query_terms = set(query.lower().split())
        results = []
        
        for doc in self.documents:
            doc_terms = set(doc["content"].lower().split())
            overlap = len(query_terms & doc_terms)
            
            if overlap > 0:
                results.append({
                    "doc_id": doc["doc_id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": overlap / len(query_terms | doc_terms)
                })
        
        # Sort by relevance score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific document."""
        if 0 <= doc_id < len(self.documents):
            return self.documents[doc_id]
        return None


class RAGPipeline:
    """RAG pipeline combining retrieval and generation."""
    
    def __init__(self, vector_store: VectorStore = None):
        self.vector_store = vector_store or VectorStore()
    
    def load_documents(self, file_path: str) -> int:
        """Load documents from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        
        count = 0
        if isinstance(docs, list):
            for doc in docs:
                content = doc.get("content", "") or doc.get("text", "")
                metadata = {k: v for k, v in doc.items() if k not in ["content", "text"]}
                if content:
                    self.vector_store.add_document(content, metadata)
                    count += 1
        
        return count
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant documents for a query."""
        results = self.vector_store.search(query, top_k)
        return [result["content"] for result in results]
    
    def format_context(self, documents: List[str]) -> str:
        """Format retrieved documents into context."""
        if not documents:
            return "No relevant documents found."
        
        context = "Relevant Documents:\n"
        for i, doc in enumerate(documents, 1):
            # Truncate long documents
            truncated = doc[:200] + "..." if len(doc) > 200 else doc
            context += f"\n{i}. {truncated}\n"
        return context
    
    def augment_prompt(self, query: str, top_k: int = 5) -> str:
        """Augment a query with retrieved context."""
        documents = self.retrieve(query, top_k)
        context = self.format_context(documents)
        
        augmented = f"""Use the following context to answer the question:

{context}

Question: {query}

Answer:"""
        return augmented
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_documents": len(self.vector_store.documents),
            "documents": [
                {
                    "id": doc["doc_id"],
                    "preview": doc["content"][:50] + "...",
                    "metadata": doc["metadata"]
                }
                for doc in self.vector_store.documents
            ]
        }
