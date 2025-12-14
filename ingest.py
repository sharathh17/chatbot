"""Knowledge ingestion system for populating the RAG vector store."""

import os
import json
from typing import List, Dict, Any
from pathlib import Path


class DocumentProcessor:
    """Process and prepare documents for ingestion."""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    @staticmethod
    def process_txt_file(file_path: str) -> List[Dict[str, Any]]:
        """Process a text file into documents."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = DocumentProcessor.chunk_text(content)
        return [
            {
                "content": chunk,
                "source": file_path,
                "type": "text"
            }
            for chunk in chunks
        ]
    
    @staticmethod
    def process_json_file(file_path: str) -> List[Dict[str, Any]]:
        """Process a JSON file into documents."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        if isinstance(data, list):
            documents = data
        elif isinstance(data, dict):
            documents = [data]
        
        # Ensure each document has content
        for doc in documents:
            if "source" not in doc:
                doc["source"] = file_path
        
        return documents


class KnowledgeIngester:
    """Ingest and manage knowledge documents."""
    
    def __init__(self, rag_pipeline=None):
        self.rag_pipeline = rag_pipeline
        self.ingestion_log: List[Dict[str, Any]] = []
    
    def ingest_file(self, file_path: str, chunk: bool = True) -> int:
        """Ingest a single file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        _, ext = os.path.splitext(file_path)
        documents = []
        
        if ext == '.txt':
            if chunk:
                documents = DocumentProcessor.process_txt_file(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents = [{
                    "content": content,
                    "source": file_path,
                    "type": "text"
                }]
        
        elif ext == '.json':
            documents = DocumentProcessor.process_json_file(file_path)
        
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Add to RAG pipeline if available
        if self.rag_pipeline:
            for doc in documents:
                content = doc.get("content") or doc.get("text", "")
                metadata = {k: v for k, v in doc.items() if k != "content"}
                if content:
                    self.rag_pipeline.vector_store.add_document(content, metadata)
        
        # Log ingestion
        self.ingestion_log.append({
            "file": file_path,
            "documents_added": len(documents),
            "type": ext
        })
        
        return len(documents)
    
    def ingest_directory(self, dir_path: str, pattern: str = "*", chunk: bool = True) -> Dict[str, Any]:
        """Ingest all files in a directory."""
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        results = {
            "total_files": 0,
            "total_documents": 0,
            "files": []
        }
        
        for file_path in Path(dir_path).glob(pattern):
            if file_path.is_file():
                try:
                    doc_count = self.ingest_file(str(file_path), chunk=chunk)
                    results["files"].append({
                        "name": file_path.name,
                        "documents": doc_count
                    })
                    results["total_documents"] += doc_count
                    results["total_files"] += 1
                except Exception as e:
                    results["files"].append({
                        "name": file_path.name,
                        "error": str(e)
                    })
        
        return results
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get statistics about ingested documents."""
        return {
            "total_ingestions": len(self.ingestion_log),
            "recent": self.ingestion_log[-5:] if self.ingestion_log else [],
            "total_documents_ingested": sum(log.get("documents_added", 0) for log in self.ingestion_log)
        }
