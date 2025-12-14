"""FastAPI application entry point for the AI Agent."""

import os
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from agent import Agent
from tools import Tool, ToolType
from ingest import KnowledgeIngester


# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    """Request model for agent queries."""
    query: str
    use_rag: bool = True
    max_iterations: int = 10


class QueryResponse(BaseModel):
    """Response model for agent queries."""
    response: str
    state: str
    tokens_used: Optional[int] = None


class DocumentIngestionRequest(BaseModel):
    """Request model for document ingestion."""
    file_path: str
    chunk: bool = True


class ToolCallRequest(BaseModel):
    """Request model for direct tool execution."""
    tool_name: str
    parameters: Dict[str, Any]


# Global agent instance
agent: Optional[Agent] = None
ingester: Optional[KnowledgeIngester] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    # Startup
    global agent, ingester
    
    print("🚀 Initializing AI Agent...")
    agent = Agent(
        name="JanoBot",
        model="gpt-3.5-turbo",
        use_rag=True,
        use_memory=True
    )
    
    ingester = KnowledgeIngester(rag_pipeline=agent.rag_pipeline)
    
    # Register some default tools
    def search_web(query: str) -> str:
        """Mock web search tool."""
        return f"Search results for '{query}': [mock results]"
    
    def calculator(expression: str) -> str:
        """Basic calculator tool."""
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    search_tool = Tool(
        name="search",
        description="Search the web for information",
        tool_type=ToolType.SEARCH,
        func=search_web,
        parameters={"query": "string"}
    )
    
    calc_tool = Tool(
        name="calculator",
        description="Perform mathematical calculations",
        tool_type=ToolType.CALCULATOR,
        func=calculator,
        parameters={"expression": "string"}
    )
    
    agent.register_tool(search_tool)
    agent.register_tool(calc_tool)
    
    print("✅ Agent initialized successfully")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down AI Agent...")


app = FastAPI(
    title="AI Agent API",
    description="FastAPI backend for AI Agent with RAG and tool calling",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Agent API is running",
        "endpoints": {
            "query": "/query",
            "status": "/status",
            "tools": "/tools",
            "ingest": "/ingest",
            "memory": "/memory",
            "rag": "/rag"
        }
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query through the agent."""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        response = agent.think(
            query=request.query,
            max_iterations=request.max_iterations
        )
        
        return QueryResponse(
            response=response,
            state=agent.state.value,
            tokens_used=None
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/status")
async def get_status():
    """Get agent status."""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    return agent.get_status()


@app.get("/tools")
async def list_tools():
    """List all available tools."""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    return {
        "tools": agent.tool_registry.list_tools(),
        "count": len(agent.tool_registry.tools)
    }


@app.post("/tool-call", response_model=Dict[str, Any])
async def call_tool(request: ToolCallRequest):
    """Execute a tool directly."""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        result = agent.execute_tool(request.tool_name, **request.parameters)
        return {
            "tool": request.tool_name,
            "result": result,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ingest")
async def ingest_document(request: DocumentIngestionRequest):
    """Ingest a document into the RAG pipeline."""
    if ingester is None:
        raise HTTPException(status_code=500, detail="Ingester not initialized")
    
    try:
        doc_count = ingester.ingest_file(
            file_path=request.file_path,
            chunk=request.chunk
        )
        
        return {
            "file": request.file_path,
            "documents_added": doc_count,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ingest-stats")
async def ingest_stats():
    """Get ingestion statistics."""
    if ingester is None:
        raise HTTPException(status_code=500, detail="Ingester not initialized")
    
    return ingester.get_ingestion_stats()


@app.get("/rag")
async def get_rag_stats():
    """Get RAG pipeline statistics."""
    if agent or agent.rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    
    return agent.rag_pipeline.get_stats()


@app.post("/memory/clear")
async def clear_memory():
    """Clear conversation memory."""
    if agent is None or agent.memory is None:
        raise HTTPException(status_code=500, detail="Memory not available")
    
    agent.memory.clear()
    return {"success": True, "message": "Memory cleared"}


@app.get("/memory")
async def get_memory(last_n: int = Query(10, ge=1, le=100)):
    """Get conversation memory."""
    if agent is None or agent.memory is None:
        raise HTTPException(status_code=500, detail="Memory not available")
    
    return {
        "history": agent.memory.get_history(last_n),
        "stats": agent.memory.summary_stats()
    }


@app.post("/reset")
async def reset_agent():
    """Reset agent state."""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    agent.reset()
    return {"success": True, "message": "Agent reset"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "agent_state": agent.state.value if agent else None
    }


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
