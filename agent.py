"""Agent logic - Core reasoning and decision-making system."""

import os
from typing import Dict, Any, List, Optional
from enum import Enum

from memory import ConversationMemory
from tools import ToolRegistry
from rag import RAGPipeline


class AgentState(str, Enum):
    """Agent execution states."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    RETRIEVING = "retrieving"
    COMPLETE = "complete"
    ERROR = "error"


class Agent:
    """Main agent for reasoning, retrieval, and tool execution."""
    
    def __init__(
        self,
        name: str = "AIAgent",
        model: str = "gpt-3.5-turbo",
        use_rag: bool = True,
        use_memory: bool = True
    ):
        self.name = name
        self.model = model
        self.state = AgentState.IDLE
        
        # Initialize systems
        self.tool_registry = ToolRegistry()
        self.memory = ConversationMemory() if use_memory else None
        self.rag_pipeline = RAGPipeline() if use_rag else None
        
        # Agent configuration
        self.max_iterations = 10
        self.temperature = 0.7
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        prompt = f"""You are {self.name}, an AI assistant capable of:
- Answering questions and providing explanations
- Using tools to retrieve information or perform actions
- Maintaining conversation context and memory
"""
        
        if self.rag_pipeline:
            prompt += "- Retrieving relevant documents from a knowledge base\n"
        
        if self.tool_registry.tools:
            prompt += f"- Using available tools: {', '.join(self.tool_registry.tools.keys())}\n"
        
        prompt += "\nRespond helpfully and accurately. Use tools when necessary."
        return prompt
    
    def register_tool(self, tool) -> None:
        """Register a tool with the agent."""
        self.tool_registry.register(tool)
    
    def _prepare_context(self, query: str) -> str:
        """Prepare context for the LLM including memory and RAG."""
        context_parts = []
        
        # Add memory context
        if self.memory:
            history = self.memory.get_context()
            if history:
                context_parts.append(f"Conversation History:\n{history}\n")
        
        # Add RAG context
        if self.rag_pipeline:
            context = self.rag_pipeline.augment_prompt(query)
            context_parts.append(context)
        
        # Add tool context
        tool_context = self.tool_registry.get_tools_for_context()
        if tool_context:
            context_parts.append(tool_context)
        
        return "\n".join(context_parts)
    
    def _parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse tool calls from model response.
        Expected format: [TOOL_CALL] tool_name: param1=value1, param2=value2 [/TOOL_CALL]
        """
        if "[TOOL_CALL]" not in response:
            return None
        
        try:
            start = response.find("[TOOL_CALL]") + len("[TOOL_CALL]")
            end = response.find("[/TOOL_CALL]")
            tool_section = response[start:end].strip()
            
            # Parse tool name and parameters
            parts = tool_section.split(":", 1)
            tool_name = parts[0].strip()
            
            params = {}
            if len(parts) > 1:
                param_str = parts[1].strip()
                for param in param_str.split(","):
                    if "=" in param:
                        k, v = param.split("=", 1)
                        params[k.strip()] = v.strip()
            
            return {
                "tool": tool_name,
                "params": params
            }
        except Exception:
            return None
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool and return results."""
        try:
            self.state = AgentState.EXECUTING
            result = self.tool_registry.execute_tool(tool_name, **kwargs)
            self.state = AgentState.IDLE
            return str(result)
        except Exception as e:
            self.state = AgentState.ERROR
            return f"Tool execution error: {str(e)}"
    
    def think(self, query: str, max_iterations: int = None) -> str:
        """Main reasoning loop."""
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        self.state = AgentState.THINKING
        
        # Add user message to memory
        if self.memory:
            self.memory.add_message("user", query)
        
        # Prepare context
        context = self._prepare_context(query)
        
        # Simulate reasoning (in production, this would call an LLM)
        response = self._generate_response(context, query)
        
        # Check for tool calls
        for _ in range(max_iterations):
            tool_call = self._parse_tool_call(response)
            if tool_call:
                tool_result = self.execute_tool(
                    tool_call["tool"],
                    **tool_call["params"]
                )
                # In production, feed tool result back to LLM for refinement
                response = self._continue_reasoning(response, tool_result)
            else:
                break
        
        # Add assistant response to memory
        if self.memory:
            self.memory.add_message("assistant", response)
        
        self.state = AgentState.COMPLETE
        return response
    
    def _generate_response(self, context: str, query: str) -> str:
        """Generate response (placeholder - implement LLM integration)."""
        # This is a mock implementation
        # In production, call OpenAI API or other LLM
        return f"Based on the context, here's my response to '{query}':\n\nI need more context to provide a comprehensive answer. Please ensure the LLM is properly configured."
    
    def _continue_reasoning(self, current: str, tool_result: str) -> str:
        """Continue reasoning with tool results (placeholder)."""
        return current
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status and statistics."""
        status = {
            "name": self.name,
            "model": self.model,
            "state": self.state.value,
            "tools_registered": len(self.tool_registry.tools),
        }
        
        if self.memory:
            status["memory"] = self.memory.summary_stats()
        
        if self.rag_pipeline:
            status["rag"] = self.rag_pipeline.get_stats()
        
        return status
    
    def reset(self) -> None:
        """Reset agent state."""
        self.state = AgentState.IDLE
        if self.memory:
            self.memory.clear()
