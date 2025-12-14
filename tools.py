"""Tool calling system for agent interactions."""

from typing import List, Dict, Any, Callable
from enum import Enum


class ToolType(str, Enum):
    """Types of tools available to the agent."""
    SEARCH = "search"
    CALCULATOR = "calculator"
    SUMMARIZER = "summarizer"
    RETRIEVER = "retriever"
    CUSTOM = "custom"


class Tool:
    """Represents a tool that the agent can call."""
    
    def __init__(
        self,
        name: str,
        description: str,
        tool_type: ToolType,
        func: Callable,
        parameters: Dict[str, Any] = None
    ):
        self.name = name
        self.description = description
        self.tool_type = tool_type
        self.func = func
        self.parameters = parameters or {}
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        return self.func(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary format for model consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.tool_type.value,
            "parameters": self.parameters
        }


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Tool:
        """Retrieve a tool by name."""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        return self.tools[name]
    
    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        return tool.execute(**kwargs)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools."""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def get_tools_for_context(self) -> str:
        """Format tools for inclusion in model context."""
        tools_str = "Available Tools:\n"
        for tool in self.tools.values():
            tools_str += f"\n- {tool.name}: {tool.description}\n"
        return tools_str
