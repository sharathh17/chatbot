# Chatbot

An AI-powered chatbot application built with FastAPI, LangChain, and OpenAI. This project implements an intelligent agent with RAG (Retrieval-Augmented Generation) capabilities for answering questions based on ingested documents.

## Features

- **FastAPI Backend**: High-performance async API for chatbot interactions
- **AI Agent**: Intelligent agent with tool calling capabilities
- **RAG Support**: Retrieve relevant information from a knowledge base
- **Document Ingestion**: Load and process documents for the knowledge base
- **Memory Management**: Maintains conversation context and state
- **Tool Integration**: Extensible tool system for custom functionality

## Project Structure

```
├── main.py           # FastAPI application entry point
├── agent.py          # Core AI agent implementation
├── tools.py          # Tool definitions and execution
├── rag.py            # RAG (Retrieval-Augmented Generation) system
├── memory.py         # Memory and state management
├── ingest.py         # Document ingestion and processing
├── requirements.txt  # Project dependencies
└── README.md         # This file
```

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sharathh17/chatbot.git
cd chatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Running the Server

Start the FastAPI application:
```bash
python main.py
```

The server will run on `http://localhost:8000`

### API Endpoints

- **POST /query** - Submit a query to the agent
  ```json
  {
    "query": "Your question here",
    "use_rag": true,
    "max_iterations": 10
  }
  ```

- **POST /ingest** - Ingest documents into the knowledge base
  ```json
  {
    "file_path": "path/to/document.txt",
    "chunk": true
  }
  ```

- **POST /tool** - Execute a tool directly
  ```json
  {
    "tool_name": "tool_name",
    "parameters": { "key": "value" }
  }
  ```

### Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation powered by Swagger UI.

## Configuration

Key environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key

## Technologies Used

- **FastAPI**: Modern web framework for building APIs
- **LangChain**: Framework for building with LLMs
- **OpenAI**: GPT models for AI capabilities
- **FAISS**: Vector database for semantic search
- **Pydantic**: Data validation using Python type annotations

## Architecture

### Agent System
The agent uses an agentic loop to:
1. Process user queries
2. Decide which tools to use
3. Execute tools with appropriate parameters
4. Iterate until reaching a conclusion

### RAG System
Retrieves relevant documents from the knowledge base to provide context-aware responses.

### Memory System
Maintains conversation history and state management across interactions.

## Development

To extend the chatbot:

1. **Add new tools**: Modify `tools.py` to register new tools
2. **Customize RAG**: Adjust retrieval settings in `rag.py`
3. **Enhance memory**: Modify memory handling in `memory.py`

## License

This project is open source and available under the MIT License.

## Author

[sharathh17](https://github.com/sharathh17)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
