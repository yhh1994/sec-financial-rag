# SEC Financial Statement RAG Agent 📊

An agentic RAG pipeline built with LangChain and local open-weight models via Ollama to query SEC 10-K and 10-Q filings.

## Prerequisites
1. Install [Ollama](https://ollama.com/) on your machine.
2. Pull the model you want to use:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
