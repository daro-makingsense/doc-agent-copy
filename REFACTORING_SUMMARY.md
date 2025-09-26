# RAG Chat Agent Refactoring Summary

## Overview
Refactored the RAG chat agent implementation to use LangChain's tool calling capabilities instead of the explicit `use_rag` parameter. The agent now automatically decides when to search for relevant documents based on the user's query.

## Key Changes

### 1. New Chat Agent with Tool Calling (`app/agents/chat_agent_with_tools.py`)
- Created a new agent that uses LangChain's tool calling mechanism
- Implemented a `search_document` tool that the LLM can invoke when needed
- The agent automatically decides when to search based on the query content
- Returns structured outputs including sources used and tool calls made

### 2. Backend API Updates
- **Removed `use_rag` parameter** from the chat endpoint
- Updated `ChatRequest` schema to remove the `use_rag` field
- Added `tool_calls` field to `ChatResponse` schema
- Modified routes to use the new `chat_agent_with_tools` instead of `chat_rag_agent`
- Chat history now includes sources for assistant messages

### 3. Frontend UI Updates
- **Removed the "Use RAG" checkbox** from the chat interface
- Added `MessageSources` component to display sources when available
- Updated API client to remove `use_rag` parameter from requests
- Modified chat UI to display sources beneath assistant messages when the RAG tool was used

### 4. Structured Output Features
- Sources are now included in responses when the search tool is used
- Each source includes:
  - Page number
  - Similarity score
  - Preview of the content
  - Chunk ID for reference

## How It Works

1. **Automatic Tool Usage**: When a user sends a message, the LLM analyzes it and decides whether to use the document search tool based on the query content.

2. **Tool Calling Flow**:
   - User sends a message
   - LLM processes the message and may call the `search_document` tool
   - If called, the tool searches the vector database for relevant chunks
   - The LLM incorporates the search results into its response
   - Sources are tracked and returned with the response

3. **Source Display**: When sources are used, they appear beneath the assistant's message in the chat UI, showing:
   - Page number
   - Relevance score (as percentage)
   - Preview of the source content

## Benefits

1. **More Natural Interaction**: The agent intelligently decides when to search documents
2. **Transparency**: Users can see which tools were called and what sources were used
3. **Cleaner API**: No need to specify `use_rag` - the agent handles it automatically
4. **Better Architecture**: Follows LangChain best practices for tool calling
5. **Extensibility**: Easy to add more tools in the future

## Migration Notes

- The old `chat_rag_agent` is still available but deprecated
- All new features should use `chat_agent_with_tools`
- Existing chat sessions will continue to work
- No database migrations required