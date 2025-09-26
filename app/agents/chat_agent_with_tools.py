"""
Chat Agent with Tool Calling for conversational interaction with financial documents.
Uses LangChain's tool calling capabilities for retrieval-augmented generation.
"""
import logging
from typing import Dict, Any, List, Optional, Type, Tuple
import asyncio
from datetime import datetime
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.pydantic_v1 import BaseModel as PydanticV1BaseModel, Field as V1Field
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.embedding_service import embedding_service
from app.tools import vector_search_tool
from app.database.models import Document, ChatSession, ChatMessage
from app.database.connection import get_db_session
from app.prompts.chat_agent import FINANCIAL_ANALYST_SYSTEM_PROMPT

settings = get_settings()
logger = logging.getLogger(__name__)


class SearchDocumentInput(PydanticV1BaseModel):
    """Input schema for document search tool."""
    query: str = V1Field(description="The search query to find relevant information")
    top_k: int = V1Field(default=5, description="Number of results to return")


class ChatResponse(BaseModel):
    """Structured output for chat responses."""
    message: str = Field(description="The main response message")
    sources_used: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="List of document sources used in the response"
    )
    tool_calls: List[str] = Field(
        default_factory=list,
        description="List of tools that were called during response generation"
    )


class ChatState(BaseModel):
    """State for the Chat Agent with tools."""
    messages: List[BaseMessage] = Field(default_factory=list)
    session_id: Optional[str] = None
    document_id: Optional[str] = None
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    sources_used: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[str] = Field(default_factory=list)
    response_metadata: Dict[str, Any] = Field(default_factory=dict)
    current_step: str = "start"
    errors: List[str] = Field(default_factory=list)


class ChatAgentWithTools:
    """Agent for conversational interaction with financial documents using tool calling."""
    
    def __init__(self):
        """Initialize the Chat Agent with tools."""
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.7,
            openai_api_key=settings.OPENAI_API_KEY
        ).bind_tools(self.get_tools())
        
        # System prompt for financial document conversations
        self.system_prompt = FINANCIAL_ANALYST_SYSTEM_PROMPT
        
        # Create the chat workflow
        self.workflow = self._create_workflow()
    
    def get_tools(self) -> List[BaseTool]:
        """Get the list of available tools."""
        return [self.create_search_tool()]
    
    def create_search_tool(self) -> BaseTool:
        """Create the document search tool."""
        @tool(args_schema=SearchDocumentInput)
        async def search_document(query: str, top_k: int = 5) -> Dict[str, Any]:
            """Search for relevant information in the financial documents.
            
            Args:
                query: The search query
                top_k: Number of results to return
                
            Returns:
                Dictionary containing search results and formatted sources
            """
            # Get document_id from the current context (will be set during chat)
            document_id = getattr(search_document, '_document_id', None)
            
            result = await vector_search_tool.search(
                query=query,
                document_id=document_id,
                top_k=top_k,
                similarity_threshold=0.6,
            )
            
            # Store sources for later use
            if hasattr(search_document, '_store_sources'):
                search_document._store_sources(result.get('similar_chunks', []))
            
            return {
                "results": result.get('formatted_results', ''),
                "sources": result.get('formatted_sources', ''),
                "chunk_count": len(result.get('similar_chunks', []))
            }
        
        return search_document
    
    def _create_workflow(self) -> StateGraph:
        """Create the langraph workflow for chat interactions with tools."""
        workflow = StateGraph(ChatState)
        
        # Add nodes for each step of the chat process
        workflow.add_node("load_context", self._load_chat_context)
        workflow.add_node("agent", self._run_agent)
        workflow.add_node("tools", self._run_tools)
        workflow.add_node("save_interaction", self._save_chat_interaction)
        
        # Define edges
        workflow.add_edge("load_context", "agent")
        
        # Conditional edge from agent to either tools or save
        workflow.add_conditional_edges(
            "agent",
            self._should_use_tools,
            {
                "tools": "tools",
                "save": "save_interaction"
            }
        )
        
        # After tools, go back to agent
        workflow.add_edge("tools", "agent")
        workflow.add_edge("save_interaction", END)
        
        # Set entry point
        workflow.set_entry_point("load_context")
        
        return workflow.compile()
    
    def _should_use_tools(self, state: ChatState) -> str:
        """Determine if tools should be used based on the last message."""
        if state.messages and hasattr(state.messages[-1], 'tool_calls') and state.messages[-1].tool_calls:
            return "tools"
        return "save"
    
    async def _load_chat_context(self, state: ChatState) -> ChatState:
        """Load chat history and context."""
        try:
            if state.session_id:
                # Load recent chat history
                with get_db_session() as db:
                    recent_messages = db.query(ChatMessage).filter(
                        ChatMessage.session_id == state.session_id
                    ).order_by(ChatMessage.created_at.desc()).limit(10).all()
                    
                    # Convert to chat history format (reverse order)
                    chat_history = []
                    for msg in reversed(recent_messages):
                        chat_history.append({
                            'role': msg.role,
                            'content': msg.content
                        })
                    
                    state.chat_history = chat_history
            
            state.current_step = "context_loaded"
            return state
            
        except Exception as e:
            logger.error(f"Error loading chat context: {str(e)}")
            state.errors.append(f"Context loading error: {str(e)}")
            return state
    
    async def _run_agent(self, state: ChatState) -> ChatState:
        """Run the agent to generate a response."""
        try:
            # Set up the search tool with the current document context
            search_tool = self.get_tools()[0]
            search_tool._document_id = state.document_id
            
            # Create a list to store sources
            sources_storage = []
            search_tool._store_sources = lambda sources: sources_storage.extend(sources)
            
            # Prepare messages
            messages = [SystemMessage(content=self.system_prompt)]
            
            # Add chat history
            for msg in state.chat_history[-6:]:  # Last 6 messages for context
                if msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    messages.append(AIMessage(content=msg['content']))
            
            # Add current messages from state
            messages.extend(state.messages)
            
            # Generate response with potential tool calls
            response = await self.llm.ainvoke(messages)
            
            # Add response to messages
            state.messages.append(response)
            
            # Check if tools were called
            if hasattr(response, 'tool_calls') and response.tool_calls:
                state.tool_calls.extend([tc['name'] for tc in response.tool_calls])
            
            # Store sources if any were collected
            if sources_storage:
                state.sources_used = sources_storage
            
            state.current_step = "agent_complete"
            return state
            
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            state.errors.append(f"Agent error: {str(e)}")
            # Create error response
            error_msg = AIMessage(content=f"I apologize, but I encountered an error: {str(e)}")
            state.messages.append(error_msg)
            return state
    
    async def _run_tools(self, state: ChatState) -> ChatState:
        """Execute tool calls from the agent."""
        try:
            last_message = state.messages[-1]
            if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                return state
            
            # Execute each tool call
            for tool_call in last_message.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                # Find and execute the tool
                tools = self.get_tools()
                tool_to_run = next((t for t in tools if t.name == tool_name), None)
                
                if tool_to_run:
                    # Set document context for the tool
                    tool_to_run._document_id = state.document_id
                    
                    # Create sources storage
                    sources_storage = []
                    tool_to_run._store_sources = lambda sources: sources_storage.extend(sources)
                    
                    # Execute tool
                    result = await tool_to_run.ainvoke(tool_args)
                    
                    # Create tool message with result
                    tool_message = ToolMessage(
                        content=json.dumps(result),
                        tool_call_id=tool_call['id']
                    )
                    state.messages.append(tool_message)
                    
                    # Store sources if any
                    if sources_storage:
                        state.sources_used.extend(sources_storage)
            
            state.current_step = "tools_complete"
            return state
            
        except Exception as e:
            logger.error(f"Error running tools: {str(e)}")
            state.errors.append(f"Tool execution error: {str(e)}")
            return state
    
    async def _save_chat_interaction(self, state: ChatState) -> ChatState:
        """Save the chat interaction to the database."""
        try:
            if state.session_id and state.messages:
                with get_db_session() as db:
                    # Find the original user message and final assistant response
                    user_content = None
                    assistant_content = None
                    
                    for msg in state.messages:
                        if isinstance(msg, HumanMessage) and not user_content:
                            user_content = msg.content
                        elif isinstance(msg, AIMessage) and not isinstance(msg, ToolMessage):
                            assistant_content = msg.content
                    
                    if user_content and assistant_content:
                        # Save user message
                        user_message = ChatMessage(
                            session_id=state.session_id,
                            role="user",
                            content=user_content,
                            token_count=self._estimate_tokens(user_content)
                        )
                        db.add(user_message)
                        
                        # Save assistant response with sources
                        assistant_message = ChatMessage(
                            session_id=state.session_id,
                            role="assistant",
                            content=assistant_content,
                            token_count=self._estimate_tokens(assistant_content),
                            model_used=settings.OPENAI_MODEL,
                            retrieved_chunks=state.sources_used if state.sources_used else None,
                            similarity_scores=[s.get('similarity_score') for s in state.sources_used] if state.sources_used else None
                        )
                        db.add(assistant_message)
                        
                        # Update session last activity
                        session = db.query(ChatSession).filter(ChatSession.id == state.session_id).first()
                        if session:
                            session.last_activity = datetime.now()
            
            state.current_step = "interaction_saved"
            return state
            
        except Exception as e:
            logger.error(f"Error saving chat interaction: {str(e)}")
            state.errors.append(f"Save interaction error: {str(e)}")
            return state
    
    async def chat(
            self, 
            message: str, 
            session_id: Optional[str] = None,
            document_id: Optional[str] = None
        ) -> Dict[str, Any]:
        """
        Process a chat message and generate a response using tool calling.
        
        Args:
            message: User's message
            session_id: Optional chat session ID
            document_id: Optional document ID for context
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            start_time = asyncio.get_event_loop().time()
            logger.info(f"Processing chat message: {message[:50]}...")
            
            # Create or get session
            if not session_id:
                session_id = await self._create_chat_session(document_id)
            
            # Initialize chat state with user message
            initial_state = ChatState(
                messages=[HumanMessage(content=message)],
                session_id=session_id,
                document_id=document_id
            )
            
            # Execute the chat workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            logger.info(f"Chat response generated in {response_time:.2f}s")
            
            # Extract the final assistant message
            assistant_response = None
            for msg in reversed(final_state.messages):
                if isinstance(msg, AIMessage) and not isinstance(msg, ToolMessage):
                    assistant_response = msg.content
                    break
            
            if not assistant_response:
                assistant_response = "I apologize, but I couldn't generate a response."
            
            # Format sources for response
            formatted_sources = None
            if final_state.sources_used:
                formatted_sources = []
                for source in final_state.sources_used:
                    formatted_sources.append({
                        'chunk_id': source.get('chunk_id'),
                        'page_number': source.get('page_number'),
                        'similarity_score': source.get('similarity_score'),
                        'preview': source.get('content', '')[:200] + "..." if source.get('content') else ""
                    })
            
            return {
                'message': assistant_response,
                'session_id': session_id,
                'sources_used': formatted_sources,
                'tool_calls': final_state.tool_calls,
                'response_time': response_time,
                'token_count': self._estimate_tokens(assistant_response),
                'errors': final_state.errors if final_state.errors else None
            }
            
        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}")
            return {
                'message': f"I apologize, but I encountered an error processing your message: {str(e)}",
                'session_id': session_id,
                'sources_used': None,
                'tool_calls': [],
                'response_time': 0,
                'error': str(e)
            }
    
    async def _create_chat_session(self, document_id: Optional[str] = None) -> str:
        """Create a new chat session."""
        with get_db_session() as db:
            session = ChatSession(
                document_id=document_id,
                session_name=f"Chat Session - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                system_prompt=self.system_prompt[:1000]  # Truncate if too long
            )
            db.add(session)
            db.flush()
            session_id = str(session.id)
        
        logger.info(f"Created new chat session: {session_id}")
        return session_id
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return int(len(text.split()) * 1.3)  # Rough approximation
    
    async def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chat history for a session."""
        try:
            with get_db_session() as db:
                messages = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                ).order_by(ChatMessage.created_at).limit(limit).all()
                
                history = []
                for msg in messages:
                    msg_data = {
                        'id': str(msg.id),
                        'role': msg.role,
                        'content': msg.content,
                        'created_at': msg.created_at.isoformat(),
                        'token_count': msg.token_count,
                    }
                    
                    # Include sources for assistant messages
                    if msg.role == 'assistant' and msg.retrieved_chunks:
                        msg_data['sources'] = []
                        for chunk in msg.retrieved_chunks:
                            msg_data['sources'].append({
                                'chunk_id': chunk.get('chunk_id'),
                                'page_number': chunk.get('page_number'),
                                'similarity_score': chunk.get('similarity_score'),
                                'preview': chunk.get('content', '')[:200] + '...' if chunk.get('content') else ''
                            })
                    
                    history.append(msg_data)
                
                return history
                
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return []
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a chat session."""
        try:
            with get_db_session() as db:
                session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
                if session:
                    message_count = db.query(ChatMessage).filter(
                        ChatMessage.session_id == session_id
                    ).count()
                    
                    return {
                        'id': str(session.id),
                        'document_id': str(session.document_id) if session.document_id else None,
                        'session_name': session.session_name,
                        'created_at': session.created_at.isoformat(),
                        'last_activity': session.last_activity.isoformat(),
                        'message_count': message_count,
                        'is_active': session.is_active
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting session info: {str(e)}")
            return None
    
    async def list_user_sessions(self, user_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """List chat sessions for a user."""
        try:
            with get_db_session() as db:
                query = db.query(ChatSession)
                if user_id:
                    query = query.filter(ChatSession.user_id == user_id)
                
                sessions = query.order_by(ChatSession.last_activity.desc()).limit(limit).all()
                
                session_list = []
                for session in sessions:
                    message_count = db.query(ChatMessage).filter(
                        ChatMessage.session_id == session.id
                    ).count()
                    
                    session_list.append({
                        'id': str(session.id),
                        'session_name': session.session_name,
                        'document_id': str(session.document_id) if session.document_id else None,
                        'created_at': session.created_at.isoformat(),
                        'last_activity': session.last_activity.isoformat(),
                        'message_count': message_count,
                        'is_active': session.is_active
                    })
                
                return session_list
                
        except Exception as e:
            logger.error(f"Error listing user sessions: {str(e)}")
            return []


# Global instance
chat_agent_with_tools = ChatAgentWithTools()