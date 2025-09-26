"""
Pydantic schemas for API serialization and validation.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID


class DocumentBase(BaseModel):
    """Base schema for document operations."""
    filename: str
    original_filename: str


class DocumentCreate(DocumentBase):
    """Schema for creating a new document."""
    file_path: str
    file_size: int
    mime_type: str


class DocumentUpdate(BaseModel):
    """Schema for updating document metadata."""
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    financial_facts: Optional[Dict[str, Any]] = None
    investment_data: Optional[Dict[str, Any]] = None
    key_metrics: Optional[Dict[str, Any]] = None
    is_processed: Optional[bool] = None
    is_embedded: Optional[bool] = None
    processing_error: Optional[str] = None
    pinecone_namespace: Optional[str] = None
    embedding_count: Optional[int] = None


class Document(DocumentBase):
    """Schema for document responses."""
    id: UUID
    file_path: str
    file_size: int
    mime_type: str
    page_count: Optional[int]
    word_count: Optional[int]
    financial_facts: Optional[Dict[str, Any]]
    investment_data: Optional[Dict[str, Any]]
    key_metrics: Optional[Dict[str, Any]]
    is_processed: bool
    is_embedded: bool
    processing_error: Optional[str]
    pinecone_namespace: Optional[str]
    embedding_count: int
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime]

    class Config:
        from_attributes = True


class DocumentSummary(BaseModel):
    """Simplified document schema for lists."""
    id: UUID
    filename: str
    original_filename: str
    file_size: int
    page_count: Optional[int]
    is_processed: bool
    is_embedded: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ChatSessionCreate(BaseModel):
    """Schema for creating a chat session."""
    document_id: Optional[UUID] = None
    session_name: Optional[str] = None
    user_id: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0, le=4000)
    system_prompt: Optional[str] = None


class ChatSessionUpdate(BaseModel):
    """Schema for updating chat session."""
    session_name: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0, le=4000)
    system_prompt: Optional[str] = None
    is_active: Optional[bool] = None


class ChatSession(BaseModel):
    """Schema for chat session responses."""
    id: UUID
    document_id: Optional[UUID]
    session_name: Optional[str]
    user_id: Optional[str]
    temperature: float
    max_tokens: int
    system_prompt: Optional[str]
    is_active: bool
    created_at: datetime
    last_activity: datetime

    class Config:
        from_attributes = True


class ChatMessageCreate(BaseModel):
    """Schema for creating a chat message."""
    session_id: UUID
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatMessage(BaseModel):
    """Schema for chat message responses."""
    id: UUID
    session_id: UUID
    role: str
    content: str
    token_count: Optional[int] = None
    model_used: Optional[str] = None
    response_time: Optional[float] = None
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    similarity_scores: Optional[List[float]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    """Schema for chat requests."""
    message: str
    session_id: Optional[UUID] = None
    document_id: Optional[UUID] = None


class ChatResponse(BaseModel):
    """Schema for chat responses."""
    message: str
    session_id: UUID
    sources_used: Optional[List[Dict[str, Any]]] = None
    tool_calls: List[str] = Field(default_factory=list)
    response_time: float
    token_count: Optional[int] = None


class ResearchTaskCreate(BaseModel):
    """Schema for creating a research task."""
    document_id: UUID
    topic: str
    research_query: str


class ResearchTask(BaseModel):
    """Schema for research task responses."""
    id: UUID
    document_id: UUID
    topic: str
    research_query: str
    status: str
    content_outline: Optional[Dict[str, Any]]
    research_findings: Optional[Dict[str, Any]]
    sources_used: Optional[List[Dict[str, Any]]]
    processing_time: Optional[float]
    model_used: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class ResearchRequest(BaseModel):
    """Schema for research requests."""
    document_id: UUID
    topic: str = Field(..., description="Topic to research (e.g., 'Key Investment Highlights')")
    custom_query: Optional[str] = None


class ResearchRequestBody(BaseModel):
    """Schema for research request body (without document_id since it comes from path)."""
    topic: str = Field(..., description="Topic to research (e.g., 'Key Investment Highlights')")
    custom_query: Optional[str] = None


class ResearchResponse(BaseModel):
    """Schema for research responses."""
    task_id: UUID
    content_outline: Dict[str, Any]
    research_findings: Dict[str, Any]
    sources_used: List[Dict[str, Any]]
    processing_time: float


class UploadResponse(BaseModel):
    """Schema for file upload responses."""
    document_id: UUID
    filename: str
    file_size: int
    status: str
    processing_started: bool


class ProcessingStatus(BaseModel):
    """Schema for processing status responses."""
    document_id: UUID
    filename: str
    is_processed: bool
    is_embedded: bool
    processing_error: Optional[str]
    embedding_count: int
    progress_percentage: float


# Financial metadata extraction schemas
class RevenueData(BaseModel):
    """Revenue information."""
    current_year: Optional[float] = Field(None, description="Current year revenue")
    previous_year: Optional[float] = Field(None, description="Previous year revenue")
    currency: str = Field("USD", description="Currency code")
    period: str = Field("annual", description="Period: annual, quarterly, or monthly")


class ProfitLossData(BaseModel):
    """Profit and loss information."""
    net_income: Optional[float] = Field(None, description="Net income")
    gross_profit: Optional[float] = Field(None, description="Gross profit")
    operating_profit: Optional[float] = Field(None, description="Operating profit")
    currency: str = Field("USD", description="Currency code")


class CashFlowData(BaseModel):
    """Cash flow information."""
    operating_cash_flow: Optional[float] = Field(None, description="Operating cash flow")
    free_cash_flow: Optional[float] = Field(None, description="Free cash flow")
    currency: str = Field("USD", description="Currency code")


class DebtEquityData(BaseModel):
    """Debt and equity information."""
    total_debt: Optional[float] = Field(None, description="Total debt")
    equity: Optional[float] = Field(None, description="Total equity")
    debt_to_equity_ratio: Optional[float] = Field(None, description="Debt to equity ratio")


class OtherMetrics(BaseModel):
    """Other financial metrics."""
    ebitda: Optional[float] = Field(None, description="EBITDA")
    margin_percentage: Optional[float] = Field(None, description="Margin percentage")
    growth_rate: Optional[float] = Field(None, description="Growth rate percentage")


class FinancialFacts(BaseModel):
    """Complete financial facts extracted from document."""
    revenue: RevenueData = Field(default_factory=RevenueData)
    profit_loss: ProfitLossData = Field(default_factory=ProfitLossData)
    cash_flow: CashFlowData = Field(default_factory=CashFlowData)
    debt_equity: DebtEquityData = Field(default_factory=DebtEquityData)
    other_metrics: OtherMetrics = Field(default_factory=OtherMetrics)


class MarketOpportunity(BaseModel):
    """Market opportunity information."""
    market_size: Optional[float] = Field(None, description="Market size")
    growth_rate: Optional[float] = Field(None, description="Market growth rate percentage")
    competitive_position: Optional[str] = Field(None, description="Competitive position description")


class BusinessModel(BaseModel):
    """Business model information."""
    type: Optional[str] = Field(None, description="Business model type")
    revenue_streams: List[str] = Field(default_factory=list, description="Revenue streams")
    key_customers: List[str] = Field(default_factory=list, description="Key customers")


class ExitStrategy(BaseModel):
    """Exit strategy information."""
    timeline: Optional[str] = Field(None, description="Timeline for exit")
    target_multiple: Optional[float] = Field(None, description="Target multiple for exit")
    potential_buyers: List[str] = Field(default_factory=list, description="Potential buyers")


class InvestmentData(BaseModel):
    """Complete investment data extracted from document."""
    investment_highlights: List[str] = Field(default_factory=list, description="Key investment highlights")
    risk_factors: List[str] = Field(default_factory=list, description="Risk factors")
    market_opportunity: MarketOpportunity = Field(default_factory=MarketOpportunity)
    business_model: BusinessModel = Field(default_factory=BusinessModel)
    strategic_initiatives: List[str] = Field(default_factory=list, description="Strategic initiatives")
    exit_strategy: ExitStrategy = Field(default_factory=ExitStrategy)