"""
Refactored metadata extraction service using vector search and regex patterns.
Only uses LLM for document summaries.
"""
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import asyncio

from app.config import get_settings
from app.database.models import Document, DocumentChunk
from app.database.connection import get_db_session
from app.services.embedding_service import embedding_service
from app.services.metadata_extractor import MetadataExtractor
from app.utils.logging_config import extraction_metrics

settings = get_settings()
logger = logging.getLogger(__name__)


class VectorRegexMetadataExtractor:
    """Enhanced metadata extractor using vector search and regex patterns."""
    
    def __init__(self):
        """Initialize the enhanced metadata extractor."""
        # Initialize the original extractor for summary generation only
        self.legacy_extractor = MetadataExtractor()
        
        # Enhanced financial patterns with more comprehensive coverage
        self.financial_patterns = {
            'revenue': {
                'patterns': [
                    # Standard revenue patterns
                    r'(?:total\s+)?(?:net\s+)?revenues?\s*(?:of|were|was|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                    r'revenues?\s+(?:increased|decreased|grew|declined)\s+(?:to|by)\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                    r'(?:fiscal|fy|year)\s*\d{4}\s*revenues?\s*(?:of|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                    # Revenue in tables
                    r'revenues?\s*\n+\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                    # Sales patterns
                    r'(?:net\s+)?sales\s*(?:of|were|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                ],
                'keywords': ['revenue', 'sales', 'turnover', 'income statement', 'top line']
            },
            'profit': {
                'patterns': [
                    # Net income patterns
                    r'net\s+(?:income|profit|earnings?)\s*(?:of|was|were|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                    r'(?:net\s+)?(?:income|profit)\s+(?:increased|decreased|grew|declined)\s+(?:to|by)\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                    # Gross profit patterns
                    r'gross\s+(?:profit|margin)\s*(?:of|was|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                    # Operating income patterns
                    r'operating\s+(?:income|profit|earnings?)\s*(?:of|was|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                    # EBITDA patterns
                    r'(?:adjusted\s+)?ebitda\s*(?:of|was|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                    # Loss patterns
                    r'net\s+loss\s*(?:of|was|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                ],
                'keywords': ['profit', 'income', 'earnings', 'ebitda', 'loss', 'margin']
            },
            'cash_flow': {
                'patterns': [
                    # Operating cash flow
                    r'(?:net\s+)?cash\s+(?:provided\s+by|from|used\s+in)\s+operating\s+activities\s*(?:of|was|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                    r'operating\s+cash\s+flows?\s*(?:of|was|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                    # Free cash flow
                    r'free\s+cash\s+flows?\s*(?:of|was|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                    # Investing/Financing cash flow
                    r'cash\s+(?:provided\s+by|from|used\s+in)\s+(?:investing|financing)\s+activities\s*(?:of|was|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                ],
                'keywords': ['cash flow', 'operating cash', 'free cash', 'cash provided', 'cash used']
            },
            'balance_sheet': {
                'patterns': [
                    # Total assets
                    r'total\s+assets\s*(?:of|were|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                    # Total liabilities
                    r'total\s+liabilities\s*(?:of|were|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                    # Total debt
                    r'(?:total\s+)?(?:long[- ]term\s+)?debt\s*(?:of|was|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                    # Shareholders equity
                    r'(?:total\s+)?(?:shareholders?\'?|stockholders?\'?)\s+equity\s*(?:of|was|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                    # Working capital
                    r'working\s+capital\s*(?:of|was|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m|b|k)?',
                ],
                'keywords': ['assets', 'liabilities', 'debt', 'equity', 'balance sheet', 'financial position']
            },
            'ratios': {
                'patterns': [
                    # Percentage patterns
                    r'(?:revenue|sales|profit|margin|growth|roi|roce)\s*(?:of|was|increased\s+by|decreased\s+by)?\s*([\d]+(?:\.\d+)?)\s*%',
                    # Debt to equity ratio
                    r'debt[- ]to[- ]equity\s+ratio\s*(?:of|was|:)?\s*([\d]+(?:\.\d+)?)',
                    # Current ratio
                    r'current\s+ratio\s*(?:of|was|:)?\s*([\d]+(?:\.\d+)?)',
                    # ROI/ROE patterns
                    r'(?:return\s+on\s+(?:investment|equity)|roi|roe)\s*(?:of|was|:)?\s*([\d]+(?:\.\d+)?)\s*%?',
                ],
                'keywords': ['ratio', 'margin', 'percentage', 'growth rate', 'return']
            }
        }
        
        # Investment data patterns
        self.investment_patterns = {
            'risk_factors': {
                'patterns': [
                    r'(?:key\s+)?risk\s+factors?\s*(?:include|are|:)\s*([^.]+(?:\.[^.]+){0,3})',
                    r'(?:principal|main|significant)\s+risks?\s*(?:include|are|:)\s*([^.]+(?:\.[^.]+){0,3})',
                    r'(?:we\s+)?(?:face|are\s+subject\s+to)\s+(?:the\s+following\s+)?risks?\s*(?::|including)\s*([^.]+(?:\.[^.]+){0,3})',
                ],
                'section_keywords': ['risk factors', 'risks', 'uncertainties', 'challenges']
            },
            'investment_highlights': {
                'patterns': [
                    r'(?:key\s+)?investment\s+highlights?\s*(?:include|are|:)\s*([^.]+(?:\.[^.]+){0,3})',
                    r'(?:investment\s+)?(?:thesis|opportunity)\s*(?:include|are|:)\s*([^.]+(?:\.[^.]+){0,3})',
                    r'(?:competitive\s+)?advantages?\s*(?:include|are|:)\s*([^.]+(?:\.[^.]+){0,3})',
                ],
                'section_keywords': ['investment highlights', 'investment thesis', 'value proposition', 'competitive advantages']
            },
            'business_model': {
                'patterns': [
                    r'(?:our\s+)?business\s+model\s*(?:is|includes?|consists?\s+of)\s*([^.]+(?:\.[^.]+){0,2})',
                    r'revenue\s+streams?\s*(?:include|are|consist\s+of)\s*([^.]+(?:\.[^.]+){0,2})',
                    r'(?:we\s+)?generate\s+revenue\s+(?:through|from|by)\s*([^.]+(?:\.[^.]+){0,2})',
                ],
                'section_keywords': ['business model', 'revenue model', 'how we make money', 'revenue streams']
            },
            'market_opportunity': {
                'patterns': [
                    r'(?:total\s+)?(?:addressable\s+)?market\s*(?:size|opportunity)?\s*(?:is|of|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|trillion|m|b|t)?',
                    r'market\s+(?:is\s+)?(?:expected\s+to\s+)?grow(?:ing)?\s+(?:at|by)\s*(?:a\s+cagr\s+of\s*)?([\d]+(?:\.\d+)?)\s*%',
                    r'(?:we\s+)?(?:operate|compete)\s+in\s+(?:a|an?)\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|trillion|m|b|t)?\s+market',
                ],
                'section_keywords': ['market opportunity', 'tam', 'market size', 'addressable market']
            }
        }
    
    def _parse_amount(self, value_str: str, scale: Optional[str]) -> Optional[float]:
        """Parse amount string with scale to float."""
        try:
            # Remove commas and convert to float
            value = float(value_str.replace(',', ''))
            
            # Apply scale
            if scale:
                scale_lower = scale.lower()
                if scale_lower in ['million', 'millions', 'm']:
                    value *= 1_000_000
                elif scale_lower in ['billion', 'billions', 'b']:
                    value *= 1_000_000_000
                elif scale_lower in ['thousand', 'thousands', 'k']:
                    value *= 1_000
                elif scale_lower in ['trillion', 'trillions', 't']:
                    value *= 1_000_000_000_000
            
            return value
        except (ValueError, AttributeError):
            return None
    
    async def extract_metadata(self, document_id: str) -> Dict[str, Any]:
        """
        Extract metadata using vector search and regex patterns.
        
        Args:
            document_id: UUID of the document
            
        Returns:
            Dictionary containing extracted metadata
        """
        try:
            logger.info(f"Starting enhanced metadata extraction for document {document_id}")
            extraction_metrics.log_extraction_start(document_id, "vector_regex")
            
            # Get document and chunks
            with get_db_session() as db:
                document = db.query(Document).filter(Document.id == document_id).first()
                if not document:
                    raise ValueError(f"Document {document_id} not found")
                
                chunks = db.query(DocumentChunk).filter(
                    DocumentChunk.document_id == document_id
                ).order_by(DocumentChunk.chunk_index).all()
                
                # Extract content from chunks
                chunk_contents = [chunk.content for chunk in chunks]
            
            # Combine all chunks for full text analysis
            full_text = "\n\n".join(chunk_contents)
            extraction_metrics.log_extraction_step(document_id, "text_combined", {
                "total_chunks": len(chunk_contents),
                "total_characters": len(full_text)
            })
            
            # Step 1: Use vector search to find relevant sections
            financial_sections = await self._find_financial_sections(document_id, full_text)
            extraction_metrics.log_extraction_step(document_id, "financial_sections_found", {
                "section_length": len(financial_sections)
            })
            
            investment_sections = await self._find_investment_sections(document_id, full_text)
            extraction_metrics.log_extraction_step(document_id, "investment_sections_found", {
                "section_length": len(investment_sections)
            })
            
            # Step 2: Extract financial data using regex on relevant sections
            financial_facts = self._extract_financial_facts_regex(financial_sections)
            extraction_metrics.log_extraction_step(document_id, "financial_facts_extracted", {
                "has_revenue": financial_facts['revenue']['current_year'] is not None,
                "has_profit": financial_facts['profit_loss']['net_income'] is not None,
                "has_cash_flow": financial_facts['cash_flow']['operating_cash_flow'] is not None
            })
            
            # Step 3: Extract investment data using regex
            investment_data = self._extract_investment_data_regex(investment_sections)
            extraction_metrics.log_extraction_step(document_id, "investment_data_extracted", {
                "highlights_count": len(investment_data.get('investment_highlights', [])),
                "risks_count": len(investment_data.get('risk_factors', [])),
                "has_market_data": investment_data['market_opportunity']['market_size'] is not None
            })
            
            # Step 4: Extract key metrics from full text
            key_metrics = self._extract_key_metrics(full_text)
            extraction_metrics.log_extraction_step(document_id, "key_metrics_extracted", {
                "metrics_count": key_metrics.get('financial_metrics_count', 0),
                "percentages_found": len(key_metrics.get('percentages_found', []))
            })
            
            # Step 5: Extract document structure
            structure_info = self._extract_document_structure(full_text)
            extraction_metrics.log_extraction_step(document_id, "structure_analyzed", {
                "sections_count": structure_info.get('sections_count', 0),
                "tables_detected": structure_info.get('tables_detected', 0)
            })
            
            # Step 6: Generate document summary (only LLM usage)
            extraction_metrics.log_extraction_step(document_id, "summary_generation_start", {"method": "LLM"})
            summary = await self.legacy_extractor.summarize_document(document_id)
            extraction_metrics.log_extraction_step(document_id, "summary_generated", {"summary_length": len(summary)})
            
            # Combine all metadata
            metadata = {
                'financial_facts': financial_facts,
                'investment_data': investment_data,
                'key_metrics': key_metrics,
                'document_structure': structure_info,
                'document_summary': summary,
                'extraction_timestamp': str(datetime.now(timezone.utc).isoformat()),
                'extraction_method': 'vector_search_regex'
            }
            
            # Update document in database
            logger.info(f"Updating document {document_id} with metadata")
            with get_db_session() as db:
                document = db.query(Document).filter(Document.id == document_id).first()
                if document:
                    document.financial_facts = financial_facts
                    document.investment_data = investment_data
                    document.key_metrics = {**key_metrics, **structure_info}
                    db.commit()
            
            # Log extraction summary
            extraction_summary = {
                "financial_data_found": any([
                    financial_facts['revenue']['current_year'] is not None,
                    financial_facts['profit_loss']['net_income'] is not None,
                    financial_facts['cash_flow']['operating_cash_flow'] is not None
                ]),
                "investment_data_found": any([
                    len(investment_data.get('investment_highlights', [])) > 0,
                    len(investment_data.get('risk_factors', [])) > 0,
                    investment_data['market_opportunity']['market_size'] is not None
                ]),
                "total_metrics": key_metrics.get('financial_metrics_count', 0),
                "document_sections": structure_info.get('sections_count', 0)
            }
            
            extraction_metrics.log_extraction_complete(document_id, True, extraction_summary)
            logger.info(f"Enhanced metadata extraction completed for document {document_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error in enhanced metadata extraction for document {document_id}: {str(e)}")
            extraction_metrics.log_extraction_complete(document_id, False, {"error": str(e)})
            raise e
    
    async def _find_financial_sections(self, document_id: str, full_text: str) -> str:
        """Use vector search to find sections containing financial data."""
        try:
            # Financial search queries
            financial_queries = [
                "financial statements income statement balance sheet cash flow",
                "revenue sales profit loss earnings EBITDA net income",
                "total assets liabilities debt equity shareholders",
                "operating cash flow free cash flow investing financing",
                "financial results fiscal year quarterly results"
            ]
            
            all_relevant_chunks = []
            
            # Search for each query
            for query in financial_queries:
                chunks = await embedding_service.search_similar_chunks(
                    query=query,
                    document_id=document_id,
                    top_k=5,
                    similarity_threshold=0.6
                )
                all_relevant_chunks.extend(chunks)
            
            # Deduplicate chunks by chunk_id
            seen_ids = set()
            unique_chunks = []
            for chunk in all_relevant_chunks:
                chunk_id = chunk.get('chunk_id')
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    unique_chunks.append(chunk)
            
            # Sort by chunk index to maintain document order
            unique_chunks.sort(key=lambda x: x.get('chunk_index', 0))
            
            # Combine chunk contents
            financial_text = "\n\n".join([chunk['content'] for chunk in unique_chunks])
            
            logger.info(f"Found {len(unique_chunks)} unique financial sections via vector search")
            
            # If not enough content found, use keyword search as fallback
            if len(financial_text) < 1000:
                logger.info("Insufficient content from vector search, using keyword search fallback")
                financial_text = self._keyword_search_sections(
                    full_text, 
                    ['financial', 'revenue', 'profit', 'income', 'cash flow', 'balance sheet', 'assets', 'liabilities']
                )
            
            return financial_text
            
        except Exception as e:
            logger.error(f"Error in financial section search: {str(e)}")
            # Fallback to keyword search
            return self._keyword_search_sections(
                full_text, 
                ['financial', 'revenue', 'profit', 'income', 'cash flow', 'balance sheet']
            )
    
    async def _find_investment_sections(self, document_id: str, full_text: str) -> str:
        """Use vector search to find sections containing investment data."""
        try:
            # Investment search queries
            investment_queries = [
                "investment highlights thesis opportunity value proposition",
                "risk factors uncertainties challenges threats",
                "business model revenue streams how we make money",
                "market opportunity TAM addressable market size growth",
                "competitive advantages differentiation moat strategy",
                "exit strategy acquisition IPO strategic buyers"
            ]
            
            all_relevant_chunks = []
            
            # Search for each query
            for query in investment_queries:
                chunks = await embedding_service.search_similar_chunks(
                    query=query,
                    document_id=document_id,
                    top_k=5,
                    similarity_threshold=0.6
                )
                all_relevant_chunks.extend(chunks)
            
            # Deduplicate chunks
            seen_ids = set()
            unique_chunks = []
            for chunk in all_relevant_chunks:
                chunk_id = chunk.get('chunk_id')
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    unique_chunks.append(chunk)
            
            # Sort by chunk index
            unique_chunks.sort(key=lambda x: x.get('chunk_index', 0))
            
            # Combine chunk contents
            investment_text = "\n\n".join([chunk['content'] for chunk in unique_chunks])
            
            logger.info(f"Found {len(unique_chunks)} unique investment sections via vector search")
            
            # Fallback to keyword search if needed
            if len(investment_text) < 1000:
                logger.info("Insufficient content from vector search, using keyword search fallback")
                investment_text = self._keyword_search_sections(
                    full_text,
                    ['investment', 'risk', 'opportunity', 'business model', 'market', 'competitive']
                )
            
            return investment_text
            
        except Exception as e:
            logger.error(f"Error in investment section search: {str(e)}")
            return self._keyword_search_sections(
                full_text,
                ['investment', 'risk', 'opportunity', 'business model', 'market']
            )
    
    def _keyword_search_sections(self, text: str, keywords: List[str], context_size: int = 500) -> str:
        """Fallback keyword-based section extraction."""
        sections = []
        text_lower = text.lower()
        
        for keyword in keywords:
            start_pos = 0
            while True:
                pos = text_lower.find(keyword.lower(), start_pos)
                if pos == -1:
                    break
                
                # Extract context around keyword
                section_start = max(0, pos - context_size)
                section_end = min(len(text), pos + context_size)
                section = text[section_start:section_end]
                sections.append(section)
                
                start_pos = pos + 1
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sections = []
        for section in sections:
            if section not in seen:
                seen.add(section)
                unique_sections.append(section)
        
        return "\n\n---\n\n".join(unique_sections)
    
    def _extract_financial_facts_regex(self, text: str) -> Dict[str, Any]:
        """Extract financial facts using comprehensive regex patterns."""
        financial_facts = {
            'revenue': {'current_year': None, 'previous_year': None, 'currency': 'USD', 'period': 'annual'},
            'profit_loss': {'net_income': None, 'gross_profit': None, 'operating_profit': None, 'currency': 'USD'},
            'cash_flow': {'operating_cash_flow': None, 'free_cash_flow': None, 'currency': 'USD'},
            'debt_equity': {'total_debt': None, 'equity': None, 'debt_to_equity_ratio': None},
            'other_metrics': {'ebitda': None, 'margin_percentage': None, 'growth_rate': None}
        }
        
        text_lower = text.lower()
        
        # Extract revenue
        for pattern in self.financial_patterns['revenue']['patterns']:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                value = self._parse_amount(match.group(1), match.group(2) if len(match.groups()) > 1 else None)
                if value and financial_facts['revenue']['current_year'] is None:
                    financial_facts['revenue']['current_year'] = value
                    logger.info(f"Found revenue: ${value:,.0f}")
                    break
        
        # Extract profit/loss metrics
        profit_patterns = self.financial_patterns['profit']['patterns']
        for i, pattern in enumerate(profit_patterns):
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                value = self._parse_amount(match.group(1), match.group(2) if len(match.groups()) > 1 else None)
                if value:
                    if i < 2 and financial_facts['profit_loss']['net_income'] is None:  # Net income patterns
                        financial_facts['profit_loss']['net_income'] = value
                        logger.info(f"Found net income: ${value:,.0f}")
                    elif i == 2 and financial_facts['profit_loss']['gross_profit'] is None:  # Gross profit
                        financial_facts['profit_loss']['gross_profit'] = value
                        logger.info(f"Found gross profit: ${value:,.0f}")
                    elif i == 3 and financial_facts['profit_loss']['operating_profit'] is None:  # Operating income
                        financial_facts['profit_loss']['operating_profit'] = value
                        logger.info(f"Found operating profit: ${value:,.0f}")
                    elif i == 4 and financial_facts['other_metrics']['ebitda'] is None:  # EBITDA
                        financial_facts['other_metrics']['ebitda'] = value
                        logger.info(f"Found EBITDA: ${value:,.0f}")
        
        # Extract cash flow
        for i, pattern in enumerate(self.financial_patterns['cash_flow']['patterns']):
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                value = self._parse_amount(match.group(1), match.group(2) if len(match.groups()) > 1 else None)
                if value:
                    if i < 2 and financial_facts['cash_flow']['operating_cash_flow'] is None:
                        financial_facts['cash_flow']['operating_cash_flow'] = value
                        logger.info(f"Found operating cash flow: ${value:,.0f}")
                    elif i == 2 and financial_facts['cash_flow']['free_cash_flow'] is None:
                        financial_facts['cash_flow']['free_cash_flow'] = value
                        logger.info(f"Found free cash flow: ${value:,.0f}")
        
        # Extract balance sheet items
        balance_patterns = self.financial_patterns['balance_sheet']['patterns']
        for i, pattern in enumerate(balance_patterns):
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                value = self._parse_amount(match.group(1), match.group(2) if len(match.groups()) > 1 else None)
                if value:
                    if i == 2 and financial_facts['debt_equity']['total_debt'] is None:  # Total debt
                        financial_facts['debt_equity']['total_debt'] = value
                        logger.info(f"Found total debt: ${value:,.0f}")
                    elif i == 3 and financial_facts['debt_equity']['equity'] is None:  # Shareholders equity
                        financial_facts['debt_equity']['equity'] = value
                        logger.info(f"Found equity: ${value:,.0f}")
        
        # Extract ratios and percentages
        for pattern in self.financial_patterns['ratios']['patterns']:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match.group(1))
                    context = text_lower[max(0, match.start()-50):match.end()+50]
                    
                    if 'margin' in context and financial_facts['other_metrics']['margin_percentage'] is None:
                        financial_facts['other_metrics']['margin_percentage'] = value
                        logger.info(f"Found margin: {value}%")
                    elif 'growth' in context and financial_facts['other_metrics']['growth_rate'] is None:
                        financial_facts['other_metrics']['growth_rate'] = value
                        logger.info(f"Found growth rate: {value}%")
                    elif 'debt' in context and 'equity' in context and financial_facts['debt_equity']['debt_to_equity_ratio'] is None:
                        financial_facts['debt_equity']['debt_to_equity_ratio'] = value
                        logger.info(f"Found debt-to-equity ratio: {value}")
                except (ValueError, IndexError):
                    continue
        
        # Calculate debt-to-equity ratio if not found but we have the components
        if (financial_facts['debt_equity']['debt_to_equity_ratio'] is None and
            financial_facts['debt_equity']['total_debt'] is not None and
            financial_facts['debt_equity']['equity'] is not None and
            financial_facts['debt_equity']['equity'] > 0):
            ratio = financial_facts['debt_equity']['total_debt'] / financial_facts['debt_equity']['equity']
            financial_facts['debt_equity']['debt_to_equity_ratio'] = round(ratio, 2)
            logger.info(f"Calculated debt-to-equity ratio: {ratio:.2f}")
        
        return financial_facts
    
    def _extract_investment_data_regex(self, text: str) -> Dict[str, Any]:
        """Extract investment data using regex patterns."""
        investment_data = {
            'investment_highlights': [],
            'risk_factors': [],
            'market_opportunity': {'market_size': None, 'growth_rate': None, 'competitive_position': None},
            'business_model': {'type': None, 'revenue_streams': [], 'key_customers': []},
            'strategic_initiatives': [],
            'exit_strategy': {'timeline': None, 'target_multiple': None, 'potential_buyers': []}
        }
        
        text_lower = text.lower()
        
        # Extract risk factors
        for pattern in self.investment_patterns['risk_factors']['patterns']:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE | re.DOTALL)
            for match in matches:
                content = match.group(1).strip()
                # Split into individual risks
                risks = re.split(r'[;•·\n]+', content)
                for risk in risks[:5]:  # Limit to 5 risks per match
                    risk = risk.strip()
                    if len(risk) > 20 and risk not in investment_data['risk_factors']:
                        investment_data['risk_factors'].append(risk)
                        logger.info(f"Found risk factor: {risk[:50]}...")
        
        # Extract investment highlights
        for pattern in self.investment_patterns['investment_highlights']['patterns']:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE | re.DOTALL)
            for match in matches:
                content = match.group(1).strip()
                highlights = re.split(r'[;•·\n]+', content)
                for highlight in highlights[:5]:
                    highlight = highlight.strip()
                    if len(highlight) > 20 and highlight not in investment_data['investment_highlights']:
                        investment_data['investment_highlights'].append(highlight)
                        logger.info(f"Found investment highlight: {highlight[:50]}...")
        
        # Extract business model
        for pattern in self.investment_patterns['business_model']['patterns']:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE | re.DOTALL)
            for match in matches:
                content = match.group(1).strip()
                if 'revenue' in pattern and 'stream' in pattern:
                    # Extract revenue streams
                    streams = re.split(r'[;,•·\n]+', content)
                    for stream in streams[:5]:
                        stream = stream.strip()
                        if len(stream) > 10 and stream not in investment_data['business_model']['revenue_streams']:
                            investment_data['business_model']['revenue_streams'].append(stream)
                            logger.info(f"Found revenue stream: {stream}")
                elif investment_data['business_model']['type'] is None:
                    # Extract business model type
                    investment_data['business_model']['type'] = content[:200]
                    logger.info(f"Found business model: {content[:50]}...")
        
        # Extract market opportunity
        for pattern in self.investment_patterns['market_opportunity']['patterns']:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if 'size' in pattern or 'market' in pattern:
                    value = self._parse_amount(match.group(1), match.group(2) if len(match.groups()) > 1 else None)
                    if value and investment_data['market_opportunity']['market_size'] is None:
                        investment_data['market_opportunity']['market_size'] = value
                        logger.info(f"Found market size: ${value:,.0f}")
                elif 'grow' in pattern:
                    try:
                        growth_rate = float(match.group(1))
                        if investment_data['market_opportunity']['growth_rate'] is None:
                            investment_data['market_opportunity']['growth_rate'] = growth_rate
                            logger.info(f"Found market growth rate: {growth_rate}%")
                    except (ValueError, IndexError):
                        continue
        
        # Extract strategic initiatives from bullet points or numbered lists
        strategic_patterns = [
            r'strategic\s+(?:initiatives?|priorities?|goals?)\s*(?:include|are|:)\s*([^.]+(?:\.[^.]+){0,3})',
            r'(?:our|the\s+company\'s)\s+(?:strategy|initiatives?)\s*(?:include|are|:)\s*([^.]+(?:\.[^.]+){0,3})',
        ]
        
        for pattern in strategic_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE | re.DOTALL)
            for match in matches:
                content = match.group(1).strip()
                initiatives = re.split(r'[;•·\n]+', content)
                for initiative in initiatives[:5]:
                    initiative = initiative.strip()
                    if len(initiative) > 20 and initiative not in investment_data['strategic_initiatives']:
                        investment_data['strategic_initiatives'].append(initiative)
                        logger.info(f"Found strategic initiative: {initiative[:50]}...")
        
        return investment_data
    
    def _extract_key_metrics(self, text: str) -> Dict[str, Any]:
        """Extract various key metrics and statistics from the document."""
        metrics = {
            'financial_metrics_count': 0,
            'percentages_found': [],
            'currency_mentions': {},
            'time_periods_mentioned': [],
            'numerical_values_count': 0
        }
        
        text_lower = text.lower()
        
        # Count financial metric mentions
        financial_keywords = ['revenue', 'profit', 'income', 'cash flow', 'debt', 'equity', 'assets', 'liabilities']
        for keyword in financial_keywords:
            count = text_lower.count(keyword)
            metrics['financial_metrics_count'] += count
        
        # Extract all percentages
        percentage_pattern = r'([\d]+(?:\.\d+)?)\s*%'
        percentages = re.findall(percentage_pattern, text)
        metrics['percentages_found'] = [float(p) for p in percentages[:20]]  # Limit to 20
        
        # Detect currencies mentioned
        currency_patterns = {
            'USD': r'\$|USD|US\s+Dollar',
            'EUR': r'€|EUR|Euro',
            'GBP': r'£|GBP|British\s+Pound',
            'JPY': r'¥|JPY|Yen',
            'CAD': r'CAD|Canadian\s+Dollar'
        }
        
        for currency, pattern in currency_patterns.items():
            count = len(re.findall(pattern, text, re.IGNORECASE))
            if count > 0:
                metrics['currency_mentions'][currency] = count
        
        # Extract time periods
        time_patterns = [
            r'(?:fiscal\s+year|fy)\s*(\d{4})',
            r'(?:year\s+ended|year\s+ending)\s*(?:december|dec|june|jun|march|mar|september|sep)?\s*\d+,?\s*(\d{4})',
            r'(\d{4})\s+(?:annual|yearly)',
            r'q[1-4]\s*(\d{4})',
            r'(?:quarter|quarterly)\s*.*?(\d{4})'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if match not in metrics['time_periods_mentioned']:
                    metrics['time_periods_mentioned'].append(match)
        
        # Count numerical values (rough estimate of data richness)
        number_pattern = r'\d+(?:,\d{3})*(?:\.\d+)?'
        metrics['numerical_values_count'] = len(re.findall(number_pattern, text))
        
        return metrics
    
    def _extract_document_structure(self, text: str) -> Dict[str, Any]:
        """Extract document structure information."""
        structure = {
            'sections_count': 0,
            'tables_detected': 0,
            'lists_detected': 0,
            'headers_found': [],
            'document_length': len(text),
            'word_count': len(text.split()),
            'average_sentence_length': 0
        }
        
        # Detect section headers
        header_patterns = [
            r'^([A-Z][A-Z\s]{3,50})$',  # ALL CAPS headers
            r'^(\d+\.?\s+[A-Z][A-Za-z\s]{3,50})$',  # Numbered sections
            r'^([IVX]+\.?\s+[A-Z][A-Za-z\s]{3,50})$',  # Roman numeral sections
        ]
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            for pattern in header_patterns:
                match = re.match(pattern, line)
                if match:
                    header = match.group(1).strip()
                    if len(header) > 3:
                        structure['headers_found'].append(header)
                        structure['sections_count'] += 1
        
        # Detect tables (look for aligned columns, pipes, or tab-separated values)
        table_indicators = [
            r'\|.*\|.*\|',  # Pipe-separated tables
            r'\t.*\t.*\t',  # Tab-separated tables
            r'^\s*\d+\s+.*\s+\d+\s*$',  # Aligned numeric columns
        ]
        
        for pattern in table_indicators:
            matches = re.findall(pattern, text, re.MULTILINE)
            structure['tables_detected'] += len(matches) // 3  # Rough estimate
        
        # Detect lists
        list_patterns = [
            r'^\s*[•·\-\*]\s+',  # Bullet points
            r'^\s*\d+\.\s+',  # Numbered lists
            r'^\s*\([a-z]\)\s+',  # Letter lists
        ]
        
        for pattern in list_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            structure['lists_detected'] += len(matches)
        
        # Calculate average sentence length
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s for s in sentences if len(s.split()) > 3]
        if valid_sentences:
            total_words = sum(len(s.split()) for s in valid_sentences)
            structure['average_sentence_length'] = round(total_words / len(valid_sentences), 1)
        
        # Limit headers found to top 20
        structure['headers_found'] = structure['headers_found'][:20]
        
        return structure


# Create a global instance
vector_regex_extractor = VectorRegexMetadataExtractor()