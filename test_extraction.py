"""
Test script for comparing old vs new metadata extraction approaches.
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.metadata_extractor import metadata_extractor
from app.services.metadata_extractor_v2 import vector_regex_extractor
from app.database.connection import get_db_session
from app.database.models import Document
import json
from datetime import datetime


async def test_extraction(document_id: str):
    """Test both extraction methods on a document."""
    print(f"\n{'='*60}")
    print(f"Testing extraction for document: {document_id}")
    print(f"{'='*60}\n")
    
    # Get document info
    with get_db_session() as db:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            print(f"Document {document_id} not found!")
            return
        
        print(f"Document: {document.original_filename}")
        print(f"Processed: {document.is_processed}")
        print(f"Embedded: {document.is_embedded}")
        print(f"Chunks: {document.embedding_count}")
    
    if not document.is_processed or not document.is_embedded:
        print("Document must be fully processed and embedded before testing extraction!")
        return
    
    # Test new vector+regex extraction
    print(f"\n{'-'*60}")
    print("Testing NEW Vector+Regex Extraction")
    print(f"{'-'*60}")
    
    try:
        start_time = datetime.now()
        vector_results = await vector_regex_extractor.extract_metadata(document_id)
        vector_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\nExtraction completed in {vector_time:.2f} seconds")
        print("\nFinancial Facts Found:")
        financial_facts = vector_results.get('financial_facts', {})
        print(f"  Revenue (current): ${financial_facts.get('revenue', {}).get('current_year', 'N/A')}")
        print(f"  Net Income: ${financial_facts.get('profit_loss', {}).get('net_income', 'N/A')}")
        print(f"  Operating Cash Flow: ${financial_facts.get('cash_flow', {}).get('operating_cash_flow', 'N/A')}")
        print(f"  Total Debt: ${financial_facts.get('debt_equity', {}).get('total_debt', 'N/A')}")
        print(f"  EBITDA: ${financial_facts.get('other_metrics', {}).get('ebitda', 'N/A')}")
        
        print("\nInvestment Data Found:")
        investment_data = vector_results.get('investment_data', {})
        print(f"  Investment Highlights: {len(investment_data.get('investment_highlights', []))}")
        print(f"  Risk Factors: {len(investment_data.get('risk_factors', []))}")
        print(f"  Market Size: ${investment_data.get('market_opportunity', {}).get('market_size', 'N/A')}")
        print(f"  Revenue Streams: {len(investment_data.get('business_model', {}).get('revenue_streams', []))}")
        
        print("\nKey Metrics:")
        key_metrics = vector_results.get('key_metrics', {})
        print(f"  Financial Metrics Count: {key_metrics.get('financial_metrics_count', 0)}")
        print(f"  Percentages Found: {len(key_metrics.get('percentages_found', []))}")
        print(f"  Numerical Values: {key_metrics.get('numerical_values_count', 0)}")
        
        # Save results
        with open(f'/workspace/logs/vector_extraction_{document_id}.json', 'w') as f:
            json.dump(vector_results, f, indent=2, default=str)
        
    except Exception as e:
        print(f"ERROR in vector extraction: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test old LLM-based extraction (optional)
    print(f"\n{'-'*60}")
    print("Testing OLD LLM-based Extraction (Optional)")
    print(f"{'-'*60}")
    
    try:
        # Ask user if they want to run the expensive LLM extraction
        response = input("\nDo you want to run the LLM-based extraction for comparison? (y/n): ")
        if response.lower() == 'y':
            start_time = datetime.now()
            llm_results = await metadata_extractor.extract_metadata(document_id)
            llm_time = (datetime.now() - start_time).total_seconds()
            
            print(f"\nLLM Extraction completed in {llm_time:.2f} seconds")
            print(f"Time difference: {llm_time - vector_time:.2f} seconds")
            
            # Save results
            with open(f'/workspace/logs/llm_extraction_{document_id}.json', 'w') as f:
                json.dump(llm_results, f, indent=2, default=str)
                
            print("\nResults saved to /workspace/logs/")
            
    except Exception as e:
        print(f"ERROR in LLM extraction: {str(e)}")
        import traceback
        traceback.print_exc()


async def list_documents():
    """List available documents for testing."""
    with get_db_session() as db:
        documents = db.query(Document).filter(
            Document.is_processed == True,
            Document.is_embedded == True
        ).all()
        
        if not documents:
            print("No processed and embedded documents found!")
            return None
        
        print("\nAvailable documents for testing:")
        print(f"{'Index':<6} {'ID':<40} {'Filename':<50} {'Chunks':<10}")
        print("-" * 110)
        
        for i, doc in enumerate(documents):
            print(f"{i:<6} {str(doc.id):<40} {doc.original_filename:<50} {doc.embedding_count:<10}")
        
        return documents


async def main():
    """Main test function."""
    print("Metadata Extraction Test Utility")
    print("================================\n")
    
    documents = await list_documents()
    if not documents:
        return
    
    print("\nEnter document index to test (or 'q' to quit): ", end="")
    choice = input()
    
    if choice.lower() == 'q':
        return
    
    try:
        index = int(choice)
        if 0 <= index < len(documents):
            document_id = str(documents[index].id)
            await test_extraction(document_id)
        else:
            print("Invalid index!")
    except ValueError:
        print("Invalid input!")


if __name__ == "__main__":
    asyncio.run(main())