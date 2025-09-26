"""
Enhanced logging configuration for metadata extraction monitoring.
"""
import logging
import sys
from datetime import datetime
from typing import Dict, Any
import json


class ExtractionLogFormatter(logging.Formatter):
    """Custom formatter for extraction logs with structured data."""
    
    def format(self, record):
        # Base log format
        log_format = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extraction-specific data if available
        if hasattr(record, 'extraction_data'):
            log_format['extraction_data'] = record.extraction_data
        
        if hasattr(record, 'document_id'):
            log_format['document_id'] = record.document_id
            
        if hasattr(record, 'extraction_method'):
            log_format['extraction_method'] = record.extraction_method
            
        if hasattr(record, 'metrics'):
            log_format['metrics'] = record.metrics
        
        # Format as JSON for structured logging
        if getattr(record, 'json_format', False):
            return json.dumps(log_format)
        
        # Human-readable format
        return f"[{log_format['timestamp']}] {log_format['level']} - {log_format['logger']} - {log_format['message']}"


class ExtractionMetricsLogger:
    """Logger for tracking extraction metrics and performance."""
    
    def __init__(self, logger_name: str = "extraction_metrics"):
        self.logger = logging.getLogger(logger_name)
        self.metrics = {}
    
    def log_extraction_start(self, document_id: str, method: str):
        """Log the start of extraction process."""
        self.metrics[document_id] = {
            'start_time': datetime.utcnow(),
            'method': method,
            'steps': {}
        }
        
        extra = {
            'document_id': document_id,
            'extraction_method': method,
            'extraction_data': {'event': 'extraction_start'}
        }
        self.logger.info(f"Starting extraction for document {document_id} using {method}", extra=extra)
    
    def log_extraction_step(self, document_id: str, step: str, data: Dict[str, Any]):
        """Log a specific step in the extraction process."""
        if document_id in self.metrics:
            self.metrics[document_id]['steps'][step] = {
                'timestamp': datetime.utcnow(),
                'data': data
            }
        
        extra = {
            'document_id': document_id,
            'extraction_data': {
                'event': 'extraction_step',
                'step': step,
                'data': data
            }
        }
        self.logger.info(f"Extraction step '{step}' for document {document_id}", extra=extra)
    
    def log_extraction_result(self, document_id: str, result_type: str, count: int, details: Any = None):
        """Log extraction results."""
        extra = {
            'document_id': document_id,
            'extraction_data': {
                'event': 'extraction_result',
                'result_type': result_type,
                'count': count,
                'details': details
            }
        }
        self.logger.info(f"Found {count} {result_type} for document {document_id}", extra=extra)
    
    def log_extraction_complete(self, document_id: str, success: bool, summary: Dict[str, Any]):
        """Log completion of extraction process."""
        if document_id in self.metrics:
            start_time = self.metrics[document_id]['start_time']
            duration = (datetime.utcnow() - start_time).total_seconds()
            summary['duration_seconds'] = duration
        
        extra = {
            'document_id': document_id,
            'extraction_data': {
                'event': 'extraction_complete',
                'success': success,
                'summary': summary
            },
            'metrics': self.metrics.get(document_id, {})
        }
        
        level = logging.INFO if success else logging.ERROR
        self.logger.log(level, f"Extraction {'completed' if success else 'failed'} for document {document_id}", extra=extra)
        
        # Clean up metrics for this document
        if document_id in self.metrics:
            del self.metrics[document_id]


def setup_extraction_logging():
    """Set up enhanced logging for extraction processes."""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ExtractionLogFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler for extraction logs
    file_handler = logging.FileHandler('/workspace/logs/extraction.log', mode='a')
    file_handler.setFormatter(ExtractionLogFormatter())
    file_handler.setLevel(logging.INFO)
    
    # Add file handler to extraction-related loggers
    extraction_loggers = [
        'app.services.metadata_extractor',
        'app.services.metadata_extractor_v2',
        'extraction_metrics'
    ]
    
    for logger_name in extraction_loggers:
        logger = logging.getLogger(logger_name)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    return ExtractionMetricsLogger()


# Global metrics logger instance
extraction_metrics = setup_extraction_logging()