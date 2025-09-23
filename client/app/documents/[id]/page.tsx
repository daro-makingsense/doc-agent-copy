'use client';

import { useParams, useRouter } from 'next/navigation';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient, DocumentStatus } from '@/lib/api-client';
import { formatBytes, formatDate } from '@/lib/utils';
import { FileText, Loader2, AlertCircle, CheckCircle, Clock, Brain, MessageSquare, ArrowLeft } from 'lucide-react';
import Link from 'next/link';
import { useEffect } from 'react';

export default function DocumentDetailsPage() {
  const params = useParams();
  const router = useRouter();
  const documentId = params.id as string;

  const { data: document, isLoading: docLoading, error: docError } = useQuery({
    queryKey: ['document', documentId],
    queryFn: () => apiClient.getDocument(documentId),
  });

  const { data: status, isLoading: statusLoading } = useQuery({
    queryKey: ['document-status', documentId],
    queryFn: () => apiClient.getDocumentStatus(documentId),
    refetchInterval: (data) => {
      // Stop polling if processing is complete
      if (data?.is_processed && data?.is_embedded) {
        return false;
      }
      // Poll every 2 seconds while processing
      return 2000;
    },
  });

  // Refetch document details when processing is complete
  const queryClient = useQueryClient();
  useEffect(() => {
    if (status?.is_processed && status?.is_embedded) {
      // Refetch document to get updated financial facts, etc.
      queryClient.invalidateQueries({ queryKey: ['document', documentId] });
    }
  }, [status?.is_processed, status?.is_embedded, documentId, queryClient]);

  if (docLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    );
  }

  if (docError || !document) {
    return (
      <div className="max-w-2xl mx-auto">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-red-600" />
            <p className="text-red-800">Failed to load document details.</p>
          </div>
        </div>
      </div>
    );
  }

  const isProcessing = !status?.is_processed || !status?.is_embedded;

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-6">
        <Link
          href="/"
          className="inline-flex items-center text-gray-900 hover:text-gray-700 mb-4"
        >
          <ArrowLeft className="h-4 w-4 mr-1" />
          Back to Documents
        </Link>
        
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              {document.original_filename || document.filename}
            </h1>
            <p className="text-gray-700">Document ID: {document.id}</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <Link
              href={`/research?documentId=${document.id}`}
              className={`inline-flex items-center px-4 py-2 rounded-lg transition-colors ${
                isProcessing
                  ? 'bg-gray-200 text-gray-700 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
              aria-disabled={isProcessing}
              onClick={(e) => isProcessing && e.preventDefault()}
            >
              <Brain className="h-4 w-4 mr-2" />
              Research
            </Link>
            <Link
              href={`/chat?documentId=${document.id}`}
              className={`inline-flex items-center px-4 py-2 rounded-lg transition-colors ${
                isProcessing
                  ? 'bg-gray-200 text-gray-700 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
              aria-disabled={isProcessing}
              onClick={(e) => isProcessing && e.preventDefault()}
            >
              <MessageSquare className="h-4 w-4 mr-2" />
              Chat
            </Link>
          </div>
        </div>
      </div>

      {/* Processing Status */}
      {isProcessing && status && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
              <div>
                <h3 className="font-semibold text-blue-900">Processing Document</h3>
                <p className="text-sm text-blue-700">
                  This may take a few minutes depending on the document size...
                </p>
              </div>
            </div>
            <span className="text-2xl font-bold text-blue-900">
              {status.progress_percentage}%
            </span>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              {status.is_processed ? (
                <CheckCircle className="h-5 w-5 text-green-600" />
              ) : (
                <Clock className="h-5 w-5 text-gray-700" />
              )}
              <span className={status.is_processed ? 'text-green-700' : 'text-gray-800'}>
                Document parsing and extraction
              </span>
            </div>
            <div className="flex items-center space-x-2">
              {status.is_embedded ? (
                <CheckCircle className="h-5 w-5 text-green-600" />
              ) : (
                <Clock className="h-5 w-5 text-gray-700" />
              )}
              <span className={status.is_embedded ? 'text-green-700' : 'text-gray-800'}>
                Creating vector embeddings ({status.embedding_count} chunks)
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Document Information */}
      <div className="grid gap-6">
        {/* Basic Information */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Document Information</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="text-gray-700">File Size:</span>
              <p className="font-medium">{formatBytes(document.file_size)}</p>
            </div>
            <div>
              <span className="text-gray-700">Pages:</span>
              <p className="font-medium">{document.page_count || 'N/A'}</p>
            </div>
            <div>
              <span className="text-gray-700">Words:</span>
              <p className="font-medium">
                {document.word_count ? document.word_count.toLocaleString() : 'N/A'}
              </p>
            </div>
            <div>
              <span className="text-gray-700">Embeddings:</span>
              <p className="font-medium">{document.embedding_count || 'N/A'}</p>
            </div>
            <div>
              <span className="text-gray-700">Created:</span>
              <p className="font-medium">{formatDate(document.created_at)}</p>
            </div>
            <div>
              <span className="text-gray-700">Updated:</span>
              <p className="font-medium">{document.updated_at ? formatDate(document.updated_at) : 'N/A'}</p>
            </div>
          </div>
        </div>

        {/* Financial Facts */}
        {document.financial_facts && Object.keys(document.financial_facts).length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Financial Facts</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {Object.entries(document.financial_facts).map(([key, value]) => (
                <div key={key}>
                  <span className="text-gray-700 text-sm">
                    {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
                  </span>
                  <p className="font-medium">{String(value)}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Investment Data */}
        {document.investment_data && Object.keys(document.investment_data).length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Investment Data</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {Object.entries(document.investment_data).map(([key, value]) => (
                <div key={key}>
                  <span className="text-gray-700 text-sm">
                    {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
                  </span>
                  <p className="font-medium">{String(value)}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Key Metrics */}
        {document.key_metrics && Object.keys(document.key_metrics).length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Key Metrics</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {Object.entries(document.key_metrics).map(([key, value]) => (
                <div key={key}>
                  <span className="text-gray-700 text-sm">
                    {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
                  </span>
                  <p className="font-medium">
                    {typeof value === 'number' ? value.toLocaleString() : String(value)}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Processing Error */}
        {document.processing_error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-start space-x-2">
              <AlertCircle className="h-5 w-5 text-red-600 mt-0.5" />
              <div>
                <p className="font-medium text-red-800">Processing Error</p>
                <p className="text-sm text-red-700 mt-1">{document.processing_error}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}