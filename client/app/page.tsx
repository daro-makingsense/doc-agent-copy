'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient, Document } from '@/lib/api-client';
import { formatBytes, formatDate, formatRelativeTime } from '@/lib/utils';
import { FileText, Trash2, Eye, Brain, MessageSquare, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import Link from 'next/link';
import { useState } from 'react';
import { useToast } from '@/hooks/use-toast';

export default function DocumentsPage() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const { data: documents, isLoading, error } = useQuery({
    queryKey: ['documents'],
    queryFn: () => apiClient.listDocuments(),
  });

  const deleteMutation = useMutation({
    mutationFn: (documentId: string) => apiClient.deleteDocument(documentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      toast({
        title: 'Document deleted',
        description: 'The document has been successfully deleted.',
      });
    },
    onError: (error) => {
      toast({
        title: 'Error',
        description: 'Failed to delete document. Please try again.',
        variant: 'destructive',
      });
    },
    onSettled: () => {
      setDeletingId(null);
    },
  });

  const handleDelete = (documentId: string) => {
    if (confirm('Are you sure you want to delete this document?')) {
      setDeletingId(documentId);
      deleteMutation.mutate(documentId);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex items-center space-x-2">
          <AlertCircle className="h-5 w-5 text-red-600" />
          <p className="text-red-800">Failed to load documents. Please try again.</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Documents</h1>
        <p className="text-gray-600">Manage your uploaded financial documents</p>
      </div>

      {documents?.length === 0 ? (
        <div className="bg-white rounded-lg shadow-sm border p-12 text-center">
          <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No documents yet</h3>
          <p className="text-gray-600 mb-6">Upload your first document to get started</p>
          <Link
            href="/upload"
            className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Upload Document
          </Link>
        </div>
      ) : (
        <div className="grid gap-4">
          {documents?.map((doc) => (
            <div
              key={doc.id}
              className="bg-white rounded-lg shadow-sm border p-6 hover:shadow-md transition-shadow"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-2">
                    <FileText className="h-5 w-5 text-blue-600" />
                    <h3 className="text-lg font-semibold text-gray-900">
                      {doc.original_filename || doc.filename}
                    </h3>
                    {doc.is_processed && doc.is_embedded && (
                      <CheckCircle className="h-5 w-5 text-green-600" />
                    )}
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4 text-sm">
                    <div>
                      <span className="text-gray-500">Size:</span>{' '}
                      <span className="font-medium">{formatBytes(doc.file_size)}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Pages:</span>{' '}
                      <span className="font-medium">{doc.page_count || 'N/A'}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Words:</span>{' '}
                      <span className="font-medium">
                        {doc.word_count ? doc.word_count.toLocaleString() : 'N/A'}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">Uploaded:</span>{' '}
                      <span className="font-medium">{formatRelativeTime(doc.created_at)}</span>
                    </div>
                  </div>

                  <div className="flex items-center space-x-2 text-sm">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      doc.is_processed 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-yellow-100 text-yellow-800'
                    }`}>
                      {doc.is_processed ? 'Processed' : 'Processing'}
                    </span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      doc.is_embedded 
                        ? 'bg-blue-100 text-blue-800' 
                        : 'bg-gray-100 text-gray-800'
                    }`}>
                      {doc.is_embedded ? 'Embedded' : 'Not Embedded'}
                    </span>
                    {doc.embedding_count && (
                      <span className="text-gray-500">
                        {doc.embedding_count} embeddings
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <Link
                    href={`/documents/${doc.id}`}
                    className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
                    title="View details"
                  >
                    <Eye className="h-5 w-5" />
                  </Link>
                  <Link
                    href={`/research?documentId=${doc.id}`}
                    className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
                    title="Research"
                  >
                    <Brain className="h-5 w-5" />
                  </Link>
                  <Link
                    href={`/chat?documentId=${doc.id}`}
                    className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
                    title="Chat"
                  >
                    <MessageSquare className="h-5 w-5" />
                  </Link>
                  <button
                    onClick={() => handleDelete(doc.id)}
                    disabled={deletingId === doc.id}
                    className="p-2 text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50"
                    title="Delete"
                  >
                    {deletingId === doc.id ? (
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : (
                      <Trash2 className="h-5 w-5" />
                    )}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}