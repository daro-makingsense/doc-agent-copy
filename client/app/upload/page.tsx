'use client';

import { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useMutation } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { formatBytes } from '@/lib/utils';
import { Upload, FileText, X, Loader2, CheckCircle } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

export default function UploadPage() {
  const router = useRouter();
  const { toast } = useToast();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const uploadMutation = useMutation({
    mutationFn: (file: File) => apiClient.uploadDocument(file),
    onSuccess: (data) => {
      toast({
        title: 'Upload successful',
        description: `${selectedFile?.name} has been uploaded and processing has started.`,
      });
      // Redirect to document details or documents list
      setTimeout(() => {
        router.push(`/documents/${data.document_id}`);
      }, 1500);
    },
    onError: (error: any) => {
      toast({
        title: 'Upload failed',
        description: error.message || 'Failed to upload document. Please try again.',
        variant: 'destructive',
      });
    },
  });

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileSelect = (file: File) => {
    if (file.type !== 'application/pdf') {
      toast({
        title: 'Invalid file type',
        description: 'Please select a PDF file.',
        variant: 'destructive',
      });
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      toast({
        title: 'File too large',
        description: `Please select a file smaller than ${formatBytes(MAX_FILE_SIZE)}.`,
        variant: 'destructive',
      });
      return;
    }

    setSelectedFile(file);
  };

  const handleUpload = () => {
    if (selectedFile) {
      uploadMutation.mutate(selectedFile);
    }
  };

  const isUploading = uploadMutation.isPending;
  const isSuccess = uploadMutation.isSuccess;

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Upload Document</h1>
        <p className="text-gray-600">Upload a financial document for AI-powered analysis</p>
      </div>

      <div className="bg-white rounded-lg shadow-sm border p-8">
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            dragActive 
              ? 'border-blue-500 bg-blue-50' 
              : 'border-gray-300 hover:border-gray-400'
          } ${isUploading || isSuccess ? 'pointer-events-none opacity-60' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {!selectedFile ? (
            <>
              <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-700 mb-2">
                Drag and drop your PDF file here, or click to browse
              </p>
              <p className="text-sm text-gray-500 mb-4">
                Maximum file size: {formatBytes(MAX_FILE_SIZE)}
              </p>
              <input
                type="file"
                accept=".pdf"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFileSelect(e.target.files[0]);
                  }
                }}
                className="hidden"
                id="file-upload"
                disabled={isUploading}
              />
              <label
                htmlFor="file-upload"
                className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors cursor-pointer"
              >
                Select File
              </label>
            </>
          ) : (
            <div className="space-y-4">
              {isSuccess ? (
                <CheckCircle className="h-12 w-12 text-green-600 mx-auto mb-4" />
              ) : (
                <FileText className="h-12 w-12 text-blue-600 mx-auto mb-4" />
              )}
              
              <div className="bg-gray-50 rounded-lg p-4 flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <FileText className="h-8 w-8 text-gray-600" />
                  <div className="text-left">
                    <p className="font-medium text-gray-900">{selectedFile.name}</p>
                    <p className="text-sm text-gray-500">{formatBytes(selectedFile.size)}</p>
                  </div>
                </div>
                {!isUploading && !isSuccess && (
                  <button
                    onClick={() => setSelectedFile(null)}
                    className="p-1 hover:bg-gray-200 rounded transition-colors"
                  >
                    <X className="h-5 w-5 text-gray-600" />
                  </button>
                )}
              </div>

              {isSuccess ? (
                <p className="text-green-600 font-medium">
                  Upload successful! Redirecting...
                </p>
              ) : (
                <button
                  onClick={handleUpload}
                  disabled={isUploading}
                  className="w-full py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                >
                  {isUploading ? (
                    <>
                      <Loader2 className="h-5 w-5 animate-spin" />
                      <span>Uploading...</span>
                    </>
                  ) : (
                    <>
                      <Upload className="h-5 w-5" />
                      <span>Upload Document</span>
                    </>
                  )}
                </button>
              )}
            </div>
          )}
        </div>

        <div className="mt-6 space-y-2">
          <h3 className="font-medium text-gray-900">Supported Features:</h3>
          <ul className="text-sm text-gray-600 space-y-1">
            <li>• PDF document parsing with advanced extraction</li>
            <li>• Financial facts and metrics extraction</li>
            <li>• Investment data analysis</li>
            <li>• Semantic search with vector embeddings</li>
            <li>• AI-powered research and chat capabilities</li>
          </ul>
        </div>
      </div>
    </div>
  );
}