'use client';

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import { useQuery, useMutation } from '@tanstack/react-query';
import { apiClient, Document, ResearchTask } from '@/lib/api-client';
import { formatRelativeTime } from '@/lib/utils';
import { Brain, FileText, Loader2, AlertCircle, CheckCircle, Clock, ChevronRight, ChevronDown } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

export default function ResearchPage() {
  const searchParams = useSearchParams();
  const { toast } = useToast();
  const preselectedDocumentId = searchParams.get('documentId');
  
  const [selectedDocument, setSelectedDocument] = useState<string>(preselectedDocumentId || '');
  const [topic, setTopic] = useState('');
  const [customQuery, setCustomQuery] = useState('');
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

  const { data: documents } = useQuery({
    queryKey: ['documents'],
    queryFn: () => apiClient.listDocuments(),
  });

  const { data: researchTasks, refetch: refetchTasks } = useQuery({
    queryKey: ['research-tasks', selectedDocument],
    queryFn: () => selectedDocument ? apiClient.listResearchTasks(selectedDocument) : null,
    enabled: !!selectedDocument,
  });

  const startResearchMutation = useMutation({
    mutationFn: ({ documentId, topic, customQuery }: { documentId: string; topic: string; customQuery?: string }) =>
      apiClient.startResearch(documentId, topic, customQuery),
    onSuccess: (data) => {
      toast({
        title: 'Research started',
        description: `Research task "${data.topic}" has been initiated.`,
      });
      refetchTasks();
      // Clear form
      setTopic('');
      setCustomQuery('');
    },
    onError: (error: any) => {
      toast({
        title: 'Failed to start research',
        description: error.message || 'Please try again.',
        variant: 'destructive',
      });
    },
  });

  // Poll for task updates
  useEffect(() => {
    if (!researchTasks || researchTasks.length === 0) return;

    const hasInProgressTasks = researchTasks.some(task => task.status === 'in_progress' || task.status === 'pending');
    if (!hasInProgressTasks) return;

    const interval = setInterval(() => {
      refetchTasks();
    }, 3000); // Poll every 3 seconds

    return () => clearInterval(interval);
  }, [researchTasks, refetchTasks]);

  const handleStartResearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedDocument || !topic.trim()) return;
    
    startResearchMutation.mutate({
      documentId: selectedDocument,
      topic: topic.trim(),
      customQuery: customQuery.trim() || undefined,
    });
  };

  const toggleSection = (sectionId: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(sectionId)) {
      newExpanded.delete(sectionId);
    } else {
      newExpanded.add(sectionId);
    }
    setExpandedSections(newExpanded);
  };

  const processedDocuments = documents?.filter(doc => doc.is_processed && doc.is_embedded) || [];

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Deep Research</h1>
        <p className="text-gray-600">Generate comprehensive research and analysis from your documents</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Research Form */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">New Research Task</h2>
            
            <form onSubmit={handleStartResearch} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Select Document
                </label>
                <select
                  value={selectedDocument}
                  onChange={(e) => setSelectedDocument(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                >
                  <option value="">Choose a document...</option>
                  {processedDocuments.map((doc) => (
                    <option key={doc.id} value={doc.id}>
                      {doc.original_filename || doc.filename}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Research Topic
                </label>
                <input
                  type="text"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  placeholder="e.g., Key Investment Highlights"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Custom Query (Optional)
                </label>
                <textarea
                  value={customQuery}
                  onChange={(e) => setCustomQuery(e.target.value)}
                  placeholder="Add specific questions or areas to focus on..."
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <button
                type="submit"
                disabled={!selectedDocument || !topic.trim() || startResearchMutation.isPending}
                className="w-full py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
              >
                {startResearchMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>Starting Research...</span>
                  </>
                ) : (
                  <>
                    <Brain className="h-4 w-4" />
                    <span>Start Research</span>
                  </>
                )}
              </button>
            </form>

            {processedDocuments.length === 0 && (
              <div className="mt-4 p-4 bg-amber-50 border border-amber-200 rounded-lg">
                <p className="text-sm text-amber-800">
                  No processed documents available. Please upload and wait for documents to be fully processed before starting research.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Research Tasks */}
        <div className="lg:col-span-2">
          {!selectedDocument ? (
            <div className="bg-gray-50 rounded-lg p-12 text-center">
              <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">Select a document to view research tasks</p>
            </div>
          ) : researchTasks && researchTasks.length > 0 ? (
            <div className="space-y-4">
              {researchTasks.map((task) => (
                <div key={task.id} className="bg-white rounded-lg shadow-sm border overflow-hidden">
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900">{task.topic}</h3>
                        {task.custom_query && (
                          <p className="text-sm text-gray-600 mt-1">{task.custom_query}</p>
                        )}
                      </div>
                      <div className="flex items-center space-x-2">
                        {task.status === 'completed' && (
                          <CheckCircle className="h-5 w-5 text-green-600" />
                        )}
                        {task.status === 'in_progress' && (
                          <Loader2 className="h-5 w-5 animate-spin text-blue-600" />
                        )}
                        {task.status === 'pending' && (
                          <Clock className="h-5 w-5 text-gray-400" />
                        )}
                        {task.status === 'failed' && (
                          <AlertCircle className="h-5 w-5 text-red-600" />
                        )}
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          task.status === 'completed' ? 'bg-green-100 text-green-800' :
                          task.status === 'in_progress' ? 'bg-blue-100 text-blue-800' :
                          task.status === 'pending' ? 'bg-gray-100 text-gray-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {task.status.replace('_', ' ')}
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center space-x-4 text-sm text-gray-500">
                      <span>Created {formatRelativeTime(task.created_at)}</span>
                      {task.processing_time && (
                        <span>• {task.processing_time.toFixed(1)}s processing time</span>
                      )}
                    </div>

                    {/* Content Outline */}
                    {task.content_outline && Object.keys(task.content_outline).length > 0 && (
                      <div className="mt-6">
                        <h4 className="font-medium text-gray-900 mb-3">Content Outline</h4>
                        <div className="space-y-2">
                          {Object.entries(task.content_outline).map(([key, section]) => {
                            const sectionId = `${task.id}-${key}`;
                            const isExpanded = expandedSections.has(sectionId);
                            const findings = task.research_findings?.[key];

                            return (
                              <div key={key} className="border rounded-lg">
                                <button
                                  onClick={() => toggleSection(sectionId)}
                                  className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
                                >
                                  <div className="flex items-start text-left">
                                    <span className="font-medium text-gray-700 mr-2">{key}.</span>
                                    <div>
                                      <p className="font-medium text-gray-900">{section.title}</p>
                                      <p className="text-sm text-gray-600">{section.description}</p>
                                    </div>
                                  </div>
                                  {isExpanded ? (
                                    <ChevronDown className="h-5 w-5 text-gray-400 flex-shrink-0" />
                                  ) : (
                                    <ChevronRight className="h-5 w-5 text-gray-400 flex-shrink-0" />
                                  )}
                                </button>

                                {isExpanded && findings && (
                                  <div className="px-4 pb-4 border-t">
                                    <div className="pt-4 prose prose-sm max-w-none">
                                      <p className="text-gray-700">{findings.content}</p>
                                      {findings.key_points && findings.key_points.length > 0 && (
                                        <div className="mt-3">
                                          <p className="font-medium text-gray-900">Key Points:</p>
                                          <ul className="mt-1 space-y-1">
                                            {findings.key_points.map((point, idx) => (
                                              <li key={idx} className="text-gray-700">{point}</li>
                                            ))}
                                          </ul>
                                        </div>
                                      )}
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* Sources */}
                    {task.sources_used && task.sources_used.length > 0 && (
                      <div className="mt-6">
                        <h4 className="font-medium text-gray-900 mb-3">Sources Used</h4>
                        <div className="space-y-2">
                          {task.sources_used.slice(0, 5).map((source, idx) => (
                            <div key={idx} className="bg-gray-50 rounded p-3 text-sm">
                              <div className="flex items-center justify-between mb-1">
                                <span className="font-medium text-gray-700">Page {source.page}</span>
                                <span className="text-xs text-gray-500">
                                  Relevance: {(source.relevance_score * 100).toFixed(0)}%
                                </span>
                              </div>
                              <p className="text-gray-600 line-clamp-2">{source.content}</p>
                            </div>
                          ))}
                          {task.sources_used.length > 5 && (
                            <p className="text-sm text-gray-500 text-center">
                              and {task.sources_used.length - 5} more sources...
                            </p>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="bg-gray-50 rounded-lg p-12 text-center">
              <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">No research tasks yet for this document</p>
              <p className="text-sm text-gray-500 mt-2">Create your first research task using the form</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}