'use client';

import { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'next/navigation';
import { useQuery, useMutation } from '@tanstack/react-query';
import { apiClient, Document, ChatSession, ChatMessage } from '@/lib/api-client';
import { formatRelativeTime, formatDate } from '@/lib/utils';
import { MessageSquare, Send, Loader2, Plus, FileText, Bot, User, Settings } from 'lucide-react';
import { useToast } from '@/components/ui/toast';

export default function ChatPage() {
  const searchParams = useSearchParams();
  const { toast } = useToast();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const preselectedDocumentId = searchParams.get('documentId');
  
  const [selectedDocument, setSelectedDocument] = useState<string>(preselectedDocumentId || '');
  const [selectedSession, setSelectedSession] = useState<string>('');
  const [message, setMessage] = useState('');
  const [useRag, setUseRag] = useState(true);
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [sessionName, setSessionName] = useState('');

  const { data: documents } = useQuery({
    queryKey: ['documents'],
    queryFn: () => apiClient.listDocuments(),
  });

  const { data: sessions, refetch: refetchSessions } = useQuery({
    queryKey: ['chat-sessions'],
    queryFn: () => apiClient.listChatSessions(),
  });

  const { data: messages, refetch: refetchMessages } = useQuery({
    queryKey: ['chat-messages', selectedSession],
    queryFn: () => selectedSession ? apiClient.getChatHistory(selectedSession) : null,
    enabled: !!selectedSession,
  });

  const createSessionMutation = useMutation({
    mutationFn: ({ documentId, name }: { documentId: string; name?: string }) =>
      apiClient.createChatSession(documentId, name),
    onSuccess: (data) => {
      toast({
        title: 'Session created',
        description: 'New chat session has been created.',
      });
      setSelectedSession(data.id);
      refetchSessions();
      setIsCreatingSession(false);
      setSessionName('');
    },
    onError: (error: any) => {
      toast({
        title: 'Failed to create session',
        description: error.message || 'Please try again.',
        variant: 'destructive',
      });
    },
  });

  const sendMessageMutation = useMutation({
    mutationFn: ({ message, documentId, sessionId, useRag }: { 
      message: string; 
      documentId?: string; 
      sessionId?: string;
      useRag: boolean;
    }) => apiClient.sendMessage(message, documentId, sessionId, useRag),
    onSuccess: () => {
      setMessage('');
      refetchMessages();
    },
    onError: (error: any) => {
      toast({
        title: 'Failed to send message',
        description: error.message || 'Please try again.',
        variant: 'destructive',
      });
    },
  });

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Filter sessions for selected document
  const documentSessions = sessions?.filter(
    session => !selectedDocument || session.document_id === selectedDocument
  ) || [];

  const processedDocuments = documents?.filter(doc => doc.is_processed && doc.is_embedded) || [];

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim()) return;

    const currentMessage = message.trim();
    setMessage(''); // Clear immediately for better UX

    sendMessageMutation.mutate({
      message: currentMessage,
      documentId: selectedDocument || undefined,
      sessionId: selectedSession || undefined,
      useRag,
    });
  };

  const handleCreateSession = () => {
    if (!selectedDocument) {
      toast({
        title: 'Select a document',
        description: 'Please select a document before creating a session.',
        variant: 'destructive',
      });
      return;
    }
    createSessionMutation.mutate({
      documentId: selectedDocument,
      name: sessionName.trim() || undefined,
    });
  };

  const currentSession = sessions?.find(s => s.id === selectedSession);

  return (
    <div className="h-[calc(100vh-8rem)] flex">
      {/* Sidebar */}
      <div className="w-80 bg-white border-r flex flex-col">
        <div className="p-4 border-b">
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Document
            </label>
            <select
              value={selectedDocument}
              onChange={(e) => {
                setSelectedDocument(e.target.value);
                setSelectedSession('');
              }}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
            >
              <option value="">All documents</option>
              {processedDocuments.map((doc) => (
                <option key={doc.id} value={doc.id}>
                  {doc.original_filename || doc.filename}
                </option>
              ))}
            </select>
          </div>

          {isCreatingSession ? (
            <div className="space-y-2">
              <input
                type="text"
                value={sessionName}
                onChange={(e) => setSessionName(e.target.value)}
                placeholder="Session name (optional)"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                autoFocus
              />
              <div className="flex space-x-2">
                <button
                  onClick={handleCreateSession}
                  disabled={!selectedDocument || createSessionMutation.isPending}
                  className="flex-1 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                >
                  {createSessionMutation.isPending ? 'Creating...' : 'Create'}
                </button>
                <button
                  onClick={() => {
                    setIsCreatingSession(false);
                    setSessionName('');
                  }}
                  className="px-3 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors text-sm"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <button
              onClick={() => setIsCreatingSession(true)}
              className="w-full py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors flex items-center justify-center space-x-2 text-sm"
            >
              <Plus className="h-4 w-4" />
              <span>New Session</span>
            </button>
          )}
        </div>

        <div className="flex-1 overflow-y-auto">
          {documentSessions.length === 0 ? (
            <div className="p-4 text-center text-gray-700 text-sm">
              No chat sessions yet
            </div>
          ) : (
            <div className="p-2 space-y-1">
              {documentSessions.map((session) => (
                <button
                  key={session.id}
                  onClick={() => setSelectedSession(session.id)}
                  className={`w-full text-left p-3 rounded-lg transition-colors ${
                    selectedSession === session.id
                      ? 'bg-blue-50 border-blue-200 border'
                      : 'hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-gray-900 text-sm">
                      {session.session_name || 'Untitled Session'}
                    </span>
                    <MessageSquare className="h-4 w-4 text-gray-700" />
                  </div>
                  <div className="text-xs text-gray-700">
                    {session.message_count} messages • {formatRelativeTime(session.last_activity)}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        {selectedSession ? (
          <>
            {/* Chat Header */}
            <div className="bg-white border-b px-6 py-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-gray-900">
                    {currentSession?.session_name || 'Chat Session'}
                  </h2>
                  <p className="text-sm text-gray-700">
                    {currentSession && formatDate(currentSession.created_at)}
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <label className="flex items-center space-x-2 text-sm">
                    <input
                      type="checkbox"
                      checked={useRag}
                      onChange={(e) => setUseRag(e.target.checked)}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-gray-700">Use RAG</span>
                  </label>
                </div>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto bg-gray-50 px-6 py-4">
              {messages && messages.length > 0 ? (
                <div className="space-y-4 max-w-4xl mx-auto">
                  {messages.map((msg) => (
                    <div
                      key={msg.id}
                      className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div className={`flex items-start space-x-2 max-w-[80%] ${
                        msg.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                      }`}>
                        <div className={`p-2 rounded-full ${
                          msg.role === 'user' ? 'bg-blue-600' : 'bg-gray-600'
                        }`}>
                          {msg.role === 'user' ? (
                            <User className="h-4 w-4 text-white" />
                          ) : (
                            <Bot className="h-4 w-4 text-white" />
                          )}
                        </div>
                        <div className={`px-4 py-2 rounded-lg ${
                          msg.role === 'user'
                            ? 'bg-blue-600 text-white'
                            : 'bg-white border'
                        }`}>
                          <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                          <p className={`text-xs mt-1 ${
                            msg.role === 'user' ? 'text-blue-100' : 'text-gray-700'
                          }`}>
                            {formatRelativeTime(msg.created_at)}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                  {sendMessageMutation.isPending && (
                    <div className="flex justify-start">
                      <div className="flex items-start space-x-2 max-w-[80%]">
                        <div className="p-2 rounded-full bg-gray-600">
                          <Bot className="h-4 w-4 text-white" />
                        </div>
                        <div className="px-4 py-2 rounded-lg bg-white border">
                          <Loader2 className="h-4 w-4 animate-spin text-gray-700" />
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              ) : (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <MessageSquare className="h-12 w-12 text-gray-700 mx-auto mb-4" />
                    <p className="text-gray-800">Start a conversation</p>
                    <p className="text-sm text-gray-700 mt-2">
                      Ask questions about your document
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Message Input */}
            <div className="bg-white border-t px-6 py-4">
              <form onSubmit={handleSendMessage} className="flex space-x-2">
                <input
                  type="text"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Type your message..."
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={sendMessageMutation.isPending}
                />
                <button
                  type="submit"
                  disabled={!message.trim() || sendMessageMutation.isPending}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                >
                  {sendMessageMutation.isPending ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  ) : (
                    <Send className="h-5 w-5" />
                  )}
                </button>
              </form>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center bg-gray-50">
            <div className="text-center">
              <MessageSquare className="h-16 w-16 text-gray-700 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No session selected</h3>
              <p className="text-gray-700 mb-4">
                Select an existing session or create a new one to start chatting
              </p>
              {selectedDocument && (
                <button
                  onClick={() => setIsCreatingSession(true)}
                  className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Create New Session
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}