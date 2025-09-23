'use client';

import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { Activity, Database, Brain, MessageSquare, HardDrive, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { formatDate } from '@/lib/utils';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444'];

export default function SystemPage() {
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['system-stats'],
    queryFn: () => apiClient.getSystemStats(),
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['system-health'],
    queryFn: () => apiClient.getSystemHealth(),
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  if (statsLoading || healthLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    );
  }

  const documentChartData = stats ? [
    { name: 'Processed', value: stats.documents.processed },
    { name: 'Processing', value: stats.documents.total - stats.documents.processed },
  ] : [];

  const researchChartData = stats ? [
    { name: 'Completed', value: stats.research.completed_tasks },
    { name: 'Pending', value: stats.research.total_tasks - stats.research.completed_tasks },
  ] : [];

  const metricsData = stats ? [
    { name: 'Documents', value: stats.documents.total },
    { name: 'Sessions', value: stats.chat.total_sessions },
    { name: 'Research', value: stats.research.total_tasks },
    { name: 'Vectors', value: Math.round(stats.vector_store.total_vectors / 1000) },
  ] : [];

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">System Monitoring</h1>
        <p className="text-gray-700">Monitor system health and performance metrics</p>
      </div>

      {/* Health Status */}
      <div className="bg-white rounded-lg shadow-sm border p-6 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">System Health</h2>
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full ${
            health?.status === 'healthy' 
              ? 'bg-green-100 text-green-800' 
              : 'bg-red-100 text-red-800'
          }`}>
            {health?.status === 'healthy' ? (
              <CheckCircle className="h-4 w-4" />
            ) : (
              <AlertCircle className="h-4 w-4" />
            )}
            <span className="text-sm font-medium capitalize">{health?.status}</span>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {health?.services && Object.entries(health.services).map(([service, status]) => (
            <div key={service} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-3">
                {service === 'database' && <Database className="h-5 w-5 text-gray-700" />}
                {service === 'openai' && <Brain className="h-5 w-5 text-gray-700" />}
                {service === 'pinecone' && <HardDrive className="h-5 w-5 text-gray-700" />}
                <span className="font-medium capitalize">{service}</span>
              </div>
              <span className={`text-sm font-medium ${
                status === 'connected' || status === 'configured' 
                  ? 'text-green-600' 
                  : 'text-red-600'
              }`}>
                {status}
              </span>
            </div>
          ))}
        </div>
        
        {health && (
          <div className="mt-4 flex items-center justify-between text-sm text-gray-700">
            <span>Version: {health.version}</span>
            <span>Last updated: {formatDate(health.timestamp)}</span>
          </div>
        )}
      </div>

      {/* Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-700">Documents</span>
            <Activity className="h-5 w-5 text-blue-600" />
          </div>
          <p className="text-2xl font-bold text-gray-900">{stats?.documents.total || 0}</p>
          <p className="text-sm text-gray-700 mt-1">
            {stats?.documents.processing_rate} processed
          </p>
        </div>
        
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-700">Chat Sessions</span>
            <MessageSquare className="h-5 w-5 text-green-600" />
          </div>
          <p className="text-2xl font-bold text-gray-900">{stats?.chat.total_sessions || 0}</p>
          <p className="text-sm text-gray-700 mt-1">
            {stats?.chat.active_sessions} active
          </p>
        </div>
        
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-700">Research Tasks</span>
            <Brain className="h-5 w-5 text-purple-600" />
          </div>
          <p className="text-2xl font-bold text-gray-900">{stats?.research.total_tasks || 0}</p>
          <p className="text-sm text-gray-700 mt-1">
            {stats?.research.completion_rate} completed
          </p>
        </div>
        
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-700">Vector Store</span>
            <HardDrive className="h-5 w-5 text-amber-600" />
          </div>
          <p className="text-2xl font-bold text-gray-900">
            {stats?.vector_store.total_vectors.toLocaleString() || 0}
          </p>
          <p className="text-sm text-gray-700 mt-1">
            {((stats?.vector_store.index_fullness || 0) * 100).toFixed(1)}% full
          </p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Document Processing Chart */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Document Processing</h3>
          <div className="h-64">
            {documentChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={documentChartData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {documentChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-gray-700">
                No data available
              </div>
            )}
          </div>
        </div>

        {/* Research Tasks Chart */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Research Tasks</h3>
          <div className="h-64">
            {researchChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={researchChartData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {researchChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-gray-700">
                No data available
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Activity Overview */}
      <div className="bg-white rounded-lg shadow-sm border p-6 mt-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Activity Overview</h3>
        <div className="h-64">
          {metricsData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metricsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#3B82F6" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-gray-500">
              No data available
            </div>
          )}
        </div>
      </div>

      {/* Vector Store Details */}
      {stats?.vector_store && (
        <div className="bg-white rounded-lg shadow-sm border p-6 mt-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Vector Store Details</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <span className="text-gray-700 text-sm">Total Vectors</span>
              <p className="font-medium">{stats.vector_store.total_vectors.toLocaleString()}</p>
            </div>
            <div>
              <span className="text-gray-700 text-sm">Dimension</span>
              <p className="font-medium">{stats.vector_store.dimension}</p>
            </div>
            <div>
              <span className="text-gray-700 text-sm">Index Fullness</span>
              <p className="font-medium">{(stats.vector_store.index_fullness * 100).toFixed(2)}%</p>
            </div>
            <div>
              <span className="text-gray-700 text-sm">Avg per Document</span>
              <p className="font-medium">
                {stats.documents.embedded > 0 
                  ? Math.round(stats.vector_store.total_vectors / stats.documents.embedded)
                  : 0}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}