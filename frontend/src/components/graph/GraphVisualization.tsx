'use client'

import { useState } from 'react'
import { useQuery, useMutation } from 'react-query'
import { graphAPI, handleAPIError } from '@/lib/api'
import toast from 'react-hot-toast'
import { 
  MagnifyingGlassIcon, 
  CircleStackIcon,
  ChartBarIcon,
  CodeBracketIcon 
} from '@heroicons/react/24/outline'

export function GraphVisualization() {
  const [cypherQuery, setCypherQuery] = useState('')
  const [naturalQuery, setNaturalQuery] = useState('')
  const [activeTab, setActiveTab] = useState<'natural' | 'cypher' | 'analytics'>('natural')

  // Get schema information
  const { data: schemaData, isLoading: schemaLoading } = useQuery(
    'graph-schema',
    () => graphAPI.getSchema(),
    {
      onError: (error) => {
        const errorInfo = handleAPIError(error)
        toast.error(errorInfo.message)
      },
    }
  )

  // Natural language query mutation
  const naturalQueryMutation = useMutation(
    (question: string) => graphAPI.naturalLanguageQuery(question),
    {
      onSuccess: (response) => {
        toast.success('Query executed successfully')
        console.log('Query results:', response.data)
      },
      onError: (error) => {
        const errorInfo = handleAPIError(error)
        toast.error(errorInfo.message)
      },
    }
  )

  // Cypher query mutation
  const cypherQueryMutation = useMutation(
    (query: string) => graphAPI.executeCypher(query),
    {
      onSuccess: (response) => {
        toast.success('Cypher query executed successfully')
        console.log('Query results:', response.data)
      },
      onError: (error) => {
        const errorInfo = handleAPIError(error)
        toast.error(errorInfo.message)
      },
    }
  )

  // Centrality analysis mutation
  const centralityMutation = useMutation(
    (algorithm: string) => graphAPI.calculateCentrality(algorithm),
    {
      onSuccess: (response) => {
        toast.success('Centrality analysis completed')
        console.log('Centrality results:', response.data)
      },
      onError: (error) => {
        const errorInfo = handleAPIError(error)
        toast.error(errorInfo.message)
      },
    }
  )

  const handleNaturalQuery = () => {
    if (!naturalQuery.trim()) return
    naturalQueryMutation.mutate(naturalQuery)
  }

  const handleCypherQuery = () => {
    if (!cypherQuery.trim()) return
    cypherQueryMutation.mutate(cypherQuery)
  }

  const schema = schemaData?.data?.schema

  return (
    <div className="flex h-full bg-gray-50">
      {/* Left panel - Query interface */}
      <div className="w-1/3 bg-white border-r border-gray-200 flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900 flex items-center">
            <CircleStackIcon className="h-6 w-6 mr-2" />
            Graph Database
          </h2>
          <p className="text-sm text-gray-500 mt-1">
            Query and visualize your graph data
          </p>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6" aria-label="Tabs">
            {[
              { id: 'natural', name: 'Natural Language', icon: MagnifyingGlassIcon },
              { id: 'cypher', name: 'Cypher', icon: CodeBracketIcon },
              { id: 'analytics', name: 'Analytics', icon: ChartBarIcon },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center`}
              >
                <tab.icon className="h-4 w-4 mr-2" />
                {tab.name}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab content */}
        <div className="flex-1 p-6 overflow-y-auto">
          {activeTab === 'natural' && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Ask a question about your data
                </label>
                <textarea
                  value={naturalQuery}
                  onChange={(e) => setNaturalQuery(e.target.value)}
                  placeholder="e.g., Find all people connected to suspicious transactions"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500"
                  rows={4}
                />
              </div>
              <button
                onClick={handleNaturalQuery}
                disabled={!naturalQuery.trim() || naturalQueryMutation.isLoading}
                className="w-full btn-primary"
              >
                {naturalQueryMutation.isLoading ? 'Processing...' : 'Ask Question'}
              </button>
            </div>
          )}

          {activeTab === 'cypher' && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Cypher Query
                </label>
                <textarea
                  value={cypherQuery}
                  onChange={(e) => setCypherQuery(e.target.value)}
                  placeholder="MATCH (n) RETURN n LIMIT 10"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500 font-mono text-sm"
                  rows={6}
                />
              </div>
              <button
                onClick={handleCypherQuery}
                disabled={!cypherQuery.trim() || cypherQueryMutation.isLoading}
                className="w-full btn-primary"
              >
                {cypherQueryMutation.isLoading ? 'Executing...' : 'Execute Query'}
              </button>
            </div>
          )}

          {activeTab === 'analytics' && (
            <div className="space-y-4">
              <h3 className="text-lg font-medium text-gray-900">Graph Analytics</h3>
              
              <div className="space-y-3">
                <button
                  onClick={() => centralityMutation.mutate('pagerank')}
                  disabled={centralityMutation.isLoading}
                  className="w-full btn-secondary text-left"
                >
                  PageRank Centrality
                </button>
                
                <button
                  onClick={() => centralityMutation.mutate('betweenness')}
                  disabled={centralityMutation.isLoading}
                  className="w-full btn-secondary text-left"
                >
                  Betweenness Centrality
                </button>
                
                <button
                  onClick={() => centralityMutation.mutate('degree')}
                  disabled={centralityMutation.isLoading}
                  className="w-full btn-secondary text-left"
                >
                  Degree Centrality
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Schema info */}
        <div className="border-t border-gray-200 p-6">
          <h3 className="text-sm font-medium text-gray-900 mb-3">Schema Info</h3>
          {schemaLoading ? (
            <div className="text-sm text-gray-500">Loading schema...</div>
          ) : schema ? (
            <div className="space-y-2 text-xs text-gray-600">
              <div>
                <span className="font-medium">Labels:</span> {schema.labels?.join(', ') || 'None'}
              </div>
              <div>
                <span className="font-medium">Relationships:</span> {schema.relationship_types?.join(', ') || 'None'}
              </div>
              <div>
                <span className="font-medium">Nodes:</span> {schema.node_count || 0}
              </div>
              <div>
                <span className="font-medium">Relationships:</span> {schema.relationship_count || 0}
              </div>
            </div>
          ) : (
            <div className="text-sm text-gray-500">No schema data available</div>
          )}
        </div>
      </div>

      {/* Right panel - Visualization */}
      <div className="flex-1 flex flex-col">
        {/* Visualization header */}
        <div className="p-6 border-b border-gray-200 bg-white">
          <h3 className="text-lg font-medium text-gray-900">Graph Visualization</h3>
          <p className="text-sm text-gray-500">
            Interactive graph visualization will appear here
          </p>
        </div>

        {/* Visualization area */}
        <div className="flex-1 p-6">
          <div className="h-full bg-white rounded-lg border-2 border-dashed border-gray-300 flex items-center justify-center">
            <div className="text-center">
              <CircleStackIcon className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">
                Graph Visualization
              </h3>
              <p className="mt-1 text-sm text-gray-500">
                Execute a query to see the graph visualization
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
