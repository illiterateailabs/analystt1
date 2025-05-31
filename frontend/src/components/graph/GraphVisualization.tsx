'use client'

import { useState, useEffect, useRef } from 'react'
import { useQuery, useMutation } from 'react-query'
import { graphAPI, crewAPI, handleAPIError } from '@/lib/api'
import toast from 'react-hot-toast'
import { 
  MagnifyingGlassIcon, 
  CircleStackIcon,
  ChartBarIcon,
  CodeBracketIcon,
  UsersIcon,
  FunnelIcon,
  ArrowDownTrayIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline'
import { Network } from 'vis-network'
import { DataSet } from 'vis-data'

// Define types for crew result
interface CrewNode {
  id: string
  label: string
  type: string
  properties: Record<string, any>
  risk_score?: number
}

interface CrewEdge {
  id: string
  from: string
  to: string
  label: string
  properties?: Record<string, any>
}

interface FraudPattern {
  id: string
  name: string
  description: string
  confidence: number
  affected_entities: string[]
}

interface ComplianceFinding {
  regulation: string
  status: 'compliant' | 'non-compliant' | 'warning'
  description: string
  recommendation?: string
}

interface CrewResult {
  graph_data?: {
    nodes: CrewNode[]
    edges: CrewEdge[]
  }
  fraud_patterns?: FraudPattern[]
  risk_assessment?: {
    overall_score: number
    factors: Record<string, number>
    summary: string
  }
  compliance_findings?: ComplianceFinding[]
  executive_summary?: string
  recommendations?: string[]
}

// Color mapping for node types
const nodeColors = {
  Person: '#4299e1',
  Account: '#48bb78',
  Transaction: '#ed8936',
  Organization: '#9f7aea',
  Asset: '#f56565',
  Event: '#667eea',
  Location: '#38b2ac',
  default: '#a0aec0'
}

// Color mapping for risk scores
const riskColors = {
  high: '#f56565',
  medium: '#ed8936',
  low: '#48bb78',
  unknown: '#a0aec0'
}

export function GraphVisualization() {
  const [cypherQuery, setCypherQuery] = useState('')
  const [naturalQuery, setNaturalQuery] = useState('')
  const [crewName, setCrewName] = useState('fraud_investigation')
  const [crewInput, setCrewInput] = useState('')
  const [activeTab, setActiveTab] = useState<'natural' | 'cypher' | 'analytics' | 'crew'>('natural')
  const [crewResult, setCrewResult] = useState<CrewResult | null>(null)
  const [filterType, setFilterType] = useState<string>('all')
  const [filterRisk, setFilterRisk] = useState<string>('all')
  const [filterPattern, setFilterPattern] = useState<string>('all')
  const [selectedNode, setSelectedNode] = useState<CrewNode | null>(null)
  
  // Refs for vis network
  const networkContainer = useRef<HTMLDivElement>(null)
  const networkInstance = useRef<Network | null>(null)

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

  // Get available crews
  const { data: crewsData, isLoading: crewsLoading } = useQuery(
    'available-crews',
    () => crewAPI.listCrews(),
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

  // Crew run mutation
  const crewRunMutation = useMutation(
    (request: { crew_name: string, inputs: any }) => crewAPI.runCrew(request),
    {
      onSuccess: (response) => {
        toast.success('Crew execution completed')
        console.log('Crew results:', response.data)
        if (response.data?.result) {
          setCrewResult(response.data.result)
        }
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

  const handleCrewRun = () => {
    const inputs = crewInput.trim() 
      ? { query: crewInput } 
      : { query: "Analyze recent transactions for suspicious patterns" }
    
    crewRunMutation.mutate({
      crew_name: crewName,
      inputs
    })
  }

  // Function to initialize network visualization
  useEffect(() => {
    if (activeTab === 'crew' && crewResult?.graph_data && networkContainer.current) {
      // Create nodes dataset
      const nodes = new DataSet(
        crewResult.graph_data.nodes.map(node => {
          const riskLevel = node.risk_score 
            ? (node.risk_score > 0.7 ? 'high' : node.risk_score > 0.4 ? 'medium' : 'low') 
            : 'unknown'
          
          // Filter nodes if filters are active
          if (
            (filterType !== 'all' && node.type !== filterType) ||
            (filterRisk !== 'all' && riskLevel !== filterRisk) ||
            (filterPattern !== 'all' && 
              !crewResult.fraud_patterns?.some(p => 
                p.id === filterPattern && p.affected_entities.includes(node.id)
              ))
          ) {
            return {
              ...node,
              hidden: true
            }
          }
          
          return {
            id: node.id,
            label: node.label,
            title: `${node.type}: ${node.label}`,
            group: node.type,
            color: {
              background: nodeColors[node.type as keyof typeof nodeColors] || nodeColors.default,
              border: riskColors[riskLevel as keyof typeof riskColors],
              highlight: {
                background: nodeColors[node.type as keyof typeof nodeColors] || nodeColors.default,
                border: '#000000'
              }
            },
            borderWidth: node.risk_score && node.risk_score > 0.7 ? 3 : 1,
            font: { size: 12 }
          }
        })
      )
      
      // Create edges dataset
      const edges = new DataSet(
        crewResult.graph_data.edges.map(edge => ({
          id: edge.id,
          from: edge.from,
          to: edge.to,
          label: edge.label,
          arrows: 'to',
          font: { align: 'middle', size: 11 }
        }))
      )
      
      // Configure network options
      const options = {
        nodes: {
          shape: 'dot',
          size: 16,
          shadow: true
        },
        edges: {
          width: 1,
          shadow: true,
          smooth: { type: 'continuous' }
        },
        physics: {
          stabilization: true,
          barnesHut: {
            gravitationalConstant: -80000,
            springConstant: 0.001,
            springLength: 200
          }
        },
        interaction: {
          navigationButtons: true,
          keyboard: true,
          tooltipDelay: 200
        },
        groups: {
          Person: { shape: 'icon', icon: { face: 'FontAwesome', code: '\uf007', size: 50, color: nodeColors.Person } },
          Account: { shape: 'icon', icon: { face: 'FontAwesome', code: '\uf19c', size: 50, color: nodeColors.Account } },
          Transaction: { shape: 'icon', icon: { face: 'FontAwesome', code: '\uf0ec', size: 50, color: nodeColors.Transaction } },
          Organization: { shape: 'icon', icon: { face: 'FontAwesome', code: '\uf1ad', size: 50, color: nodeColors.Organization } },
          Asset: { shape: 'icon', icon: { face: 'FontAwesome', code: '\uf0d6', size: 50, color: nodeColors.Asset } },
          Event: { shape: 'icon', icon: { face: 'FontAwesome', code: '\uf073', size: 50, color: nodeColors.Event } },
          Location: { shape: 'icon', icon: { face: 'FontAwesome', code: '\uf041', size: 50, color: nodeColors.Location } }
        }
      }
      
      // Create network
      networkInstance.current = new Network(
        networkContainer.current,
        { nodes, edges },
        options
      )
      
      // Add event listener for node clicks
      networkInstance.current.on('click', (params) => {
        if (params.nodes.length > 0) {
          const nodeId = params.nodes[0]
          const node = crewResult.graph_data?.nodes.find(n => n.id === nodeId)
          if (node) {
            setSelectedNode(node)
          }
        } else {
          setSelectedNode(null)
        }
      })
      
      return () => {
        if (networkInstance.current) {
          networkInstance.current.destroy()
          networkInstance.current = null
        }
      }
    }
  }, [activeTab, crewResult, filterType, filterRisk, filterPattern])
  
  // Function to export graph as image
  const exportGraph = () => {
    if (networkInstance.current) {
      const dataUrl = networkInstance.current.canvas.canvas.toDataURL('image/png')
      const link = document.createElement('a')
      link.download = `fraud-investigation-${new Date().toISOString().slice(0, 10)}.png`
      link.href = dataUrl
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      toast.success('Graph exported as PNG image')
    }
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
              { id: 'crew', name: 'Crew Results', icon: UsersIcon },
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

          {activeTab === 'crew' && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Crew Selection
                </label>
                <select
                  value={crewName}
                  onChange={(e) => setCrewName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500"
                >
                  {crewsLoading ? (
                    <option>Loading crews...</option>
                  ) : (
                    crewsData?.data?.crews?.map((crew: any) => (
                      <option key={crew.name} value={crew.name}>
                        {crew.name} - {crew.description}
                      </option>
                    )) || [
                      <option key="fraud_investigation" value="fraud_investigation">
                        fraud_investigation - Fraud detection and analysis
                      </option>
                    ]
                  )}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Investigation Query
                </label>
                <textarea
                  value={crewInput}
                  onChange={(e) => setCrewInput(e.target.value)}
                  placeholder="e.g., Analyze transactions between accounts A and B for potential layering patterns"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500"
                  rows={4}
                />
              </div>

              <button
                onClick={handleCrewRun}
                disabled={crewRunMutation.isLoading}
                className="w-full btn-primary"
              >
                {crewRunMutation.isLoading ? 'Running investigation...' : 'Run Investigation'}
              </button>

              {crewResult && (
                <div className="mt-6 space-y-6">
                  {/* Filtering controls */}
                  <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                    <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center">
                      <FunnelIcon className="h-4 w-4 mr-1" />
                      Filter Graph
                    </h4>
                    <div className="grid grid-cols-3 gap-3">
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">Entity Type</label>
                        <select
                          value={filterType}
                          onChange={(e) => setFilterType(e.target.value)}
                          className="w-full px-2 py-1 text-sm border border-gray-300 rounded-md"
                        >
                          <option value="all">All Types</option>
                          {crewResult.graph_data?.nodes
                            .map(node => node.type)
                            .filter((value, index, self) => self.indexOf(value) === index)
                            .map(type => (
                              <option key={type} value={type}>{type}</option>
                            ))
                          }
                        </select>
                      </div>
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">Risk Level</label>
                        <select
                          value={filterRisk}
                          onChange={(e) => setFilterRisk(e.target.value)}
                          className="w-full px-2 py-1 text-sm border border-gray-300 rounded-md"
                        >
                          <option value="all">All Risks</option>
                          <option value="high">High Risk</option>
                          <option value="medium">Medium Risk</option>
                          <option value="low">Low Risk</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">Fraud Pattern</label>
                        <select
                          value={filterPattern}
                          onChange={(e) => setFilterPattern(e.target.value)}
                          className="w-full px-2 py-1 text-sm border border-gray-300 rounded-md"
                        >
                          <option value="all">All Patterns</option>
                          {crewResult.fraud_patterns?.map(pattern => (
                            <option key={pattern.id} value={pattern.id}>{pattern.name}</option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>

                  {/* Export button */}
                  <button
                    onClick={exportGraph}
                    className="flex items-center text-sm text-gray-700 bg-white border border-gray-300 rounded-md px-3 py-2 hover:bg-gray-50"
                  >
                    <ArrowDownTrayIcon className="h-4 w-4 mr-2" />
                    Export Graph as Image
                  </button>

                  {/* Executive summary */}
                  {crewResult.executive_summary && (
                    <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                      <h4 className="text-sm font-medium text-blue-800 mb-2">Executive Summary</h4>
                      <p className="text-sm text-blue-700">{crewResult.executive_summary}</p>
                    </div>
                  )}

                  {/* Fraud patterns */}
                  {crewResult.fraud_patterns && crewResult.fraud_patterns.length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Detected Fraud Patterns</h4>
                      <div className="space-y-2">
                        {crewResult.fraud_patterns.map((pattern) => (
                          <div key={pattern.id} className="bg-red-50 p-3 rounded-md border border-red-200">
                            <h5 className="text-sm font-medium text-red-800 flex items-center">
                              <ExclamationTriangleIcon className="h-4 w-4 mr-1 text-red-600" />
                              {pattern.name} 
                              <span className="ml-2 text-xs bg-red-200 text-red-800 px-2 py-0.5 rounded-full">
                                {Math.round(pattern.confidence * 100)}% confidence
                              </span>
                            </h5>
                            <p className="text-xs text-red-700 mt-1">{pattern.description}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Compliance findings */}
                  {crewResult.compliance_findings && crewResult.compliance_findings.length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Compliance Findings</h4>
                      <div className="space-y-2">
                        {crewResult.compliance_findings.map((finding, index) => (
                          <div 
                            key={index} 
                            className={`p-3 rounded-md border ${
                              finding.status === 'compliant' 
                                ? 'bg-green-50 border-green-200' 
                                : finding.status === 'warning'
                                ? 'bg-yellow-50 border-yellow-200'
                                : 'bg-red-50 border-red-200'
                            }`}
                          >
                            <h5 className={`text-sm font-medium flex items-center ${
                              finding.status === 'compliant' 
                                ? 'text-green-800' 
                                : finding.status === 'warning'
                                ? 'text-yellow-800'
                                : 'text-red-800'
                            }`}>
                              {finding.status === 'compliant' 
                                ? <CheckCircleIcon className="h-4 w-4 mr-1 text-green-600" />
                                : finding.status === 'warning'
                                ? <ExclamationTriangleIcon className="h-4 w-4 mr-1 text-yellow-600" />
                                : <ExclamationTriangleIcon className="h-4 w-4 mr-1 text-red-600" />
                              }
                              {finding.regulation}
                            </h5>
                            <p className="text-xs mt-1">{finding.description}</p>
                            {finding.recommendation && (
                              <p className="text-xs mt-1 font-medium">
                                Recommendation: {finding.recommendation}
                              </p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Recommendations */}
                  {crewResult.recommendations && crewResult.recommendations.length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Recommendations</h4>
                      <ul className="list-disc list-inside space-y-1">
                        {crewResult.recommendations.map((recommendation, index) => (
                          <li key={index} className="text-sm text-gray-700">{recommendation}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
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
            {activeTab === 'crew' && crewResult 
              ? 'Fraud investigation results visualization' 
              : 'Interactive graph visualization will appear here'}
          </p>
        </div>

        {/* Visualization area */}
        <div className="flex-1 p-6 flex">
          {activeTab === 'crew' && crewResult?.graph_data ? (
            <div className="flex-1 flex flex-col">
              {/* Graph visualization */}
              <div className="flex-1 rounded-lg border border-gray-300 overflow-hidden relative">
                <div ref={networkContainer} className="w-full h-full"></div>
                
                {/* Legend */}
                <div className="absolute bottom-4 right-4 bg-white p-3 rounded-lg shadow-md border border-gray-200 text-xs">
                  <h4 className="font-medium text-gray-700 mb-2">Legend</h4>
                  <div className="space-y-1.5">
                    {Object.entries(nodeColors).filter(([key]) => key !== 'default').map(([type, color]) => (
                      <div key={type} className="flex items-center">
                        <span 
                          className="w-3 h-3 rounded-full mr-2" 
                          style={{ backgroundColor: color }}
                        ></span>
                        <span>{type}</span>
                      </div>
                    ))}
                    <div className="border-t border-gray-200 my-1 pt-1">
                      <div className="flex items-center">
                        <span className="w-3 h-3 rounded-full mr-2 border-2 border-red-500"></span>
                        <span>High Risk</span>
                      </div>
                      <div className="flex items-center">
                        <span className="w-3 h-3 rounded-full mr-2 border-2 border-orange-500"></span>
                        <span>Medium Risk</span>
                      </div>
                      <div className="flex items-center">
                        <span className="w-3 h-3 rounded-full mr-2 border-2 border-green-500"></span>
                        <span>Low Risk</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Selected node details */}
              {selectedNode && (
                <div className="mt-4 p-4 bg-white rounded-lg border border-gray-300 max-h-64 overflow-y-auto">
                  <div className="flex justify-between items-start">
                    <h4 className="text-sm font-medium text-gray-900 flex items-center">
                      <InformationCircleIcon className="h-4 w-4 mr-1" />
                      {selectedNode.type}: {selectedNode.label}
                    </h4>
                    <button 
                      onClick={() => setSelectedNode(null)}
                      className="text-gray-500 hover:text-gray-700"
                    >
                      ×
                    </button>
                  </div>
                  
                  <div className="mt-2 space-y-2">
                    {selectedNode.risk_score && (
                      <div className="flex items-center">
                        <span className="text-xs font-medium w-24">Risk Score:</span>
                        <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div 
                            className={`h-full ${
                              selectedNode.risk_score > 0.7 
                                ? 'bg-red-500' 
                                : selectedNode.risk_score > 0.4 
                                ? 'bg-orange-500' 
                                : 'bg-green-500'
                            }`} 
                            style={{ width: `${selectedNode.risk_score * 100}%` }}
                          ></div>
                        </div>
                        <span className="ml-2 text-xs">{Math.round(selectedNode.risk_score * 100)}%</span>
                      </div>
                    )}
                    
                    <div className="text-xs">
                      <h5 className="font-medium mb-1">Properties:</h5>
                      <div className="grid grid-cols-2 gap-x-2 gap-y-1">
                        {Object.entries(selectedNode.properties).map(([key, value]) => (
                          <div key={key}>
                            <span className="font-medium">{key}:</span> {String(value)}
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    {/* Connected fraud patterns */}
                    {crewResult.fraud_patterns && (
                      <div className="text-xs">
                        <h5 className="font-medium mb-1">Related Fraud Patterns:</h5>
                        <div className="space-y-1">
                          {crewResult.fraud_patterns
                            .filter(pattern => pattern.affected_entities.includes(selectedNode.id))
                            .map(pattern => (
                              <div key={pattern.id} className="bg-red-50 p-1 rounded text-red-700">
                                {pattern.name}
                              </div>
                            ))}
                          {!crewResult.fraud_patterns.some(pattern => 
                            pattern.affected_entities.includes(selectedNode.id)
                          ) && (
                            <div className="text-gray-500">No related fraud patterns</div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="w-full h-full bg-white rounded-lg border-2 border-dashed border-gray-300 flex items-center justify-center">
              <div className="text-center">
                <CircleStackIcon className="mx-auto h-12 w-12 text-gray-400" />
                <h3 className="mt-2 text-sm font-medium text-gray-900">
                  Graph Visualization
                </h3>
                <p className="mt-1 text-sm text-gray-500">
                  {activeTab === 'crew' 
                    ? 'Run a crew investigation to see the graph visualization' 
                    : 'Execute a query to see the graph visualization'}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
