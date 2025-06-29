'use client'

import { useState, useRef, useEffect } from 'react'
import { useQuery, useMutation } from 'react-query'
import { graphAPI, handleAPIError, GraphData, GraphNode, GraphRelationship } from '@/lib/api'
import toast from 'react-hot-toast'
import { Network, DataSet } from 'vis-network/standalone/esm/vis-network'
import { 
  MagnifyingGlassIcon, 
  CircleStackIcon,
  ChartBarIcon,
  CodeBracketIcon,
  ArrowDownTrayIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline'

// Enhanced Styling Logic
const getNodeStyle = (node: GraphNode) => {
  const properties = node.properties || {};
  const riskScore = properties.risk_score || 0;
  const isWashTrading = properties.wash_trading_indicator === true;
  const isSuspicious = properties.suspicious_pattern === true;

  let color = '#97C2FC'; // Default blue
  let shape = 'dot';
  let size = 25;

  // Default label-based styling (backward compatible)
  if (node.labels.includes('Person')) { color = '#FB7E81'; shape = 'diamond'; }
  else if (node.labels.includes('Account')) { color = '#7BE141'; shape = 'square'; }
  else if (node.labels.includes('Transaction')) { color = '#FFA807'; shape = 'triangle'; }
  else if (node.labels.includes('Wallet')) { color = '#6E6EFD'; shape = 'hexagon'; }

  // Fraud-specific overrides
  if (riskScore > 0.8 || properties.fraud_indicator === true) {
    color = '#ff3b30'; // Red for high risk
  } else if (riskScore > 0.5) {
    color = '#ff9500'; // Orange for medium risk
  } else if (isWashTrading || isSuspicious) {
    color = '#ff69b4'; // Pink for suspicious patterns
  }

  // Size scaling
  if (riskScore > 0) {
    size = 15 + Math.min(riskScore * 20, 40); // Scale size based on risk
  } else if (properties.transaction_volume) {
    size = 10 + Math.min(Math.log(properties.transaction_volume) * 5, 40);
  }

  return {
    id: node.id,
    label: node.labels[0] + '\n' + (properties.name || properties.id || ''),
    title: `<b>${node.labels.join(', ')}</b><br><pre>${JSON.stringify(properties, null, 2)}</pre>`,
    color,
    shape,
    size,
    font: { size: 12 }
  };
};

const getEdgeStyle = (rel: GraphRelationship) => {
  const properties = rel.properties || {};
  const isFraudLink = properties.is_fraud_link === true;
  const isWashTrade = properties.is_wash_trade === true;
  const isSuspicious = properties.is_suspicious_flow === true;
  const transactionValue = properties.value_usd || properties.value || 0;

  let color = { color: '#848484', highlight: '#6E6EFD' }; // Default gray
  let dashes = false;
  let width = 1;

  // Fraud-specific styling
  if (isFraudLink) {
    color = { color: '#ff3b30', highlight: '#ff3b30' };
  } else if (isWashTrade) {
    color = { color: '#ff69b4', highlight: '#ff69b4' };
    dashes = true;
  } else if (isSuspicious) {
    color = { color: '#cccccc', highlight: '#aaaaaa' };
    dashes = [5, 5];
  }

  // Width scaling
  if (transactionValue > 0) {
    width = 1 + Math.min(Math.log10(transactionValue + 1), 10);
  }

  return {
    id: rel.id,
    from: rel.startNode,
    to: rel.endNode,
    label: rel.type,
    arrows: 'to',
    font: { size: 10, align: 'middle' },
    title: `<b>${rel.type}</b><br><pre>${JSON.stringify(properties, null, 2)}</pre>`,
    color,
    dashes,
    width,
  };
};

const GraphLegend = () => (
  <div className="absolute bottom-4 left-4 bg-white bg-opacity-80 p-3 rounded-lg shadow-md border border-gray-200">
    <h4 className="font-semibold text-sm mb-2 flex items-center">
      <InformationCircleIcon className="h-4 w-4 mr-1" />
      Legend
    </h4>
    <div className="space-y-1 text-xs">
      <div className="flex items-center"><div className="w-3 h-3 rounded-full mr-2" style={{ backgroundColor: '#ff3b30' }}></div> High Risk / Fraud Link</div>
      <div className="flex items-center"><div className="w-3 h-3 rounded-full mr-2" style={{ backgroundColor: '#ff9500' }}></div> Medium Risk</div>
      <div className="flex items-center"><div className="w-3 h-3 rounded-full mr-2" style={{ backgroundColor: '#ff69b4' }}></div> Suspicious Pattern</div>
      <div className="flex items-center"><div className="w-3 h-0.5 mr-2 border-t-2 border-dashed" style={{ borderColor: '#ff69b4' }}></div> Wash Trading Flow</div>
      <div className="flex items-center"><div className="w-3 h-0.5 mr-2 border-t-2 border-dashed" style={{ borderColor: '#cccccc' }}></div> Suspicious Flow</div>
    </div>
  </div>
);


export function GraphVisualization() {
  const [cypherQuery, setCypherQuery] = useState('')
  const [naturalQuery, setNaturalQuery] = useState('')
  const [activeTab, setActiveTab] = useState<'natural' | 'cypher' | 'analytics'>('natural')
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  
  // Refs for vis-network
  const networkRef = useRef<Network | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Get schema information
  const { data: schemaData, isLoading: schemaLoading } = useQuery(
    'graph-schema',
    () => graphAPI.getGraphSchema(),
    {
      onError: (error) => {
        const errorInfo = handleAPIError(error)
        toast.error(errorInfo.message)
      },
    }
  )

  // Natural language query mutation
  const naturalQueryMutation = useMutation(
    (question: string) => graphAPI.executeGraphQuery({ query: question, parameters: { natural: true } }),
    {
      onSuccess: (response) => {
        toast.success('Query executed successfully')
        console.log('Query results:', response.data)
        setGraphData(response.data.data || response.data)
      },
      onError: (error) => {
        const errorInfo = handleAPIError(error)
        toast.error(errorInfo.message)
      },
    }
  )

  // Cypher query mutation
  const cypherQueryMutation = useMutation(
    (query: string) => graphAPI.executeGraphQuery({ query }),
    {
      onSuccess: (response) => {
        toast.success('Cypher query executed successfully')
        console.log('Query results:', response.data)
        setGraphData(response.data.data || response.data)
      },
      onError: (error) => {
        const errorInfo = handleAPIError(error)
        toast.error(errorInfo.message)
      },
    }
  )

  // Centrality analysis mutation
  const centralityMutation = useMutation(
    (algorithm: string) => graphAPI.executeGraphQuery({ 
      query: `CALL gds.${algorithm}.stream('graph') YIELD nodeId, score RETURN gds.util.asNode(nodeId) AS node, score ORDER BY score DESC LIMIT 20`
    }),
    {
      onSuccess: (response) => {
        toast.success('Centrality analysis completed')
        console.log('Centrality results:', response.data)
        setGraphData(response.data.data || response.data)
      },
      onError: (error) => {
        const errorInfo = handleAPIError(error)
        toast.error(errorInfo.message)
      },
    }
  )

  // Initialize and update network when graphData changes
  useEffect(() => {
    if (!containerRef.current || !graphData) return

    // Prepare nodes and edges with enhanced styling
    const nodes = new DataSet(graphData.nodes.map(getNodeStyle));
    const edges = new DataSet(graphData.relationships.map(getEdgeStyle));

    // Network options
    const options = {
      layout: {
        improvedLayout: true,
        hierarchical: false
      },
      physics: {
        enabled: true,
        barnesHut: {
          gravitationalConstant: -2000,
          centralGravity: 0.3,
          springLength: 95,
          springConstant: 0.04,
          damping: 0.09
        }
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        navigationButtons: true,
        keyboard: true
      },
      nodes: {
        borderWidth: 2,
        borderWidthSelected: 4,
      },
      edges: {
        smooth: {
          type: 'continuous'
        }
      }
    }

    // Create or update network
    if (networkRef.current) {
      networkRef.current.setData({ nodes, edges })
    } else if (containerRef.current) {
      networkRef.current = new Network(containerRef.current, { nodes, edges }, options)
      
      // Add event listeners
      networkRef.current.on('click', (params) => {
        if (params.nodes.length > 0) {
          const nodeId = params.nodes[0]
          const node = nodes.get(nodeId)
          console.log('Node clicked:', node)
          toast.success(`Clicked node: ${node.label}`)
        }
      })
    }

    return () => {
      // Cleanup if component unmounts
      if (networkRef.current) {
        networkRef.current.destroy()
        networkRef.current = null
      }
    }
  }, [graphData])

  // Function to export network as PNG
  const exportNetworkAsPNG = () => {
    if (!networkRef.current) return
    
    try {
      // Get canvas with current network state
      const dataUrl = networkRef.current.canvas.body.container.getElementsByTagName('canvas')[0].toDataURL('image/png')
      
      // Create download link
      const link = document.createElement('a')
      link.href = dataUrl
      link.download = `graph-export-${new Date().toISOString().slice(0, 10)}.png`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      
      toast.success('Graph exported as PNG')
    } catch (error) {
      console.error('Error exporting graph:', error)
      toast.error('Failed to export graph')
    }
  }

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
      <div className="flex-1 flex flex-col relative">
        {/* Visualization header */}
        <div className="p-6 border-b border-gray-200 bg-white flex justify-between items-center">
          <div>
            <h3 className="text-lg font-medium text-gray-900">Graph Visualization</h3>
            <p className="text-sm text-gray-500">
              {graphData ? 
                `Showing ${graphData.nodes.length} nodes and ${graphData.relationships.length} relationships` : 
                'Execute a query to visualize the graph'}
            </p>
          </div>
          
          {graphData && (
            <button
              onClick={exportNetworkAsPNG}
              className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
            >
              <ArrowDownTrayIcon className="h-4 w-4 mr-2" />
              Export PNG
            </button>
          )}
        </div>

        {/* Visualization area */}
        <div className="flex-1 p-6">
          {graphData ? (
            <div ref={containerRef} className="h-full w-full border border-gray-200 rounded-lg" />
          ) : (
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
          )}
        </div>
        {graphData && <GraphLegend />}
      </div>
    </div>
  )
}
