import React, { useCallback, useEffect, useRef, useState } from 'react';
import ForceGraph2D, { ForceGraphMethods } from 'react-force-graph-2d';
import { scaleLinear } from 'd3-scale';
import { schemeRdYlGn } from 'd3-scale-chromatic';
import { Modal, Button, Tooltip, Spin, Select, Slider, Switch, Card, Tag, Badge } from 'antd';
import { ZoomInOutlined, ZoomOutOutlined, FilterOutlined, InfoCircleOutlined, ReloadOutlined, SaveOutlined, WarningOutlined } from '@ant-design/icons';
import axios from 'axios';

// Types for graph data
interface GraphNode {
  id: string;
  address?: string;
  label?: string;
  type?: string;
  risk_score?: number;
  fraud_risk?: 'low' | 'medium' | 'high' | 'critical' | 'unknown';
  anomaly_count?: number;
  size?: number;
  color?: string;
  x?: number;
  y?: number;
  metadata?: Record<string, any>;
}

interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  type?: string;
  fraud_type?: string;
  value?: number;
  hash?: string;
  timestamp?: string;
  width?: number;
  color?: string;
  dashed?: boolean;
  metadata?: Record<string, any>;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

// Props for the component
interface FraudVisualizationProps {
  initialData?: GraphData;
  entityId?: string;
  entityType?: string;
  height?: number;
  width?: number;
  onNodeClick?: (node: GraphNode) => void;
  onLinkClick?: (link: GraphLink) => void;
  className?: string;
  style?: React.CSSProperties;
  autoFetch?: boolean;
}

/**
 * FraudVisualization Component
 * 
 * A React component for visualizing blockchain fraud patterns using an interactive graph.
 * Features include:
 * - Color-coded nodes based on fraud risk scores
 * - Different edge styles for various fraud types
 * - Zoom, pan, and node selection capabilities
 * - Hover/click details for fraud information
 * - Integration with the anomaly detection API
 * - Filtering options for different fraud types
 */
const FraudVisualization: React.FC<FraudVisualizationProps> = ({
  initialData,
  entityId,
  entityType = 'address',
  height = 600,
  width = 800,
  onNodeClick,
  onLinkClick,
  className,
  style,
  autoFetch = true
}) => {
  // State for graph data and UI
  const [graphData, setGraphData] = useState<GraphData>(initialData || { nodes: [], links: [] });
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [selectedLink, setSelectedLink] = useState<GraphLink | null>(null);
  const [showNodeDetails, setShowNodeDetails] = useState<boolean>(false);
  const [showLinkDetails, setShowLinkDetails] = useState<boolean>(false);
  
  // Filters state
  const [filters, setFilters] = useState({
    fraudTypes: [] as string[],
    riskLevels: ['low', 'medium', 'high', 'critical'],
    minRiskScore: 0,
    maxRiskScore: 1,
    showOnlyAnomalies: false
  });
  
  // Ref for the force graph instance
  const graphRef = useRef<ForceGraphMethods>();
  
  // Color scale for risk scores (green to red)
  const riskColorScale = scaleLinear<string>()
    .domain([0, 0.3, 0.7, 1])
    .range(['#00cc00', '#ffcc00', '#ff6600', '#cc0000'])
    .clamp(true);
  
  // Fetch graph data from API
  const fetchGraphData = useCallback(async () => {
    if (!entityId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Fetch node data
      const nodeResponse = await axios.get(`/api/v1/graph/entity/${entityType}/${entityId}`);
      
      // Fetch anomaly data
      const anomalyResponse = await axios.get(`/api/v1/anomaly/entities/${entityType}/${entityId}/results`);
      
      // Process node data
      const nodes: GraphNode[] = nodeResponse.data.nodes.map((node: any) => ({
        id: node.id,
        address: node.address || node.id,
        label: node.label || node.address || node.id,
        type: node.type || 'unknown',
        risk_score: node.risk_score || 0,
        fraud_risk: node.fraud_risk || 'unknown',
        anomaly_count: 0, // Will be updated with anomaly data
        size: 5 + (node.importance || 1) * 3,
        metadata: { ...node }
      }));
      
      // Process link data
      const links: GraphLink[] = nodeResponse.data.relationships.map((rel: any) => ({
        source: rel.source,
        target: rel.target,
        type: rel.type || 'unknown',
        fraud_type: rel.fraud_type || null,
        value: rel.value || 1,
        hash: rel.hash || null,
        timestamp: rel.timestamp || null,
        metadata: { ...rel }
      }));
      
      // Update nodes with anomaly data
      if (anomalyResponse.data && Array.isArray(anomalyResponse.data)) {
        const anomalyMap = new Map<string, number>();
        
        anomalyResponse.data.forEach((anomaly: any) => {
          const entityId = anomaly.entity_id;
          anomalyMap.set(entityId, (anomalyMap.get(entityId) || 0) + 1);
        });
        
        nodes.forEach(node => {
          node.anomaly_count = anomalyMap.get(node.id) || 0;
        });
      }
      
      // Apply colors based on risk scores and anomalies
      nodes.forEach(node => {
        // Determine base color from risk score
        let baseColor: string;
        
        if (node.fraud_risk === 'high' || node.fraud_risk === 'critical') {
          baseColor = riskColorScale(0.9);
        } else if (node.fraud_risk === 'medium') {
          baseColor = riskColorScale(0.5);
        } else if (node.fraud_risk === 'low') {
          baseColor = riskColorScale(0.2);
        } else {
          // Use risk_score if available, otherwise default to low risk
          baseColor = riskColorScale(node.risk_score || 0.1);
        }
        
        node.color = baseColor;
      });
      
      // Apply styles to links based on fraud type
      links.forEach(link => {
        if (link.fraud_type === 'wash_trading') {
          link.color = '#ff3366';
          link.width = 2;
          link.dashed = true;
        } else if (link.fraud_type === 'smurfing') {
          link.color = '#9933cc';
          link.width = 1.5;
        } else if (link.fraud_type === 'layering') {
          link.color = '#ff9900';
          link.width = 2;
        } else if (link.fraud_type === 'round_amount') {
          link.color = '#33ccff';
          link.width = 1.5;
        } else if (link.fraud_type === 'high_frequency') {
          link.color = '#cc66ff';
          link.width = 1;
          link.dashed = true;
        } else {
          link.color = '#999999';
          link.width = 1;
        }
      });
      
      setGraphData({ nodes, links });
    } catch (err) {
      console.error('Error fetching graph data:', err);
      setError('Failed to fetch graph data. Please try again.');
    } finally {
      setLoading(false);
    }
  }, [entityId, entityType, riskColorScale]);
  
  // Fetch data on component mount or when entityId changes
  useEffect(() => {
    if (autoFetch && entityId) {
      fetchGraphData();
    }
  }, [autoFetch, entityId, fetchGraphData]);
  
  // Apply filters to graph data
  const filteredGraphData = useCallback(() => {
    if (!graphData) return { nodes: [], links: [] };
    
    // Filter nodes
    const filteredNodes = graphData.nodes.filter(node => {
      // Filter by risk level
      if (node.fraud_risk && !filters.riskLevels.includes(node.fraud_risk)) {
        return false;
      }
      
      // Filter by risk score
      if (node.risk_score !== undefined && 
          (node.risk_score < filters.minRiskScore || node.risk_score > filters.maxRiskScore)) {
        return false;
      }
      
      // Filter for anomalies only
      if (filters.showOnlyAnomalies && (!node.anomaly_count || node.anomaly_count === 0)) {
        return false;
      }
      
      return true;
    });
    
    // Get filtered node IDs for link filtering
    const filteredNodeIds = new Set(filteredNodes.map(node => node.id));
    
    // Filter links
    const filteredLinks = graphData.links.filter(link => {
      // Filter by fraud type
      if (filters.fraudTypes.length > 0 && 
          (!link.fraud_type || !filters.fraudTypes.includes(link.fraud_type))) {
        return false;
      }
      
      // Ensure both source and target nodes are in the filtered nodes
      const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
      const targetId = typeof link.target === 'string' ? link.target : link.target.id;
      
      return filteredNodeIds.has(sourceId) && filteredNodeIds.has(targetId);
    });
    
    return { nodes: filteredNodes, links: filteredLinks };
  }, [graphData, filters]);
  
  // Handle node click
  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNode(node);
    setShowNodeDetails(true);
    if (onNodeClick) onNodeClick(node);
  }, [onNodeClick]);
  
  // Handle link click
  const handleLinkClick = useCallback((link: GraphLink) => {
    setSelectedLink(link);
    setShowLinkDetails(true);
    if (onLinkClick) onLinkClick(link);
  }, [onLinkClick]);
  
  // Handle zoom in
  const handleZoomIn = useCallback(() => {
    if (graphRef.current) {
      const currentZoom = graphRef.current.zoom();
      graphRef.current.zoom(currentZoom * 1.5, 400); // Zoom in by 50% with animation
    }
  }, []);
  
  // Handle zoom out
  const handleZoomOut = useCallback(() => {
    if (graphRef.current) {
      const currentZoom = graphRef.current.zoom();
      graphRef.current.zoom(currentZoom / 1.5, 400); // Zoom out by 33% with animation
    }
  }, []);
  
  // Handle center view
  const handleCenterView = useCallback(() => {
    if (graphRef.current) {
      graphRef.current.centerAt(0, 0, 1000);
      graphRef.current.zoom(1, 1000);
    }
  }, []);
  
  // Handle filter changes
  const handleFraudTypeFilterChange = useCallback((values: string[]) => {
    setFilters(prev => ({ ...prev, fraudTypes: values }));
  }, []);
  
  const handleRiskLevelFilterChange = useCallback((values: string[]) => {
    setFilters(prev => ({ ...prev, riskLevels: values }));
  }, []);
  
  const handleRiskScoreFilterChange = useCallback((values: [number, number]) => {
    setFilters(prev => ({ ...prev, minRiskScore: values[0], maxRiskScore: values[1] }));
  }, []);
  
  const handleAnomalyFilterChange = useCallback((checked: boolean) => {
    setFilters(prev => ({ ...prev, showOnlyAnomalies: checked }));
  }, []);
  
  // Custom node rendering function
  const nodeCanvasObject = useCallback((node: GraphNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const { x, y, color, size = 5, label, anomaly_count = 0 } = node;
    if (x === undefined || y === undefined) return;
    
    const fontSize = 12 / globalScale;
    const nodeSize = size / globalScale;
    
    // Draw node circle
    ctx.beginPath();
    ctx.arc(x, y, nodeSize, 0, 2 * Math.PI);
    ctx.fillStyle = color || '#888888';
    ctx.fill();
    
    // Draw border for nodes with anomalies
    if (anomaly_count > 0) {
      const borderWidth = 2 / globalScale;
      ctx.strokeStyle = '#ff0000';
      ctx.lineWidth = borderWidth;
      ctx.stroke();
      
      // Draw small badge with anomaly count
      const badgeSize = nodeSize * 0.7;
      ctx.beginPath();
      ctx.arc(x + nodeSize, y - nodeSize, badgeSize, 0, 2 * Math.PI);
      ctx.fillStyle = '#ff0000';
      ctx.fill();
      
      // Draw anomaly count text
      ctx.font = `${fontSize}px Sans-Serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#ffffff';
      ctx.fillText(anomaly_count.toString(), x + nodeSize, y - nodeSize);
    }
    
    // Draw label if zoom level is sufficient
    if (globalScale > 0.8 && label) {
      ctx.font = `${fontSize}px Sans-Serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label.substring(0, 10), x, y);
    }
  }, []);
  
  // Custom link rendering function
  const linkCanvasObject = useCallback((link: GraphLink, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const source = typeof link.source === 'object' ? link.source : { x: 0, y: 0 };
    const target = typeof link.target === 'object' ? link.target : { x: 0, y: 0 };
    
    if (!source.x || !source.y || !target.x || !target.y) return;
    
    const { color = '#999999', width = 1, dashed = false } = link;
    
    // Draw link line
    ctx.beginPath();
    ctx.moveTo(source.x!, source.y!);
    ctx.lineTo(target.x!, target.y!);
    ctx.strokeStyle = color;
    ctx.lineWidth = width / globalScale;
    
    // Apply dashed line style for certain fraud types
    if (dashed) {
      ctx.setLineDash([5 / globalScale, 5 / globalScale]);
    } else {
      ctx.setLineDash([]);
    }
    
    ctx.stroke();
    
    // Reset line dash
    ctx.setLineDash([]);
    
    // Draw arrow for direction
    const dx = target.x! - source.x!;
    const dy = target.y! - source.y!;
    const angle = Math.atan2(dy, dx);
    
    const arrowLength = 10 / globalScale;
    const arrowWidth = 5 / globalScale;
    
    // Calculate position for arrow (slightly before target)
    const targetRadius = 5 / globalScale; // Approximate node radius
    const dist = Math.sqrt(dx * dx + dy * dy);
    const offsetRatio = (dist - targetRadius) / dist;
    
    const arrowX = source.x! + dx * offsetRatio;
    const arrowY = source.y! + dy * offsetRatio;
    
    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(
      arrowX - arrowLength * Math.cos(angle - Math.PI / 6),
      arrowY - arrowLength * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      arrowX - arrowLength * Math.cos(angle + Math.PI / 6),
      arrowY - arrowLength * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }, []);
  
  // Render node details modal
  const renderNodeDetailsModal = () => {
    if (!selectedNode) return null;
    
    return (
      <Modal
        title={`Node Details: ${selectedNode.label || selectedNode.id}`}
        open={showNodeDetails}
        onCancel={() => setShowNodeDetails(false)}
        footer={[
          <Button key="close" onClick={() => setShowNodeDetails(false)}>
            Close
          </Button>,
          <Button 
            key="investigate" 
            type="primary"
            onClick={() => {
              // Navigate to investigation page or open investigation panel
              window.open(`/investigation/${selectedNode.id}`, '_blank');
            }}
          >
            Investigate
          </Button>
        ]}
        width={600}
      >
        <Card>
          <div style={{ marginBottom: 16 }}>
            <Tag color={selectedNode.color}>
              Risk Level: {selectedNode.fraud_risk || 'Unknown'}
            </Tag>
            {selectedNode.anomaly_count && selectedNode.anomaly_count > 0 && (
              <Tag color="red">
                <WarningOutlined /> {selectedNode.anomaly_count} Anomalies Detected
              </Tag>
            )}
          </div>
          
          <h4>Basic Information</h4>
          <table className="details-table">
            <tbody>
              <tr>
                <td><strong>ID:</strong></td>
                <td>{selectedNode.id}</td>
              </tr>
              <tr>
                <td><strong>Address:</strong></td>
                <td>{selectedNode.address || 'N/A'}</td>
              </tr>
              <tr>
                <td><strong>Type:</strong></td>
                <td>{selectedNode.type || 'Unknown'}</td>
              </tr>
              <tr>
                <td><strong>Risk Score:</strong></td>
                <td>{selectedNode.risk_score !== undefined ? (selectedNode.risk_score * 100).toFixed(2) + '%' : 'N/A'}</td>
              </tr>
            </tbody>
          </table>
          
          {selectedNode.anomaly_count && selectedNode.anomaly_count > 0 && (
            <>
              <h4>Detected Anomalies</h4>
              <Button type="primary" size="small" onClick={() => window.open(`/anomalies/${selectedNode.id}`, '_blank')}>
                View All Anomalies
              </Button>
            </>
          )}
          
          {selectedNode.metadata && (
            <>
              <h4>Additional Properties</h4>
              <pre style={{ maxHeight: 200, overflow: 'auto' }}>
                {JSON.stringify(selectedNode.metadata, null, 2)}
              </pre>
            </>
          )}
        </Card>
      </Modal>
    );
  };
  
  // Render link details modal
  const renderLinkDetailsModal = () => {
    if (!selectedLink) return null;
    
    const source = typeof selectedLink.source === 'object' ? selectedLink.source : { id: selectedLink.source };
    const target = typeof selectedLink.target === 'object' ? selectedLink.target : { id: selectedLink.target };
    
    return (
      <Modal
        title={`Transaction Details: ${selectedLink.hash || 'Unknown'}`}
        open={showLinkDetails}
        onCancel={() => setShowLinkDetails(false)}
        footer={[
          <Button key="close" onClick={() => setShowLinkDetails(false)}>
            Close
          </Button>,
          selectedLink.hash && (
            <Button 
              key="explorer" 
              type="primary"
              onClick={() => {
                // Open transaction in blockchain explorer
                window.open(`https://etherscan.io/tx/${selectedLink.hash}`, '_blank');
              }}
            >
              View in Explorer
            </Button>
          )
        ]}
        width={600}
      >
        <Card>
          {selectedLink.fraud_type && (
            <div style={{ marginBottom: 16 }}>
              <Tag color={selectedLink.color}>
                Fraud Type: {selectedLink.fraud_type}
              </Tag>
            </div>
          )}
          
          <h4>Transaction Information</h4>
          <table className="details-table">
            <tbody>
              <tr>
                <td><strong>From:</strong></td>
                <td>{source.id}</td>
              </tr>
              <tr>
                <td><strong>To:</strong></td>
                <td>{target.id}</td>
              </tr>
              <tr>
                <td><strong>Type:</strong></td>
                <td>{selectedLink.type || 'Unknown'}</td>
              </tr>
              <tr>
                <td><strong>Value:</strong></td>
                <td>{selectedLink.value !== undefined ? selectedLink.value + ' ETH' : 'N/A'}</td>
              </tr>
              <tr>
                <td><strong>Timestamp:</strong></td>
                <td>{selectedLink.timestamp || 'N/A'}</td>
              </tr>
              <tr>
                <td><strong>Hash:</strong></td>
                <td>{selectedLink.hash || 'N/A'}</td>
              </tr>
            </tbody>
          </table>
          
          {selectedLink.metadata && (
            <>
              <h4>Additional Properties</h4>
              <pre style={{ maxHeight: 200, overflow: 'auto' }}>
                {JSON.stringify(selectedLink.metadata, null, 2)}
              </pre>
            </>
          )}
        </Card>
      </Modal>
    );
  };
  
  // Render filter controls
  const renderFilterControls = () => (
    <div className="fraud-visualization-filters">
      <Card size="small" title={<><FilterOutlined /> Filters</>} style={{ marginBottom: 16 }}>
        <div style={{ marginBottom: 12 }}>
          <label>Fraud Types:</label>
          <Select
            mode="multiple"
            style={{ width: '100%' }}
            placeholder="Filter by fraud type"
            value={filters.fraudTypes}
            onChange={handleFraudTypeFilterChange}
            options={[
              { value: 'wash_trading', label: 'Wash Trading' },
              { value: 'smurfing', label: 'Smurfing' },
              { value: 'layering', label: 'Layering' },
              { value: 'round_amount', label: 'Round Amount' },
              { value: 'high_frequency', label: 'High Frequency' }
            ]}
          />
        </div>
        
        <div style={{ marginBottom: 12 }}>
          <label>Risk Levels:</label>
          <Select
            mode="multiple"
            style={{ width: '100%' }}
            placeholder="Filter by risk level"
            value={filters.riskLevels}
            onChange={handleRiskLevelFilterChange}
            options={[
              { value: 'low', label: 'Low Risk' },
              { value: 'medium', label: 'Medium Risk' },
              { value: 'high', label: 'High Risk' },
              { value: 'critical', label: 'Critical Risk' }
            ]}
          />
        </div>
        
        <div style={{ marginBottom: 12 }}>
          <label>Risk Score Range:</label>
          <Slider
            range
            min={0}
            max={1}
            step={0.05}
            value={[filters.minRiskScore, filters.maxRiskScore]}
            onChange={handleRiskScoreFilterChange}
            marks={{
              0: '0%',
              0.25: '25%',
              0.5: '50%',
              0.75: '75%',
              1: '100%'
            }}
          />
        </div>
        
        <div>
          <Switch
            checked={filters.showOnlyAnomalies}
            onChange={handleAnomalyFilterChange}
          />
          <span style={{ marginLeft: 8 }}>Show only nodes with anomalies</span>
        </div>
      </Card>
      
      <Card size="small" title="Legend" style={{ marginBottom: 16 }}>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: riskColorScale(0.1) }}></div>
          <span>Low Risk</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: riskColorScale(0.5) }}></div>
          <span>Medium Risk</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: riskColorScale(0.8) }}></div>
          <span>High Risk</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: riskColorScale(1.0) }}></div>
          <span>Critical Risk</span>
        </div>
        
        <hr style={{ margin: '8px 0' }} />
        
        <div className="legend-item">
          <div className="legend-line" style={{ backgroundColor: '#ff3366', height: 2 }}></div>
          <span>Wash Trading</span>
        </div>
        <div className="legend-item">
          <div className="legend-line" style={{ backgroundColor: '#9933cc', height: 2 }}></div>
          <span>Smurfing</span>
        </div>
        <div className="legend-item">
          <div className="legend-line" style={{ backgroundColor: '#ff9900', height: 2 }}></div>
          <span>Layering</span>
        </div>
        <div className="legend-item">
          <div className="legend-line" style={{ backgroundColor: '#33ccff', height: 2 }}></div>
          <span>Round Amount</span>
        </div>
        <div className="legend-item">
          <div className="legend-line" style={{ backgroundColor: '#cc66ff', height: 2 }}></div>
          <span>High Frequency</span>
        </div>
      </Card>
      
      <div className="fraud-visualization-controls">
        <Button icon={<ZoomInOutlined />} onClick={handleZoomIn} title="Zoom In" />
        <Button icon={<ZoomOutOutlined />} onClick={handleZoomOut} title="Zoom Out" />
        <Button icon={<ReloadOutlined />} onClick={handleCenterView} title="Reset View" />
        <Button icon={<SaveOutlined />} onClick={() => {
          // Export current graph as PNG
          const canvas = document.querySelector('canvas');
          if (canvas) {
            const link = document.createElement('a');
            link.download = `fraud-graph-${entityId || 'export'}.png`;
            link.href = canvas.toDataURL('image/png');
            link.click();
          }
        }} title="Export as PNG" />
      </div>
    </div>
  );
  
  // Main render function
  return (
    <div className={`fraud-visualization-container ${className || ''}`} style={style}>
      <div className="fraud-visualization-layout">
        <div className="fraud-visualization-sidebar">
          {renderFilterControls()}
        </div>
        
        <div className="fraud-visualization-graph" style={{ height, width: '100%' }}>
          {loading ? (
            <div className="loading-container">
              <Spin size="large" tip="Loading graph data..." />
            </div>
          ) : error ? (
            <div className="error-container">
              <h3>Error Loading Graph</h3>
              <p>{error}</p>
              <Button type="primary" onClick={fetchGraphData}>
                Retry
              </Button>
            </div>
          ) : (
            <ForceGraph2D
              ref={graphRef}
              graphData={filteredGraphData()}
              nodeLabel={node => `${node.label || node.id} (Risk: ${node.fraud_risk || 'Unknown'})`}
              linkLabel={link => {
                const fraudType = link.fraud_type ? `Fraud Type: ${link.fraud_type}` : '';
                const value = link.value !== undefined ? `Value: ${link.value} ETH` : '';
                return [fraudType, value].filter(Boolean).join('\n');
              }}
              nodeCanvasObject={nodeCanvasObject}
              linkCanvasObject={linkCanvasObject}
              onNodeClick={handleNodeClick}
              onLinkClick={handleLinkClick}
              nodeRelSize={6}
              linkWidth={1}
              linkDirectionalArrowLength={3}
              linkDirectionalArrowRelPos={0.9}
              linkDirectionalParticles={link => link.fraud_type ? 4 : 0}
              linkDirectionalParticleSpeed={link => link.fraud_type === 'high_frequency' ? 0.02 : 0.005}
              cooldownTime={3000}
              d3AlphaDecay={0.02}
              d3VelocityDecay={0.3}
              width={width}
              height={height}
            />
          )}
        </div>
      </div>
      
      {renderNodeDetailsModal()}
      {renderLinkDetailsModal()}
      
      <style jsx>{`
        .fraud-visualization-container {
          display: flex;
          flex-direction: column;
          height: 100%;
        }
        
        .fraud-visualization-layout {
          display: flex;
          flex: 1;
        }
        
        .fraud-visualization-sidebar {
          width: 250px;
          padding: 16px;
          overflow-y: auto;
          background-color: #f5f5f5;
          border-right: 1px solid #e8e8e8;
        }
        
        .fraud-visualization-graph {
          flex: 1;
          position: relative;
          overflow: hidden;
        }
        
        .fraud-visualization-controls {
          display: flex;
          gap: 8px;
          margin-top: 16px;
        }
        
        .loading-container,
        .error-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
          width: 100%;
        }
        
        .legend-item {
          display: flex;
          align-items: center;
          margin-bottom: 4px;
        }
        
        .legend-color {
          width: 16px;
          height: 16px;
          border-radius: 50%;
          margin-right: 8px;
        }
        
        .legend-line {
          width: 24px;
          height: 2px;
          margin-right: 8px;
        }
        
        .details-table {
          width: 100%;
          border-collapse: collapse;
        }
        
        .details-table td {
          padding: 4px 8px;
          border-bottom: 1px solid #f0f0f0;
        }
        
        .details-table td:first-child {
          width: 120px;
        }
      `}</style>
    </div>
  );
};

export default FraudVisualization;
