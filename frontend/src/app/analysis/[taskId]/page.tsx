'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import { BarChart, CheckCircle, Download, AlertTriangle, Info, FileText, Share2, Loader2, ExternalLink } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Ensuring GraphVisualization import is correct for a default export
import GraphVisualization from '../../../components/graph/GraphVisualization'; 
// Ensuring fetchAnalysisResults import is correct for a named export
import { fetchAnalysisResults } from '../../../lib/api'; 

interface AnalysisNode {
  id: string;
  label: string;
  type?: string;
  properties?: Record<string, any>;
  risk_score?: number;
  size?: number;
  color?: string;
  [key: string]: any;
}

interface AnalysisEdge {
  from: string;
  to: string;
  label?: string;
  properties?: Record<string, any>;
  weight?: number;
  color?: string;
  [key: string]: any;
}
interface GraphData {
  nodes: AnalysisNode[];
  edges: AnalysisEdge[];
}

interface GeneratedVisualization {
  filename: string;
  content: string; // base64 encoded image data
  type: 'image/png' | 'image/jpeg' | 'image/svg+xml' | 'text/html'; // Mime type
}

interface AnalysisData {
  task_id: string;
  status: string;
  title?: string;
  executive_summary?: string;
  risk_score?: number;
  confidence?: number;
  detailed_findings?: string; // Markdown or plain text
  graph_data?: GraphData;
  visualizations?: GeneratedVisualization[];
  recommendations?: string[];
  code_generated?: string;
  execution_details?: any; // Could be more specific
  error?: string;
  // CrewAI specific fields if available
  crew_name?: string;
  crew_inputs?: Record<string, any>;
  crew_result?: any; 
}

const AnalysisResultsPage = () => {
  const params = useParams();
  const taskId = params.taskId as string;

  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (taskId) {
      setLoading(true);
      setError(null);
      fetchAnalysisResults(taskId)
        .then((data: AnalysisData) => {
          // Ensure nested objects are initialized if not present
          setAnalysisData({
            ...data,
            graph_data: data.graph_data || { nodes: [], edges: [] },
            visualizations: data.visualizations || [],
            recommendations: data.recommendations || [],
          });
        })
        .catch((err) => {
          console.error('Error fetching analysis results:', err);
          setError(err.message || 'Failed to load analysis results.');
          setAnalysisData(null); // Clear data on error
        })
        .finally(() => {
          setLoading(false);
        });
    }
  }, [taskId]);

  const handleExportJSON = () => {
    if (!analysisData) return;
    const jsonString = JSON.stringify(analysisData, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis_results_${taskId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleExportPDF = () => {
    // Placeholder for PDF export functionality
    // This would typically use a library like jsPDF and html2canvas
    // or a server-side PDF generation service.
    alert('PDF export functionality is not yet implemented.');
    console.log('Attempting to print to PDF via browser');
    window.print();
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-gray-100 p-4">
        <Loader2 className="h-16 w-16 animate-spin text-blue-500 mb-4" />
        <p className="text-xl">Loading Analysis Results for Task ID: {taskId}...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-gray-100 p-4">
        <AlertTriangle className="h-16 w-16 text-red-500 mb-4" />
        <h2 className="text-2xl font-semibold mb-2">Error Loading Analysis</h2>
        <p className="text-red-400 mb-4">{error}</p>
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-md text-white transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!analysisData) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-gray-100 p-4">
        <Info className="h-16 w-16 text-yellow-500 mb-4" />
        <p className="text-xl">No analysis data found for Task ID: {taskId}.</p>
         <p className="text-sm text-gray-400">The task might still be processing or encountered an issue.</p>
      </div>
    );
  }
  
  const riskScoreColor = (score?: number) => {
    if (score === undefined) return 'text-gray-400';
    if (score >= 0.75) return 'text-red-500';
    if (score >= 0.5) return 'text-yellow-500';
    if (score >= 0.25) return 'text-blue-500';
    return 'text-green-500';
  };
  
  const confidenceColor = (score?: number) => {
    if (score === undefined) return 'text-gray-400';
    if (score >= 0.8) return 'text-green-500';
    if (score >= 0.6) return 'text-blue-500';
    return 'text-yellow-500';
  };


  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-4 md:p-8">
      <header className="mb-8">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold text-blue-400 break-all">
              Analysis Report: {analysisData.title || `Task ${taskId}`}
            </h1>
            <p className="text-sm text-gray-400 mt-1">Task ID: <span className="font-mono">{taskId}</span></p>
            {analysisData.crew_name && <p className="text-sm text-gray-400">Crew: {analysisData.crew_name}</p>}
          </div>
          <div className="flex space-x-2 mt-4 md:mt-0">
            <button
              onClick={handleExportJSON}
              className="flex items-center px-4 py-2 bg-green-600 hover:bg-green-700 rounded-md text-white transition-colors text-sm"
            >
              <Download className="h-4 w-4 mr-2" /> Export JSON
            </button>
            <button
              onClick={handleExportPDF}
              className="flex items-center px-4 py-2 bg-red-600 hover:bg-red-700 rounded-md text-white transition-colors text-sm"
            >
              <FileText className="h-4 w-4 mr-2" /> Export PDF
            </button>
            {/* <button
              className="flex items-center px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-md text-white transition-colors text-sm"
            >
              <Share2 className="h-4 w-4 mr-2" /> Share
            </button> */}
          </div>
        </div>
         {analysisData.status && (
          <p className={`mt-2 text-sm ${analysisData.status === 'completed' ? 'text-green-400' : 'text-yellow-400'}`}>
            Status: {analysisData.status}
          </p>
        )}
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column / Main Column on smaller screens */}
        <main className="lg:col-span-2 space-y-6">
          {analysisData.executive_summary && (
            <section className="bg-gray-800 p-6 rounded-lg shadow-lg">
              <h2 className="text-2xl font-semibold mb-3 text-blue-300 border-b border-gray-700 pb-2">Executive Summary</h2>
              <div className="prose prose-sm prose-invert max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{analysisData.executive_summary}</ReactMarkdown>
              </div>
            </section>
          )}

          {analysisData.detailed_findings && (
            <section className="bg-gray-800 p-6 rounded-lg shadow-lg">
              <h2 className="text-2xl font-semibold mb-3 text-blue-300 border-b border-gray-700 pb-2">Detailed Findings</h2>
              <div className="prose prose-sm prose-invert max-w-none">
                 <ReactMarkdown remarkPlugins={[remarkGfm]}>{analysisData.detailed_findings}</ReactMarkdown>
              </div>
            </section>
          )}

          {analysisData.graph_data && (analysisData.graph_data.nodes.length > 0 || analysisData.graph_data.edges.length > 0) && (
            <section className="bg-gray-800 p-6 rounded-lg shadow-lg">
              <h2 className="text-2xl font-semibold mb-3 text-blue-300 border-b border-gray-700 pb-2">Graph Visualization</h2>
              <div className="h-[500px] md:h-[600px] w-full bg-gray-700 rounded">
                <GraphVisualization
                  graphData={analysisData.graph_data}
                  isLoading={loading}
                  error={error}
                  // Optional: Pass specific layout or interaction options
                />
              </div>
            </section>
          )}

          {analysisData.visualizations && analysisData.visualizations.length > 0 && (
            <section className="bg-gray-800 p-6 rounded-lg shadow-lg">
              <h2 className="text-2xl font-semibold mb-3 text-blue-300 border-b border-gray-700 pb-2">Generated Visualizations</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {analysisData.visualizations.map((viz, index) => (
                  <div key={index} className="border border-gray-700 p-3 rounded-md">
                    <h3 className="text-lg font-medium mb-2 text-gray-300">{viz.filename}</h3>
                    {viz.type.startsWith('image/') ? (
                      <img
                        src={`data:${viz.type};base64,${viz.content}`}
                        alt={viz.filename}
                        className="max-w-full h-auto rounded-md"
                      />
                    ) : viz.type === 'text/html' ? (
                       <div className="w-full h-64 overflow-auto border border-gray-600 rounded">
                         <iframe 
                            srcDoc={viz.content} 
                            title={viz.filename} 
                            className="w-full h-full"
                            sandbox="allow-scripts allow-same-origin" // Be cautious with sandbox attributes
                         />
                       </div>
                    ) : (
                      <p className="text-sm text-gray-400">Unsupported visualization type: {viz.type}</p>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}
          
          {analysisData.code_generated && (
            <section className="bg-gray-800 p-6 rounded-lg shadow-lg">
              <h2 className="text-2xl font-semibold mb-3 text-blue-300 border-b border-gray-700 pb-2">Generated Code</h2>
              <div className="prose prose-sm prose-invert max-w-none bg-gray-700 p-4 rounded-md overflow-x-auto">
                <pre><code>{analysisData.code_generated}</code></pre>
              </div>
            </section>
          )}

        </main>

        {/* Right Sidebar / Second Column on smaller screens */}
        <aside className="lg:col-span-1 space-y-6">
          {(analysisData.risk_score !== undefined || analysisData.confidence !== undefined) && (
            <section className="bg-gray-800 p-6 rounded-lg shadow-lg">
              <h2 className="text-xl font-semibold mb-3 text-blue-300 border-b border-gray-700 pb-2">Risk Assessment</h2>
              <div className="space-y-3">
                {analysisData.risk_score !== undefined && (
                  <div>
                    <p className="text-sm text-gray-400">Overall Risk Score</p>
                    <p className={`text-3xl font-bold ${riskScoreColor(analysisData.risk_score)}`}>
                      {(analysisData.risk_score * 100).toFixed(1)}%
                    </p>
                  </div>
                )}
                {analysisData.confidence !== undefined && (
                   <div>
                    <p className="text-sm text-gray-400">Confidence Level</p>
                    <p className={`text-2xl font-semibold ${confidenceColor(analysisData.confidence)}`}>
                      {(analysisData.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                )}
              </div>
            </section>
          )}

          {analysisData.recommendations && analysisData.recommendations.length > 0 && (
            <section className="bg-gray-800 p-6 rounded-lg shadow-lg">
              <h2 className="text-xl font-semibold mb-3 text-blue-300 border-b border-gray-700 pb-2">Recommendations</h2>
              <ul className="space-y-2">
                {analysisData.recommendations.map((rec, index) => (
                  <li key={index} className="flex items-start text-sm">
                    <CheckCircle className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {analysisData.execution_details && (
             <section className="bg-gray-800 p-6 rounded-lg shadow-lg">
              <h2 className="text-xl font-semibold mb-3 text-blue-300 border-b border-gray-700 pb-2">Execution Details</h2>
              <div className="text-xs space-y-1 text-gray-400">
                {Object.entries(analysisData.execution_details).map(([key, value]) => (
                  <p key={key}>
                    <span className="font-semibold text-gray-300">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}: </span> 
                    <span className="font-mono break-all">{typeof value === 'object' ? JSON.stringify(value) : String(value)}</span>
                  </p>
                ))}
              </div>
            </section>
          )}
          
          {analysisData.crew_inputs && (
             <section className="bg-gray-800 p-6 rounded-lg shadow-lg">
              <h2 className="text-xl font-semibold mb-3 text-blue-300 border-b border-gray-700 pb-2">Crew Inputs</h2>
              <div className="text-xs space-y-1 text-gray-400">
                {Object.entries(analysisData.crew_inputs).map(([key, value]) => (
                  <p key={key}>
                    <span className="font-semibold text-gray-300">{key}: </span> 
                    <span className="font-mono break-all">{typeof value === 'object' ? JSON.stringify(value) : String(value)}</span>
                  </p>
                ))}
              </div>
            </section>
          )}

        </aside>
      </div>
    </div>
  );
};

export default AnalysisResultsPage;
