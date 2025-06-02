'use client';

import { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import { PlusCircle, Search, ListFilter, RefreshCw, AlertTriangle, CheckCircle, Loader2, ExternalLink } from 'lucide-react';
import { useRouter } from 'next/navigation'; // For navigation after creating a task
// Assuming an API function to run a crew and list tasks will exist
// For now, we'll use mock data and a placeholder for runCrew
import { runCrew, /* listAnalysisTasks */ } from '../../lib/api'; 

type AnalysisStatus = 'running' | 'completed' | 'failed' | 'pending';

interface AnalysisTaskSummary {
  id: string;
  title: string;
  shortDescription?: string;
  status: AnalysisStatus;
  createdAt: string; // ISO string
  updatedAt: string; // ISO string
  crewName?: string;
  resultSummary?: string; // e.g., "3 patterns found", "Risk: High"
}

// Mock data for now
const MOCK_ANALYSIS_TASKS: AnalysisTaskSummary[] = [
  { id: 'task_001', title: 'Q1 Financial Anomaly Detection', shortDescription: 'Analyze transaction data for Q1 to find anomalies.', status: 'completed', createdAt: new Date(Date.now() - 86400000 * 2).toISOString(), updatedAt: new Date(Date.now() - 86400000 * 2).toISOString(), crewName: 'fraud_investigation', resultSummary: 'Risk: Medium, 2 patterns found' },
  { id: 'task_002', title: 'Crypto Wallet Tracing - Case #C456', shortDescription: 'Trace funds from suspicious wallet 0xABC...', status: 'running', createdAt: new Date(Date.now() - 3600000 * 5).toISOString(), updatedAt: new Date(Date.now() - 3600000 * 1).toISOString(), crewName: 'crypto_tracer' },
  { id: 'task_003', title: 'AML Policy Document Review', shortDescription: 'Review new AML policy updates for compliance gaps.', status: 'failed', createdAt: new Date(Date.now() - 86400000 * 5).toISOString(), updatedAt: new Date(Date.now() - 86400000 * 5 + 3600000).toISOString(), crewName: 'compliance_checker', resultSummary: 'Error: Document parsing failed' },
  { id: 'task_004', title: 'Market Sentiment Analysis - BTC', shortDescription: 'Analyze recent news and social media for BTC sentiment.', status: 'completed', createdAt: new Date(Date.now() - 86400000 * 1).toISOString(), updatedAt: new Date(Date.now() - 86400000 * 1).toISOString(), crewName: 'market_analyst', resultSummary: 'Sentiment: Neutral to Positive' },
  { id: 'task_005', title: 'Insider Trading Detection - Stock XYZ', shortDescription: 'Monitor trading patterns for stock XYZ for potential insider activity.', status: 'pending', createdAt: new Date(Date.now() - 3600000 * 1).toISOString(), updatedAt: new Date(Date.now() - 3600000 * 1).toISOString(), crewName: 'insider_trading_detector' },
];

const AnalysisOverviewPage = () => {
  const router = useRouter();
  const [tasks, setTasks] = useState<AnalysisTaskSummary[]>(MOCK_ANALYSIS_TASKS);
  const [isLoadingTasks, setIsLoadingTasks] = useState<boolean>(false);
  const [taskError, setTaskError] = useState<string | null>(null);

  const [searchTerm, setSearchTerm] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<AnalysisStatus | 'all'>('all');
  
  const [newAnalysisDescription, setNewAnalysisDescription] = useState<string>('');
  const [selectedCrew, setSelectedCrew] = useState<string>('fraud_investigation'); // Default crew
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Placeholder for available crews - would come from API
  const availableCrews = [
    { id: 'fraud_investigation', name: 'Fraud Investigation' },
    { id: 'crypto_tracer', name: 'Crypto Tracer' },
    { id: 'compliance_checker', name: 'Compliance Checker' },
    { id: 'market_analyst', name: 'Market Analyst' },
    { id: 'insider_trading_detector', name: 'Insider Trading Detector' },
    { id: 'generic_data_analyzer', name: 'Generic Data Analyzer' },
  ];

  const fetchTasks = async () => {
    setIsLoadingTasks(true);
    setTaskError(null);
    try {
      // Replace with actual API call: const fetchedTasks = await listAnalysisTasks({ filter: statusFilter, search: searchTerm });
      // For now, simulate API call with mock data
      await new Promise(resolve => setTimeout(resolve, 500)); 
      let filteredTasks = MOCK_ANALYSIS_TASKS;
      if (statusFilter !== 'all') {
        filteredTasks = filteredTasks.filter(task => task.status === statusFilter);
      }
      if (searchTerm) {
        filteredTasks = filteredTasks.filter(task => 
          task.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
          task.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
          (task.shortDescription && task.shortDescription.toLowerCase().includes(searchTerm.toLowerCase()))
        );
      }
      setTasks(filteredTasks);
    } catch (error) {
      console.error("Failed to fetch tasks:", error);
      setTaskError("Could not load analysis tasks. Please try again.");
    } finally {
      setIsLoadingTasks(false);
    }
  };

  useEffect(() => {
    fetchTasks();
  }, [statusFilter]); // Re-fetch when filter changes. Search is handled client-side after initial fetch or on demand.

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  };
  
  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetchTasks(); // Re-fetch with search term
  };

  const handleRunAnalysis = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newAnalysisDescription.trim()) {
      setSubmitError("Analysis description cannot be empty.");
      return;
    }
    setIsSubmitting(true);
    setSubmitError(null);
    try {
      const response = await runCrew({ 
        crew_name: selectedCrew, 
        inputs: { query: newAnalysisDescription, description: newAnalysisDescription } 
      });
      
      if (response.success && response.task_id) {
        // Add to tasks list optimistically or re-fetch
        const newTask: AnalysisTaskSummary = {
          id: response.task_id,
          title: newAnalysisDescription.substring(0, 50) + (newAnalysisDescription.length > 50 ? '...' : ''),
          shortDescription: newAnalysisDescription,
          status: 'pending', // Or 'running' if backend confirms immediately
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          crewName: selectedCrew,
        };
        setTasks(prevTasks => [newTask, ...prevTasks]);
        setNewAnalysisDescription(''); // Clear form
        // Optionally navigate to the new task's page or refresh list
        // router.push(`/analysis/${response.task_id}`); 
        fetchTasks(); // Refresh list to get latest status
      } else {
        throw new Error(response.error || "Failed to start analysis task.");
      }
    } catch (error: any) {
      console.error("Failed to run analysis:", error);
      setSubmitError(error.message || "An unexpected error occurred while submitting the analysis.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const filteredTasks = useMemo(() => {
    return tasks.filter(task => {
      const matchesSearch = searchTerm === '' || 
        task.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        task.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (task.shortDescription && task.shortDescription.toLowerCase().includes(searchTerm.toLowerCase()));
      const matchesStatus = statusFilter === 'all' || task.status === statusFilter;
      return matchesSearch && matchesStatus;
    });
  }, [tasks, searchTerm, statusFilter]);

  const getStatusIcon = (status: AnalysisStatus) => {
    switch (status) {
      case 'completed': return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'running': return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />;
      case 'failed': return <AlertTriangle className="h-5 w-5 text-red-500" />;
      case 'pending': return <RefreshCw className="h-5 w-5 text-yellow-500" />; // Using RefreshCw for pending
      default: return <Info className="h-5 w-5 text-gray-500" />;
    }
  };
  
  const getStatusColor = (status: AnalysisStatus) => {
    switch (status) {
      case 'completed': return 'border-green-500';
      case 'running': return 'border-blue-500';
      case 'failed': return 'border-red-500';
      case 'pending': return 'border-yellow-500';
      default: return 'border-gray-500';
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-4 md:p-8">
      <header className="mb-8">
        <h1 className="text-3xl md:text-4xl font-bold text-blue-400">Analysis Dashboard</h1>
        <p className="text-gray-400 mt-1">Manage and initiate your data analysis tasks.</p>
      </header>

      {/* Section to Run New Analysis */}
      <section className="mb-10 bg-gray-800 p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-semibold mb-4 text-blue-300 border-b border-gray-700 pb-2">
          <PlusCircle className="inline-block h-6 w-6 mr-2 align-text-bottom" />
          Run New Analysis
        </h2>
        <form onSubmit={handleRunAnalysis} className="space-y-4">
          <div>
            <label htmlFor="analysisDescription" className="block text-sm font-medium text-gray-300 mb-1">
              Analysis Description / Query
            </label>
            <textarea
              id="analysisDescription"
              value={newAnalysisDescription}
              onChange={(e) => setNewAnalysisDescription(e.target.value)}
              placeholder="e.g., 'Analyze transaction patterns for client X between Jan 1 and Mar 31'"
              rows={3}
              className="w-full p-2 bg-gray-700 border border-gray-600 rounded-md focus:ring-blue-500 focus:border-blue-500"
              disabled={isSubmitting}
            />
          </div>
          <div>
            <label htmlFor="crewSelect" className="block text-sm font-medium text-gray-300 mb-1">
              Select Analysis Crew/Type
            </label>
            <select
              id="crewSelect"
              value={selectedCrew}
              onChange={(e) => setSelectedCrew(e.target.value)}
              className="w-full p-2 bg-gray-700 border border-gray-600 rounded-md focus:ring-blue-500 focus:border-blue-500"
              disabled={isSubmitting}
            >
              {availableCrews.map(crew => (
                <option key={crew.id} value={crew.id}>{crew.name}</option>
              ))}
            </select>
          </div>
          {submitError && <p className="text-sm text-red-400">{submitError}</p>}
          <button
            type="submit"
            disabled={isSubmitting || !newAnalysisDescription.trim()}
            className="w-full md:w-auto px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-md shadow-md transition-colors duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {isSubmitting ? (
              <Loader2 className="h-5 w-5 mr-2 animate-spin" />
            ) : (
              <PlusCircle className="h-5 w-5 mr-2" />
            )}
            {isSubmitting ? 'Submitting...' : 'Run Analysis'}
          </button>
        </form>
      </section>

      {/* Section for Analysis Tasks List */}
      <section className="bg-gray-800 p-6 rounded-lg shadow-lg">
        <div className="flex flex-col md:flex-row justify-between items-center mb-6 gap-4">
          <h2 className="text-2xl font-semibold text-blue-300">
            <ListFilter className="inline-block h-6 w-6 mr-2 align-text-bottom" />
            Analysis Tasks
          </h2>
          <div className="flex flex-col sm:flex-row gap-2 w-full md:w-auto">
            <form onSubmit={handleSearchSubmit} className="flex-grow sm:flex-grow-0">
              <div className="relative">
                <input
                  type="text"
                  placeholder="Search tasks by ID, title..."
                  value={searchTerm}
                  onChange={handleSearchChange}
                  className="w-full sm:w-64 p-2 pl-10 bg-gray-700 border border-gray-600 rounded-md focus:ring-blue-500 focus:border-blue-500"
                />
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              </div>
            </form>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as AnalysisStatus | 'all')}
              className="p-2 bg-gray-700 border border-gray-600 rounded-md focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="all">All Statuses</option>
              <option value="pending">Pending</option>
              <option value="running">Running</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
            </select>
            <button
                onClick={fetchTasks}
                disabled={isLoadingTasks}
                className="p-2 bg-gray-700 hover:bg-gray-600 border border-gray-600 rounded-md transition-colors flex items-center justify-center"
                title="Refresh Tasks"
              >
                <RefreshCw className={`h-5 w-5 ${isLoadingTasks ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {isLoadingTasks && (
          <div className="flex justify-center items-center py-10">
            <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
            <p className="ml-2 text-gray-300">Loading tasks...</p>
          </div>
        )}
        {taskError && !isLoadingTasks && (
          <div className="text-center py-10">
            <AlertTriangle className="h-10 w-10 text-red-500 mx-auto mb-2" />
            <p className="text-red-400">{taskError}</p>
          </div>
        )}
        {!isLoadingTasks && !taskError && filteredTasks.length === 0 && (
          <div className="text-center py-10">
            <Info className="h-10 w-10 text-gray-500 mx-auto mb-2" />
            <p className="text-gray-400">No analysis tasks found matching your criteria.</p>
          </div>
        )}

        {!isLoadingTasks && !taskError && filteredTasks.length > 0 && (
          <div className="space-y-4">
            {filteredTasks.map(task => (
              <div
                key={task.id}
                className={`bg-gray-750 p-4 rounded-md shadow-md border-l-4 ${getStatusColor(task.status)} hover:shadow-blue-500/30 transition-shadow duration-200`}
              >
                <div className="flex flex-col sm:flex-row justify-between items-start">
                  <div>
                    <Link href={`/analysis/${task.id}`} legacyBehavior>
                      <a className="text-lg font-semibold text-blue-400 hover:underline hover:text-blue-300 break-all">
                        {task.title || `Analysis Task ${task.id}`}
                      </a>
                    </Link>
                    <p className="text-xs text-gray-500 font-mono mt-0.5">ID: {task.id}</p>
                    {task.shortDescription && <p className="text-sm text-gray-300 mt-1 truncate max-w-md">{task.shortDescription}</p>}
                  </div>
                  <div className="flex items-center space-x-3 mt-2 sm:mt-0 text-sm text-gray-400 flex-shrink-0">
                    {getStatusIcon(task.status)}
                    <span className="capitalize">{task.status}</span>
                  </div>
                </div>
                <div className="mt-3 pt-3 border-t border-gray-700 flex flex-col sm:flex-row justify-between items-start sm:items-center text-xs text-gray-500">
                  <div className="space-y-1 sm:space-y-0">
                    {task.crewName && <p>Type: <span className="text-gray-400">{task.crewName}</span></p>}
                    <p>Created: <span className="text-gray-400">{new Date(task.createdAt).toLocaleString()}</span></p>
                    <p>Updated: <span className="text-gray-400">{new Date(task.updatedAt).toLocaleString()}</span></p>
                  </div>
                  <Link href={`/analysis/${task.id}`} legacyBehavior>
                    <a className="mt-2 sm:mt-0 inline-flex items-center px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium rounded-md transition-colors">
                      View Details <ExternalLink className="h-3 w-3 ml-1.5" />
                    </a>
                  </Link>
                </div>
                {task.resultSummary && (
                    <p className="mt-2 text-xs text-gray-400 bg-gray-700 p-2 rounded-md">
                        <strong>Result Summary:</strong> {task.resultSummary}
                    </p>
                )}
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};

export default AnalysisOverviewPage;
