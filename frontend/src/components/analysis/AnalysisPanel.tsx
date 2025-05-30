'use client'

import { useState } from 'react'
import { useQuery, useMutation } from 'react-query'
import { analysisAPI, handleAPIError } from '@/lib/api'
import toast from 'react-hot-toast'
import { 
  ChartBarIcon, 
  ExclamationTriangleIcon,
  PlayIcon,
  DocumentTextIcon 
} from '@heroicons/react/24/outline'

export function AnalysisPanel() {
  const [analysisTask, setAnalysisTask] = useState('')
  const [codeToExecute, setCodeToExecute] = useState('')
  const [activeTab, setActiveTab] = useState<'analysis' | 'fraud' | 'code'>('analysis')

  // Fraud detection query
  const { data: fraudData, isLoading: fraudLoading, refetch: refetchFraud } = useQuery(
    'fraud-patterns',
    () => analysisAPI.detectFraudPatterns('money_laundering', 50),
    {
      enabled: false, // Don't auto-fetch
      onError: (error) => {
        const errorInfo = handleAPIError(error)
        toast.error(errorInfo.message)
      },
    }
  )

  // Analysis mutation
  const analysisMutation = useMutation(
    (task: string) => analysisAPI.performAnalysis(task, 'graph'),
    {
      onSuccess: (response) => {
        toast.success('Analysis completed successfully')
        console.log('Analysis results:', response.data)
      },
      onError: (error) => {
        const errorInfo = handleAPIError(error)
        toast.error(errorInfo.message)
      },
    }
  )

  // Code execution mutation
  const codeExecutionMutation = useMutation(
    (code: string) => analysisAPI.executeCode(code, ['pandas', 'numpy', 'matplotlib']),
    {
      onSuccess: (response) => {
        if (response.data.success) {
          toast.success('Code executed successfully')
        } else {
          toast.error('Code execution failed')
        }
        console.log('Execution results:', response.data)
      },
      onError: (error) => {
        const errorInfo = handleAPIError(error)
        toast.error(errorInfo.message)
      },
    }
  )

  const handleAnalysis = () => {
    if (!analysisTask.trim()) return
    analysisMutation.mutate(analysisTask)
  }

  const handleCodeExecution = () => {
    if (!codeToExecute.trim()) return
    codeExecutionMutation.mutate(codeToExecute)
  }

  const handleFraudDetection = () => {
    refetchFraud()
  }

  return (
    <div className="flex h-full bg-gray-50">
      {/* Left panel - Controls */}
      <div className="w-1/3 bg-white border-r border-gray-200 flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900 flex items-center">
            <ChartBarIcon className="h-6 w-6 mr-2" />
            Data Analysis
          </h2>
          <p className="text-sm text-gray-500 mt-1">
            AI-powered analysis and fraud detection
          </p>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6" aria-label="Tabs">
            {[
              { id: 'analysis', name: 'Analysis', icon: ChartBarIcon },
              { id: 'fraud', name: 'Fraud Detection', icon: ExclamationTriangleIcon },
              { id: 'code', name: 'Code Execution', icon: PlayIcon },
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
          {activeTab === 'analysis' && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Analysis Task Description
                </label>
                <textarea
                  value={analysisTask}
                  onChange={(e) => setAnalysisTask(e.target.value)}
                  placeholder="e.g., Analyze transaction patterns to identify potential money laundering schemes"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500"
                  rows={4}
                />
              </div>
              
              <button
                onClick={handleAnalysis}
                disabled={!analysisTask.trim() || analysisMutation.isLoading}
                className="w-full btn-primary"
              >
                {analysisMutation.isLoading ? 'Analyzing...' : 'Start Analysis'}
              </button>

              {/* Quick analysis templates */}
              <div className="mt-6">
                <h3 className="text-sm font-medium text-gray-700 mb-3">Quick Templates</h3>
                <div className="space-y-2">
                  {[
                    'Identify high-risk transaction patterns',
                    'Analyze network centrality for key actors',
                    'Detect circular money flows',
                    'Find suspicious account clustering',
                  ].map((template) => (
                    <button
                      key={template}
                      onClick={() => setAnalysisTask(template)}
                      className="w-full text-left px-3 py-2 text-sm bg-gray-50 hover:bg-gray-100 rounded border"
                    >
                      {template}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'fraud' && (
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">
                  Fraud Detection Patterns
                </h3>
                
                <button
                  onClick={handleFraudDetection}
                  disabled={fraudLoading}
                  className="w-full btn-primary mb-4"
                >
                  {fraudLoading ? 'Detecting...' : 'Detect Money Laundering Patterns'}
                </button>

                {fraudData?.data && (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <h4 className="font-medium text-yellow-800 mb-2">
                      Detection Results
                    </h4>
                    <p className="text-sm text-yellow-700">
                      Found {fraudData.data.patterns_found} potential patterns
                    </p>
                    {fraudData.data.explanation && (
                      <p className="text-sm text-yellow-700 mt-2">
                        {fraudData.data.explanation}
                      </p>
                    )}
                  </div>
                )}
              </div>

              {/* Fraud detection options */}
              <div className="space-y-3">
                <h4 className="text-sm font-medium text-gray-700">Detection Types</h4>
                {[
                  'Money Laundering Schemes',
                  'Circular Transactions',
                  'Suspicious Velocity',
                  'Account Takeover Patterns',
                ].map((type) => (
                  <button
                    key={type}
                    className="w-full text-left px-3 py-2 text-sm bg-gray-50 hover:bg-gray-100 rounded border"
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'code' && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Python Code
                </label>
                <textarea
                  value={codeToExecute}
                  onChange={(e) => setCodeToExecute(e.target.value)}
                  placeholder="import pandas as pd&#10;import numpy as np&#10;&#10;# Your analysis code here"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500 font-mono text-sm"
                  rows={8}
                />
              </div>
              
              <button
                onClick={handleCodeExecution}
                disabled={!codeToExecute.trim() || codeExecutionMutation.isLoading}
                className="w-full btn-primary"
              >
                {codeExecutionMutation.isLoading ? 'Executing...' : 'Execute Code'}
              </button>

              {/* Code templates */}
              <div className="mt-6">
                <h3 className="text-sm font-medium text-gray-700 mb-3">Code Templates</h3>
                <div className="space-y-2">
                  {[
                    'Basic data analysis',
                    'Graph metrics calculation',
                    'Visualization generation',
                    'Statistical analysis',
                  ].map((template) => (
                    <button
                      key={template}
                      className="w-full text-left px-3 py-2 text-sm bg-gray-50 hover:bg-gray-100 rounded border"
                    >
                      {template}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Right panel - Results */}
      <div className="flex-1 flex flex-col">
        {/* Results header */}
        <div className="p-6 border-b border-gray-200 bg-white">
          <h3 className="text-lg font-medium text-gray-900">Analysis Results</h3>
          <p className="text-sm text-gray-500">
            Results and visualizations will appear here
          </p>
        </div>

        {/* Results area */}
        <div className="flex-1 p-6">
          <div className="h-full bg-white rounded-lg border-2 border-dashed border-gray-300 flex items-center justify-center">
            <div className="text-center">
              <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">
                Analysis Results
              </h3>
              <p className="mt-1 text-sm text-gray-500">
                Run an analysis to see results and visualizations
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
