import React, { useState, useEffect, useCallback } from 'react';
import { listAgents, getAgentPrompt, updateAgentPrompt, resetAgentPrompt, AgentListItem, PromptResponse, PromptUpdate } from '../../lib/api';

// Simple toast notification component
const Toast = ({ message, type, onClose }: { message: string; type: 'success' | 'error'; onClose: () => void }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose();
    }, 5000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className={`fixed bottom-4 right-4 p-4 rounded-md shadow-lg z-50 ${type === 'success' ? 'bg-green-500' : 'bg-red-500'} text-white`}>
      <div className="flex items-center">
        <span>{message}</span>
        <button onClick={onClose} className="ml-4 text-white hover:text-gray-200">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
        </button>
      </div>
    </div>
  );
};

const PromptsManager: React.FC = () => {
  // State for agents list
  const [agents, setAgents] = useState<AgentListItem[]>([]);
  const [isLoadingAgents, setIsLoadingAgents] = useState(true);
  const [agentsError, setAgentsError] = useState<string | null>(null);

  // State for selected agent and prompt
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [promptData, setPromptData] = useState<PromptResponse | null>(null);
  const [isLoadingPrompt, setIsLoadingPrompt] = useState(false);
  const [promptError, setPromptError] = useState<string | null>(null);

  // State for prompt editing
  const [editedPrompt, setEditedPrompt] = useState<string>('');
  const [editedDescription, setEditedDescription] = useState<string>('');
  const [isSaving, setIsSaving] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  // Toast notification state
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);

  // Fetch agents list on component mount
  useEffect(() => {
    const fetchAgents = async () => {
      try {
        setIsLoadingAgents(true);
        setAgentsError(null);
        const response = await listAgents();
        setAgents(response.agents);
      } catch (error) {
        console.error('Error fetching agents:', error);
        setAgentsError('Failed to load agents. Please try again.');
      } finally {
        setIsLoadingAgents(false);
      }
    };

    fetchAgents();
  }, []);

  // Fetch selected agent prompt
  const fetchAgentPrompt = useCallback(async (agentId: string) => {
    try {
      setIsLoadingPrompt(true);
      setPromptError(null);
      const response = await getAgentPrompt(agentId);
      setPromptData(response);
      setEditedPrompt(response.system_prompt);
      setEditedDescription(response.description || '');
      setHasChanges(false);
    } catch (error) {
      console.error(`Error fetching prompt for agent ${agentId}:`, error);
      setPromptError('Failed to load agent prompt. Please try again.');
      setPromptData(null);
    } finally {
      setIsLoadingPrompt(false);
    }
  }, []);

  // Handle agent selection
  const handleAgentSelect = useCallback((agentId: string) => {
    // Check for unsaved changes
    if (hasChanges && selectedAgentId && promptData) {
      if (!window.confirm('You have unsaved changes. Do you want to discard them?')) {
        return;
      }
    }
    
    setSelectedAgentId(agentId);
    fetchAgentPrompt(agentId);
  }, [fetchAgentPrompt, hasChanges, promptData, selectedAgentId]);

  // Handle prompt changes
  const handlePromptChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setEditedPrompt(e.target.value);
    setHasChanges(true);
  };

  // Handle description changes
  const handleDescriptionChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEditedDescription(e.target.value);
    setHasChanges(true);
  };

  // Save prompt changes
  const handleSavePrompt = async () => {
    if (!selectedAgentId || !promptData) return;

    try {
      setIsSaving(true);
      const updateData: PromptUpdate = {
        system_prompt: editedPrompt,
        description: editedDescription,
        metadata: promptData.metadata,
      };

      const response = await updateAgentPrompt(selectedAgentId, updateData);
      setPromptData(response);
      setHasChanges(false);
      
      // Update the agent in the list to show custom prompt status
      setAgents(prevAgents => 
        prevAgents.map(agent => 
          agent.agent_id === selectedAgentId 
            ? { ...agent, has_custom_prompt: true } 
            : agent
        )
      );

      setToast({ message: 'Prompt saved successfully!', type: 'success' });
    } catch (error) {
      console.error('Error saving prompt:', error);
      setToast({ message: 'Failed to save prompt. Please try again.', type: 'error' });
    } finally {
      setIsSaving(false);
    }
  };

  // Reset prompt to default
  const handleResetPrompt = async () => {
    if (!selectedAgentId) return;
    
    if (!window.confirm('Are you sure you want to reset this prompt to the default? This cannot be undone.')) {
      return;
    }

    try {
      setIsResetting(true);
      const response = await resetAgentPrompt(selectedAgentId);
      setPromptData(response);
      setEditedPrompt(response.system_prompt);
      setEditedDescription(response.description || '');
      setHasChanges(false);
      
      // Update the agent in the list to show default prompt status
      setAgents(prevAgents => 
        prevAgents.map(agent => 
          agent.agent_id === selectedAgentId 
            ? { ...agent, has_custom_prompt: false } 
            : agent
        )
      );

      setToast({ message: 'Prompt reset to default successfully!', type: 'success' });
    } catch (error) {
      console.error('Error resetting prompt:', error);
      setToast({ message: 'Failed to reset prompt. Please try again.', type: 'error' });
    } finally {
      setIsResetting(false);
    }
  };

  // Close toast notification
  const handleCloseToast = () => {
    setToast(null);
  };

  return (
    <div className="flex flex-col md:flex-row h-full min-h-screen bg-gray-50">
      {/* Agent list sidebar */}
      <div className="w-full md:w-1/4 lg:w-1/5 bg-white border-r border-gray-200 p-4 overflow-y-auto">
        <h2 className="text-xl font-bold mb-4">Agents</h2>
        
        {isLoadingAgents ? (
          <div className="flex justify-center items-center h-32">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
          </div>
        ) : agentsError ? (
          <div className="text-red-500 p-4 rounded-md bg-red-50">
            {agentsError}
            <button 
              className="mt-2 text-blue-500 hover:text-blue-700"
              onClick={() => window.location.reload()}
            >
              Retry
            </button>
          </div>
        ) : agents.length === 0 ? (
          <div className="text-gray-500 p-4">No agents found.</div>
        ) : (
          <ul className="space-y-2">
            {agents.map((agent) => (
              <li key={agent.agent_id}>
                <button
                  className={`w-full text-left p-3 rounded-md transition-colors ${
                    selectedAgentId === agent.agent_id
                      ? 'bg-blue-100 text-blue-800'
                      : 'hover:bg-gray-100'
                  }`}
                  onClick={() => handleAgentSelect(agent.agent_id)}
                >
                  <div className="font-medium">{agent.agent_id}</div>
                  {agent.description && (
                    <div className="text-sm text-gray-600 truncate">{agent.description}</div>
                  )}
                  <div className="mt-1">
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      agent.has_custom_prompt
                        ? 'bg-purple-100 text-purple-800'
                        : 'bg-gray-100 text-gray-800'
                    }`}>
                      {agent.has_custom_prompt ? 'Custom' : 'Default'}
                    </span>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Prompt editor */}
      <div className="flex-1 p-4 overflow-y-auto">
        {selectedAgentId ? (
          isLoadingPrompt ? (
            <div className="flex justify-center items-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
            </div>
          ) : promptError ? (
            <div className="text-red-500 p-4 rounded-md bg-red-50">
              {promptError}
              <button 
                className="mt-2 text-blue-500 hover:text-blue-700"
                onClick={() => fetchAgentPrompt(selectedAgentId)}
              >
                Retry
              </button>
            </div>
          ) : promptData ? (
            <div>
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold">{selectedAgentId}</h2>
                <div className="space-x-2">
                  <button
                    className={`px-4 py-2 rounded-md ${
                      hasChanges
                        ? 'bg-blue-500 hover:bg-blue-600 text-white'
                        : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    }`}
                    onClick={handleSavePrompt}
                    disabled={!hasChanges || isSaving}
                  >
                    {isSaving ? (
                      <span className="flex items-center">
                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Saving...
                      </span>
                    ) : (
                      'Save Changes'
                    )}
                  </button>
                  <button
                    className="px-4 py-2 bg-red-100 text-red-700 hover:bg-red-200 rounded-md"
                    onClick={handleResetPrompt}
                    disabled={isResetting || promptData.is_default}
                  >
                    {isResetting ? (
                      <span className="flex items-center">
                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-red-700" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Resetting...
                      </span>
                    ) : (
                      'Reset to Default'
                    )}
                  </button>
                </div>
              </div>

              <div className="mb-4">
                <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <input
                  type="text"
                  id="description"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  value={editedDescription}
                  onChange={handleDescriptionChange}
                  placeholder="Agent description"
                />
              </div>

              <div className="mb-4">
                <div className="flex justify-between items-center mb-1">
                  <label htmlFor="system_prompt" className="block text-sm font-medium text-gray-700">
                    System Prompt
                  </label>
                  <span className="text-xs text-gray-500">
                    {promptData.is_default ? 'Default Prompt' : 'Custom Prompt'}
                  </span>
                </div>
                <textarea
                  id="system_prompt"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 font-mono"
                  value={editedPrompt}
                  onChange={handlePromptChange}
                  rows={20}
                  placeholder="Enter the system prompt for this agent..."
                />
              </div>

              {hasChanges && (
                <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-4">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <svg className="h-5 w-5 text-yellow-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div className="ml-3">
                      <p className="text-sm text-yellow-700">
                        You have unsaved changes. Click "Save Changes" to apply them.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {promptData.metadata && Object.keys(promptData.metadata).length > 0 && (
                <div className="mt-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Metadata</h3>
                  <div className="bg-gray-50 p-4 rounded-md overflow-x-auto">
                    <pre className="text-sm text-gray-700">
                      {JSON.stringify(promptData.metadata, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          ) : null
        ) : (
          <div className="flex flex-col items-center justify-center h-64 text-gray-500">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
            <p className="text-xl">Select an agent to view or edit its prompt</p>
          </div>
        )}
      </div>

      {/* Toast notification */}
      {toast && <Toast message={toast.message} type={toast.type} onClose={handleCloseToast} />}
    </div>
  );
};

export default PromptsManager;
