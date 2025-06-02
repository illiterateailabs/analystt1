import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Button, 
  Card, 
  CardContent, 
  Stepper, 
  Step, 
  StepLabel, 
  Typography, 
  TextField, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  Chip, 
  FormHelperText,
  Divider,
  IconButton,
  Alert,
  CircularProgress,
  Tooltip,
  Switch,
  FormControlLabel,
  Grid,
  Paper
} from '@mui/material';
import { 
  Add as AddIcon, 
  Delete as DeleteIcon, 
  Save as SaveIcon, 
  Preview as PreviewIcon,
  Help as HelpIcon,
  ArrowBack as BackIcon,
  ArrowForward as NextIcon,
  Check as CheckIcon
} from '@mui/icons-material';
import { useRouter } from 'next/navigation';
import { useToast } from '@/hooks/useToast';
import { useAuth } from '@/hooks/useAuth';
import { api } from '@/lib/api';

// Types
interface Agent {
  name: string;
  role: string;
  goal: string;
  backstory?: string;
  tools?: string[];
  verbose?: boolean;
}

interface Task {
  description: string;
  agent: string;
  expected_output?: string;
  tools?: string[];
  async_execution?: boolean;
}

interface TemplateData {
  name: string;
  description: string;
  agents: Agent[];
  tasks: Task[];
  workflow?: any;
  verbose?: boolean;
  memory?: any;
  max_rpm?: number;
  sla_seconds?: number;
  hitl_triggers?: string[];
}

interface Tool {
  name: string;
  description: string;
}

interface UseCaseSuggestion {
  name: string;
  description: string;
  suggested_agents: string[];
  suggested_tools: string[];
  estimated_creation_time: string;
  confidence: number;
}

// Main component
const TemplateCreator: React.FC = () => {
  const router = useRouter();
  const { showToast } = useToast();
  const { user } = useAuth();
  
  // State for wizard steps
  const [activeStep, setActiveStep] = useState(0);
  const steps = ['Use Case', 'Agents', 'Tools', 'Tasks', 'Workflow', 'Review'];
  
  // State for template data
  const [templateData, setTemplateData] = useState<TemplateData>({
    name: '',
    description: '',
    agents: [],
    tasks: [],
    workflow: {},
    verbose: true,
    memory: { memory_type: 'buffer', memory_key: 'chat_history' },
    max_rpm: 10,
    sla_seconds: 30,
    hitl_triggers: []
  });
  
  // State for available tools and agents
  const [availableTools, setAvailableTools] = useState<Tool[]>([]);
  const [availableAgents, setAvailableAgents] = useState<string[]>([]);
  
  // State for use case input and suggestions
  const [useCase, setUseCase] = useState('');
  const [useCaseSuggestions, setUseCaseSuggestions] = useState<UseCaseSuggestion[]>([]);
  const [selectedSuggestion, setSelectedSuggestion] = useState<UseCaseSuggestion | null>(null);
  
  // Loading states
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  
  // Validation states
  const [errors, setErrors] = useState<Record<string, string>>({});
  
  // Fetch available tools and agents on component mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch available tools
        const toolsResponse = await api.get('/tools');
        setAvailableTools(toolsResponse.data.tools || []);
        
        // Fetch available agents
        const agentsResponse = await api.get('/agents');
        setAvailableAgents(agentsResponse.data.agents || []);
      } catch (error) {
        console.error('Error fetching data:', error);
        showToast('Failed to load tools and agents', 'error');
      }
    };
    
    fetchData();
  }, [showToast]);
  
  // Get suggestions when use case changes
  useEffect(() => {
    if (useCase.length > 5) {
      const getSuggestions = async () => {
        setIsLoading(true);
        try {
          const response = await api.get(`/templates/suggestions?use_case=${encodeURIComponent(useCase)}`);
          setUseCaseSuggestions(response.data || []);
        } catch (error) {
          console.error('Error getting suggestions:', error);
        } finally {
          setIsLoading(false);
        }
      };
      
      const debounce = setTimeout(() => {
        getSuggestions();
      }, 500);
      
      return () => clearTimeout(debounce);
    }
  }, [useCase]);
  
  // Apply suggestion to template data
  const applySuggestion = (suggestion: UseCaseSuggestion) => {
    setSelectedSuggestion(suggestion);
    
    // Create agents from suggestion
    const suggestedAgents = suggestion.suggested_agents.map(agentName => ({
      name: agentName,
      role: `${agentName.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}`,
      goal: `Perform ${agentName.replace(/_/g, ' ')} tasks effectively`,
      tools: suggestion.suggested_tools.slice(0, 2) // Assign first 2 tools by default
    }));
    
    // Create initial tasks
    const initialTasks = suggestion.suggested_agents.map(agentName => ({
      description: `Perform ${agentName.replace(/_/g, ' ')} analysis`,
      agent: agentName
    }));
    
    // Update template data
    setTemplateData(prev => ({
      ...prev,
      name: suggestion.name,
      description: suggestion.description,
      agents: suggestedAgents,
      tasks: initialTasks
    }));
  };
  
  // Handle next step
  const handleNext = () => {
    // Validate current step
    if (!validateStep(activeStep)) {
      return;
    }
    
    setActiveStep(prevStep => prevStep + 1);
  };
  
  // Handle back step
  const handleBack = () => {
    setActiveStep(prevStep => prevStep - 1);
  };
  
  // Validate current step
  const validateStep = (step: number): boolean => {
    const newErrors: Record<string, string> = {};
    
    switch (step) {
      case 0: // Use Case
        if (!templateData.name) {
          newErrors.name = 'Template name is required';
        } else if (!/^[a-z0-9_]+$/.test(templateData.name)) {
          newErrors.name = 'Template name should be lowercase with underscores only';
        }
        
        if (!templateData.description) {
          newErrors.description = 'Description is required';
        }
        break;
        
      case 1: // Agents
        if (templateData.agents.length === 0) {
          newErrors.agents = 'At least one agent is required';
        } else {
          templateData.agents.forEach((agent, index) => {
            if (!agent.name) {
              newErrors[`agent_${index}_name`] = 'Agent name is required';
            }
            if (!agent.role) {
              newErrors[`agent_${index}_role`] = 'Agent role is required';
            }
            if (!agent.goal) {
              newErrors[`agent_${index}_goal`] = 'Agent goal is required';
            }
          });
        }
        break;
        
      case 2: // Tools
        // No specific validation for tools
        break;
        
      case 3: // Tasks
        if (templateData.tasks.length === 0) {
          newErrors.tasks = 'At least one task is required';
        } else {
          templateData.tasks.forEach((task, index) => {
            if (!task.description) {
              newErrors[`task_${index}_description`] = 'Task description is required';
            }
            if (!task.agent) {
              newErrors[`task_${index}_agent`] = 'Task agent is required';
            }
          });
        }
        break;
        
      case 4: // Workflow
        // No specific validation for workflow
        break;
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };
  
  // Handle agent changes
  const handleAgentChange = (index: number, field: keyof Agent, value: any) => {
    const updatedAgents = [...templateData.agents];
    updatedAgents[index] = { ...updatedAgents[index], [field]: value };
    setTemplateData({ ...templateData, agents: updatedAgents });
  };
  
  // Add new agent
  const addAgent = () => {
    setTemplateData({
      ...templateData,
      agents: [
        ...templateData.agents,
        {
          name: `agent_${templateData.agents.length + 1}`,
          role: '',
          goal: ''
        }
      ]
    });
  };
  
  // Remove agent
  const removeAgent = (index: number) => {
    const updatedAgents = [...templateData.agents];
    updatedAgents.splice(index, 1);
    setTemplateData({ ...templateData, agents: updatedAgents });
  };
  
  // Handle task changes
  const handleTaskChange = (index: number, field: keyof Task, value: any) => {
    const updatedTasks = [...templateData.tasks];
    updatedTasks[index] = { ...updatedTasks[index], [field]: value };
    setTemplateData({ ...templateData, tasks: updatedTasks });
  };
  
  // Add new task
  const addTask = () => {
    setTemplateData({
      ...templateData,
      tasks: [
        ...templateData.tasks,
        {
          description: '',
          agent: templateData.agents.length > 0 ? templateData.agents[0].name : ''
        }
      ]
    });
  };
  
  // Remove task
  const removeTask = (index: number) => {
    const updatedTasks = [...templateData.tasks];
    updatedTasks.splice(index, 1);
    setTemplateData({ ...templateData, tasks: updatedTasks });
  };
  
  // Save template
  const saveTemplate = async () => {
    // Validate all steps
    for (let i = 0; i < steps.length - 1; i++) {
      if (!validateStep(i)) {
        setActiveStep(i);
        showToast('Please fix the errors before saving', 'error');
        return;
      }
    }
    
    setIsSaving(true);
    
    try {
      // Format template data for API
      const formattedAgents = templateData.agents.map(agent => ({
        name: agent.name,
        role: agent.role,
        goal: agent.goal,
        backstory: agent.backstory,
        tools: agent.tools,
        verbose: agent.verbose
      }));
      
      const formattedTasks = templateData.tasks.map(task => ({
        description: task.description,
        agent: task.agent,
        expected_output: task.expected_output,
        tools: task.tools,
        async_execution: task.async_execution
      }));
      
      const templatePayload = {
        name: templateData.name,
        description: templateData.description,
        agents: formattedAgents,
        tasks: formattedTasks,
        workflow: templateData.workflow,
        verbose: templateData.verbose,
        memory: templateData.memory,
        max_rpm: templateData.max_rpm,
        sla_seconds: templateData.sla_seconds,
        hitl_triggers: templateData.hitl_triggers
      };
      
      // Send to API
      await api.post('/templates', templatePayload);
      
      showToast('Template created successfully', 'success');
      router.push('/templates');
    } catch (error: any) {
      console.error('Error saving template:', error);
      showToast(error.response?.data?.detail || 'Failed to save template', 'error');
    } finally {
      setIsSaving(false);
    }
  };
  
  // Render step content
  const getStepContent = (step: number) => {
    switch (step) {
      case 0:
        return renderUseCaseStep();
      case 1:
        return renderAgentsStep();
      case 2:
        return renderToolsStep();
      case 3:
        return renderTasksStep();
      case 4:
        return renderWorkflowStep();
      case 5:
        return renderReviewStep();
      default:
        return 'Unknown step';
    }
  };
  
  // Render Use Case step
  const renderUseCaseStep = () => {
    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="h6" gutterBottom>
          Define Your Investigation Template
        </Typography>
        
        <TextField
          fullWidth
          label="Template Name"
          value={templateData.name}
          onChange={(e) => setTemplateData({ ...templateData, name: e.target.value })}
          margin="normal"
          variant="outlined"
          error={!!errors.name}
          helperText={errors.name || 'Use lowercase with underscores (e.g., crypto_investigation)'}
        />
        
        <TextField
          fullWidth
          label="Template Description"
          value={templateData.description}
          onChange={(e) => setTemplateData({ ...templateData, description: e.target.value })}
          margin="normal"
          variant="outlined"
          multiline
          rows={2}
          error={!!errors.description}
          helperText={errors.description}
        />
        
        <Box sx={{ mt: 3, mb: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            Describe Your Use Case
          </Typography>
          
          <TextField
            fullWidth
            label="What type of investigation do you need?"
            value={useCase}
            onChange={(e) => setUseCase(e.target.value)}
            margin="normal"
            variant="outlined"
            placeholder="e.g., Cryptocurrency mixer investigation, DeFi exploit analysis, Banking fraud detection"
            multiline
            rows={3}
          />
        </Box>
        
        {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
            <CircularProgress size={24} />
          </Box>
        )}
        
        {useCaseSuggestions.length > 0 && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Suggested Templates
            </Typography>
            
            <Grid container spacing={2}>
              {useCaseSuggestions.map((suggestion, index) => (
                <Grid item xs={12} md={6} key={index}>
                  <Card 
                    variant="outlined" 
                    sx={{ 
                      cursor: 'pointer',
                      border: selectedSuggestion?.name === suggestion.name ? '2px solid #1976d2' : undefined
                    }}
                    onClick={() => applySuggestion(suggestion)}
                  >
                    <CardContent>
                      <Typography variant="h6" component="div">
                        {suggestion.name}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        {suggestion.description}
                      </Typography>
                      <Box sx={{ mt: 1 }}>
                        <Typography variant="caption" display="block">
                          Suggested Agents:
                        </Typography>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {suggestion.suggested_agents.map((agent, i) => (
                            <Chip key={i} label={agent} size="small" />
                          ))}
                        </Box>
                      </Box>
                      <Box sx={{ mt: 1 }}>
                        <Typography variant="caption" display="block">
                          Confidence: {Math.round(suggestion.confidence * 100)}%
                        </Typography>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        )}
      </Box>
    );
  };
  
  // Render Agents step
  const renderAgentsStep = () => {
    return (
      <Box sx={{ mt: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6" gutterBottom>
            Configure Agents
          </Typography>
          <Button
            variant="outlined"
            startIcon={<AddIcon />}
            onClick={addAgent}
          >
            Add Agent
          </Button>
        </Box>
        
        {errors.agents && (
          <Alert severity="error" sx={{ mt: 1, mb: 2 }}>
            {errors.agents}
          </Alert>
        )}
        
        {templateData.agents.map((agent, index) => (
          <Paper key={index} elevation={1} sx={{ p: 2, mt: 2, position: 'relative' }}>
            <IconButton
              size="small"
              sx={{ position: 'absolute', top: 8, right: 8 }}
              onClick={() => removeAgent(index)}
            >
              <DeleteIcon />
            </IconButton>
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Agent Name"
                  value={agent.name}
                  onChange={(e) => handleAgentChange(index, 'name', e.target.value)}
                  margin="normal"
                  variant="outlined"
                  error={!!errors[`agent_${index}_name`]}
                  helperText={errors[`agent_${index}_name`] || 'Use lowercase with underscores'}
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Agent Role"
                  value={agent.role}
                  onChange={(e) => handleAgentChange(index, 'role', e.target.value)}
                  margin="normal"
                  variant="outlined"
                  error={!!errors[`agent_${index}_role`]}
                  helperText={errors[`agent_${index}_role`]}
                />
              </Grid>
              
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Agent Goal"
                  value={agent.goal}
                  onChange={(e) => handleAgentChange(index, 'goal', e.target.value)}
                  margin="normal"
                  variant="outlined"
                  error={!!errors[`agent_${index}_goal`]}
                  helperText={errors[`agent_${index}_goal`]}
                />
              </Grid>
              
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Agent Backstory (Optional)"
                  value={agent.backstory || ''}
                  onChange={(e) => handleAgentChange(index, 'backstory', e.target.value)}
                  margin="normal"
                  variant="outlined"
                  multiline
                  rows={2}
                />
              </Grid>
              
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={agent.verbose || false}
                      onChange={(e) => handleAgentChange(index, 'verbose', e.target.checked)}
                    />
                  }
                  label="Verbose Mode"
                />
              </Grid>
            </Grid>
          </Paper>
        ))}
      </Box>
    );
  };
  
  // Render Tools step
  const renderToolsStep = () => {
    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="h6" gutterBottom>
          Assign Tools to Agents
        </Typography>
        
        {templateData.agents.map((agent, agentIndex) => (
          <Paper key={agentIndex} elevation={1} sx={{ p: 2, mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              {agent.name} ({agent.role})
            </Typography>
            
            <FormControl fullWidth margin="normal" variant="outlined">
              <InputLabel id={`tools-label-${agentIndex}`}>Tools</InputLabel>
              <Select
                labelId={`tools-label-${agentIndex}`}
                multiple
                value={agent.tools || []}
                onChange={(e) => handleAgentChange(agentIndex, 'tools', e.target.value)}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {(selected as string[]).map((value) => (
                      <Chip key={value} label={value} />
                    ))}
                  </Box>
                )}
                label="Tools"
              >
                {availableTools.map((tool) => (
                  <MenuItem key={tool.name} value={tool.name}>
                    <Box>
                      {tool.name}
                      <Typography variant="caption" display="block" color="text.secondary">
                        {tool.description}
                      </Typography>
                    </Box>
                  </MenuItem>
                ))}
              </Select>
              <FormHelperText>Select tools for this agent</FormHelperText>
            </FormControl>
          </Paper>
        ))}
        
        <Box sx={{ mt: 3 }}>
          <Alert severity="info">
            Tools enable agents to perform specific actions. Assign appropriate tools to each agent based on their role.
          </Alert>
        </Box>
      </Box>
    );
  };
  
  // Render Tasks step
  const renderTasksStep = () => {
    return (
      <Box sx={{ mt: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6" gutterBottom>
            Define Tasks
          </Typography>
          <Button
            variant="outlined"
            startIcon={<AddIcon />}
            onClick={addTask}
          >
            Add Task
          </Button>
        </Box>
        
        {errors.tasks && (
          <Alert severity="error" sx={{ mt: 1, mb: 2 }}>
            {errors.tasks}
          </Alert>
        )}
        
        {templateData.tasks.map((task, index) => (
          <Paper key={index} elevation={1} sx={{ p: 2, mt: 2, position: 'relative' }}>
            <IconButton
              size="small"
              sx={{ position: 'absolute', top: 8, right: 8 }}
              onClick={() => removeTask(index)}
            >
              <DeleteIcon />
            </IconButton>
            
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Task Description"
                  value={task.description}
                  onChange={(e) => handleTaskChange(index, 'description', e.target.value)}
                  margin="normal"
                  variant="outlined"
                  error={!!errors[`task_${index}_description`]}
                  helperText={errors[`task_${index}_description`]}
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <FormControl fullWidth margin="normal" variant="outlined" error={!!errors[`task_${index}_agent`]}>
                  <InputLabel id={`task-agent-label-${index}`}>Assigned Agent</InputLabel>
                  <Select
                    labelId={`task-agent-label-${index}`}
                    value={task.agent}
                    onChange={(e) => handleTaskChange(index, 'agent', e.target.value)}
                    label="Assigned Agent"
                  >
                    {templateData.agents.map((agent) => (
                      <MenuItem key={agent.name} value={agent.name}>
                        {agent.name} ({agent.role})
                      </MenuItem>
                    ))}
                  </Select>
                  <FormHelperText>{errors[`task_${index}_agent`]}</FormHelperText>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Expected Output (Optional)"
                  value={task.expected_output || ''}
                  onChange={(e) => handleTaskChange(index, 'expected_output', e.target.value)}
                  margin="normal"
                  variant="outlined"
                />
              </Grid>
              
              <Grid item xs={12}>
                <FormControl fullWidth margin="normal" variant="outlined">
                  <InputLabel id={`task-tools-label-${index}`}>Task-Specific Tools (Optional)</InputLabel>
                  <Select
                    labelId={`task-tools-label-${index}`}
                    multiple
                    value={task.tools || []}
                    onChange={(e) => handleTaskChange(index, 'tools', e.target.value)}
                    renderValue={(selected) => (
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {(selected as string[]).map((value) => (
                          <Chip key={value} label={value} />
                        ))}
                      </Box>
                    )}
                    label="Task-Specific Tools (Optional)"
                  >
                    {availableTools.map((tool) => (
                      <MenuItem key={tool.name} value={tool.name}>
                        {tool.name}
                      </MenuItem>
                    ))}
                  </Select>
                  <FormHelperText>Override agent's tools for this specific task (optional)</FormHelperText>
                </FormControl>
              </Grid>
              
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={task.async_execution || false}
                      onChange={(e) => handleTaskChange(index, 'async_execution', e.target.checked)}
                    />
                  }
                  label="Asynchronous Execution"
                />
              </Grid>
            </Grid>
          </Paper>
        ))}
        
        <Box sx={{ mt: 3 }}>
          <Alert severity="info">
            Tasks define what each agent will do. The order of tasks in this list will determine their execution order unless modified in the Workflow step.
          </Alert>
        </Box>
      </Box>
    );
  };
  
  // Render Workflow step
  const renderWorkflowStep = () => {
    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="h6" gutterBottom>
          Configure Workflow & Settings
        </Typography>
        
        <Paper elevation={1} sx={{ p: 2, mt: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            General Settings
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={templateData.verbose || false}
                    onChange={(e) => setTemplateData({ ...templateData, verbose: e.target.checked })}
                  />
                }
                label="Verbose Mode"
              />
              <FormHelperText>Enable detailed logging for the entire crew</FormHelperText>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="Max Requests Per Minute"
                value={templateData.max_rpm || 10}
                onChange={(e) => setTemplateData({ ...templateData, max_rpm: parseInt(e.target.value) })}
                margin="normal"
                variant="outlined"
                InputProps={{ inputProps: { min: 1, max: 60 } }}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="SLA in Seconds"
                value={templateData.sla_seconds || 30}
                onChange={(e) => setTemplateData({ ...templateData, sla_seconds: parseInt(e.target.value) })}
                margin="normal"
                variant="outlined"
                InputProps={{ inputProps: { min: 1 } }}
              />
            </Grid>
          </Grid>
        </Paper>
        
        <Paper elevation={1} sx={{ p: 2, mt: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            HITL (Human-in-the-Loop) Triggers
          </Typography>
          
          <TextField
            fullWidth
            label="HITL Triggers"
            value={templateData.hitl_triggers?.join(', ') || ''}
            onChange={(e) => {
              const triggers = e.target.value.split(',').map(t => t.trim()).filter(t => t);
              setTemplateData({ ...templateData, hitl_triggers: triggers });
            }}
            margin="normal"
            variant="outlined"
            helperText="Comma-separated list of keywords that will trigger HITL review (e.g., 'pii, sanctions, high_risk')"
          />
        </Paper>
        
        <Paper elevation={1} sx={{ p: 2, mt: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Memory Configuration
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth margin="normal" variant="outlined">
                <InputLabel id="memory-type-label">Memory Type</InputLabel>
                <Select
                  labelId="memory-type-label"
                  value={templateData.memory?.memory_type || 'buffer'}
                  onChange={(e) => setTemplateData({ 
                    ...templateData, 
                    memory: { ...templateData.memory, memory_type: e.target.value } 
                  })}
                  label="Memory Type"
                >
                  <MenuItem value="buffer">Buffer Memory</MenuItem>
                  <MenuItem value="summary">Summary Memory</MenuItem>
                  <MenuItem value="vector">Vector Memory</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Memory Key"
                value={templateData.memory?.memory_key || 'chat_history'}
                onChange={(e) => setTemplateData({ 
                  ...templateData, 
                  memory: { ...templateData.memory, memory_key: e.target.value } 
                })}
                margin="normal"
                variant="outlined"
              />
            </Grid>
          </Grid>
        </Paper>
        
        <Box sx={{ mt: 3 }}>
          <Alert severity="info">
            The workflow settings control how the crew operates. Advanced workflow configurations can be added later through the YAML editor.
          </Alert>
        </Box>
      </Box>
    );
  };
  
  // Render Review step
  const renderReviewStep = () => {
    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="h6" gutterBottom>
          Review Template
        </Typography>
        
        <Paper elevation={1} sx={{ p: 2, mt: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            Basic Information
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="body2" color="text.secondary">
                Template Name:
              </Typography>
              <Typography variant="body1">
                {templateData.name}
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography variant="body2" color="text.secondary">
                SLA:
              </Typography>
              <Typography variant="body1">
                {templateData.sla_seconds} seconds
              </Typography>
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="body2" color="text.secondary">
                Description:
              </Typography>
              <Typography variant="body1">
                {templateData.description}
              </Typography>
            </Grid>
          </Grid>
        </Paper>
        
        <Paper elevation={1} sx={{ p: 2, mt: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Agents ({templateData.agents.length})
          </Typography>
          
          {templateData.agents.map((agent, index) => (
            <Box key={index} sx={{ mt: 2 }}>
              <Typography variant="body1" fontWeight="bold">
                {agent.name}
              </Typography>
              <Typography variant="body2">
                Role: {agent.role}
              </Typography>
              <Typography variant="body2">
                Goal: {agent.goal}
              </Typography>
              <Typography variant="body2">
                Tools: {agent.tools?.join(', ') || 'None'}
              </Typography>
              {index < templateData.agents.length - 1 && <Divider sx={{ mt: 1 }} />}
            </Box>
          ))}
        </Paper>
        
        <Paper elevation={1} sx={{ p: 2, mt: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Tasks ({templateData.tasks.length})
          </Typography>
          
          {templateData.tasks.map((task, index) => (
            <Box key={index} sx={{ mt: 2 }}>
              <Typography variant="body1" fontWeight="bold">
                Task {index + 1}
              </Typography>
              <Typography variant="body2">
                Description: {task.description}
              </Typography>
              <Typography variant="body2">
                Agent: {task.agent}
              </Typography>
              {task.expected_output && (
                <Typography variant="body2">
                  Expected Output: {task.expected_output}
                </Typography>
              )}
              {index < templateData.tasks.length - 1 && <Divider sx={{ mt: 1 }} />}
            </Box>
          ))}
        </Paper>
        
        <Paper elevation={1} sx={{ p: 2, mt: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            HITL Configuration
          </Typography>
          
          <Typography variant="body2">
            Triggers: {templateData.hitl_triggers?.join(', ') || 'None'}
          </Typography>
        </Paper>
        
        <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
          <Button
            variant="contained"
            color="primary"
            startIcon={<SaveIcon />}
            onClick={saveTemplate}
            disabled={isSaving}
            sx={{ minWidth: 200 }}
          >
            {isSaving ? <CircularProgress size={24} /> : 'Save Template'}
          </Button>
        </Box>
      </Box>
    );
  };
  
  return (
    <Box sx={{ width: '100%', p: 2 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Create New Template
      </Typography>
      
      <Stepper activeStep={activeStep} sx={{ pt: 3, pb: 5 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>
      
      <Box>
        {getStepContent(activeStep)}
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
          <Button
            variant="outlined"
            onClick={handleBack}
            disabled={activeStep === 0}
            startIcon={<BackIcon />}
          >
            Back
          </Button>
          
          {activeStep === steps.length - 1 ? (
            <Button
              variant="contained"
              color="primary"
              onClick={saveTemplate}
              disabled={isSaving}
              startIcon={<SaveIcon />}
            >
              {isSaving ? <CircularProgress size={24} /> : 'Save Template'}
            </Button>
          ) : (
            <Button
              variant="contained"
              onClick={handleNext}
              endIcon={<NextIcon />}
            >
              Next
            </Button>
          )}
        </Box>
      </Box>
    </Box>
  );
};

export default TemplateCreator;
