import React, { useState, useEffect } from 'react';
import './App.css';

const BASE_PATH = '/Users/seojune/Desktop/tau_results';

function App() {
  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {
      // Fallback for browsers that don't support clipboard API
      const textArea = document.createElement('textarea');
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
    }
  };

  const updateURL = (model, taskId, domain) => {
    const url = new URL(window.location);
    if (model) url.searchParams.set('model', model);
    else url.searchParams.delete('model');
    
    if (taskId !== '' && taskId !== null && taskId !== undefined) {
      url.searchParams.set('taskId', taskId);
    } else {
      url.searchParams.delete('taskId');
    }
    
    if (domain) url.searchParams.set('domain', domain);
    else url.searchParams.delete('domain');
    
    window.history.pushState({}, '', url);
  };

  const readFromURL = () => {
    const url = new URL(window.location);
    return {
      model: url.searchParams.get('model') || '',
      taskId: url.searchParams.get('taskId') || '',
      domain: url.searchParams.get('domain') || 'retail'
    };
  };
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedDomain, setSelectedDomain] = useState('retail');
  const [tasks, setTasks] = useState([]);
  const [selectedTaskId, setSelectedTaskId] = useState('');
  const [initialLoadDone, setInitialLoadDone] = useState(false);
  const [taskContent, setTaskContent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [summary, setSummary] = useState([]);

  // const getFailedSamples = async () => {
  //   try {
  //     const response = await fetch('/api/failed-samples');
  //     if (!response.ok) throw new Error('Failed to fetch failed samples');
  //     const samples = await response.json();
  //     console.log('Samples with no successful models:', samples);
  //     setFailedSamples(samples);
  //   } catch (err) {
  //     console.error('Failed to load failed samples:', err);
  //   }
  // };

  const getModels = async (domain) => {
    try {
      const response = await fetch(`/api/models?domain=${domain}`);
      if (!response.ok) throw new Error('Failed to fetch models');
      const modelNames = await response.json();
      
      const modelMap = {
        'claude-4-sonnet-thinking-off': 'Claude 4 Sonnet (thinking off)',
        'claude-4-sonnet-thinking-on-10k': 'Claude 4 Sonnet (thinking on 10k)',
        'DeepSeek-R1-0528': 'DeepSeek R1-0528',
        'DeepSeek-V3-0324': 'DeepSeek V3-0324',
        'gpt-4.1': 'GPT-4.1',
        'gpt-4o-mini': 'GPT-4o Mini',
        'o3-high': 'o3 High',
        'o4-mini-high': 'o4 Mini High',
        'Qwen3-8B': 'Qwen3-8B',
        'Qwen3-32B': 'Qwen3-32B',
        'Qwen3-235B-A22B-FP8': 'Qwen3-235B-A22B-FP8',
        'Qwen3-235B-A22B-Instruct-2507-FP8': 'Qwen3-235B-A22B-Instruct-2507-FP8',
        'Qwen3-235B-A22B-Thinking-2507-FP8': 'Qwen3-235B-A22B-Thinking-2507-FP8',
        'Kimi-K2-Instruct': 'Kimi-K2-Instruct',
        'grok-4': 'Grok-4'
      };
      
      const modelList = modelNames.map(name => ({
        path: name,
        name: modelMap[name] || name.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
      }));
      
      setModels(modelList);
      
      // Set default model selection to first model
      if (modelList.length > 0 && !selectedModel) {
        setSelectedModel(modelList[0].path);
      }
    } catch (err) {
      setError('Failed to load models');
    }
  };

  const getTasks = async (modelName, domain) => {
    if (!modelName) {
      setTasks([]);
      return;
    }

    try {
      const response = await fetch(`/api/tasks?model=${encodeURIComponent(modelName)}&domain=${domain}`);
      if (!response.ok) throw new Error('Failed to fetch tasks');
      const data = await response.json();
      setTasks(data);
      
      // If current task doesn't exist in new model, reset to first task
      if (selectedTaskId && !data.find(t => t.taskId === parseInt(selectedTaskId))) {
        setSelectedTaskId(data.length > 0 ? data[0].taskId.toString() : '');
      } else if (data.length > 0 && !selectedTaskId) {
        // Set default task selection to first task only if none selected
        setSelectedTaskId(data[0].taskId.toString());
      }
    } catch (err) {
      setError('Failed to load tasks');
      setTasks([]);
    }
  };

  const getSummary = async (domain) => {
    try {
      const response = await fetch(`/api/task-summary?domain=${domain}`);
      if (!response.ok) throw new Error('Failed to fetch summary');
      const data = await response.json();
      setSummary(data);
    } catch (err) {
      console.error('Failed to load summary:', err);
      setSummary([]);
    }
  };

  const loadTaskContent = async (modelName, taskId, domain) => {
    if (!modelName || taskId === '') {
      setTaskContent(null);
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch(`/api/task-content?model=${encodeURIComponent(modelName)}&taskId=${taskId}&domain=${domain}`);
      
      if (!response.ok) throw new Error('Failed to fetch task content');
      const content = await response.json();
      setTaskContent(content);
    } catch (err) {
      setError('Failed to load task content');
      setTaskContent(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const initializeFromURL = async () => {
      const urlParams = readFromURL();
      setSelectedDomain(urlParams.domain);
      await getModels(urlParams.domain);
      await getSummary(urlParams.domain);
      
      if (urlParams.model) {
        setSelectedModel(urlParams.model);
        if (urlParams.taskId) {
          setSelectedTaskId(urlParams.taskId);
        }
      }
      setInitialLoadDone(true);
    };
    
    initializeFromURL();
  }, []);

  useEffect(() => {
    if (selectedModel && selectedDomain) {
      getTasks(selectedModel, selectedDomain);
      setTaskContent(null);
    }
  }, [selectedModel, selectedDomain]);

  // Update models and summary when domain changes
  useEffect(() => {
    if (selectedDomain && initialLoadDone) {
      getModels(selectedDomain);
      getSummary(selectedDomain);
      setSelectedModel('');
      setSelectedTaskId('');
      setTaskContent(null);
    }
  }, [selectedDomain, initialLoadDone]);

  useEffect(() => {
    loadTaskContent(selectedModel, selectedTaskId, selectedDomain);
  }, [selectedModel, selectedTaskId, selectedDomain]);

  // Update URL when selections change (but not during initial load)
  useEffect(() => {
    if (initialLoadDone) {
      updateURL(selectedModel, selectedTaskId, selectedDomain);
    }
  }, [selectedModel, selectedTaskId, selectedDomain, initialLoadDone]);

  // Handle browser back/forward navigation
  useEffect(() => {
    const handlePopState = () => {
      const urlParams = readFromURL();
      setSelectedModel(urlParams.model);
      setSelectedTaskId(urlParams.taskId);
      setSelectedDomain(urlParams.domain);
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (!tasks.length) return;
      
      const currentIndex = tasks.findIndex(t => t.taskId === parseInt(selectedTaskId));
      if (currentIndex === -1) return;
      
      // Model navigation with h/l keys
      if (models.length > 0) {
        const currentModelIndex = models.findIndex(model => model.path === selectedModel);
        
        if (event.key === 'h' || event.key === 'ㅗ') {
          event.preventDefault();
          if (currentModelIndex > 0) {
            setSelectedModel(models[currentModelIndex - 1].path);
          } else {
            setSelectedModel(models[models.length - 1].path);
          }
          return;
        } else if (event.key === 'l' || event.key === 'ㅣ') {
          event.preventDefault();
          if (currentModelIndex !== -1 && currentModelIndex < models.length - 1) {
            setSelectedModel(models[currentModelIndex + 1].path);
          } else {
            setSelectedModel(models[0].path);
          }
          return;
        }
      }
      
      // Task navigation with j/k keys
      if ((event.key === 'k' || event.key === 'ㅏ') && currentIndex > 0) {
        event.preventDefault();
        setSelectedTaskId(tasks[currentIndex - 1].taskId.toString());
      } else if ((event.key === 'j' || event.key === 'ㅓ') && currentIndex < tasks.length - 1) {
        event.preventDefault();
        setSelectedTaskId(tasks[currentIndex + 1].taskId.toString());
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [tasks, selectedTaskId, models, selectedModel]);

  return (
    <div className="container">
      <div className="header">
        <h3>Tau-Bench Results Viewer</h3>
        
        
        <div className="controls">
          <div className="control-group">
            <label htmlFor="domain-select">Domain:</label>
            <select
              id="domain-select"
              value={selectedDomain}
              onChange={(e) => setSelectedDomain(e.target.value)}
            >
              <option value="retail">Retail</option>
              <option value="airline">Airline</option>
            </select>
          </div>

          <div className="control-group">
            <label htmlFor="model-select">Model:</label>
            <select
              id="model-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              <option value="">Select a model...</option>
              {models.map((model) => (
                <option key={model.path} value={model.path}>
                  {model.name}
                </option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label htmlFor="task-select">
              Task: 
              {tasks.length > 0 && (
                <span style={{ fontSize: '0.8em', color: '#666', margin: '0 8px' }}>
                  ({tasks.filter(t => t.reward === 1.0).length}/{tasks.length} successful)
                </span>
              )}
            </label>
            <select
              id="task-select"
              value={selectedTaskId}
              onChange={(e) => setSelectedTaskId(e.target.value)}
              disabled={!selectedModel}
            >
              <option value="">Select a task...</option>
              {tasks.map((task) => (
                <option key={task.taskId} value={task.taskId.toString()} style={{
                  color: task.reward === 1.0 ? '#22c55e' : task.reward === 0 ? '#ef4444' : '#f59e0b',
                  fontWeight: task.reward === 1.0 ? 'bold' : 'normal'
                }}>
                  Task {task.taskId} (Reward: {task.reward})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="log-display">
        {error && <div className="error">{error}</div>}
        {loading && <div className="loading">Loading task content...</div>}
        {!loading && !error && !taskContent && (
          <div className="no-selection">
            Select a model and task to view the task content.
          </div>
        )}
        {!loading && !error && taskContent && (
          <div className="log-content-wrapper">
            {/* Task Status */}
            <div 
              className="log-status" 
              style={{ 
                backgroundColor: taskContent.reward === 1.0 ? '#22c55e' : taskContent.reward === 0 ? '#ef4444' : '#f59e0b', 
                color: 'white', 
                padding: '8px 16px', 
                borderRadius: '4px', 
                fontWeight: 'bold',
                flexShrink: 0
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>Task {taskContent.task_id} - Reward: {taskContent.reward} ({selectedDomain.toUpperCase()})</span>
              </div>
              {taskContent.info && taskContent.info.task && taskContent.info.task.instruction && (
                <div style={{ fontSize: '0.9em', fontWeight: 'normal', marginTop: '4px' }}>
                  <strong>Instruction:</strong> {taskContent.info.task.instruction.substring(0, 200)}...
                </div>
              )}
              {taskContent.reward === 0 && taskContent.info && taskContent.info.reward_info && taskContent.info.reward_info.info && (
                <div style={{ fontSize: '0.9em', fontWeight: 'normal', marginTop: '4px', backgroundColor: 'rgba(255,255,255,0.2)', padding: '4px 8px', borderRadius: '3px' }}>
                  <strong>Failure Reason:</strong> 
                  {taskContent.info.reward_info.info.r_actions === 0 ? ' Action sequence mismatch' : ''}
                  {taskContent.info.reward_info.info.r_outputs === 0 ? ' Output validation failed' : ''}
                  {taskContent.info.reward_info.info.outputs && Object.keys(taskContent.info.reward_info.info.outputs).length > 0 ? 
                    ` (Expected outputs: ${Object.keys(taskContent.info.reward_info.info.outputs).join(', ')})` : ''}
                </div>
              )}
            </div>
            
            <div className="main-content-split">
              <div className="left-panel">
                {/* Actions Split - Actual on left, Expected on right with synchronized scrolling */}
                <div style={{ display: 'flex', gap: '20px', marginBottom: '20px', height: '100%' }}>
                  {/* Create synchronized scrolling container */}
                  <div style={{ flex: 1, height: '100%', display: 'flex', gap: '20px' }}>
                    {/* Actual Actions from Traj - LEFT */}
                    <div style={{ flex: 1, height: '100%' }}>
                      {taskContent.traj && (
                        <div className="function-calls-section" style={{ height: '100%' }}>
                          <div className="function-calls-header">Actual Actions</div>
                          <div 
                            id="actions-scroll-container"
                            style={{ height: 'calc(100% - 40px)', overflowY: 'auto', overflowX: 'hidden' }}
                            onScroll={(e) => {
                              const expectedContainer = document.getElementById('expected-actions-content');
                              if (expectedContainer) {
                                expectedContainer.scrollTop = e.target.scrollTop;
                              }
                            }}
                          >
                            <div id="actual-actions-content">
                              {taskContent.traj
                                .filter(action => action.role === 'assistant' && action.tool_calls)
                                .flatMap(action => action.tool_calls)
                                .map((toolCall, index) => (
                                <div key={index} className="function-call-item golden">
                                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                    <strong>{toolCall.function?.name || toolCall.name}</strong>
                                    <button 
                                      onClick={() => {
                                        const args = toolCall.function?.arguments || toolCall.arguments || {};
                                        const parsedArgs = typeof args === 'string' ? JSON.parse(args) : args;
                                        copyToClipboard(JSON.stringify(parsedArgs, null, 2));
                                      }}
                                      className="copy-button"
                                      title="Copy arguments"
                                    >
                                      copy
                                    </button>
                                  </div>
                                  <pre>
                                    {(() => {
                                      const args = toolCall.function?.arguments || toolCall.arguments || {};
                                      const parsedArgs = typeof args === 'string' ? JSON.parse(args) : args;
                                      return JSON.stringify(parsedArgs, null, 2);
                                    })()}
                                  </pre>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Expected Actions from Reward Info - RIGHT */}
                    <div style={{ flex: 1, height: '100%' }}>
                      {taskContent.info && taskContent.info.reward_info && taskContent.info.reward_info.actions && (
                        <div className="function-calls-section" style={{ height: '100%' }}>
                          <div className="function-calls-header">Expected Actions</div>
                          <div 
                            id="expected-actions-content"
                            style={{ height: 'calc(100% - 40px)', overflowY: 'auto', overflowX: 'hidden' }}
                            onScroll={(e) => {
                              const actualContainer = document.getElementById('actions-scroll-container');
                              if (actualContainer) {
                                actualContainer.scrollTop = e.target.scrollTop;
                              }
                            }}
                          >
                            {taskContent.info.reward_info.actions.map((action, index) => (
                              <div key={index} className="function-call-item actual">
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                  <strong>{action.name}</strong>
                                  <button 
                                    onClick={() => copyToClipboard(JSON.stringify(action.kwargs, null, 2))}
                                    className="copy-button"
                                    title="Copy arguments"
                                  >
                                    copy
                                  </button>
                                </div>
                                <pre>
                                  {JSON.stringify(action.kwargs, null, 2)}
                                </pre>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="right-panel">
                <div style={{ 
                  fontWeight: 'bold', 
                  marginBottom: '5px', 
                  padding: '5px 10px', 
                  backgroundColor: '#f8f9fa', 
                  borderRadius: '4px',
                  fontSize: '14px',
                  flexShrink: 0,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  Conversation Trajectory
                  <button 
                    onClick={() => copyToClipboard(JSON.stringify(taskContent.traj, null, 2))}
                    className="copy-button"
                    title="Copy conversation"
                  >
                    copy
                  </button>
                </div>
                <div className="log-content" style={{ maxHeight: '600px', overflowY: 'auto' }}>
                  {taskContent.traj && taskContent.traj.map((message, index) => (
                    <div key={index} style={{ 
                      marginBottom: '15px', 
                      padding: '10px', 
                      backgroundColor: message.role === 'user' ? '#e3f2fd' : message.role === 'assistant' ? '#f3e5f5' : message.role === 'system' ? '#fff3e0' : '#f5f5f5',
                      borderRadius: '4px',
                      borderLeft: `4px solid ${message.role === 'user' ? '#1976d2' : message.role === 'assistant' ? '#7b1fa2' : message.role === 'system' ? '#f57c00' : '#666'}`
                    }}>
                      <div style={{ fontWeight: 'bold', fontSize: '12px', marginBottom: '5px', textTransform: 'uppercase' }}>
                        {message.role}
                      </div>
                      <div style={{ fontSize: '14px', whiteSpace: 'pre-wrap' }}>
                        {message.content}
                      </div>
                      {message.tool_calls && (
                        <div style={{ marginTop: '10px' }}>
                          <div style={{ fontWeight: 'bold', fontSize: '12px', marginBottom: '5px' }}>Tool Calls:</div>
                          {message.tool_calls.map((call, callIndex) => (
                            <div key={callIndex} style={{ 
                              backgroundColor: 'rgba(255,255,255,0.7)', 
                              padding: '5px', 
                              borderRadius: '3px', 
                              marginBottom: '5px' 
                            }}>
                              <div style={{ fontWeight: 'bold' }}>{call.function.name}</div>
                              <pre style={{ fontSize: '12px', margin: '5px 0' }}>
                                {call.function.arguments}
                              </pre>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;