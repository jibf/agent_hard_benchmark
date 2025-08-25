const express = require('express');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = 3001;
const BASE_PATH = path.join(__dirname, '../results');

app.use(cors());
app.use(express.json());

const getResultFiles = (domain = null) => {
  try {
    const files = fs.readdirSync(BASE_PATH);
    let filteredFiles = files.filter(file => file.endsWith('.json') && file.startsWith('tool-calling-'));
    
    if (domain) {
      filteredFiles = filteredFiles.filter(file => file.includes(`tool-calling-${domain}-`));
    }
    
    return filteredFiles;
  } catch (error) {
    console.error(`Error reading directory ${BASE_PATH}:`, error);
    return [];
  }
};

const parseModelFromFilename = (filename) => {
  // Extract model name from filename
  // Handle both old and new formats:
  // Old: tool-calling-airline-claude-4-sonnet-thinking-off-0.0_range_0--1_user-openai-gpt-4o-20240806-llm_0816020105.json
  // New: tool-calling-airline-DeepSeek-R1-0528-0.0_range_0--1_user-openai-gpt-4o-20240806-llm_0819143202.json
  
  if (filename.startsWith('tool-calling-')) {
    // Remove prefix and suffix
    let modelPart = filename.replace('tool-calling-', '').replace('.json', '');
    
    // Split by domain (airline/retail) if present
    const domainMatch = modelPart.match(/^(airline|retail)-(.+)/);
    if (domainMatch) {
      modelPart = domainMatch[2];
    }
    
    // Extract model name up to the pattern -0.0_range
    const match = modelPart.match(/^(.+?)-0\.0_range/);
    return match ? match[1] : modelPart;
  }
  
  return filename.replace('.json', '');
};

const getModelList = (domain = null) => {
  try {
    const files = getResultFiles(domain);
    const models = new Set();
    
    files.forEach(file => {
      const modelName = parseModelFromFilename(file);
      models.add(modelName);
    });
    
    return Array.from(models).sort();
  } catch (error) {
    console.error('Error getting models:', error);
    return [];
  }
};

const getTasksForModel = (modelName, domain = null) => {
  try {
    const files = getResultFiles(domain);
    const modelFiles = files.filter(file => parseModelFromFilename(file) === modelName);
    
    const allTasks = [];
    
    for (const file of modelFiles) {
      try {
        const filePath = path.join(BASE_PATH, file);
        const content = fs.readFileSync(filePath, 'utf8');
        const tasks = JSON.parse(content);

        tasks.forEach((task, index) => {
          // Determine task domain
          const actions = task?.info?.task?.actions || [];
          let taskDomain = 'unknown';
          if (actions.length > 0) {
            const actionNames = actions.map(action => action.name);
            if (actionNames.some(name => name.includes('book_reservation'))) {
              taskDomain = 'airline';
            } else if (actionNames.some(name => 
              name.includes('get_product') || name.includes('order') || name.includes('user'))) {
              taskDomain = 'retail';
            }
          }
          
          // Filter by domain if specified
          allTasks.push({
            taskId: task.task_id,
            fileName: file,
            taskIndex: index,
            reward: task.reward,
            info: task.info,
            domain: domain 
          });
        });
      } catch (parseError) {
        console.error(`Error parsing file ${file}:`, parseError);
      }
    }
    
    return allTasks.sort((a, b) => a.taskId - b.taskId);
  } catch (error) {
    console.error('Error getting tasks for model:', error);
    return [];
  }
};

app.get('/api/models', (req, res) => {
  const { domain } = req.query;
  try {
    const models = getModelList(domain);
    res.json(models);
  } catch (error) {
    console.error('Error getting models:', error);
    res.status(500).json({ error: 'Failed to get models' });
  }
});

app.get('/api/tasks', (req, res) => {
  const { model, domain } = req.query;
  
  if (!model) {
    return res.status(400).json({ error: 'Model parameter is required' });
  }
  
  try {
    const tasks = getTasksForModel(model, domain);
    res.json(tasks);
  } catch (error) {
    console.error('Error getting tasks:', error);
    res.status(500).json({ error: 'Failed to get tasks' });
  }
});

app.get('/api/task-content', (req, res) => {
  const { model, taskId, domain } = req.query;
  
  if (!model || taskId === undefined) {
    return res.status(400).json({ error: 'Model and taskId parameters are required' });
  }
  
  try {
    const tasks = getTasksForModel(model, domain);
    const task = tasks.find(t => t.taskId === parseInt(taskId));
    
    if (!task) {
      return res.status(404).json({ error: 'Task not found' });
    }
    
    const filePath = path.join(BASE_PATH, task.fileName);
    const content = fs.readFileSync(filePath, 'utf8');
    const allTasks = JSON.parse(content);
    const fullTask = allTasks[task.taskIndex];
    
    res.json(fullTask);
  } catch (error) {
    console.error('Error reading task content:', error);
    res.status(500).json({ error: 'Failed to read task content' });
  }
});

app.get('/api/task-summary', (req, res) => {
  const { domain } = req.query;
  try {
    const models = getModelList(domain);
    const summary = [];
    
    for (const model of models) {
      const tasks = getTasksForModel(model, domain);
      const totalTasks = tasks.length;
      const successfulTasks = tasks.filter(t => t.reward === 1.0).length;
      const averageReward = tasks.length > 0 ? tasks.reduce((sum, t) => sum + t.reward, 0) / tasks.length : 0;
      
      summary.push({
        model,
        totalTasks,
        successfulTasks,
        averageReward: Math.round(averageReward * 1000) / 1000,
        successRate: Math.round((successfulTasks / totalTasks) * 1000) / 10, // percentage with 1 decimal
        domain: domain || 'all'
      });
    }
    
    res.json(summary.sort((a, b) => b.averageReward - a.averageReward));
  } catch (error) {
    console.error('Error getting task summary:', error);
    res.status(500).json({ error: 'Failed to get task summary' });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`Base path: ${BASE_PATH}`);
});