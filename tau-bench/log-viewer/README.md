# Tau-Bench Results Viewer

A React-based web application for viewing and analyzing results from different AI models in the Tau-Bench benchmark.

## Features

- **Model Performance Summary**: Overview of all models with success rates and average rewards
- **Model Selection**: Choose from various AI models including Claude, DeepSeek, GPT, Qwen, and more
- **Task Navigation**: Browse through individual tasks and their results
- **Conversation Viewing**: View complete conversation trajectories with proper formatting
- **Action Comparison**: Compare expected actions vs actual actions performed by the model
- **Success/Failure Analysis**: Visual indicators for task success/failure rates

## Directory Structure

The application expects tau-bench results to be organized as JSON files:
```
/Users/seojune/Desktop/tau_results/results/
├── tool-calling-claude-4-sonnet-thinking-off-0.0_range_0--1_user-openai-gpt-4o-20240806-llm_0816020105.json
├── tool-calling-DeepSeek-V3-0324-0.0_range_0--1_user-openai-gpt-4o-20240806-llm_0813191451.json
├── tool-calling-gpt-4.1-0.0_range_0--1_user-openai-gpt-4o-20240806-llm_0816022412.json
└── ...
```

## Getting Started

### Prerequisites

- Node.js (version 14 or higher)
- npm or yarn

### Installation

1. Navigate to the log-viewer directory:
   ```bash
   cd /Users/seojune/Desktop/tau_results/log-viewer
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

### Running the Application

1. Start both the backend server and frontend development server:
   ```bash
   npm run dev
   ```

   Or run them separately:
   ```bash
   # Terminal 1 - Backend server
   npm run server

   # Terminal 2 - Frontend development server
   npm start
   ```

2. Open your browser and navigate to `http://localhost:3000`

### Available Scripts

- `npm start` - Runs the React development server
- `npm run server` - Runs the Express backend server
- `npm run dev` - Runs both frontend and backend concurrently
- `npm run build` - Builds the app for production
- `npm test` - Launches the test runner

## API Endpoints

The backend server provides the following API endpoints:

- `GET /api/models` - Returns available models
- `GET /api/tasks?model={modelName}` - Returns tasks for a specific model
- `GET /api/task-content?model={modelName}&taskId={taskId}` - Returns complete task data
- `GET /api/task-summary` - Returns performance summary for all models

## Usage

1. **View Performance Summary**: See overview of all models with success rates
2. **Select a Model**: Choose from the dropdown list of available AI models
3. **Browse Tasks**: Select a specific task to analyze
4. **View Results**: See conversation trajectories, expected vs actual actions, and task outcomes
5. **Navigate**: Use j/k keys for task navigation, h/l keys for model switching

## Technologies Used

- **Frontend**: React, CSS3
- **Backend**: Node.js, Express.js
- **File System**: Native Node.js fs module for reading log files
- **CORS**: Enabled for cross-origin requests

## Development

### File Structure

```
log-viewer/
├── public/
│   └── index.html
├── src/
│   ├── App.js          # Main React component
│   ├── App.css         # Component styles
│   ├── index.js        # React entry point
│   └── index.css       # Global styles
├── server.js           # Express backend server
├── package.json        # Dependencies and scripts
└── README.md          # This file
```

### Customization

- **Base Path**: Modify the `BASE_PATH` constant in `server.js` to point to your results directory
- **Model Display Names**: Update the `modelMap` object in `App.js` to customize how model names are displayed  
- **Styling**: Modify CSS files to customize the appearance
- **Data Format**: The viewer expects tau-bench JSON format with task_id, reward, info, and traj fields

## Troubleshooting

1. **Port Conflicts**: If port 3000 or 3001 are in use, the application will prompt you to use different ports
2. **File Access**: Ensure the application has read permissions for the results directory
3. **CORS Issues**: The backend includes CORS middleware to handle cross-origin requests
4. **Data Format**: Ensure JSON files follow the expected tau-bench format structure

## License

This project is private and intended for internal use only.