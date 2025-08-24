# Taubench Results Viewer

A lightweight tool to visualize and analyze taubench JSON result files. This viewer provides an organized way to examine task information, rewards, and conversation flows between users and assistants.

## Features

- 📊 **Summary Statistics**: Overall performance metrics and cost analysis
- 📋 **Task List**: View all tasks with their status, rewards, and costs
- 🔍 **Task Details**: Detailed view of specific tasks including instructions and expected actions
- 💬 **Conversation Flow**: Step-by-step view of user-assistant interactions including tool calls
- 📈 **Detailed Metrics**: Comprehensive statistics and reward distribution analysis
- 🎯 **Interactive Mode**: Menu-driven interface for exploring results

## Usage

### Basic Commands

```bash
# Show summary statistics
python simple_viewer.py results_file.json --summary

# List all tasks
python simple_viewer.py results_file.json --list

# List only failed tasks
python simple_viewer.py results_file.json --failed

# View details for a specific task (1-based indexing)
python simple_viewer.py results_file.json --task 5

# View conversation for a specific task
python simple_viewer.py results_file.json --conversation 5

# Show detailed metrics and statistics
python simple_viewer.py results_file.json --metrics

# Run in interactive mode
python simple_viewer.py results_file.json --interactive
```

### Interactive Mode

The interactive mode provides a menu-driven interface:

```
==================================================
🎯 TAUBENCH VIEWER MENU
==================================================
1. Show summary
2. List all tasks
3. List failed tasks only
4. View task details
5. View conversation
6. Show metrics
7. Exit
==================================================
```

## Output Examples

### Summary View
```
==================== RESULTS SUMMARY ====================
📊 Total Tasks: 50
✅ Successful Tasks: 27 (54.0%)
📈 Average Reward: 0.540
💰 Total Cost: $0.0385
📁 File: gpt-4o-20240806-llm_0816022412.json
```

### Task List View
```
==================== TASK LIST ====================
#    Task ID  Reward   Status       Cost       User ID        
----------------------------------------------------------------------
1    0        1.000    ✅ Success    $0.0009    mia_li_3668    
2    1        1.000    ✅ Success    $0.0009    olivia_gonzalez_2305
3    2        0.000    ❌ Failed     $0.0008    omar_davis_3817
```

### Conversation View
```
==================== CONVERSATION FOR TASK 0 ====================

👤 USER (Message 2):
==================================================
Hi! I'm looking to book a one-way flight from New York to Seattle on May 20th.

🤖 ASSISTANT (Message 3):
==================================================
I can help you with that! To proceed, could you please provide your user ID?

🔧 ASSISTANT TOOL CALL (Message 7):
==================================================
Tool: get_user_details
Arguments: {"user_id":"mia_li_3668"}

⚙️ TOOL RESPONSE (Message 8):
==================================================
Tool: get_user_details
Response: {"name": {"first_name": "Mia", "last_name": "Li"}, ...}
```

## File Format

The viewer expects taubench JSON files with the following structure:

```json
[
  {
    "task_id": 0,
    "reward": 1.0,
    "info": {
      "task": {
        "user_id": "user_id",
        "actions": [...],
        "instruction": "user instruction"
      },
      "user_cost": 0.0009,
      "reward_info": {...}
    },
    "traj": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "...", "tool_calls": [...]},
      {"role": "tool", "name": "...", "content": "..."}
    ]
  }
]
```

## Requirements

- Python 3.6+
- No external dependencies (uses only standard library)

## Installation

No installation required! Just download the `simple_viewer.py` file and run it directly.

```bash
# Run directly
python simple_viewer.py your_results.json
```

## Tips

1. **Use `--failed` to focus on problematic tasks** when debugging
2. **Use `--conversation` to understand the flow** of successful vs failed interactions
3. **Use interactive mode** for exploratory analysis
4. **Pipe to `head` or `less`** for long outputs: `python simple_viewer.py file.json --list | head -20`

## Troubleshooting

- **File not found**: Ensure the JSON file path is correct
- **Invalid task number**: Task numbers are 1-based (not 0-based)
- **Large files**: Use `head` or `less` to view partial output for very large result files 