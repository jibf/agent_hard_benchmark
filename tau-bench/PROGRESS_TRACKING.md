# Progress Tracking in Tau-Bench

This document explains the enhanced progress tracking features available in tau-bench, which help you monitor the execution of your benchmark runs and estimate completion times.

## Overview

Tau-bench now includes comprehensive progress tracking with multiple levels of detail:

1. **Overall Progress**: Shows total progress across all trials and tasks
2. **Trial Progress**: Shows progress within each trial
3. **Task Progress**: Shows individual task execution with action tracking
4. **Time Estimation**: Real-time ETA calculations and performance metrics

## Features

### Multi-Level Progress Bars

The progress tracking system provides three levels of progress bars:

```
Overall Progress (ETA: 0:15:30): 45%|████▌     | 9/20 [02:30<03:05, 12.5s/task]
Trial 1/2: 100%|██████████| 3/3 [00:45<00:00, 15.0s/task]
Task 2: 67%|██████▋   | 20/30 [00:12<00:06, action=search_products, reward=0.00, done=False]
```

### Real-Time Information

Each progress bar provides real-time information:

- **Overall Progress**: Shows percentage complete, ETA, and average time per task
- **Trial Progress**: Shows current trial and tasks completed within that trial
- **Task Progress**: Shows current action, reward, and completion status

### Time Estimation

The system calculates and updates time estimates in real-time:

- **ETA**: Estimated time to completion based on current progress
- **Average Time**: Average time per task for performance analysis
- **Tasks per Second**: Throughput metric for optimization

## Usage

### Basic Usage

The progress tracking is automatically enabled when you run tau-bench:

```bash
python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --user-model gpt-4o --user-model-provider openai --user-strategy llm --max-concurrency 5
```

### Example with Progress Tracking

```python
from tau_bench.run import run
from tau_bench.types import RunConfig

config = RunConfig(
    model_provider="openai",
    user_model_provider="openai",
    model="gpt-4o",
    user_model="gpt-4o",
    num_trials=3,
    env="retail",
    agent_strategy="tool-calling",
    temperature=0.0,
    task_split="test",
    start_index=0,
    end_index=10,
    max_concurrency=2,
    seed=42,
    user_strategy="llm",
)

results = run(config)
```

### Running the Example

To see the progress tracking in action, run the example script:

```bash
python example_progress_tracking.py
```

This will run a small subset of tasks to demonstrate all the progress tracking features.

## Progress Bar Details

### Overall Progress Bar

```
Overall Progress (ETA: 0:15:30): 45%|████▌     | 9/20 [02:30<03:05, 12.5s/task]
```

- **Description**: Shows "Overall Progress" with current ETA
- **Progress**: Visual progress bar with percentage
- **Count**: Completed tasks / total tasks
- **Time**: Elapsed time and estimated remaining time
- **Rate**: Average time per task

### Trial Progress Bar

```
Trial 1/2: 100%|██████████| 3/3 [00:45<00:00, 15.0s/task]
```

- **Description**: Shows current trial number
- **Progress**: Visual progress bar for current trial
- **Count**: Tasks completed in current trial / total tasks in trial
- **Time**: Time spent on current trial
- **Rate**: Average time per task in current trial

### Task Progress Bar

```
Task 2: 67%|██████▋   | 20/30 [00:12<00:06, action=search_products, reward=0.00, done=False]
```

- **Description**: Shows current task ID
- **Progress**: Visual progress bar for current task
- **Count**: Steps completed / maximum steps
- **Time**: Time spent on current task
- **Postfix**: Current action, reward, and completion status

## Performance Metrics

At the end of each run, the system displays comprehensive performance metrics:

```
⏱️  Total execution time: 0:45:30
📊 Average time per task: 136.5 seconds
🚀 Tasks per second: 0.007
```

## Configuration Options

### Concurrency Control

The `--max-concurrency` parameter affects how progress bars are displayed:

- **Low concurrency (1-2)**: Shows detailed task-level progress
- **High concurrency (5+)**: Focuses on overall and trial progress

### Task Selection

Use `--task-ids` to run specific tasks and see progress for those tasks:

```bash
python run.py --task-ids 1 3 5 --max-concurrency 1
```

### Trial Configuration

The `--num-trials` parameter determines how many trials are run:

```bash
python run.py --num-trials 5 --max-concurrency 2
```

## Troubleshooting

### Progress Bars Not Showing

1. **Check tqdm installation**: Ensure tqdm is installed: `pip install tqdm`
2. **Terminal compatibility**: Some terminals may not display progress bars correctly
3. **Concurrency issues**: High concurrency may limit detailed progress display

### ETA Not Updating

- **Early in run**: ETA may not be accurate until several tasks complete
- **Variable task times**: ETA assumes similar task completion times
- **Network delays**: API response times can vary significantly

### Performance Issues

- **Reduce concurrency**: Lower `--max-concurrency` for more detailed progress
- **Monitor API limits**: High concurrency may hit API rate limits
- **Check network**: Slow network can affect progress tracking accuracy

## Advanced Usage

### Custom Progress Tracking

You can implement custom progress tracking by modifying the agent's solve method:

```python
def solve(self, env, task_index=None, max_num_steps=30, progress_bar=None):
    # Your custom progress tracking logic
    if progress_bar is not None:
        progress_bar.update(1)
        progress_bar.set_postfix({
            'custom_metric': value,
            'status': status
        })
```

### Integration with External Monitoring

The progress tracking can be integrated with external monitoring systems by capturing the progress bar output or implementing custom callbacks.

## Dependencies

The progress tracking feature requires:

- `tqdm>=4.65.0`: For progress bar display
- `time`: For timing calculations
- `datetime`: For ETA formatting

These are automatically installed when you install tau-bench.

## Future Enhancements

Planned improvements to the progress tracking system:

- **Web-based dashboard**: Real-time web interface for monitoring
- **Detailed analytics**: Performance breakdowns by task type
- **Resource monitoring**: CPU, memory, and API usage tracking
- **Alert system**: Notifications for completion or errors
- **Export capabilities**: Progress data export for analysis 