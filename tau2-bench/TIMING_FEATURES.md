# Timing Features and Progress Tracking in Tau2-Bench

This document explains the enhanced timing features and progress tracking available in tau2-bench, which help you monitor the execution of your benchmark runs and estimate completion times, especially useful for long-running domains like telecom.

## Overview

Tau2-bench now includes comprehensive progress tracking with multiple levels of detail:

1. **Overall Progress**: Shows total progress across all trials and tasks
2. **Individual Task Progress**: Shows step-by-step progress within each task
3. **Time Estimation**: Real-time ETA calculations and performance metrics
4. **Task Completion Status**: Real-time display of task success/failure

## Features

### Multi-Level Progress Bars

The progress tracking system provides two levels of progress bars:

```
Overall Progress (ETA: 0:15:30): 45%|████▌     | 9/20 [02:30<03:05, 12.5s/task]
Task telecom_task_1: 67%|██████▋   | 33/50 [00:12<00:06, step=33, errors=0, done=False]
```

### Real-Time Information

Each progress bar provides real-time information:

- **Overall Progress**: Shows percentage complete, ETA, and average time per task
- **Task Progress**: Shows current step, errors, and completion status

### Time Estimation

The system calculates and updates time estimates in real-time:

- **ETA**: Estimated time to completion based on current progress
- **Average Time**: Average time per task for performance analysis
- **Tasks per Second**: Throughput metric for optimization

## Usage

### Basic Usage

The progress tracking is automatically enabled when you run tau2-bench:

```bash
python -m tau2 run --domain telecom --agent llm_agent --user user_simulator \
  --agent-llm openai/gpt-4o --user-llm openai/gpt-4o \
  --max-concurrency 2 --num-trials 1
```

### Example with Progress Tracking

```python
from tau2.data_model.simulation import RunConfig
from tau2.run import run_domain

config = RunConfig(
    domain="telecom",
    agent="llm_agent",
    user="user_simulator",
    task_ids=["telecom_task_1", "telecom_task_2"],
    llm_agent="openai/gpt-4o",
    llm_args_agent={"temperature": 0.0},
    llm_user="openai/gpt-4o",
    llm_args_user={"temperature": 0.0},
    num_trials=2,
    max_steps=100,
    max_errors=10,
    max_concurrency=2,
    seed=42,
    save_to="telecom_results",
)

results = run_domain(config)
```

### Running the Example

To see the progress tracking in action, run the example script:

```bash
python example_telecom_timing.py
```

This will run a small subset of telecom tasks to demonstrate all the progress tracking features.

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

### Task Progress Bar

```
Task telecom_task_1: 67%|██████▋   | 33/50 [00:12<00:06, step=33, errors=0, done=False]
```

- **Description**: Shows current task ID
- **Progress**: Visual progress bar for current task
- **Count**: Steps completed / maximum steps
- **Time**: Time spent on current task
- **Postfix**: Current step, errors, and completion status

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
- **High concurrency (5+)**: Focuses on overall progress only

### Task Selection

Use `--task-ids` to run specific tasks and see progress for those tasks:

```bash
python -m tau2 run --domain telecom --task-ids telecom_task_1 telecom_task_2 --max-concurrency 1
```

### Trial Configuration

The `--num-trials` parameter determines how many trials are run:

```bash
python -m tau2 run --domain telecom --num-trials 5 --max-concurrency 2
```

## Telecom Domain Specific Features

### Long-Running Task Handling

Telecom tasks can be particularly time-consuming due to their complexity. The progress tracking system provides:

- **Step-by-step progress**: See exactly which step the simulation is on
- **Error tracking**: Monitor error counts in real-time
- **ETA updates**: Get accurate time estimates as the simulation progresses

### Performance Optimization

For telecom domain specifically:

- **Use lower concurrency**: Telecom tasks benefit from detailed progress tracking
- **Monitor ETA**: Long-running tasks need accurate time estimates
- **Check step progress**: Complex tasks may take many steps to complete

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

You can implement custom progress tracking by modifying the orchestrator:

```python
def run(self, progress_bar=None) -> SimulationRun:
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

These are automatically installed when you install tau2-bench.

## Future Enhancements

Planned improvements to the progress tracking system:

- **Web-based dashboard**: Real-time web interface for monitoring
- **Detailed analytics**: Performance breakdowns by task type
- **Resource monitoring**: CPU, memory, and API usage tracking
- **Alert system**: Notifications for completion or errors
- **Export capabilities**: Progress data export for analysis 