# Copyright Sierra

import os
import json
import random
import traceback
import time
from math import comb
import multiprocessing
from typing import List, Dict, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from tau_bench.envs import get_env
from tau_bench.agents.base import Agent
from tau_bench.types import EnvRunResult, RunConfig
from litellm import provider_list
from tau_bench.envs.user import UserStrategy
from tau_bench.types import RunMetadata




def run(config: RunConfig) -> List[EnvRunResult]:
    assert config.model_provider in provider_list, "Invalid model provider"
    assert config.user_model_provider in provider_list, "Invalid user model provider"
    assert config.agent_strategy in ["tool-calling", "act", "react", "few-shot"], "Invalid agent strategy"
    assert config.task_split in ["train", "test", "dev"], "Invalid task split"
    assert config.user_strategy in [item.value for item in UserStrategy], "Invalid user strategy"

    random.seed(config.seed)
    time_str = datetime.now().strftime("%m%d%H%M%S")
    ckpt_path = f"{config.log_dir}/{config.agent_strategy}-{config.model.split('/')[-1]}-{config.temperature}_range_{config.start_index}-{config.end_index}_user-{config.user_model}-{config.user_strategy}_{time_str}.json"
    # Ensure the log directory exists
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Also ensure the parent directory of the checkpoint file exists
    checkpoint_dir = os.path.dirname(ckpt_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)


    

    print(f"Loading user with strategy: {config.user_strategy}")
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=config,
    )
    end_index = (
        len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))
    )
    results: List[EnvRunResult] = []
    # lock = multiprocessing.Lock()
    
    # Calculate total tasks to run
    if config.task_ids and len(config.task_ids) > 0:
        total_tasks = len(config.task_ids) * config.num_trials
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
        print(f"Total tasks to run: {total_tasks} (tasks: {len(config.task_ids)}, trials: {config.num_trials})")
    else:
        total_tasks = (end_index - config.start_index) * config.num_trials
        print(
            f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})"
        )
        print(f"Total tasks to run: {total_tasks} (tasks: {end_index - config.start_index}, trials: {config.num_trials})")
    
    # Start overall progress tracking
    overall_start_time = time.time()
    overall_pbar = tqdm(
        total=total_tasks,
        desc="Overall Progress",
        unit="task",
        position=0,
        leave=True,
        ncols=100
    )
    
    try:
        for i in range(config.num_trials):
            if config.task_ids and len(config.task_ids) > 0:
                idxs = config.task_ids
            else:
                idxs = list(range(config.start_index, end_index))
            if config.shuffle:
                random.shuffle(idxs)

            # Trial progress tracking
            trial_pbar = tqdm(
                total=len(idxs),
                desc=f"Trial {i+1}/{config.num_trials}",
                unit="task",
                position=1,
                leave=False,
                ncols=100
            )

        def _run(idx: int) -> EnvRunResult:
            task_start_time = time.time()
            
            isolated_env = get_env(
                config.env,
                user_strategy=config.user_strategy,
                user_model=config.user_model,
                task_split=config.task_split,
                user_provider=config.user_model_provider,
                task_index=idx,
            )

            # Individual task progress tracking
            task_pbar = tqdm(
                total=30,  # max_num_steps
                desc=f"Task {idx}",
                unit="step",
                position=2,
                leave=False,
                ncols=100
            )
            
            try:
                res = agent.solve(
                    env=isolated_env,
                    task_index=idx,
                    progress_bar=task_pbar,
                )
                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=i,
                    # metadata=metadata,
                )
            except Exception as e:
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=i,
                    # metadata=metadata,
                )
            
            task_duration = time.time() - task_start_time
            task_pbar.close()
            
            # Update progress bars
            trial_pbar.update(1)
            overall_pbar.update(1)
            
            # Calculate and display ETA
            elapsed_time = time.time() - overall_start_time
            completed_tasks = overall_pbar.n
            if completed_tasks > 0:
                avg_time_per_task = elapsed_time / completed_tasks
                remaining_tasks = total_tasks - completed_tasks
                eta_seconds = avg_time_per_task * remaining_tasks
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                
                # Update progress bar description with ETA
                overall_pbar.set_description(
                    f"Overall Progress (ETA: {eta_str})"
                )
            
            print(
                "✅" if result.reward == 1 else "❌",
                f"task_id={idx}",
                f"(trial {i+1}/{config.num_trials})",
                f"[{task_duration:.1f}s]",
                result.info,
            )
            print("-----")
            
            # with lock:
            #     data = []
            #     if os.path.exists(ckpt_path):
            #         with open(ckpt_path, "r") as f:
            #             data = json.load(f)
            #     with open(ckpt_path, "w") as f:
            #         json.dump(data + [result.model_dump()], f, indent=2)
            return result

        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            res = list(executor.map(_run, idxs))
            results.extend(res)
        
            trial_pbar.close()

        overall_pbar.close()
        
        # Update metadata with final timing information
        total_duration = time.time() - overall_start_time
        avg_time_per_task = total_duration / total_tasks if total_tasks > 0 else 0
        
        # metadata.timestamp_end = datetime.now()
        # metadata.total_duration_seconds = total_duration
        
        print(f"\n⏱️  Total execution time: {timedelta(seconds=int(total_duration))}")
        print(f"📊 Average time per task: {avg_time_per_task:.1f} seconds")
        print(f"🚀 Tasks per second: {total_tasks / total_duration:.2f}")

        display_metrics(results)

        # Save results with updated metadata
        with open(ckpt_path, "w") as f:
            # # Update metadata in all results
            # for result in results:
            #     result.metadata = metadata
            json.dump([result.model_dump() for result in results], f, indent=2)
            print(f"\n📄 Results saved to {ckpt_path}\n")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print(f"💾 Saving partial results ({len(results)} tasks completed)")
        
        # Save partial results
        with open(ckpt_path, "w") as f:
            json.dump([result.model_dump() for result in results], f, indent=2)
        print(f"📄 Partial results saved to {ckpt_path}")
        
        # Re-raise the exception
        raise
    
    return results


def agent_factory(
    tools_info: List[Dict[str, Any]], wiki, config: RunConfig
) -> Agent:
    if config.agent_strategy == "tool-calling":
        # native tool calling
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent

        return ToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
            base_url=config.base_url,
        )
    elif config.agent_strategy == "act":
        # `act` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=False,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "react":
        # `react` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "few-shot":
        from tau_bench.agents.few_shot_agent import FewShotToolCallingAgent
        assert config.few_shot_displays_path is not None, "Few shot displays path is required for few-shot agent strategy"
        with open(config.few_shot_displays_path, "r") as f:
            few_shot_displays = [json.loads(line)["messages_display"] for line in f]

        return FewShotToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            few_shot_displays=few_shot_displays,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"Unknown agent strategy: {config.agent_strategy}")


def display_metrics(results: List[EnvRunResult]) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r.trial for r in results]))
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")
