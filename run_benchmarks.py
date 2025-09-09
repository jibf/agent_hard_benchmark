#!/usr/bin/env python3
"""
Universal Benchmark Runner for AgentHard Suite
A one-click solution to run all LLM tool-use benchmarks
"""

import os
import sys
import json
import argparse
import subprocess
import logging
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple
import shutil


class BenchmarkRunner:
    """Main benchmark runner class"""
    
    def __init__(self, api_key: str, base_url: str, model_name: str, output_dir: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) if output_dir else Path(f"results_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Store root directory
        self.root_dir = Path.cwd()
        
        # Track results
        self.results = {}
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "benchmark_run.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        
    def get_provider_from_model(self) -> str:
        """Determine provider from model name"""
        model_lower = self.model_name.lower()
        if 'claude' in model_lower:
            return 'anthropic'
        elif 'gemini' in model_lower:
            return 'google'
        elif 'mistral' in model_lower:
            return 'mistral'
        elif 'together' in model_lower:
            return 'together_ai'
        else:
            return 'openai'
    
    def setup_environment(self, benchmark_dir: Path):
        """Setup environment variables for benchmark execution"""
        env = os.environ.copy()
        
        # Common environment variables
        env_vars = {
            'OPENAI_API_KEY': self.api_key,
            'OPENAI_BASE_URL': self.base_url,
            'OPENAI_API_BASE': self.base_url,
            'VLLM_API_BASE': self.base_url,
            'GPT_AGENT_API_KEY': self.api_key,
            'GPT_BASE_URL': self.base_url,
            'GPT_API_KEY': self.api_key,
            'API_KEY': self.api_key,
            'BASE_URL': self.base_url,
        }
        
        env.update(env_vars)
        return env
    
    def create_env_file(self, benchmark_dir: Path, custom_vars: Optional[Dict] = None):
        """Create .env file for benchmarks that need it"""
        env_file = benchmark_dir / '.env'
        
        default_vars = {
            'OPENAI_API_KEY': self.api_key,
            'OPENAI_BASE_URL': self.base_url,
            'OPENAI_API_BASE': self.base_url,
            'API_KEY': self.api_key,
            'BASE_URL': self.base_url,
        }
        
        if custom_vars:
            default_vars.update(custom_vars)
            
        with open(env_file, 'w') as f:
            for key, value in default_vars.items():
                f.write(f"{key}={value}\n")
                
        self.logger.info(f"Created .env file in {benchmark_dir}")
    
    def run_command(self, command: str, cwd: Path, timeout: int = 3600) -> Tuple[bool, str]:
        """Execute a command with real-time output and return success status and output"""
        try:
            self.logger.info(f"Executing: {command}")
            self.logger.info(f"Working directory: {cwd}")
            
            env = self.setup_environment(cwd)
            
            # Use Popen for real-time output
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            output_lines = []
            
            # Read output line by line in real-time
            try:
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(output.rstrip())  # Print to terminal in real-time
                        output_lines.append(output)
                
                # Wait for process to complete
                return_code = process.wait(timeout=timeout)
                full_output = ''.join(output_lines)
                
                if return_code == 0:
                    self.logger.info(f"Command completed successfully")
                    return True, full_output
                else:
                    self.logger.error(f"Command failed with return code {return_code}")
                    return False, full_output
                    
            except subprocess.TimeoutExpired:
                process.kill()
                self.logger.error(f"Command timed out after {timeout} seconds")
                return False, f"Command timed out after {timeout} seconds"
                
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            return False, f"Error: {str(e)}"
    
    def copy_results(self, benchmark_dir: Path, benchmark_name: str):
        """Copy benchmark results to output directory"""
        result_dirs = ['results', 'result', 'data']
        
        for result_dir in result_dirs:
            src_path = benchmark_dir / result_dir
            if src_path.exists():
                dst_path = self.output_dir / f"{benchmark_name}_{result_dir}"
                try:
                    if src_path.is_dir():
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_path, dst_path)
                    self.logger.info(f"Copied results from {src_path} to {dst_path}")
                except Exception as e:
                    self.logger.error(f"Failed to copy results: {e}")
    
    def run_drafterbench(self) -> bool:
        """Run DrafterBench"""
        benchmark_name = "DrafterBench"
        benchmark_dir = self.root_dir / "DrafterBench"
        
        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False
        
        self.logger.info(f"Running {benchmark_name}...")
        
        provider = self.get_provider_from_model()
        command = f"python evaluation.py --model {self.model_name} --model-provider {provider} --temperature 0.0 --vllm_url {self.base_url}"
        
        success, output = self.run_command(command, benchmark_dir)
        
        # Save output
        output_file = self.output_dir / f"{benchmark_name}_output.log"
        with open(output_file, 'w') as f:
            f.write(output)

        self.copy_results(benchmark_dir, benchmark_name)
        return success

    def run_toolsandbox(self) -> bool:
        """Run ToolSandbox"""
        benchmark_name = "ToolSandbox"
        benchmark_dir = self.root_dir / "ToolSandbox"
        
        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False
        
        self.logger.info(f"Running {benchmark_name}...")

        # TODO: implement api_key and base_url and setup a mapping from model_path to agent
        command = "tool_sandbox --user GPT_4_o_2024_08_06 --agent GPT_4_o_2024_08_06"
        
        success, output = self.run_command(command, benchmark_dir)
        
        # Save output
        output_file = self.output_dir / f"{benchmark_name}_output.log"
        with open(output_file, 'w') as f:
            f.write(output)
        
        # Run average calculation if available
        if success:
            avg_script = benchmark_dir / "cal_avg_benchmark.py"
            if avg_script.exists():
                self.logger.info("Calculating ToolSandbox average score...")
                avg_success, avg_output = self.run_command("python cal_avg_benchmark.py", benchmark_dir)
                
                avg_file = self.output_dir / f"{benchmark_name}_avg_score.log"
                with open(avg_file, 'w') as f:
                    f.write(avg_output)
        
        self.copy_results(benchmark_dir, benchmark_name)
        return success
    
    def run_nexusbench(self) -> bool:
        """Run NexusBench"""
        benchmark_name = "NexusBench"
        benchmark_dir = self.root_dir / "NexusBench"
        
        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False
            
        self.logger.info(f"Running {benchmark_name}...")
        
        # Create .env file
        self.create_env_file(benchmark_dir)

        command = f"nexusbench --client OpenAI --base_url {self.base_url} --api_key {self.api_key} --model {self.model_name} --benchmarks all"

        success, output = self.run_command(command, benchmark_dir)
        
        # Save output
        output_file = self.output_dir / f"{benchmark_name}_output.log"
        with open(output_file, 'w') as f:
            f.write(output)
        
        self.copy_results(benchmark_dir, benchmark_name)
        return success
    
    # TODO: check how to run cfbench
    def run_cfbench(self) -> bool:
        """Run CFBench"""
        benchmark_name = "CFBench"
        benchmark_dir = self.root_dir / "ComplexFuncBench"

        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} source directory not found, skipping...")
            return False
            
        self.logger.info(f"Running {benchmark_name}...")
        
        command = f"python evaluation.py --model_name={self.model_name}"
        
        success, output = self.run_command(command, benchmark_dir)
        
        # Save output
        output_file = self.output_dir / f"{benchmark_name}_output.log"
        with open(output_file, 'w') as f:
            f.write(output)
        
        self.copy_results(benchmark_dir, benchmark_name)
        return success
    
    def run_multichallenge(self) -> bool:
        """Run MultiChallenge"""
        benchmark_name = "MultiChallenge"
        benchmark_dir = self.root_dir / "multi_challenge"
        
        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False
            
        self.logger.info(f"Running {benchmark_name}...")
        
        provider = self.get_provider_from_model()
        
        # Extract model name without provider prefix
        model_safe_name = self.model_name.replace('/', '_')
        
        command = f"python main.py --model-provider {provider} --provider-args model={self.model_name} temp=0 --attempts 1 --output-file results/{model_safe_name}_evaluation_results.txt --raw results/{model_safe_name}_detailed_results.csv"
        
        success, output = self.run_command(command, benchmark_dir)
        
        # Save output
        output_file = self.output_dir / f"{benchmark_name}_output.log"
        with open(output_file, 'w') as f:
            f.write(output)
        
        self.copy_results(benchmark_dir, benchmark_name)
        return success
    
    def run_acebench(self) -> bool:
        """Run ACEBench"""
        benchmark_name = "ACEBench"
        benchmark_dir = self.root_dir / "ACEBench"
        
        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False
            
        self.logger.info(f"Running {benchmark_name}...")
        
        # Create comprehensive .env file
        custom_vars = {
            'GPT_AGENT_API_KEY': self.api_key,
            'GPT_BASE_URL': self.base_url,
            'GPT_API_KEY': self.api_key,
            'DEEPSEEK_API_KEY': self.api_key,
            'DEEPSEEK_BASE_URL': self.base_url,
            'QWEN_API_KEY': self.api_key,
            'QWEN_BASE_URL': self.base_url,
        }
        self.create_env_file(benchmark_dir, custom_vars)
        
        # Run generation
        if 'local' in self.model_name.lower():
            self.logger.warning("Local model detected. Please ensure model path is available.")
            command = f"python generate.py --model {self.model_name} --model-path /path/to/model --category normal --language en --num-gpus 4"
        else:
            command = f"python generate.py --model {self.model_name} --category normal --language en"
        
        success, output = self.run_command(command, benchmark_dir)
        
        # Save generation output
        output_file = self.output_dir / f"{benchmark_name}_generate_output.log"
        with open(output_file, 'w') as f:
            f.write(output)
        
        # Run evaluation if generation succeeded
        if success:
            self.logger.info("Running ACEBench evaluation...")
            model_for_eval = self.model_name.replace('/', '-')
            eval_command = f"python eval_main.py --model {model_for_eval} --category normal --language en"
            
            eval_success, eval_output = self.run_command(eval_command, benchmark_dir)
            
            # Save evaluation output
            eval_output_file = self.output_dir / f"{benchmark_name}_eval_output.log"
            with open(eval_output_file, 'w') as f:
                f.write(eval_output)
        
        self.copy_results(benchmark_dir, benchmark_name)
        return success
    
    def run_taubench(self) -> bool:
        """Run TauBench"""
        benchmark_name = "TauBench"
        benchmark_dir = self.root_dir / "tau-bench"
        
        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False
            
        self.logger.info(f"Running {benchmark_name}...")
        
        # Create .env file with specific variables for TauBench
        custom_vars = {
            'ANTHROPIC_API_BASE': self.base_url,
            'VLLM_API_BASE': self.base_url,
        }
        self.create_env_file(benchmark_dir, custom_vars)
        
        provider = self.get_provider_from_model()
        user_model = "openai/gpt-4o-20240806"  # Use stable user model
        
        command = f"python run.py --agent-strategy tool-calling --env retail --model {self.model_name} --model-provider {provider} --user-model {user_model} --user-model-provider openai --user-strategy llm --max-concurrency 10"
        
        success, output = self.run_command(command, benchmark_dir, timeout=7200)  # 2 hours timeout
        
        # Save output
        output_file = self.output_dir / f"{benchmark_name}_output.log"
        with open(output_file, 'w') as f:
            f.write(output)
        
        self.copy_results(benchmark_dir, benchmark_name)
        return success
    
    def run_bfcl(self) -> bool:
        """Run BFCL-v3"""
        benchmark_name = "BFCL"
        benchmark_dir = self.root_dir / "gorilla" / "berkeley-function-call-leaderboard"
        
        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False
            
        self.logger.info(f"Running {benchmark_name}...")
        
        # Run generation
        # TODO: make num_threads configurable
        command = f"bfcl generate --model {self.model_name} --num-threads 4"
        
        success, output = self.run_command(command, benchmark_dir, timeout=7200)  # 2 hours timeout
        
        # Save generation output
        output_file = self.output_dir / f"{benchmark_name}_generate_output.log"
        with open(output_file, 'w') as f:
            f.write(output)
        
        # Run evaluation if generation succeeded
        if success:
            self.logger.info("Running BFCL evaluation...")
            eval_command = f"bfcl evaluate --model {self.model_name}"

            eval_success, eval_output = self.run_command(eval_command, benchmark_dir)
            
            # Save evaluation output
            eval_output_file = self.output_dir / f"{benchmark_name}_eval_output.log"
            with open(eval_output_file, 'w') as f:
                f.write(eval_output)
        
        self.copy_results(benchmark_dir, benchmark_name)
        return success
    
    def run_single_benchmark(self, benchmark_name: str) -> bool:
        """Run a single benchmark"""
        benchmark_methods = {
            'drafterbench': self.run_drafterbench,
            'toolsandbox': self.run_toolsandbox,
            'nexusbench': self.run_nexusbench,
            'cfbench': self.run_cfbench,
            'multichallenge': self.run_multichallenge,
            'acebench': self.run_acebench,
            'taubench': self.run_taubench,
            'bfcl': self.run_bfcl,
        }
        
        if benchmark_name not in benchmark_methods:
            self.logger.error(f"Unknown benchmark: {benchmark_name}")
            return False
        
        try:
            return benchmark_methods[benchmark_name]()
        except Exception as e:
            self.logger.error(f"Error running {benchmark_name}: {e}")
            return False
    
    def run_all_benchmarks(self, concurrent: bool = False) -> Dict[str, bool]:
        """Run all benchmarks"""
        benchmarks = [
            'drafterbench',
            'toolsandbox', 
            'nexusbench',
            'cfbench',
            'multichallenge',
            'acebench',
            'taubench',
            'bfcl'
        ]
        
        results = {}
        start_time = time.time()
        
        if concurrent:
            # Run benchmarks concurrently (be careful with API limits)
            self.logger.info("Running benchmarks concurrently...")
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_benchmark = {
                    executor.submit(self.run_single_benchmark, benchmark): benchmark 
                    for benchmark in benchmarks
                }
                
                for future in as_completed(future_to_benchmark):
                    benchmark = future_to_benchmark[future]
                    try:
                        result = future.result()
                        results[benchmark] = result
                        status = '✓' if result else '✗'
                        self.logger.info(f"Completed: {benchmark} {status}")
                            
                    except Exception as e:
                        self.logger.error(f"{benchmark} failed with exception: {e}")
                        results[benchmark] = False
                        
        else:
            # Run benchmarks sequentially
            self.logger.info("Running benchmarks sequentially...")
            for i, benchmark in enumerate(benchmarks, 1):
                self.logger.info(f"Starting benchmark {i}/{len(benchmarks)}: {benchmark}")
                
                results[benchmark] = self.run_single_benchmark(benchmark)
                status = '✓' if results[benchmark] else '✗'
                self.logger.info(f"Completed {benchmark}: {status}")
        
        # Final summary
        total_duration = time.time() - start_time
        successful = sum(results.values())
        self.logger.info(f"All benchmarks completed: {successful}/{len(benchmarks)} successful (took {total_duration:.0f}s)")
        return results
    
    def generate_summary(self, results: Dict[str, bool]):
        """Generate execution summary"""
        summary_file = self.output_dir / "summary.json"
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'base_url': self.base_url,
            'results': results,
            'total_benchmarks': len(results),
            'successful_benchmarks': sum(results.values()),
            'failed_benchmarks': len(results) - sum(results.values()),
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also create human-readable summary
        text_summary_file = self.output_dir / "summary.txt"
        with open(text_summary_file, 'w') as f:
            f.write("Benchmark Execution Summary\n")
            f.write("==========================\n\n")
            f.write(f"Timestamp: {summary['timestamp']}\n")
            f.write(f"Model: {summary['model']}\n")
            f.write(f"Base URL: {summary['base_url']}\n")
            f.write(f"Total Benchmarks: {summary['total_benchmarks']}\n")
            f.write(f"Successful: {summary['successful_benchmarks']}\n")
            f.write(f"Failed: {summary['failed_benchmarks']}\n\n")
            
            f.write("Individual Results:\n")
            f.write("------------------\n")
            for benchmark, success in results.items():
                status = "✓ PASSED" if success else "✗ FAILED"
                f.write(f"{benchmark:15} {status}\n")
        
        self.logger.info(f"Summary saved to {summary_file} and {text_summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Universal Benchmark Runner for AgentHard Suite')
    parser.add_argument('api_key', help='API key for the model provider')
    parser.add_argument('base_url', help='Base URL for the API (e.g., https://api.openai.com/v1)')
    parser.add_argument('model_name', help='Name of the model (e.g., openai/gpt-4o-20240806)')
    parser.add_argument('--benchmark', help='Specific benchmark to run (default: all)', default='all')
    parser.add_argument('--output-dir', help='Output directory for results')
    parser.add_argument('--concurrent', action='store_true', help='Run benchmarks concurrently (use with caution for API limits)')
    parser.add_argument('--list-benchmarks', action='store_true', help='List available benchmarks')
    
    args = parser.parse_args()
    
    if args.list_benchmarks:
        print("Available benchmarks:")
        benchmarks = [
            'drafterbench - DrafterBench technical drawing revision tasks',
            'toolsandbox - ToolSandbox stateful tool use evaluation', 
            'nexusbench - NexusBench function calling benchmark',
            'cfbench - CFBench comprehensive function calling evaluation',
            'multichallenge - MultiChallenge conversation evaluation',
            'acebench - ACEBench comprehensive agent evaluation',
            'taubench - TauBench tool-agent-user interaction',
            'bfcl - BFCL-v3 multi-turn function calling'
        ]
        for benchmark in benchmarks:
            print(f"  {benchmark}")
        return
    
    runner = BenchmarkRunner(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    runner.logger.info("Starting benchmark execution...")
    runner.logger.info(f"Model: {args.model_name}")
    runner.logger.info(f"Base URL: {args.base_url}")
    runner.logger.info(f"Output directory: {runner.output_dir}")
    
    try:
        if args.benchmark == 'all':
            results = runner.run_all_benchmarks(concurrent=args.concurrent)
        else:
            benchmark_result = runner.run_single_benchmark(args.benchmark)
            results = {args.benchmark: benchmark_result}
        
        # Generate summary
        runner.generate_summary(results)
        
        # Print results
        print(f"\n{'='*50}")
        print("BENCHMARK EXECUTION COMPLETED")
        print(f"{'='*50}")
        print(f"Results saved in: {runner.output_dir}")
        
        successful = sum(results.values())
        total = len(results)
        
        print(f"\nOverall Results: {successful}/{total} benchmarks passed")
        
        for benchmark, success in results.items():
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"  {benchmark:15} {status}")
        
        if successful < total:
            print(f"\n⚠️  {total - successful} benchmark(s) failed. Check logs in {runner.output_dir}")
            sys.exit(1)
        else:
            print(f"\n🎉 All benchmarks completed successfully!")
            
    except KeyboardInterrupt:
        runner.logger.info("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        runner.logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()