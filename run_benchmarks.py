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
from typing import Dict, Optional, Tuple, List
from dotenv import load_dotenv


class BenchmarkRunner:
    """Main benchmark runner class"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        output_dir: Optional[str] = None,
        temperature: float = 0.0,
        proc_num: int = 4,
        user_model: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.proc_num = proc_num
        self.user_model = user_model or "openai/gpt-4o-20240806"  # Default user model

        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = (
            Path(output_dir).resolve()
            if output_dir
            else Path(f"results_{timestamp}").resolve()
        )
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
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

    def get_provider_from_model(self) -> str:
        """Determine provider from model name"""
        return self.get_provider_from_model_name(self.model_name)

    def get_provider_from_model_name(self, model_name: str) -> str:
        """Determine provider from any model name"""
        model_lower = model_name.lower()
        if "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "google"
        elif "mistral" in model_lower:
            return "mistral"
        elif "together" in model_lower:
            return "together_ai"
        else:
            return "openai"

    def setup_environment(self, benchmark_dir: Path):
        """Setup environment variables for benchmark execution"""
        env = os.environ.copy()

        # Unified API key pattern - all benchmarks use OpenAI-compatible API_KEY/BASE_URL
        env_vars = {
            "API_KEY": self.api_key,  # Primary agent model (OpenAI-compatible)
            "BASE_URL": self.base_url,  # Primary agent model base URL
            "USER_API_KEY": os.getenv("USER_API_KEY", self.api_key),  # User model (defaults to same as agent)
            "USER_BASE_URL": os.getenv("USER_BASE_URL", self.base_url),  # User model base URL
        }

        # Add RAPID_API_KEY if it exists in environment
        rapid_api_key = os.getenv("RAPID_API_KEY")
        if rapid_api_key:
            env_vars["RAPID_API_KEY"] = rapid_api_key

        env.update(env_vars)
        return env

    def create_env_file(self, benchmark_dir: Path, custom_vars: Optional[Dict] = None):
        """Create .env file for benchmarks that need it"""
        env_file = benchmark_dir / ".env"

        default_vars = {
            "API_KEY": self.api_key,  # Primary agent model (OpenAI-compatible)
            "BASE_URL": self.base_url,  # Primary agent model base URL
            "USER_API_KEY": os.getenv("USER_API_KEY", self.api_key),  # User model (OpenAI-compatible)
            "USER_BASE_URL": os.getenv("USER_BASE_URL", self.base_url),  # User model base URL
        }

        if custom_vars:
            default_vars.update(custom_vars)

        with open(env_file, "w") as f:
            for key, value in default_vars.items():
                f.write(f"{key}={value}\n")

        self.logger.info(f"Created .env file in {benchmark_dir}")

    def run_command(
        self, command: str, cwd: Path, timeout: int = 3600
    ) -> Tuple[bool, str]:
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
                universal_newlines=True,
            )

            output_lines = []

            # Read output line by line in real-time
            try:
                while True:
                    output = process.stdout.readline()
                    if output == "" and process.poll() is not None:
                        break
                    if output:
                        print(output.rstrip())  # Print to terminal in real-time
                        output_lines.append(output)

                # Wait for process to complete
                return_code = process.wait(timeout=timeout)
                full_output = "".join(output_lines)

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

    def run_drafterbench(self) -> bool:
        """Run DrafterBench"""
        benchmark_name = "DrafterBench"
        benchmark_dir = self.root_dir / "DrafterBench"

        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False

        self.logger.info(f"Running {benchmark_name}...")

        # Create benchmark-specific output directory
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        provider = self.get_provider_from_model()
        command = f"python evaluation.py --model {self.model_name} --model-provider {provider} --temperature {self.temperature} --vllm_url {self.base_url} --proc_num {self.proc_num} --result_dir {benchmark_output_dir}"

        success, output = self.run_command(command, benchmark_dir)

        # Save output
        output_file = benchmark_output_dir / "output.log"
        with open(output_file, "w") as f:
            f.write(output)
        return success

    def run_toolsandbox(self) -> bool:
        """Run ToolSandbox"""
        benchmark_name = "ToolSandbox"
        benchmark_dir = self.root_dir / "ToolSandbox"

        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False

        self.logger.info(f"Running {benchmark_name}...")

        # Create benchmark-specific output directory
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        # Note: ToolSandbox does not have --proc_num parameter
        command = f"tool_sandbox --user Dynamic --user_model_name {self.user_model} --agent Dynamic --model_name {self.model_name} --output_dir {benchmark_output_dir} --parallel {self.proc_num}"

        success, output = self.run_command(command, benchmark_dir)

        # Save output
        output_file = benchmark_output_dir / "output.log"
        with open(output_file, "w") as f:
            f.write(output)

        # Run average calculation if available
        # if success:
        #     avg_script = benchmark_dir / "cal_avg_benchmark.py"
        #     if avg_script.exists():
        #         self.logger.info("Calculating ToolSandbox average score...")
        #         avg_success, avg_output = self.run_command("python cal_avg_benchmark.py --log ", benchmark_dir)

        #         avg_file = self.output_dir / f"{benchmark_name}_avg_score.log"
        #         with open(avg_file, 'w') as f:
        #             f.write(avg_output)

        return success

    def run_nexusbench(self) -> bool:
        """Run NexusBench"""
        benchmark_name = "NexusBench"
        benchmark_dir = self.root_dir / "NexusBench"

        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False

        self.logger.info(f"Running {benchmark_name}...")

        # Create benchmark-specific output directory
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        # Create .env file
        self.create_env_file(benchmark_dir)

        command = f"nexusbench --client OpenAI --base_url {self.base_url} --api_key {self.api_key} --model {self.model_name} --benchmarks all --num_samples_parallel {self.proc_num} --output-dir {benchmark_output_dir}"

        success, output = self.run_command(command, benchmark_dir)

        # Save output
        output_file = benchmark_output_dir / "output.log"
        with open(output_file, "w") as f:
            f.write(output)

        # NexusBench now saves results locally via --output-dir parameter
        return success

    def run_cfbench(self) -> bool:
        """Run CFBench"""
        benchmark_name = "CFBench"
        benchmark_dir = self.root_dir / "ComplexFuncBench"

        if not benchmark_dir.exists():
            self.logger.warning(
                f"{benchmark_name} source directory not found, skipping..."
            )
            return False

        self.logger.info(f"Running {benchmark_name}...")

        # Create .env file for ComplexFuncBench
        self.create_env_file(benchmark_dir)

        # Create benchmark-specific output directory
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        command = f"python evaluation.py --model_name={self.model_name} --proc_num {self.proc_num} --log_dir {benchmark_output_dir} --eval_model {self.user_model}"

        success, output = self.run_command(command, benchmark_dir)

        # Save output
        output_file = benchmark_output_dir / "output.log"
        with open(output_file, "w") as f:
            f.write(output)
        return success

    def run_multichallenge(self) -> bool:
        """Run MultiChallenge"""
        benchmark_name = "MultiChallenge"
        benchmark_dir = self.root_dir / "multi_challenge"

        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False

        self.logger.info(f"Running {benchmark_name}...")

        # Create .env file for MultiChallenge
        self.create_env_file(benchmark_dir)

        # Create benchmark-specific output directory
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        provider = "openai"

        # Extract model name without provider prefix
        model_safe_name = self.model_name.replace("/", "_")

        command = f"python main.py --model-provider {provider} --provider-args model={self.model_name} temp={self.temperature} --attempts 1 --output-file {benchmark_output_dir}/{model_safe_name}_evaluation_results.txt --raw {benchmark_output_dir}/{model_safe_name}_detailed_results.csv --max-workers_response_gen {self.proc_num} --max-workers_eval {self.proc_num}"

        success, output = self.run_command(command, benchmark_dir)

        # Save output
        output_file = benchmark_output_dir / "output.log"
        with open(output_file, "w") as f:
            f.write(output)
        return success

    def run_acebench(self) -> bool:
        """Run ACEBench"""
        benchmark_name = "ACEBench"
        benchmark_dir = self.root_dir / "ACEBench"

        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False

        # Create benchmark-specific output directory if not exists
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Running {benchmark_name}...")

        # ACEBench uses standard API_KEY/BASE_URL pattern
        self.create_env_file(benchmark_dir)

        command = f"python generate.py --model {self.model_name} --user-model {self.user_model} --temperature {self.temperature} --category normal special agent --language en --num-threads {self.proc_num} --output-dir {benchmark_output_dir}"

        success, output = self.run_command(command, benchmark_dir)

        # Save generation output
        output_file = benchmark_output_dir / "generate_output.log"
        with open(output_file, "w") as f:
            f.write(output)

        # Run evaluation if generation succeeded
        if success:
            self.logger.info("Running ACEBench evaluation...")
            eval_command = f"python eval_main.py --model {self.model_name} --category normal special agent --language en --output-dir {benchmark_output_dir}"

            eval_success, eval_output = self.run_command(eval_command, benchmark_dir)

            # Save evaluation output
            eval_output_file = benchmark_output_dir / "eval_output.log"
            with open(eval_output_file, "w") as f:
                f.write(eval_output)

        return success

    def run_taubench(self) -> bool:
        """Run TauBench"""
        benchmark_name = "TauBench"
        benchmark_dir = self.root_dir / "tau-bench"

        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False

        self.logger.info(f"Running {benchmark_name}...")

        # Create benchmark-specific output directory
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        # TauBench uses standard API_KEY/BASE_URL pattern
        self.create_env_file(benchmark_dir)

        provider = self.get_provider_from_model()
        user_provider = self.get_provider_from_model_name(self.user_model)

        # Run both retail and airline environments
        all_success = True
        all_outputs = []

        for env_type in ["retail", "airline"]:
            self.logger.info(f"Running TauBench on {env_type} environment...")
            command = f"python run.py --agent-strategy tool-calling --env {env_type} --model {self.model_name} --model-provider {provider} --user-model {self.user_model} --user-model-provider {user_provider} --user-strategy llm --max-concurrency {self.proc_num} --temperature {self.temperature} --base-url {self.base_url} --log-dir {benchmark_output_dir}"

            env_success, env_output = self.run_command(
                command, benchmark_dir, timeout=7200
            )
            all_success = all_success and env_success
            all_outputs.append(f"=== {env_type.upper()} ENVIRONMENT ===\n{env_output}")

            if not env_success:
                self.logger.error(f"TauBench failed on {env_type} environment")

        # Combine all outputs
        output = "\n\n".join(all_outputs)
        success = all_success

        # Save output
        output_file = benchmark_output_dir / "output.log"
        with open(output_file, "w") as f:
            f.write(output)

        return success

    def run_bfcl(self) -> bool:
        """Run BFCL-v3"""
        benchmark_name = "BFCL"
        benchmark_dir = self.root_dir / "gorilla" / "berkeley-function-call-leaderboard"

        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False

        self.logger.info(f"Running {benchmark_name}...")

        # Create .env file for BFCL
        self.create_env_file(benchmark_dir)

        # Create benchmark-specific output directory
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        # Run generation
        command = f"bfcl generate --model {self.model_name} --num-threads {self.proc_num} --temperature {self.temperature} --result-dir {benchmark_output_dir}/result"

        success, output = self.run_command(
            command, benchmark_dir, timeout=7200
        )  # 2 hours timeout

        # Save generation output
        # Create benchmark-specific output directory if not exists
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        output_file = benchmark_output_dir / "generate_output.log"
        with open(output_file, "w") as f:
            f.write(output)

        # Run evaluation if generation succeeded
        if success:
            self.logger.info("Running BFCL evaluation...")
            eval_command = f"bfcl evaluate --model {self.model_name} --result-dir {benchmark_output_dir}/result --score-dir {benchmark_output_dir}/score"

            eval_success, eval_output = self.run_command(eval_command, benchmark_dir)

            # Save evaluation output
            eval_output_file = benchmark_output_dir / "eval_output.log"
            with open(eval_output_file, "w") as f:
                f.write(eval_output)

        return success

    def run_tau2bench(self) -> bool:
        """Run Tau2Bench (tau2-bench)"""
        benchmark_name = "Tau2Bench"
        benchmark_dir = self.root_dir / "tau2-bench"

        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False

        self.logger.info(f"Running {benchmark_name}...")

        # Create benchmark-specific output directory
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        # Create .env file
        self.create_env_file(benchmark_dir)

        # Run all three domains: retail, airline, telecom
        all_success = True
        all_outputs = []

        domains = ["retail", "airline", "telecom"]

        for domain in domains:
            self.logger.info(f"Running Tau2Bench on {domain} domain...")
            command = f"tau2 run --domain {domain} --agent-llm {self.model_name} --user-llm {self.user_model} --num-trials 1 --max-concurrency {self.proc_num} --base-url {self.base_url} --save-to {benchmark_output_dir}/{domain}_results"

            domain_success, domain_output = self.run_command(
                command, benchmark_dir, timeout=7200
            )
            all_success = all_success and domain_success
            all_outputs.append(f"=== {domain.upper()} DOMAIN ===\n{domain_output}")

            if not domain_success:
                self.logger.error(f"Tau2Bench failed on {domain} domain")

        # Combine all outputs
        output = "\n\n".join(all_outputs)

        # Save output
        output_file = benchmark_output_dir / "output.log"
        with open(output_file, "w") as f:
            f.write(output)

        self.copy_results(benchmark_dir, benchmark_name)
        return all_success

    def run_single_benchmark(self, benchmark_name: str) -> bool:
        """Run a single benchmark"""
        benchmark_methods = {
            "drafterbench": self.run_drafterbench,
            "toolsandbox": self.run_toolsandbox,
            "nexusbench": self.run_nexusbench,
            "cfbench": self.run_cfbench,
            "multichallenge": self.run_multichallenge,
            "acebench": self.run_acebench,
            "taubench": self.run_taubench,
            "tau2bench": self.run_tau2bench,
            "bfcl": self.run_bfcl,
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
            "drafterbench",
            "bfcl",
            "toolsandbox",
            "nexusbench",
            "cfbench",
            "acebench",
            "multichallenge",
            "taubench",
            "tau2bench",
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
                        status = "✓" if result else "✗"
                        self.logger.info(f"Completed: {benchmark} {status}")

                    except Exception as e:
                        self.logger.error(f"{benchmark} failed with exception: {e}")
                        results[benchmark] = False

        else:
            # Run benchmarks sequentially
            self.logger.info("Running benchmarks sequentially...")
            for i, benchmark in enumerate(benchmarks, 1):
                self.logger.info(
                    f"Starting benchmark {i}/{len(benchmarks)}: {benchmark}"
                )

                results[benchmark] = self.run_single_benchmark(benchmark)
                status = "✓" if results[benchmark] else "✗"
                self.logger.info(f"Completed {benchmark}: {status}")

        # Final summary
        total_duration = time.time() - start_time
        successful = sum(results.values())
        self.logger.info(
            f"All benchmarks completed: {successful}/{len(benchmarks)} successful (took {total_duration:.0f}s)"
        )
        return results

    def generate_summary(self, results: Dict[str, bool]):
        """Generate execution summary"""
        summary_file = self.output_dir / "summary.json"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "base_url": self.base_url,
            "results": results,
            "total_benchmarks": len(results),
            "successful_benchmarks": sum(results.values()),
            "failed_benchmarks": len(results) - sum(results.values()),
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Also create human-readable summary
        text_summary_file = self.output_dir / "summary.txt"
        with open(text_summary_file, "w") as f:
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


def load_env_config():
    """Load configuration from .env file if it exists"""
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file, override=True)
        print(f"Loaded configuration from {env_file}")
    else:
        print(
            "No .env file found. Please copy .env.example to .env and configure API credentials."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Universal Benchmark Runner for AgentHard Suite"
    )
    parser.add_argument(
        "model_name", help="Name of the model (e.g., openai/gpt-4o-20240806)"
    )
    parser.add_argument(
        "--benchmark", help="Specific benchmark to run (default: all)", default="all"
    )
    parser.add_argument(
        "--output-dir", default="./results", help="Output directory for results"
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Run benchmarks concurrently (use with caution for API limits)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model generation (default: 0.0)",
    )
    parser.add_argument(
        "--proc-num",
        type=int,
        default=4,
        help="Number of processes/threads for parallel execution (default: 4)",
    )
    parser.add_argument(
        "--user-model",
        help="User model for benchmarks that need it (default: from .env USER_MODEL or openai/gpt-4o-20240806)",
    )
    parser.add_argument(
        "--list-benchmarks", action="store_true", help="List available benchmarks"
    )

    # Load .env configuration first
    load_env_config()

    args = parser.parse_args()

    if args.list_benchmarks:
        print("Available benchmarks:")
        benchmarks = [
            "drafterbench - DrafterBench technical drawing revision tasks",
            "toolsandbox - ToolSandbox stateful tool use evaluation",
            "nexusbench - NexusBench function calling benchmark",
            "cfbench - CFBench comprehensive function calling evaluation",
            "multichallenge - MultiChallenge conversation evaluation",
            "acebench - ACEBench comprehensive agent evaluation",
            "taubench - TauBench tool-agent-user interaction (retail + airline)",
            "tau2bench - Tau2Bench conversational agents (retail + airline + telecom)",
            "bfcl - BFCL-v3 multi-turn function calling",
        ]
        for benchmark in benchmarks:
            print(f"  {benchmark}")
        return

    # Get API credentials from .env file (required)
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        print("Error: API_KEY not found in .env file. Please set API_KEY in .env file")
        sys.exit(1)

    if not base_url:
        print(
            "Error: BASE_URL not found in .env file. Please set BASE_URL in .env file"
        )
        sys.exit(1)

    # Get user model from command line or .env file
    user_model = args.user_model or os.getenv("USER_MODEL")

    runner = BenchmarkRunner(
        api_key=api_key,
        base_url=base_url,
        model_name=args.model_name,
        output_dir=args.output_dir,
        temperature=args.temperature,
        proc_num=args.proc_num,
        user_model=user_model,
    )

    runner.logger.info("Starting benchmark execution...")
    runner.logger.info(f"Model: {args.model_name}")
    runner.logger.info(f"User Model: {runner.user_model}")
    runner.logger.info(f"Base URL: {base_url}")
    runner.logger.info(f"Temperature: {args.temperature}")
    runner.logger.info(f"Processes/Threads: {args.proc_num}")
    runner.logger.info(f"Output directory: {runner.output_dir}")

    try:
        if args.benchmark == "all":
            results = runner.run_all_benchmarks(concurrent=args.concurrent)
        else:
            benchmark_result = runner.run_single_benchmark(args.benchmark)
            results = {args.benchmark: benchmark_result}

        # Generate summary
        runner.generate_summary(results)

        # Print results
        print(f"\n{'=' * 50}")
        print("BENCHMARK EXECUTION COMPLETED")
        print(f"{'=' * 50}")
        print(f"Results saved in: {runner.output_dir}")

        successful = sum(results.values())
        total = len(results)

        print(f"\nOverall Results: {successful}/{total} benchmarks passed")

        for benchmark, success in results.items():
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"  {benchmark:15} {status}")

        if successful < total:
            print(
                f"\n  {total - successful} benchmark(s) failed. Check logs in {runner.output_dir}"
            )
            sys.exit(1)
        else:
            print(f"\n All benchmarks completed successfully!")

    except KeyboardInterrupt:
        runner.logger.info("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        runner.logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
