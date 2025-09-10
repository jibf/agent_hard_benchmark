#!/usr/bin/env python3
"""
Server Deployment Script for Enhanced BFCL Functionality Analysis
Optimized for high-throughput parallel processing with proper error handling
"""

import json
import os
import sys
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import argparse
import psutil
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bfcl_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ServerDeploymentManager:
    def __init__(self, config_file: Optional[str] = None):
        self.config = self.load_config(config_file)
        self.pid_file = Path("bfcl_analysis.pid")
        self.start_time = datetime.now()
        
    def load_config(self, config_file: Optional[str]) -> dict:
        """Load deployment configuration"""
        default_config = {
            "num_workers": min(32, psutil.cpu_count()),  # Use all CPU cores up to 32
            "api_rate_limit": 60,  # requests per minute
            "timeout_per_case": 60,  # seconds
            "batch_size": 50,
            "max_retries": 3,
            "output_dir": "score",
            "backup_interval": 300,  # 5 minutes
            "memory_limit_gb": 16,
            "target_tasks": [
                'live_multiple',
                'multi_turn_long_context',
                'multi_turn_miss_func', 
                'live_irrelevance',
                'multi_turn_miss_param',
                'multi_turn_base',
                'irrelevance',
                'live_simple'
            ]
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                    logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}, using defaults")
        
        return default_config
    
    def check_system_resources(self) -> bool:
        """Check if system has adequate resources"""
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        logger.info(f"System Resources:")
        logger.info(f"  CPU cores: {psutil.cpu_count()}")
        logger.info(f"  Total memory: {memory_gb:.1f} GB")
        logger.info(f"  Available memory: {memory.available / (1024**3):.1f} GB")
        logger.info(f"  CPU usage: {psutil.cpu_percent()}%")
        
        if memory.available / (1024**3) < 4:  # Less than 4GB available
            logger.warning("Low memory available, reducing worker count")
            self.config["num_workers"] = min(8, self.config["num_workers"])
            return False
        
        return True
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def create_pid_file(self) -> bool:
        """Create PID file to prevent multiple instances"""
        if self.pid_file.exists():
            try:
                with open(self.pid_file, 'r') as f:
                    old_pid = int(f.read().strip())
                
                # Check if process is still running
                if psutil.pid_exists(old_pid):
                    logger.error(f"Another instance is running with PID {old_pid}")
                    return False
                else:
                    logger.info(f"Removing stale PID file (PID {old_pid} not running)")
                    self.pid_file.unlink()
            except Exception as e:
                logger.warning(f"Error checking PID file: {e}")
        
        # Create new PID file
        with open(self.pid_file, 'w') as f:
            f.write(str(os.getpid()))
        
        logger.info(f"Created PID file with PID {os.getpid()}")
        return True
    
    def cleanup_pid_file(self):
        """Remove PID file"""
        if self.pid_file.exists():
            self.pid_file.unlink()
            logger.info("Removed PID file")
    
    def monitor_progress(self, output_dir: Path) -> dict:
        """Monitor analysis progress"""
        progress = {}
        
        for task in self.config["target_tasks"]:
            result_file = output_dir / f"enhanced_functionality_analysis_{task}.json"
            temp_file = output_dir / f"temp_enhanced_functionality_analysis_{task}.json"
            
            if result_file.exists():
                progress[task] = "completed"
            elif temp_file.exists():
                try:
                    with open(temp_file, 'r') as f:
                        data = json.load(f)
                        progress[task] = f"in_progress_{len(data)}_cases"
                except:
                    progress[task] = "in_progress"
            else:
                progress[task] = "pending"
        
        return progress
    
    def backup_results(self, output_dir: Path):
        """Create backup of current results"""
        backup_dir = output_dir / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup completed results
        for file_path in output_dir.glob("enhanced_functionality_analysis_*.json"):
            if file_path.name != "enhanced_functionality_analysis_all_tasks.json":
                backup_path = backup_dir / file_path.name
                backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')
        
        logger.info(f"Created backup at {backup_dir}")
    
    def estimate_completion_time(self, progress: dict, start_time: datetime) -> str:
        """Estimate completion time based on current progress"""
        completed = sum(1 for status in progress.values() if status == "completed")
        total = len(progress)
        
        if completed == 0:
            return "Unable to estimate"
        
        elapsed = datetime.now() - start_time
        estimated_total = elapsed * (total / completed)
        remaining = estimated_total - elapsed
        
        return str(remaining).split('.')[0]  # Remove microseconds
    
    def run_analysis(self, resume_from: Optional[str] = None):
        """Run the enhanced analysis"""
        logger.info("="*80)
        logger.info("ENHANCED BFCL FUNCTIONALITY ANALYSIS - SERVER DEPLOYMENT")
        logger.info("="*80)
        
        # System checks
        if not self.check_system_resources():
            logger.warning("System resources may be insufficient for optimal performance")
        
        if not self.create_pid_file():
            logger.error("Failed to create PID file, exiting")
            return False
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        try:
            # Import and setup analyzer
            from enhanced_functionality_analyzer import EnhancedFunctionalityAnalyzer
            
            analyzer = EnhancedFunctionalityAnalyzer(
                num_workers=self.config["num_workers"]
            )
            
            # Setup output directory
            output_dir = Path(self.config["output_dir"])
            output_dir.mkdir(exist_ok=True)
            
            logger.info(f"Configuration:")
            logger.info(f"  Workers: {self.config['num_workers']}")
            logger.info(f"  Batch size: {self.config['batch_size']}")
            logger.info(f"  Rate limit: {self.config['api_rate_limit']} req/min")
            logger.info(f"  Output directory: {output_dir}")
            
            if resume_from:
                logger.info(f"  Resuming from: {resume_from}")
            
            # Start progress monitoring in background
            import threading
            
            def progress_monitor():
                last_backup = time.time()
                while True:
                    time.sleep(60)  # Check every minute
                    
                    progress = self.monitor_progress(output_dir)
                    completed = sum(1 for status in progress.values() if status == "completed")
                    total = len(progress)
                    
                    logger.info(f"Progress: {completed}/{total} tasks completed")
                    
                    # Create periodic backups
                    if time.time() - last_backup > self.config["backup_interval"]:
                        self.backup_results(output_dir)
                        last_backup = time.time()
                    
                    # Estimate completion
                    eta = self.estimate_completion_time(progress, self.start_time)
                    logger.info(f"ETA: {eta}")
                    
                    if completed == total:
                        logger.info("All tasks completed!")
                        break
            
            monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
            monitor_thread.start()
            
            # Run the actual analysis
            logger.info("Starting enhanced analysis...")
            analyzer.run_enhanced_analysis(resume_from=resume_from)
            
            # Final backup
            self.backup_results(output_dir)
            
            logger.info("Analysis completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return False
        
        finally:
            self.cleanup_pid_file()
    
    def shutdown(self):
        """Graceful shutdown procedure"""
        logger.info("Initiating graceful shutdown...")
        
        # Give processes time to finish current work
        time.sleep(5)
        
        # Final backup if needed
        output_dir = Path(self.config["output_dir"])
        if output_dir.exists():
            self.backup_results(output_dir)
        
        self.cleanup_pid_file()
        logger.info("Shutdown complete")

def main():
    parser = argparse.ArgumentParser(description='Enhanced BFCL Functionality Analysis - Server Deployment')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--resume', help='Resume from specific task')
    parser.add_argument('--workers', type=int, help='Number of workers to use')
    parser.add_argument('--dry-run', action='store_true', help='Show configuration without running')
    
    args = parser.parse_args()
    
    # Create deployment manager
    manager = ServerDeploymentManager(config_file=args.config)
    
    # Override workers if specified
    if args.workers:
        manager.config["num_workers"] = args.workers
    
    if args.dry_run:
        print("Configuration:")
        print(json.dumps(manager.config, indent=2))
        print(f"System CPU cores: {psutil.cpu_count()}")
        print(f"System memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        return
    
    # Run analysis
    success = manager.run_analysis(resume_from=args.resume)
    
    if success:
        logger.info("Deployment completed successfully")
        sys.exit(0)
    else:
        logger.error("Deployment failed")
        sys.exit(1)

if __name__ == "__main__":
    main()