#!/usr/bin/env python3
"""
Gradient-based optimization for fixed probability routing policy.

This script performs iterative optimization of router probability distributions
using numerical gradients calculated from benchmark metrics.
"""

import argparse
import subprocess
import json
import time
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import signal
import sys
import os
from datetime import datetime


class GradientOptimizer:
    """Optimize fixed probability routing using gradient descent/ascent."""
    
    def __init__(
        self,
        num_workers: int,
        initial_probs: List[float],
        workers: List[str],
        router_port: int = 29001,
        router_host: str = "0.0.0.0",
        request_config: Dict = None,
        optimization_config: Dict = None
    ):
        """
        Initialize the gradient optimizer.
        
        Args:
            num_workers: Number of worker nodes
            initial_probs: Initial probability distribution
            workers: List of worker URLs
            router_port: Port for the router
            router_host: Host for the router
            request_config: Configuration for benchmark requests
            optimization_config: Configuration for optimization process
        """
        self.num_workers = num_workers
        self.current_probs = np.array(initial_probs)
        self.workers = workers
        self.router_port = router_port
        self.router_host = router_host
        
        # Default request configuration
        self.request_config = request_config or {
            "num_requests": 100,
            "request_rate": 20.0,
            "dataset": "random",
            "input_len": 512,
            "output_len": 50,
            "range_ratio": 0.25
        }
        
        # Default optimization configuration
        self.opt_config = optimization_config or {
            "perturbation": 0.01,
            "learning_rate": 0.01,
            "objective": "maximize_throughput",  # or "minimize_latency" or "balanced"
            "throughput_weight": 0.5,  # For balanced objective
            "latency_weight": 0.5,     # For balanced objective
        }
        
        # Router process handle
        self.router_process = None
        
        # Optimization history
        self.history = []
        
        # Output directory for results
        self.output_dir = Path(f"/nas/ganluo/sglang/gradient_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Optimization output directory: {self.output_dir}")
        
    def normalize_probabilities(self, probs: np.ndarray) -> np.ndarray:
        """Normalize probabilities to sum to 1.0."""
        probs = np.maximum(probs, 0.001)  # Ensure no negative or zero probabilities
        return probs / probs.sum()
    
    def start_router(self, probabilities: List[float]) -> subprocess.Popen:
        """Start the router with given probability distribution."""
        # Stop existing router if running
        self.stop_router()
        
        # Prepare router command
        cmd = [
            "python", "/nas/ganluo/sglang/start_router.py",
            "--policy", "fixed_probability",
            "--host", self.router_host,
            "--port", str(self.router_port),
            "--log-level", "INFO",
            "--workers"
        ] + self.workers + [
            "--enable-tracking",
            "--max-trace-entries", "100000",
            "--trace-ttl", "3600",
            "--fixed-probabilities"
        ] + [str(p) for p in probabilities]
        
        print(f"Starting router with probabilities: {probabilities}")
        print(f"Command: {' '.join(cmd)}")
        
        # Start router process
        self.router_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for router to start
        time.sleep(5)
        
        # Verify router is running
        try:
            import requests
            response = requests.get(f"http://{self.router_host}:{self.router_port}/health")
            if response.status_code != 200:
                raise Exception("Router health check failed")
            print("Router started successfully")
        except Exception as e:
            print(f"Router startup failed: {e}")
            self.stop_router()
            raise
        
        return self.router_process
    
    def stop_router(self):
        """Stop the router process if running."""
        if self.router_process:
            print("Stopping router...")
            self.router_process.terminate()
            try:
                self.router_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.router_process.kill()
                self.router_process.wait()
            self.router_process = None
            time.sleep(2)  # Wait for port to be released
    
    def run_benchmark(self, probabilities: List[float]) -> Dict[str, float]:
        """
        Run benchmark with given probabilities and return metrics.
        
        Returns:
            Dict containing PREFILL_THROUGHPUT, DECODE_THROUGHPUT, AVG_LATENCY
        """
        # Start router with new probabilities
        self.start_router(probabilities)
        
        # Prepare output CSV path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"benchmark_{timestamp}.csv"
        
        # Prepare benchmark command
        cmd = [
            "python", "/nas/ganluo/sglang/send_req.py",
            "--num-requests", str(self.request_config["num_requests"]),
            "--request-rate", str(self.request_config["request_rate"]),
            "--dataset", self.request_config["dataset"],
            "--router-url", f"http://{self.router_host}:{self.router_port}",
            "--output", str(csv_path)
        ]
        
        # Add dataset-specific parameters
        if self.request_config["dataset"] == "random":
            cmd.extend([
                "--input-len", str(self.request_config["input_len"]),
                "--output-len", str(self.request_config["output_len"]),
                "--range-ratio", str(self.request_config["range_ratio"])
            ])
        
        print(f"\nRunning benchmark...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run benchmark
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout
            )
            print("Benchmark completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Benchmark failed: {e}")
            print(f"stderr: {e.stderr}")
            self.stop_router()
            raise
        except subprocess.TimeoutExpired:
            print("Benchmark timed out")
            self.stop_router()
            raise
        
        # Parse CSV to get metrics from column averages
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                raise ValueError("CSV file is empty")
            
            # Calculate column averages for more stable metrics
            # Filter out NaN values and zeros for more accurate averages
            prefill_values = df["PREFILL_THROUGHPUT"].dropna()
            decode_values = df["DECODE_THROUGHPUT"].dropna()
            latency_values = df["AVG_LATENCY"].dropna()
            
            # Remove zeros (requests that haven't completed yet)
            prefill_values = prefill_values[prefill_values > 0]
            decode_values = decode_values[decode_values > 0]
            latency_values = latency_values[latency_values > 0]
            
            # Calculate averages
            metrics = {
                "PREFILL_THROUGHPUT": float(prefill_values.mean()) if len(prefill_values) > 0 else 0,
                "DECODE_THROUGHPUT": float(decode_values.mean()) if len(decode_values) > 0 else 0,
                "AVG_LATENCY": float(latency_values.mean()) if len(latency_values) > 0 else 0
            }
            
            print(f"Metrics (column averages): PREFILL={metrics['PREFILL_THROUGHPUT']:.2f} tok/s, "
                  f"DECODE={metrics['DECODE_THROUGHPUT']:.2f} tok/s, "
                  f"AVG_LATENCY={metrics['AVG_LATENCY']:.3f}s")
            print(f"  Based on {len(prefill_values)} prefill, {len(decode_values)} decode, "
                  f"{len(latency_values)} latency samples")
            
            return metrics
            
        except Exception as e:
            print(f"Failed to parse CSV: {e}")
            self.stop_router()
            raise
        finally:
            self.stop_router()
    
    def calculate_objective(self, metrics: Dict[str, float]) -> float:
        """
        Calculate objective value from metrics.
        
        Returns:
            Objective value (higher is better)
        """
        objective = self.opt_config["objective"]
        
        if objective == "maximize_throughput":
            # Maximize total throughput
            return metrics["PREFILL_THROUGHPUT"] + metrics["DECODE_THROUGHPUT"]
        
        elif objective == "minimize_latency":
            # Minimize latency (return negative for maximization)
            return -metrics["AVG_LATENCY"]
        
        elif objective == "balanced":
            # Balanced objective: weighted sum of throughput and latency
            throughput_score = metrics["PREFILL_THROUGHPUT"] + metrics["DECODE_THROUGHPUT"]
            latency_score = 1.0 / (metrics["AVG_LATENCY"] + 0.001)  # Avoid division by zero
            
            return (self.opt_config["throughput_weight"] * throughput_score + 
                   self.opt_config["latency_weight"] * latency_score)
        
        else:
            raise ValueError(f"Unknown objective: {objective}")
    
    def calculate_gradients(self, current_probs: np.ndarray) -> np.ndarray:
        """
        Calculate numerical gradients using finite differences.
        
        Returns:
            Gradient vector for the objective function
        """
        perturbation = self.opt_config["perturbation"]
        gradients = np.zeros_like(current_probs)
        
        # Get baseline metrics
        print("\n" + "="*60)
        print("Calculating baseline metrics...")
        baseline_metrics = self.run_benchmark(current_probs.tolist())
        baseline_objective = self.calculate_objective(baseline_metrics)
        print(f"Baseline objective: {baseline_objective:.4f}")
        
        # Calculate gradient for each probability
        for i in range(len(current_probs)):
            print("\n" + "-"*40)
            print(f"Calculating gradient for position {i+1}/{len(current_probs)}...")
            
            # Create perturbed probabilities
            perturbed_probs = current_probs.copy()
            perturbed_probs[i] += perturbation
            
            # Renormalize to maintain sum = 1
            perturbed_probs = self.normalize_probabilities(perturbed_probs)
            
            # Run benchmark with perturbed probabilities
            perturbed_metrics = self.run_benchmark(perturbed_probs.tolist())
            perturbed_objective = self.calculate_objective(perturbed_metrics)
            
            # Calculate gradient (finite difference)
            gradients[i] = (perturbed_objective - baseline_objective) / perturbation
            print(f"Gradient[{i}]: {gradients[i]:.6f}")
        
        return gradients
    
    def update_probabilities(
        self, 
        current_probs: np.ndarray, 
        gradients: np.ndarray, 
        step_size: float
    ) -> np.ndarray:
        """
        Update probabilities using gradient ascent.
        
        Returns:
            Updated probability vector
        """
        # Gradient ascent (to maximize objective)
        new_probs = current_probs + step_size * gradients
        
        # Normalize to ensure valid probability distribution
        new_probs = self.normalize_probabilities(new_probs)
        
        return new_probs
    
    def optimize(
        self, 
        max_iterations: int = 50, 
        tolerance: float = 1e-4,
        min_improvement: float = 0.001
    ):
        """
        Main optimization loop.
        
        Args:
            max_iterations: Maximum number of optimization iterations
            tolerance: Convergence tolerance for probability changes
            min_improvement: Minimum improvement required to continue
        """
        print("\n" + "="*60)
        print("Starting Gradient-Based Optimization")
        print("="*60)
        print(f"Initial probabilities: {self.current_probs}")
        print(f"Objective: {self.opt_config['objective']}")
        print(f"Learning rate: {self.opt_config['learning_rate']}")
        print(f"Perturbation: {self.opt_config['perturbation']}")
        
        best_probs = self.current_probs.copy()
        best_objective = float('-inf')
        no_improvement_count = 0
        
        try:
            for iteration in range(max_iterations):
                print(f"\n{'='*60}")
                print(f"ITERATION {iteration + 1}/{max_iterations}")
                print(f"{'='*60}")
                print(f"Current probabilities: {self.current_probs}")
                
                # Calculate gradients
                gradients = self.calculate_gradients(self.current_probs)
                print(f"\nGradients: {gradients}")
                
                # Update probabilities
                new_probs = self.update_probabilities(
                    self.current_probs,
                    gradients,
                    self.opt_config["learning_rate"]
                )
                print(f"Updated probabilities: {new_probs}")
                
                # Evaluate new probabilities
                print("\nEvaluating updated probabilities...")
                metrics = self.run_benchmark(new_probs.tolist())
                objective = self.calculate_objective(metrics)
                print(f"New objective: {objective:.4f}")
                
                # Record history
                self.history.append({
                    "iteration": iteration + 1,
                    "probabilities": new_probs.tolist(),
                    "objective": objective,
                    "prefill_throughput": metrics["PREFILL_THROUGHPUT"],
                    "decode_throughput": metrics["DECODE_THROUGHPUT"],
                    "avg_latency": metrics["AVG_LATENCY"],
                    "gradients": gradients.tolist()
                })
                
                # Check for improvement
                if objective > best_objective + min_improvement:
                    improvement = objective - best_objective
                    print(f"✅ Improvement: {improvement:.4f}")
                    best_objective = objective
                    best_probs = new_probs.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    print(f"⚠️ No significant improvement (count: {no_improvement_count})")
                
                # Check convergence
                prob_change = np.abs(new_probs - self.current_probs).max()
                print(f"Max probability change: {prob_change:.6f}")
                
                if prob_change < tolerance:
                    print("\n✅ Converged: probability changes below tolerance")
                    break
                
                if no_improvement_count >= 3:
                    print("\n⚠️ Stopping: no improvement for 3 iterations")
                    break
                
                # Update current probabilities
                self.current_probs = new_probs
                
                # Save intermediate results
                self.save_results()
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Optimization interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Optimization failed: {e}")
            raise
        finally:
            # Ensure router is stopped
            self.stop_router()
            
            # Save final results
            self.save_results()
            
            print("\n" + "="*60)
            print("OPTIMIZATION COMPLETE")
            print("="*60)
            print(f"Best probabilities: {best_probs}")
            print(f"Best objective: {best_objective:.4f}")
            print(f"Total iterations: {len(self.history)}")
            print(f"Results saved to: {self.output_dir}")
    
    def save_results(self):
        """Save optimization history and results."""
        # Save history as JSON
        history_path = self.output_dir / "optimization_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save history as CSV for easy analysis
        if self.history:
            csv_path = self.output_dir / "optimization_results.csv"
            rows = []
            for record in self.history:
                row = {
                    "iteration": record["iteration"],
                    "objective": record["objective"],
                    "prefill_throughput": record["prefill_throughput"],
                    "decode_throughput": record["decode_throughput"],
                    "avg_latency": record["avg_latency"]
                }
                # Add individual probabilities
                for i, prob in enumerate(record["probabilities"]):
                    row[f"prob_{i}"] = prob
                # Add gradients
                for i, grad in enumerate(record["gradients"]):
                    row[f"gradient_{i}"] = grad
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            
            # Print summary
            print(f"\nOptimization summary saved to: {csv_path}")
            print(df[["iteration", "objective", "prefill_throughput", "decode_throughput", "avg_latency"]].to_string())


def main():
    """Main entry point for the gradient optimizer."""
    parser = argparse.ArgumentParser(
        description="Gradient-based optimization for fixed probability routing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Worker configuration
    parser.add_argument(
        "--workers",
        nargs="+",
        default=["http://localhost:30001", "http://localhost:30002", "http://localhost:30003"],
        help="Worker URLs"
    )
    
    # Initial probabilities
    parser.add_argument(
        "--initial-probs",
        type=float,
        nargs="+",
        help="Initial probability distribution (must sum to 1.0)"
    )
    
    # Router configuration
    parser.add_argument(
        "--router-port",
        type=int,
        default=29001,
        help="Router port"
    )
    
    parser.add_argument(
        "--router-host",
        type=str,
        default="0.0.0.0",
        help="Router host"
    )
    
    # Request configuration
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests per benchmark"
    )
    
    parser.add_argument(
        "--request-rate",
        type=float,
        default=20.0,
        help="Request rate (req/s)"
    )
    
    parser.add_argument(
        "--input-len",
        type=int,
        default=512,
        help="Average input length for random dataset"
    )
    
    parser.add_argument(
        "--output-len",
        type=int,
        default=50,
        help="Average output length for random dataset"
    )
    
    # Optimization configuration
    parser.add_argument(
        "--objective",
        choices=["maximize_throughput", "minimize_latency", "balanced"],
        default="maximize_throughput",
        help="Optimization objective"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for gradient ascent"
    )
    
    parser.add_argument(
        "--perturbation",
        type=float,
        default=0.01,
        help="Perturbation size for gradient calculation"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum optimization iterations"
    )
    
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Convergence tolerance"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    num_workers = len(args.workers)
    
    # Set default initial probabilities if not provided
    if args.initial_probs is None:
        args.initial_probs = [1.0 / num_workers] * num_workers
    
    # Validate probability count
    if len(args.initial_probs) != num_workers:
        parser.error(f"Number of probabilities ({len(args.initial_probs)}) must match number of workers ({num_workers})")
    
    # Validate probability sum
    prob_sum = sum(args.initial_probs)
    if abs(prob_sum - 1.0) > 0.001:
        parser.error(f"Initial probabilities must sum to 1.0 (got {prob_sum})")
    
    # Create request configuration
    request_config = {
        "num_requests": args.num_requests,
        "request_rate": args.request_rate,
        "dataset": "random",
        "input_len": args.input_len,
        "output_len": args.output_len,
        "range_ratio": 0.25
    }
    
    # Create optimization configuration
    optimization_config = {
        "perturbation": args.perturbation,
        "learning_rate": args.learning_rate,
        "objective": args.objective,
        "throughput_weight": 0.5,
        "latency_weight": 0.5,
    }
    
    # Create optimizer
    optimizer = GradientOptimizer(
        num_workers=num_workers,
        initial_probs=args.initial_probs,
        workers=args.workers,
        router_port=args.router_port,
        router_host=args.router_host,
        request_config=request_config,
        optimization_config=optimization_config
    )
    
    # Set up signal handler for clean shutdown
    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal. Cleaning up...")
        optimizer.stop_router()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run optimization
    try:
        optimizer.optimize(
            max_iterations=args.max_iterations,
            tolerance=args.tolerance
        )
    except Exception as e:
        print(f"Optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()