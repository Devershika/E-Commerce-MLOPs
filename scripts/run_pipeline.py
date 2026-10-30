#!/usr/bin/env python3
"""
E-Commerce Demand Forecasting MLOps Pipeline Runner
===================================================
A unified CLI execution script to run tests, execute DVC pipelines,
launch the FastAPI backend, or start the Streamlit frontend.
"""

import argparse
import os
import subprocess
import sys


def get_python_exe():
    """Returns the virtual environment python if present, else sys.executable."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_win = os.path.join(root, ".venv", "Scripts", "python.exe")
    venv_unix = os.path.join(root, ".venv", "bin", "python")
    if os.path.exists(venv_win):
        return venv_win
    if os.path.exists(venv_unix):
        return venv_unix
    return sys.executable


def run_command(cmd, desc):
    """Executes a command and streams output."""
    print(f"\n=======================================================")
    print(f"[*] {desc}")
    print(f"[*] Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print(f"=======================================================\n")
    ret = subprocess.call(cmd, shell=isinstance(cmd, str))
    if ret != 0:
        print(f"\n[!] Command failed with exit code: {ret}")
        return False
    return True


def run_tests():
    """Executes pytest unit test suite."""
    py = get_python_exe()
    return run_command([py, "-m", "pytest", "tests/", "-v"], "Running Automated Pytest Test Suite")


def run_pipeline():
    """Executes the DVC preprocessing, training, and evaluation pipeline."""
    # Check if dvc is available, else run fallback direct scripts
    py = get_python_exe()
    print("[*] Executing Preprocessing & Feature Engineering...")
    s1 = run_command(
        [py, "scripts/preprocess_data.py", "--input", "Data/sales_data.csv", "--output", "Data/processed", "--test-ratio", "0.2"],
        "Stage 1: Preprocessing & Splitting Data",
    )
    if not s1:
        return False

    print("[*] Training Holt-Winters Champion Model...")
    s2 = run_command(
        [py, "save_model.py", "--input", "Data/processed/train.parquet", "--output", "models/holt_winters_model.joblib"],
        "Stage 2: Model Training & Serialization",
    )
    if not s2:
        return False

    print("[*] Evaluating Model on Test Split...")
    s3 = run_command(
        [py, "evaluate.py", "--input", "Data/processed/test.parquet", "--output", "reports/metrics.json"],
        "Stage 3: Generating Model Metrics Report",
    )
    return s3


def run_serving(host="127.0.0.1", port=8000):
    """Launches the FastAPI backend service."""
    py = get_python_exe()
    return run_command(
        [py, "-m", "uvicorn", "src.api.main:app", "--host", host, "--port", str(port), "--reload"],
        f"Starting FastAPI Prediction Service on http://{host}:{port}",
    )


def run_ui():
    """Launches the Streamlit frontend application."""
    py = get_python_exe()
    return run_command(
        [py, "-m", "streamlit", "run", "apps/app.py"],
        "Starting Streamlit Demand Forecasting Dashboard",
    )


def main():
    parser = argparse.ArgumentParser(
        description="E-Commerce Demand Forecasting MLOps Unified Execution Script",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--test", action="store_true", help="Run full automated unit test suite with pytest.")
    parser.add_argument("--pipeline", action="store_true", help="Run preprocessing, training, and evaluation pipeline.")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI prediction backend on port 8000.")
    parser.add_argument("--ui", action="store_true", help="Launch the Streamlit dashboard on port 8501.")
    parser.add_argument("--all", action="store_true", help="Run tests, pipeline, and launch backend & frontend.")
    parser.add_argument("--host", default="127.0.0.1", help="Host address for backend API (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port for backend API (default: 8000).")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    if args.pipeline:
        success = run_pipeline()
        sys.exit(0 if success else 1)

    if args.serve:
        run_serving(host=args.host, port=args.port)
        sys.exit(0)

    if args.ui:
        run_ui()
        sys.exit(0)

    if args.all:
        print("[*] Running end-to-end pipeline verification...")
        if not run_pipeline():
            sys.exit(1)
        if not run_tests():
            sys.exit(1)
        print("\n[+] Verification successful! To start serving, run:")
        print("    python scripts/run_pipeline.py --serve")
        print("    python scripts/run_pipeline.py --ui")


if __name__ == "__main__":
    main()
