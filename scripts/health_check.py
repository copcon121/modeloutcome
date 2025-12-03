#!/usr/bin/env python3
"""
Comprehensive System Health Check
=================================
Validates all components of the S4_LDN_ASM_LowShift pipeline.

Usage:
    python scripts/health_check.py
    python scripts/health_check.py --verbose
    python scripts/health_check.py --api http://localhost:8000
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ==============================================================================
# HEALTH CHECK RESULTS
# ==============================================================================

class HealthCheckResult:
    def __init__(self, name: str, status: str, message: str, duration: float = 0):
        self.name = name
        self.status = status  # "PASS", "FAIL", "WARN"
        self.message = message
        self.duration = duration

    def __str__(self):
        icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(self.status, "?")
        return f"{icon} {self.name}: {self.message}"


# ==============================================================================
# HEALTH CHECKS
# ==============================================================================

def check_python_version() -> HealthCheckResult:
    """Check Python version"""
    start = time.time()
    version = sys.version_info
    
    if version >= (3, 9):
        return HealthCheckResult(
            "Python Version",
            "PASS",
            f"Python {version.major}.{version.minor}.{version.micro}",
            time.time() - start
        )
    else:
        return HealthCheckResult(
            "Python Version",
            "FAIL",
            f"Python {version.major}.{version.minor} (requires 3.9+)",
            time.time() - start
        )


def check_dependencies() -> HealthCheckResult:
    """Check required dependencies"""
    start = time.time()
    missing = []
    
    required = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("requests", "requests"),
    ]
    
    for name, module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(name)
    
    if not missing:
        return HealthCheckResult(
            "Dependencies",
            "PASS",
            f"All {len(required)} packages installed",
            time.time() - start
        )
    else:
        return HealthCheckResult(
            "Dependencies",
            "FAIL",
            f"Missing: {', '.join(missing)}",
            time.time() - start
        )


def check_asm_model() -> HealthCheckResult:
    """Check ASM model file"""
    start = time.time()
    model_path = ROOT / "output/asm_models_v1/ASM-GRU64-v1.0-C3.pt"
    
    if not model_path.exists():
        return HealthCheckResult(
            "ASM Model File",
            "FAIL",
            f"Not found: {model_path}",
            time.time() - start
        )
    
    size_mb = model_path.stat().st_size / (1024 * 1024)
    
    if size_mb < 0.1:
        return HealthCheckResult(
            "ASM Model File",
            "WARN",
            f"File too small: {size_mb:.2f}MB",
            time.time() - start
        )
    
    return HealthCheckResult(
        "ASM Model File",
        "PASS",
        f"Found ({size_mb:.2f}MB)",
        time.time() - start
    )


def check_asm_model_loading() -> HealthCheckResult:
    """Check ASM model can be loaded"""
    start = time.time()
    
    try:
        from services.live_gateway.asm_inference import ASMModelLoader
        
        loader = ASMModelLoader()
        success = loader.load()
        
        if success:
            return HealthCheckResult(
                "ASM Model Loading",
                "PASS",
                f"Loaded successfully on {loader.device}",
                time.time() - start
            )
        else:
            return HealthCheckResult(
                "ASM Model Loading",
                "FAIL",
                "Failed to load model",
                time.time() - start
            )
    except Exception as e:
        return HealthCheckResult(
            "ASM Model Loading",
            "FAIL",
            f"Error: {str(e)[:50]}",
            time.time() - start
        )


def check_asm_inference() -> HealthCheckResult:
    """Check ASM inference works"""
    start = time.time()
    
    try:
        from services.live_gateway.asm_inference import ASMModelLoader
        
        loader = ASMModelLoader()
        loader.load()
        
        # Test with random data
        dummy_context = np.random.randn(60, 100).astype(np.float32)
        probs = loader.predict_proba(dummy_context)
        
        # Validate probabilities
        prob_sum = probs['p_up'] + probs['p_down'] + probs['p_neutral']
        
        if 0.99 <= prob_sum <= 1.01:
            return HealthCheckResult(
                "ASM Inference",
                "PASS",
                f"Probabilities valid (sum={prob_sum:.4f})",
                time.time() - start
            )
        else:
            return HealthCheckResult(
                "ASM Inference",
                "WARN",
                f"Probability sum: {prob_sum:.4f} (expected ~1.0)",
                time.time() - start
            )
    except Exception as e:
        return HealthCheckResult(
            "ASM Inference",
            "FAIL",
            f"Error: {str(e)[:50]}",
            time.time() - start
        )


def check_smc_context_manager() -> HealthCheckResult:
    """Check SMC Context Manager"""
    start = time.time()
    
    try:
        from src.layer2_feature_engine_v2.context_manager import SMCContextManager
        from src.layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
        
        manager = SMCContextManager(GC_M1_SMC_CONFIG, tick_size=0.1)
        
        return HealthCheckResult(
            "SMC Context Manager",
            "PASS",
            "Initialized successfully",
            time.time() - start
        )
    except Exception as e:
        return HealthCheckResult(
            "SMC Context Manager",
            "FAIL",
            f"Error: {str(e)[:50]}",
            time.time() - start
        )


def check_s4_engine() -> HealthCheckResult:
    """Check S4 rule engine"""
    start = time.time()
    
    try:
        from services.live_gateway.s4_engine import check_s4_setup, detect_session, S4_CONFIG
        
        # Test session detection
        london_hours = [detect_session(h) for h in range(24)]
        london_count = london_hours.count("London")
        
        if london_count == 6:  # 8-13 UTC
            return HealthCheckResult(
                "S4 Rule Engine",
                "PASS",
                f"Session detection OK (London={london_count}h)",
                time.time() - start
            )
        else:
            return HealthCheckResult(
                "S4 Rule Engine",
                "WARN",
                f"London hours: {london_count} (expected 6)",
                time.time() - start
            )
    except Exception as e:
        return HealthCheckResult(
            "S4 Rule Engine",
            "FAIL",
            f"Error: {str(e)[:50]}",
            time.time() - start
        )


def check_context_store() -> HealthCheckResult:
    """Check context store"""
    start = time.time()
    
    try:
        from services.live_gateway.context_store import ContextStore
        
        store = ContextStore()
        
        return HealthCheckResult(
            "Context Store",
            "PASS",
            "Initialized successfully",
            time.time() - start
        )
    except Exception as e:
        return HealthCheckResult(
            "Context Store",
            "FAIL",
            f"Error: {str(e)[:50]}",
            time.time() - start
        )


def check_data_files() -> HealthCheckResult:
    """Check data files exist"""
    start = time.time()
    
    data_dir = ROOT / "data/raw/new_data"
    
    if not data_dir.exists():
        return HealthCheckResult(
            "Data Files",
            "WARN",
            f"Directory not found: {data_dir}",
            time.time() - start
        )
    
    jsonl_files = list(data_dir.glob("*.jsonl"))
    
    if len(jsonl_files) >= 6:
        return HealthCheckResult(
            "Data Files",
            "PASS",
            f"Found {len(jsonl_files)} JSONL files",
            time.time() - start
        )
    elif len(jsonl_files) > 0:
        return HealthCheckResult(
            "Data Files",
            "WARN",
            f"Found {len(jsonl_files)} files (expected 6)",
            time.time() - start
        )
    else:
        return HealthCheckResult(
            "Data Files",
            "FAIL",
            "No JSONL files found",
            time.time() - start
        )


def check_log_directory() -> HealthCheckResult:
    """Check log directory is writable"""
    start = time.time()
    
    log_dir = ROOT / "logs"
    
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Test write
        test_file = log_dir / ".health_check_test"
        test_file.write_text("test")
        test_file.unlink()
        
        return HealthCheckResult(
            "Log Directory",
            "PASS",
            f"Writable: {log_dir}",
            time.time() - start
        )
    except Exception as e:
        return HealthCheckResult(
            "Log Directory",
            "FAIL",
            f"Not writable: {str(e)[:30]}",
            time.time() - start
        )


def check_api_endpoint(api_url: str) -> HealthCheckResult:
    """Check API endpoint (if running)"""
    start = time.time()
    
    try:
        import requests
        
        response = requests.get(f"{api_url}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return HealthCheckResult(
                "API Endpoint",
                "PASS",
                f"Healthy (model_loaded={data.get('model_loaded')})",
                time.time() - start
            )
        else:
            return HealthCheckResult(
                "API Endpoint",
                "FAIL",
                f"Status code: {response.status_code}",
                time.time() - start
            )
    except requests.exceptions.ConnectionError:
        return HealthCheckResult(
            "API Endpoint",
            "WARN",
            "Not running (connection refused)",
            time.time() - start
        )
    except Exception as e:
        return HealthCheckResult(
            "API Endpoint",
            "FAIL",
            f"Error: {str(e)[:30]}",
            time.time() - start
        )


def check_inference_performance() -> HealthCheckResult:
    """Benchmark inference performance"""
    start = time.time()
    
    try:
        from services.live_gateway.asm_inference import ASMModelLoader
        
        loader = ASMModelLoader()
        loader.load()
        
        # Benchmark
        dummy_context = np.random.randn(60, 100).astype(np.float32)
        
        times = []
        for _ in range(10):
            t0 = time.time()
            loader.predict_proba(dummy_context)
            times.append((time.time() - t0) * 1000)
        
        avg_ms = np.mean(times)
        
        if avg_ms < 50:
            status = "PASS"
        elif avg_ms < 100:
            status = "WARN"
        else:
            status = "FAIL"
        
        return HealthCheckResult(
            "Inference Performance",
            status,
            f"Avg: {avg_ms:.2f}ms (10 runs)",
            time.time() - start
        )
    except Exception as e:
        return HealthCheckResult(
            "Inference Performance",
            "FAIL",
            f"Error: {str(e)[:30]}",
            time.time() - start
        )


# ==============================================================================
# MAIN
# ==============================================================================

def run_health_checks(api_url: str = None, verbose: bool = False) -> Tuple[int, int, int]:
    """Run all health checks"""
    
    print("=" * 60)
    print("SYSTEM HEALTH CHECK")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    checks = [
        check_python_version,
        check_dependencies,
        check_asm_model,
        check_asm_model_loading,
        check_asm_inference,
        check_smc_context_manager,
        check_s4_engine,
        check_context_store,
        check_data_files,
        check_log_directory,
        check_inference_performance,
    ]
    
    results = []
    
    for check_func in checks:
        try:
            result = check_func()
            results.append(result)
            print(result)
            
            if verbose and result.duration > 0:
                print(f"   Duration: {result.duration*1000:.2f}ms")
        except Exception as e:
            results.append(HealthCheckResult(
                check_func.__name__,
                "FAIL",
                f"Check crashed: {str(e)[:30]}"
            ))
    
    # API check (optional)
    if api_url:
        result = check_api_endpoint(api_url)
        results.append(result)
        print(result)
    
    # Summary
    passed = sum(1 for r in results if r.status == "PASS")
    warned = sum(1 for r in results if r.status == "WARN")
    failed = sum(1 for r in results if r.status == "FAIL")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"⚠️  Warnings: {warned}")
    print(f"❌ Failed: {failed}")
    print(f"Total: {len(results)}")
    
    if failed == 0 and warned == 0:
        print("\n🎉 All checks passed! System is healthy.")
    elif failed == 0:
        print("\n⚠️  System operational with warnings.")
    else:
        print("\n❌ System has issues that need attention.")
    
    print("=" * 60)
    
    return passed, warned, failed


def main():
    parser = argparse.ArgumentParser(description="System Health Check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--api", type=str, default=None, help="API URL to check (e.g., http://localhost:8000)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    if args.api is None:
        args.api = "http://localhost:8000"
    
    passed, warned, failed = run_health_checks(api_url=args.api, verbose=args.verbose)
    
    # Exit code
    if failed > 0:
        sys.exit(1)
    elif warned > 0:
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
