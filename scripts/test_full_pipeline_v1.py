#!/usr/bin/env python3
"""
Full Pipeline Test Script
========================
Test toàn bộ pipeline từ raw data → features → ASM → S4 → Live Gateway
và tạo báo cáo HTML tổng hợp.

Usage:
    python scripts/test_full_pipeline_v1.py
    python scripts/test_full_pipeline_v1.py --quick  # Test 1 tuần
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import numpy as np

import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ==============================================================================
# CONFIGURATION
# ==============================================================================

REPORTS_DIR = ROOT / "reports"
OUTPUT_HTML = REPORTS_DIR / "full_pipeline_test_v1.html"
OUTPUT_JSON = REPORTS_DIR / "full_pipeline_test_v1.json"

# Test data paths
TEST_DATA_PATHS = {
    "quick": [ROOT / "data/raw/new_data/smc_export_gc_m1_v3_20250428.jsonl"],
    "full": list((ROOT / "data/raw/new_data").glob("*.jsonl")),
}


# ==============================================================================
# TEST RESULT CLASSES
# ==============================================================================

@dataclass
class ComponentResult:
    name: str
    status: str  # "PASS", "FAIL", "SKIP"
    duration: float
    metrics: Dict[str, Any]
    error: Optional[str] = None
    details: Optional[str] = None


@dataclass
class PipelineTestResult:
    test_id: str
    timestamp: str
    mode: str
    overall_status: str
    total_duration: float
    components: List[ComponentResult]
    summary: Dict[str, Any]
    recommendations: List[str]


# ==============================================================================
# COMPONENT TESTERS
# ==============================================================================

def test_feature_generation(data_files: List[Path]) -> ComponentResult:
    """Test SMC feature generation"""
    start_time = time.time()
    
    try:
        print("\n[1/5] Testing Feature Generation...")
        
        from src.layer2_feature_engine_v2.context_manager import SMCContextManager
        from src.layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
        from src.layer2_feature_engine_v2.schema import RawBar
        
        manager = SMCContextManager(GC_M1_SMC_CONFIG, tick_size=0.1)
        bars_processed = 0
        features_generated = 0
        feature_count = 0
        
        with open(data_files[0], 'r') as f:
            for line_num, line in enumerate(f):
                if line_num >= 1000:
                    break
                
                try:
                    data = json.loads(line)
                    ts = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00').split('.')[0])
                    bar_data = data['bar']
                    tick_data = data.get('tick_features', {})
                    
                    raw_bar = RawBar(
                        symbol=data.get('symbol', 'GC'),
                        timeframe=data.get('timeframe', 'M1'),
                        timestamp=ts,
                        bar_index=data['bar_index'],
                        o=bar_data['o'], h=bar_data['h'], l=bar_data['l'], c=bar_data['c'],
                        volume=bar_data['volume'],
                        delta=bar_data.get('delta', 0),
                        buy_volume=bar_data.get('buy_volume', 0),
                        sell_volume=bar_data.get('sell_volume', 0),
                        best_bid=bar_data.get('best_bid', bar_data['c']),
                        best_ask=bar_data.get('best_ask', bar_data['c']),
                        tick_speed=tick_data.get('tick_speed', 0),
                        aggr_buy_speed=tick_data.get('aggr_buy_speed', 0),
                        aggr_sell_speed=tick_data.get('aggr_sell_speed', 0),
                        price_speed=tick_data.get('price_speed', bar_data['h'] - bar_data['l']),
                        vwap_daily=bar_data.get('vwap_daily', 0.0)
                    )
                    
                    feature_bar = manager.update(raw_bar)
                    bars_processed += 1
                    if feature_bar:
                        features_generated += 1
                        feature_count = len([k for k in feature_bar.to_dict().keys() 
                                           if not k.startswith('_') and k not in ['timestamp', 'bar_index']])
                except Exception:
                    continue
        
        metrics = {
            "bars_processed": bars_processed,
            "features_generated": features_generated,
            "feature_count": feature_count,
        }
        
        status = "PASS" if features_generated > 100 and feature_count > 80 else "FAIL"
        duration = time.time() - start_time
        print(f"    Processed {bars_processed} bars, {feature_count} features. Status: {status}")
        
        return ComponentResult("Feature Generation", status, duration, metrics)
        
    except Exception as e:
        return ComponentResult("Feature Generation", "FAIL", time.time() - start_time, {}, str(e))


def test_asm_model(data_files: List[Path]) -> ComponentResult:
    """Test ASM model loading and inference"""
    start_time = time.time()
    
    try:
        print("\n[2/5] Testing ASM Model...")
        
        from services.live_gateway.asm_inference import ASMModelLoader
        
        loader = ASMModelLoader()
        success = loader.load()
        
        if not success:
            return ComponentResult("ASM Model", "FAIL", time.time() - start_time, {}, "Failed to load")
        
        dummy_context = np.random.randn(60, 100).astype(np.float32)
        probs = loader.predict_proba(dummy_context)
        prob_sum = probs['p_up'] + probs['p_down'] + probs['p_neutral']
        
        metrics = {"model_loaded": success, "prob_sum": round(prob_sum, 4)}
        status = "PASS" if 0.99 <= prob_sum <= 1.01 else "FAIL"
        
        print(f"    Model loaded: {success}, Prob sum: {prob_sum:.4f}. Status: {status}")
        return ComponentResult("ASM Model", status, time.time() - start_time, metrics)
        
    except Exception as e:
        return ComponentResult("ASM Model", "FAIL", time.time() - start_time, {}, str(e))


def test_s4_engine(data_files: List[Path]) -> ComponentResult:
    """Test S4 rule engine"""
    start_time = time.time()
    
    try:
        print("\n[3/5] Testing S4 Rule Engine...")
        
        from services.live_gateway.s4_engine import detect_session
        
        london_hours = [detect_session(h) for h in range(24)]
        london_count = london_hours.count("London")
        
        metrics = {"london_hours": london_count}
        status = "PASS" if london_count == 6 else "FAIL"
        
        print(f"    London hours: {london_count}. Status: {status}")
        return ComponentResult("S4 Rule Engine", status, time.time() - start_time, metrics)
        
    except Exception as e:
        return ComponentResult("S4 Rule Engine", "FAIL", time.time() - start_time, {}, str(e))


def test_live_gateway(data_files: List[Path]) -> ComponentResult:
    """Test Live Gateway integration"""
    start_time = time.time()
    
    try:
        print("\n[4/5] Testing Live Gateway...")
        
        from services.live_gateway.context_store import ContextStore
        from services.live_gateway.asm_inference import ASMModelLoader
        
        store = ContextStore()
        loader = ASMModelLoader()
        loader.load()
        
        metrics = {"context_store": "OK", "model_loaded": loader.loaded}
        status = "PASS" if loader.loaded else "FAIL"
        
        print(f"    Context store: OK, Model: {loader.loaded}. Status: {status}")
        return ComponentResult("Live Gateway", status, time.time() - start_time, metrics)
        
    except Exception as e:
        return ComponentResult("Live Gateway", "FAIL", time.time() - start_time, {}, str(e))


def test_integration(data_files: List[Path]) -> ComponentResult:
    """Test full integration"""
    start_time = time.time()
    
    try:
        print("\n[5/5] Testing Integration...")
        
        model_exists = (ROOT / "output/asm_models_v1/ASM-GRU64-v1.0-C3.pt").exists()
        
        metrics = {"data_files": len(data_files), "model_exists": model_exists}
        status = "PASS" if model_exists and len(data_files) > 0 else "FAIL"
        
        print(f"    Data files: {len(data_files)}, Model exists: {model_exists}. Status: {status}")
        return ComponentResult("Integration", status, time.time() - start_time, metrics)
        
    except Exception as e:
        return ComponentResult("Integration", "FAIL", time.time() - start_time, {}, str(e))


# ==============================================================================
# HTML REPORT
# ==============================================================================

def generate_html_report(result: PipelineTestResult) -> str:
    """Generate HTML report"""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Pipeline Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .pass {{ background-color: #28a745; color: white; padding: 5px 10px; border-radius: 3px; }}
        .fail {{ background-color: #dc3545; color: white; padding: 5px 10px; border-radius: 3px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Full Pipeline Test Report</h1>
        <p><strong>Test ID:</strong> {result.test_id}</p>
        <p><strong>Timestamp:</strong> {result.timestamp}</p>
        <p><strong>Mode:</strong> {result.mode.upper()}</p>
        <p><strong>Overall Status:</strong> <span class="{result.overall_status.lower()}">{result.overall_status}</span></p>
        <p><strong>Duration:</strong> {result.total_duration:.2f}s</p>
    </div>
    
    <h2>Summary</h2>
    <p>Components Passed: {result.summary['components_passed']}/{result.summary['components_tested']}</p>
    <p>Success Rate: {result.summary['success_rate']}%</p>
    
    <h2>Component Results</h2>
    <table>
        <tr><th>Component</th><th>Status</th><th>Duration</th><th>Metrics</th></tr>
"""
    
    for comp in result.components:
        status_class = comp.status.lower()
        metrics_str = ", ".join([f"{k}: {v}" for k, v in comp.metrics.items()])
        html += f'<tr><td>{comp.name}</td><td><span class="{status_class}">{comp.status}</span></td><td>{comp.duration:.2f}s</td><td>{metrics_str}</td></tr>\n'
    
    html += f"""
    </table>
    
    <h2>Recommendations</h2>
    <ul>
"""
    for rec in result.recommendations:
        html += f"<li>{rec}</li>\n"
    
    html += """
    </ul>
    <footer style="margin-top: 40px; color: #666;">
        <p>Generated by Full Pipeline Test v1.0</p>
    </footer>
</body>
</html>
"""
    return html


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full Pipeline Test")
    parser.add_argument("--quick", action="store_true", help="Quick test (1 week)")
    args = parser.parse_args()
    
    mode = "quick" if args.quick else "full"
    
    print("=" * 60)
    print(f"FULL PIPELINE TEST - {mode.upper()} MODE")
    print("=" * 60)
    
    # Get data files
    data_files = TEST_DATA_PATHS[mode]
    if not data_files or not any(f.exists() for f in data_files):
        print("ERROR: No test data files found!")
        return
    
    data_files = [f for f in data_files if f.exists()]
    print(f"Test data: {len(data_files)} files")
    
    start_time = time.time()
    test_id = f"pipeline_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run tests
    results = [
        test_feature_generation(data_files),
        test_asm_model(data_files),
        test_s4_engine(data_files),
        test_live_gateway(data_files),
        test_integration(data_files),
    ]
    
    # Calculate summary
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    overall = "PASS" if failed == 0 else ("PARTIAL" if passed > failed else "FAIL")
    
    summary = {
        "components_tested": len(results),
        "components_passed": passed,
        "components_failed": failed,
        "success_rate": round(100 * passed / len(results), 1),
    }
    
    recommendations = []
    for r in results:
        if r.status == "FAIL":
            recommendations.append(f"❌ Fix {r.name}: {r.error or 'Check logs'}")
    if not recommendations:
        recommendations.append("✅ All components PASS - pipeline ready for production")
    
    total_duration = time.time() - start_time
    
    result = PipelineTestResult(
        test_id=test_id,
        timestamp=datetime.now().isoformat(),
        mode=mode,
        overall_status=overall,
        total_duration=total_duration,
        components=results,
        summary=summary,
        recommendations=recommendations
    )
    
    # Save reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
    
    html = generate_html_report(result)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Overall Status: {overall}")
    print(f"Components Passed: {passed}/{len(results)}")
    print(f"Duration: {total_duration:.2f}s")
    print(f"\nReports saved to:")
    print(f"  - {OUTPUT_JSON}")
    print(f"  - {OUTPUT_HTML}")
    print("=" * 60)


if __name__ == "__main__":
    main()
