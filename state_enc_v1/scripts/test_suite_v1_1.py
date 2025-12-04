#!/usr/bin/env python
"""
STATE-ENC v1.1 — FULL TEST SUITE (10 Tests)

Tests:
A. Deterministic Repeatability
B. Temporal Smoothness
C. Shock Sensitivity
D. Missing Feature Stress
E. Regime Linearity (PCA)
F. Cross-Session Generalization
G. Bar Order Noise Robustness
H. Feature Importance (Occlusion)
I. Normalization Stability
J. Latent Drift Detection
"""

import json
import sys
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F

print("=" * 70)
print("STATE-ENC v1.1 — FULL TEST SUITE (10 Tests)")
print("=" * 70)


def cosine_distance(z1: torch.Tensor, z2: torch.Tensor) -> float:
    """Cosine distance between embeddings"""
    sim = F.cosine_similarity(z1.flatten().unsqueeze(0), z2.flatten().unsqueeze(0))
    return (1 - sim).item()


def generate_synthetic_sequence(num_bars: int = 64, regime: str = "random") -> List[Dict]:
    """Generate synthetic bar sequence"""
    base_price = 17250.0
    bars = []
    price = base_price
    cum_delta = 0
    
    for i in range(num_bars):
        # Regime-specific price movement
        if regime == "bull":
            change = abs(random.gauss(0.5, 1))
        elif regime == "bear":
            change = -abs(random.gauss(0.5, 1))
        elif regime == "chop":
            change = random.gauss(0, 0.5)
        elif regime == "drive":
            change = random.gauss(2, 1) if random.random() > 0.5 else random.gauss(-2, 1)
        else:
            change = random.gauss(0, 2)
        
        price += change
        
        o = price
        h = price + abs(random.gauss(0, 1.5))
        l = price - abs(random.gauss(0, 1.5))
        c = price + random.gauss(0, 1)
        h, l = max(h, o, c), min(l, o, c)
        
        volume = int(abs(random.gauss(1000, 300)))
        buy_vol = int(volume * random.uniform(0.3, 0.7))
        sell_vol = volume - buy_vol
        delta = buy_vol - sell_vol
        cum_delta += delta
        
        session = random.choice(["ASIA", "LDN", "NY"])
        
        bar = {
            "time": (datetime(2024, 1, 15, 2, 0) + timedelta(minutes=i)).isoformat(),
            "o": round(o, 2), "h": round(h, 2), "l": round(l, 2), "c": round(c, 2),
            "volume": volume, "delta": delta,
            "buy_volume": buy_vol, "sell_volume": sell_vol,
            "tick_count": int(volume * 0.8),
            "session": session, "symbol": "NQ",
            "ext_trend_dir": 1 if regime == "bull" else (-1 if regime == "bear" else 0),
            "int_trend_dir": random.choice([-1, 0, 1]),
            "ext_bos_up": random.choice([0, 0, 0, 1]),
            "ext_bos_down": random.choice([0, 0, 0, 1]),
            "swing_high": round(price + 30, 2),
            "swing_low": round(price - 30, 2),
            "vah": round(price + 15, 2),
            "val": round(price - 15, 2),
            "poc": round(price, 2),
            "cum_delta_session": cum_delta,
            "atr_m1_14": round(abs(random.gauss(4, 1)), 2),
            "asm_regime_hint": {"bull": 2, "bear": 3, "chop": 1, "drive": 4}.get(regime, 0),
        }
        bars.append(bar)
    
    return bars


def load_model_and_normalizer():
    """Load trained model and normalizer"""
    from state_enc_v1.src.model.state_enc_model import StateEncModel
    from state_enc_v1.src.normalization import FeatureNormalizer
    
    # Paths
    model_path = Path("state_enc_v1/artifacts/v1_1/final/state_enc_v1.1.pt")
    config_path = Path("state_enc_v1/artifacts/v1_1/final/model_config_v1.1.json")
    feature_path = Path("state_enc_v1/artifacts/v1_1/final/feature_config_v1.1.json")
    
    # Load config
    with open(config_path) as f:
        model_config = json.load(f)
    
    # Load model
    model = StateEncModel.from_config(model_config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Load normalizer
    normalizer = FeatureNormalizer.from_file(str(feature_path))
    
    return model, normalizer, model_config


def encode_sequence(model, normalizer, seq: List[Dict]) -> torch.Tensor:
    """Encode sequence to z_t"""
    X = normalizer.transform_sequence(seq)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        z_t = model.encode(X_tensor)
    return z_t


# =============================================================================
# TEST A: Deterministic Repeatability
# =============================================================================
def test_a_deterministic(model, normalizer, seq) -> Dict:
    print("\n[TEST A] Deterministic Repeatability...")
    
    embeddings = []
    for _ in range(50):
        z_t = encode_sequence(model, normalizer, seq)
        embeddings.append(z_t)
    
    distances = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            d = cosine_distance(embeddings[i], embeddings[j])
            distances.append(d)
    
    max_dist = max(distances) if distances else 0
    mean_dist = np.mean(distances) if distances else 0
    passed = max_dist < 0.0005
    
    print(f"  Max distance: {max_dist:.6f}, Mean: {mean_dist:.6f}, Pass: {passed}")
    
    return {
        "test": "A_Deterministic",
        "pass": passed,
        "max_distance": float(max_dist),
        "mean_distance": float(mean_dist),
        "threshold": 0.0005
    }


# =============================================================================
# TEST B: Temporal Smoothness
# =============================================================================
def test_b_temporal_smoothness(model, normalizer) -> Dict:
    print("\n[TEST B] Temporal Smoothness...")
    
    # Generate 100 bars for sliding window (reduced for speed)
    long_seq = generate_synthetic_sequence(100, "random")
    
    distances = []
    window_size = 64
    
    # Sample every 5 steps for speed
    for i in range(0, len(long_seq) - window_size - 1, 5):
        seq1 = long_seq[i:i + window_size]
        seq2 = long_seq[i + 1:i + 1 + window_size]
        
        z1 = encode_sequence(model, normalizer, seq1)
        z2 = encode_sequence(model, normalizer, seq2)
        
        d = cosine_distance(z1, z2)
        distances.append(d)
    
    median_dist = np.median(distances)
    max_dist = max(distances)
    passed = median_dist < 0.15 and max_dist < 0.35
    
    # Histogram bins
    hist, bins = np.histogram(distances, bins=10)
    
    print(f"  Median: {median_dist:.4f}, Max: {max_dist:.4f}, Pass: {passed}")
    
    return {
        "test": "B_TemporalSmoothness",
        "pass": passed,
        "median_distance": float(median_dist),
        "max_distance": float(max_dist),
        "histogram": {"counts": hist.tolist(), "bins": bins.tolist()},
        "thresholds": {"median": 0.15, "max": 0.35}
    }


# =============================================================================
# TEST C: Shock Sensitivity
# =============================================================================
def test_c_shock_sensitivity(model, normalizer, seq) -> Dict:
    print("\n[TEST C] Shock Sensitivity...")
    
    z_base = encode_sequence(model, normalizer, seq)
    
    results = {}
    
    # Shock 1: Price +0.3%
    shocked_seq = [bar.copy() for bar in seq]
    for bar in shocked_seq[-5:]:
        for k in ["o", "h", "l", "c"]:
            bar[k] = bar[k] * 1.003
    z_shock = encode_sequence(model, normalizer, shocked_seq)
    d1 = cosine_distance(z_base, z_shock)
    results["price_shock"] = {"distance": float(d1), "in_range": 0.15 <= d1 <= 0.50}
    
    # Shock 2: Delta +20%
    shocked_seq = [bar.copy() for bar in seq]
    for bar in shocked_seq[-5:]:
        bar["delta"] = int(bar.get("delta", 0) * 1.2)
    z_shock = encode_sequence(model, normalizer, shocked_seq)
    d2 = cosine_distance(z_base, z_shock)
    results["delta_shock"] = {"distance": float(d2), "in_range": 0.15 <= d2 <= 0.50}
    
    # Shock 3: Volume +35%
    shocked_seq = [bar.copy() for bar in seq]
    for bar in shocked_seq[-5:]:
        bar["volume"] = int(bar.get("volume", 0) * 1.35)
    z_shock = encode_sequence(model, normalizer, shocked_seq)
    d3 = cosine_distance(z_base, z_shock)
    results["volume_shock"] = {"distance": float(d3), "in_range": 0.15 <= d3 <= 0.50}
    
    # Overall pass: at least 2/3 in range OR all distances > 0.05 (sensitive)
    in_range_count = sum(1 for r in results.values() if r["in_range"])
    all_sensitive = all(r["distance"] > 0.05 for r in results.values())
    passed = in_range_count >= 2 or all_sensitive
    
    print(f"  Price: {d1:.4f}, Delta: {d2:.4f}, Volume: {d3:.4f}, Pass: {passed}")
    
    return {
        "test": "C_ShockSensitivity",
        "pass": passed,
        "scenarios": results
    }


# =============================================================================
# TEST D: Missing Feature Stress
# =============================================================================
def test_d_missing_features(model, normalizer, seq) -> Dict:
    print("\n[TEST D] Missing Feature Stress...")
    
    z_base = encode_sequence(model, normalizer, seq)
    
    results = {}
    mask_ratios = [0.05, 0.10, 0.20, 0.30, 0.40]
    
    for ratio in mask_ratios:
        # Create masked sequence
        masked_seq = []
        for bar in seq:
            new_bar = bar.copy()
            keys = list(new_bar.keys())
            num_mask = int(len(keys) * ratio)
            mask_keys = random.sample(keys, min(num_mask, len(keys)))
            for k in mask_keys:
                if k not in ["time", "session", "symbol"]:
                    new_bar[k] = 0.0
            masked_seq.append(new_bar)
        
        z_masked = encode_sequence(model, normalizer, masked_seq)
        d = cosine_distance(z_base, z_masked)
        results[f"mask_{int(ratio*100)}pct"] = float(d)
    
    # Check thresholds
    pass_20 = results.get("mask_20pct", 1.0) < 0.30
    pass_40 = results.get("mask_40pct", 1.0) < 0.50
    passed = pass_20 and pass_40
    
    print(f"  Distances: {results}")
    print(f"  Pass: {passed}")
    
    return {
        "test": "D_MissingFeatures",
        "pass": passed,
        "distances": results,
        "thresholds": {"mask_20pct": 0.30, "mask_40pct": 0.50}
    }


# =============================================================================
# TEST E: Regime Linearity (PCA)
# =============================================================================
def test_e_regime_pca(model, normalizer) -> Dict:
    print("\n[TEST E] Regime Linearity (PCA)...")
    
    regimes = ["bull", "bear", "chop", "drive"]
    embeddings = []
    labels = []
    
    for regime in regimes:
        for _ in range(20):  # Reduced for speed
            seq = generate_synthetic_sequence(64, regime)
            z_t = encode_sequence(model, normalizer, seq)
            embeddings.append(z_t.numpy().flatten())
            labels.append(regime)
    
    # Simple PCA (2D)
    X = np.array(embeddings)
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    pc1, pc2 = eigenvectors[:, idx[0]], eigenvectors[:, idx[1]]
    
    X_pca = np.column_stack([X_centered @ pc1, X_centered @ pc2])
    
    # Compute cluster separation
    centroids = {}
    for regime in regimes:
        mask = [l == regime for l in labels]
        centroids[regime] = X_pca[mask].mean(axis=0)
    
    # Inter-cluster distances
    inter_distances = []
    for i, r1 in enumerate(regimes):
        for r2 in regimes[i+1:]:
            d = np.linalg.norm(centroids[r1] - centroids[r2])
            inter_distances.append(d)
    
    min_separation = min(inter_distances) if inter_distances else 0
    passed = min_separation > 0.1  # At least some separation
    
    # PCA scatter data
    pca_data = [
        {"x": float(X_pca[i, 0]), "y": float(X_pca[i, 1]), "regime": labels[i]}
        for i in range(len(labels))
    ]
    
    print(f"  Min cluster separation: {min_separation:.4f}, Pass: {passed}")
    
    return {
        "test": "E_RegimePCA",
        "pass": passed,
        "min_separation": float(min_separation),
        "centroids": {k: v.tolist() for k, v in centroids.items()},
        "pca_scatter": pca_data,
        "explained_variance": [float(eigenvalues[idx[0]]), float(eigenvalues[idx[1]])]
    }


# =============================================================================
# TEST F: Cross-Session Generalization
# =============================================================================
def test_f_cross_session(model, normalizer) -> Dict:
    print("\n[TEST F] Cross-Session Generalization...")
    
    sessions = ["ASIA", "LDN", "NY"]
    session_embeddings = {s: [] for s in sessions}
    
    for session in sessions:
        for _ in range(10):
            seq = generate_synthetic_sequence(64, "random")
            # Force session
            for bar in seq:
                bar["session"] = session
            z_t = encode_sequence(model, normalizer, seq)
            session_embeddings[session].append(z_t.numpy().flatten())
    
    # Compute distance matrix
    def mean_distance(emb1_list, emb2_list):
        distances = []
        for e1 in emb1_list:
            for e2 in emb2_list:
                d = 1 - np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
                distances.append(d)
        return np.mean(distances)
    
    matrix = {}
    for s1 in sessions:
        matrix[s1] = {}
        for s2 in sessions:
            matrix[s1][s2] = float(mean_distance(session_embeddings[s1], session_embeddings[s2]))
    
    # Check: ASIA-LDN distance should be reasonable
    asia_ldn = matrix["ASIA"]["LDN"]
    passed = asia_ldn < 0.5  # Reasonable generalization
    
    print(f"  ASIA-LDN distance: {asia_ldn:.4f}, Pass: {passed}")
    
    return {
        "test": "F_CrossSession",
        "pass": passed,
        "distance_matrix": matrix
    }


# =============================================================================
# TEST G: Bar Order Noise
# =============================================================================
def test_g_bar_order_noise(model, normalizer, seq) -> Dict:
    print("\n[TEST G] Bar Order Noise Robustness...")
    
    z_base = encode_sequence(model, normalizer, seq)
    
    results = []
    swap_configs = [(5, 6), (10, 15), (20, 25), (30, 35)]
    
    for i, j in swap_configs:
        if i < len(seq) and j < len(seq):
            swapped_seq = seq.copy()
            swapped_seq[i], swapped_seq[j] = swapped_seq[j], swapped_seq[i]
            z_swapped = encode_sequence(model, normalizer, swapped_seq)
            d = cosine_distance(z_base, z_swapped)
            results.append({"swap": f"{i}-{j}", "distance": float(d), "sensitive": d > 0.05})
    
    # Pass if model is sensitive to order changes
    sensitive_count = sum(1 for r in results if r["sensitive"])
    passed = sensitive_count >= len(results) // 2
    
    print(f"  Sensitive swaps: {sensitive_count}/{len(results)}, Pass: {passed}")
    
    return {
        "test": "G_BarOrderNoise",
        "pass": passed,
        "swap_results": results
    }


# =============================================================================
# TEST H: Feature Importance (Occlusion)
# =============================================================================
def test_h_feature_importance(model, normalizer, seq) -> Dict:
    print("\n[TEST H] Feature Importance (Occlusion)...")
    
    z_base = encode_sequence(model, normalizer, seq)
    
    # Get feature names
    feature_names = normalizer.feature_names
    importance = {}
    
    for idx, fname in enumerate(feature_names[:50]):  # Limit to 50 for speed
        # Occlude feature
        occluded_seq = []
        for bar in seq:
            new_bar = bar.copy()
            if fname in new_bar:
                new_bar[fname] = 0.0
            occluded_seq.append(new_bar)
        
        z_occluded = encode_sequence(model, normalizer, occluded_seq)
        d = cosine_distance(z_base, z_occluded)
        importance[fname] = float(d)
    
    # Sort by importance
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_20 = sorted_importance[:20]
    
    passed = True  # Informational test
    
    print(f"  Top 5 important: {[x[0] for x in top_20[:5]]}")
    
    return {
        "test": "H_FeatureImportance",
        "pass": passed,
        "top_20": [{"feature": k, "importance": v} for k, v in top_20],
        "all_importance": importance
    }


# =============================================================================
# TEST I: Normalization Stability
# =============================================================================
def test_i_normalization_stability(model, normalizer, seq) -> Dict:
    print("\n[TEST I] Normalization Stability...")
    
    z_base = encode_sequence(model, normalizer, seq)
    
    # Variant 1: Add small noise to normalized values
    X_base = normalizer.transform_sequence(seq)
    
    variants = []
    
    # Standard + noise
    X_v1 = X_base + np.random.randn(*X_base.shape) * 0.01
    z_v1 = model.encode(torch.tensor(X_v1, dtype=torch.float32).unsqueeze(0))
    d1 = cosine_distance(z_base, z_v1)
    variants.append({"variant": "noise_0.01", "distance": float(d1)})
    
    # Scale variant
    X_v2 = X_base * 1.05
    z_v2 = model.encode(torch.tensor(X_v2, dtype=torch.float32).unsqueeze(0))
    d2 = cosine_distance(z_base, z_v2)
    variants.append({"variant": "scale_1.05", "distance": float(d2)})
    
    # Clip variant
    X_v3 = np.clip(X_base, -3, 3)
    z_v3 = model.encode(torch.tensor(X_v3, dtype=torch.float32).unsqueeze(0))
    d3 = cosine_distance(z_base, z_v3)
    variants.append({"variant": "clip_3", "distance": float(d3)})
    
    max_deviation = max(v["distance"] for v in variants)
    passed = max_deviation < 0.25
    
    print(f"  Max deviation: {max_deviation:.4f}, Pass: {passed}")
    
    return {
        "test": "I_NormalizationStability",
        "pass": passed,
        "variants": variants,
        "max_deviation": float(max_deviation),
        "threshold": 0.25
    }


# =============================================================================
# TEST J: Latent Drift Detection
# =============================================================================
def test_j_latent_drift(model, normalizer) -> Dict:
    print("\n[TEST J] Latent Drift Detection...")
    
    # Generate 50 sequences across "time"
    embeddings = []
    for i in range(50):
        seq = generate_synthetic_sequence(64, "random")
        z_t = encode_sequence(model, normalizer, seq)
        embeddings.append(z_t.numpy().flatten())
    
    # Compute centroid drift
    window = 10
    centroids = []
    for i in range(0, len(embeddings) - window + 1, 5):
        centroid = np.mean(embeddings[i:i + window], axis=0)
        centroids.append(centroid)
    
    # Drift between consecutive centroids
    drifts = []
    for i in range(len(centroids) - 1):
        d = np.linalg.norm(centroids[i + 1] - centroids[i])
        drifts.append(float(d))
    
    mean_drift = np.mean(drifts) if drifts else 0
    max_drift = max(drifts) if drifts else 0
    passed = max_drift < 0.4
    
    print(f"  Mean drift: {mean_drift:.4f}, Max drift: {max_drift:.4f}, Pass: {passed}")
    
    return {
        "test": "J_LatentDrift",
        "pass": passed,
        "mean_drift": float(mean_drift),
        "max_drift": float(max_drift),
        "drift_series": drifts,
        "threshold": 0.4
    }


# =============================================================================
# MAIN
# =============================================================================
def run_full_test_suite():
    start_time = time.time()
    
    # Load model
    print("\nLoading model and normalizer...")
    model, normalizer, config = load_model_and_normalizer()
    print(f"  Model loaded: z_t dim = {model.get_embedding_dim()}")
    
    # Generate base sequence
    base_seq = generate_synthetic_sequence(64, "random")
    
    # Run all tests
    results = {}
    
    results["A"] = test_a_deterministic(model, normalizer, base_seq)
    results["B"] = test_b_temporal_smoothness(model, normalizer)
    results["C"] = test_c_shock_sensitivity(model, normalizer, base_seq)
    results["D"] = test_d_missing_features(model, normalizer, base_seq)
    results["E"] = test_e_regime_pca(model, normalizer)
    results["F"] = test_f_cross_session(model, normalizer)
    results["G"] = test_g_bar_order_noise(model, normalizer, base_seq)
    results["H"] = test_h_feature_importance(model, normalizer, base_seq)
    results["I"] = test_i_normalization_stability(model, normalizer, base_seq)
    results["J"] = test_j_latent_drift(model, normalizer)
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)
    
    summary_table = []
    for key, result in results.items():
        status = "✅ PASS" if result["pass"] else "❌ FAIL"
        summary_table.append({"test": result["test"], "status": status, "pass": result["pass"]})
        print(f"  {result['test']}: {status}")
    
    passed_count = sum(1 for r in results.values() if r["pass"])
    total_count = len(results)
    overall_pass = passed_count >= 8  # 80% threshold
    
    print(f"\n  Overall: {passed_count}/{total_count} tests passed")
    print(f"  Status: {'✅ SUITE PASSED' if overall_pass else '❌ SUITE FAILED'}")
    print(f"  Total time: {total_time:.1f}s")
    
    # Save report
    report = {
        "version": "1.1",
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "summary": {
            "passed": passed_count,
            "total": total_count,
            "overall_pass": overall_pass
        },
        "tests": results
    }
    
    output_dir = Path("state_enc_v1/artifacts/v1_1/final")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy bools to Python bools
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return bool(obj) if isinstance(obj, np.bool_) else int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    report = convert_to_serializable(report)
    
    report_path = output_dir / "test_report_v1.1.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")
    
    # Print detailed table
    print("\n" + "=" * 70)
    print("DETAILED RESULTS TABLE")
    print("=" * 70)
    print(f"{'Test':<30} {'Status':<10} {'Key Metric':<30}")
    print("-" * 70)
    
    metrics = {
        "A": f"max_dist={results['A']['max_distance']:.6f}",
        "B": f"median={results['B']['median_distance']:.4f}, max={results['B']['max_distance']:.4f}",
        "C": f"price={results['C']['scenarios']['price_shock']['distance']:.4f}",
        "D": f"mask_20%={results['D']['distances'].get('mask_20pct', 0):.4f}",
        "E": f"separation={results['E']['min_separation']:.4f}",
        "F": f"ASIA-LDN={results['F']['distance_matrix']['ASIA']['LDN']:.4f}",
        "G": f"sensitive={sum(1 for r in results['G']['swap_results'] if r['sensitive'])}/4",
        "H": f"top={results['H']['top_20'][0]['feature'] if results['H']['top_20'] else 'N/A'}",
        "I": f"max_dev={results['I']['max_deviation']:.4f}",
        "J": f"max_drift={results['J']['max_drift']:.4f}",
    }
    
    for key, result in results.items():
        status = "PASS" if result["pass"] else "FAIL"
        print(f"{result['test']:<30} {status:<10} {metrics[key]:<30}")
    
    return report


if __name__ == "__main__":
    report = run_full_test_suite()
    sys.exit(0 if report["summary"]["overall_pass"] else 1)
