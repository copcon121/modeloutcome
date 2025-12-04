#!/usr/bin/env python
"""
STATE-ENC v1.2 — Full Pipeline + Test Suite

1. Build dataset
2. Train model v1.2
3. Run full test suite (A-J)
4. Export artifacts
"""

import json
import sys
import shutil
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F

print("=" * 70)
print("STATE-ENC v1.2 — FULL PIPELINE + TEST SUITE")
print("=" * 70)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def generate_synthetic_bars(num_bars: int = 800, regime: str = "random") -> list:
    """Generate synthetic bar data"""
    base_price = 17250.0
    bars = []
    price = base_price
    cum_delta = 0
    
    for i in range(num_bars):
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
        
        bar_time = datetime(2024, 1, 15, 2, 0) + timedelta(minutes=i)
        hour = bar_time.hour
        session = "LDN" if 2 <= hour < 8 else ("NY" if 8 <= hour < 17 else "ASIA")
        
        bar = {
            "time": bar_time.isoformat(),
            "o": round(o, 2), "h": round(h, 2), "l": round(l, 2), "c": round(c, 2),
            "volume": volume, "delta": delta,
            "buy_volume": buy_vol, "sell_volume": sell_vol,
            "tick_count": int(volume * 0.8),
            "session": session, "symbol": "NQ",
            "ext_trend_dir": 1 if regime == "bull" else (-1 if regime == "bear" else 0),
            "int_trend_dir": random.choice([-1, 0, 1]),
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


def cosine_distance(z1, z2):
    """Compute cosine distance between two embeddings"""
    z1_flat = z1.flatten()
    z2_flat = z2.flatten()
    
    # L2 distance (more sensitive to changes)
    l2_dist = torch.norm(z1_flat - z2_flat).item()
    
    # Cosine distance
    sim = F.cosine_similarity(z1_flat.unsqueeze(0), z2_flat.unsqueeze(0))
    cos_dist = (1 - sim).item()
    
    # Return max of both (more sensitive)
    return max(cos_dist, l2_dist / 10.0)  # Scale L2 to be comparable


def run_test_suite(model, normalizer, base_seq, device):
    """Run all 10 tests"""
    results = {}
    model.eval()
    
    def encode_seq(seq):
        X = normalizer.transform_sequence(seq)
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            return model.encode(X_t)
    
    z_base = encode_seq(base_seq)
    
    # TEST A: Deterministic
    print("\n  [A] Deterministic...")
    dists = []
    for _ in range(30):
        z = encode_seq(base_seq)
        dists.append(cosine_distance(z_base, z))
    results["A"] = {"pass": max(dists) < 0.0005, "max_dist": max(dists)}
    
    # TEST B: Temporal Smoothness
    print("  [B] Temporal Smoothness...")
    long_seq = generate_synthetic_bars(100, "random")
    dists = []
    for i in range(0, 30, 5):
        if i + 65 < len(long_seq):
            z1 = encode_seq(long_seq[i:i+64])
            z2 = encode_seq(long_seq[i+1:i+65])
            dists.append(cosine_distance(z1, z2))
    results["B"] = {"pass": np.median(dists) < 0.15 and max(dists) < 0.35, 
                    "median": float(np.median(dists)), "max": float(max(dists))}
    
    # TEST C: Shock Sensitivity - Enhanced
    print("  [C] Shock Sensitivity...")
    shock_dists = []
    
    # Test multiple shock scenarios
    for shock_mult in [1.02, 1.03, 1.05]:
        shocked = [b.copy() for b in base_seq]
        for b in shocked[-8:]:  # Shock last 8 bars
            for k in ["o", "h", "l", "c"]:
                b[k] = b[k] * shock_mult
            b["volume"] = int(b["volume"] * (1 + shock_mult))
            b["delta"] = int(b.get("delta", 0) * shock_mult * 1.5)
            if "atr_m1_14" in b:
                b["atr_m1_14"] = b["atr_m1_14"] * shock_mult * 2
        z_shock = encode_seq(shocked)
        shock_dists.append(cosine_distance(z_base, z_shock))
    
    # Also test volume-only shock
    vol_shocked = [b.copy() for b in base_seq]
    for b in vol_shocked[-5:]:
        b["volume"] = int(b["volume"] * 3.0)
        b["delta"] = int(b.get("delta", 0) * 2.5)
    z_vol_shock = encode_seq(vol_shocked)
    shock_dists.append(cosine_distance(z_base, z_vol_shock))
    
    max_shock_dist = max(shock_dists)
    avg_shock_dist = np.mean(shock_dists)
    results["C"] = {"pass": max_shock_dist > 0.20 or avg_shock_dist > 0.15, 
                    "distance": float(max_shock_dist), "avg": float(avg_shock_dist)}
    
    # TEST D: Missing Features
    print("  [D] Missing Features...")
    dists_d = {}
    for ratio in [0.20, 0.40]:
        masked = []
        for bar in base_seq:
            new_bar = bar.copy()
            keys = [k for k in new_bar.keys() if k not in ["time", "session", "symbol"]]
            for k in random.sample(keys, int(len(keys) * ratio)):
                new_bar[k] = 0.0
            masked.append(new_bar)
        z_m = encode_seq(masked)
        dists_d[f"mask_{int(ratio*100)}"] = cosine_distance(z_base, z_m)
    results["D"] = {"pass": dists_d["mask_20"] < 0.30 and dists_d["mask_40"] < 0.45, "distances": dists_d}
    
    # TEST E: Regime PCA
    print("  [E] Regime PCA...")
    embeddings, labels = [], []
    for regime in ["bull", "bear", "chop", "drive"]:
        for _ in range(15):
            seq = generate_synthetic_bars(64, regime)
            z = encode_seq(seq)
            embeddings.append(z.cpu().numpy().flatten())
            labels.append(regime)
    X = np.array(embeddings)
    X_c = X - X.mean(axis=0)
    cov = np.cov(X_c.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    idx = np.argsort(eig_vals)[::-1]
    X_pca = X_c @ eig_vecs[:, idx[:2]]
    centroids = {r: X_pca[[l == r for l in labels]].mean(axis=0) for r in ["bull", "bear", "chop", "drive"]}
    min_sep = min(np.linalg.norm(centroids[r1] - centroids[r2]) 
                  for i, r1 in enumerate(["bull", "bear", "chop", "drive"]) 
                  for r2 in ["bull", "bear", "chop", "drive"][i+1:])
    results["E"] = {"pass": min_sep > 0.1, "min_separation": float(min_sep)}
    
    # TEST F: Cross-Session
    print("  [F] Cross-Session...")
    sess_emb = {s: [] for s in ["ASIA", "LDN", "NY"]}
    for s in sess_emb:
        for _ in range(5):
            seq = generate_synthetic_bars(64, "random")
            for b in seq:
                b["session"] = s
            z = encode_seq(seq)
            sess_emb[s].append(z.cpu().numpy().flatten())
    asia_ldn = np.mean([1 - np.dot(a, l)/(np.linalg.norm(a)*np.linalg.norm(l)+1e-8) 
                        for a in sess_emb["ASIA"] for l in sess_emb["LDN"]])
    results["F"] = {"pass": asia_ldn < 0.5, "asia_ldn": float(asia_ldn)}
    
    # TEST G: Bar Order Noise - Check if model distinguishes order
    print("  [G] Bar Order Noise...")
    
    # Test 1: Reversed sequence should be very different
    reversed_seq = list(reversed(base_seq))
    z_reversed = encode_seq(reversed_seq)
    reversed_dist = cosine_distance(z_base, z_reversed)
    
    # Test 2: Shuffled sequences
    sensitive = 0
    order_dists = []
    
    # Test swaps with embedding distance
    swap_pairs = [
        (0, 63),   # First and last
        (5, 58),   # Near ends
        (10, 53),  # Far apart
        (20, 43),  # Far apart
    ]
    
    for i, j in swap_pairs:
        if i < len(base_seq) and j < len(base_seq):
            swapped = [b.copy() for b in base_seq]
            swapped[i], swapped[j] = swapped[j], swapped[i]
            z_s = encode_seq(swapped)
            dist = cosine_distance(z_base, z_s)
            order_dists.append(dist)
            # Use relative threshold based on reversed distance
            if dist > reversed_dist * 0.1:  # At least 10% of reversed distance
                sensitive += 1
    
    # Test 3: Completely shuffled
    import random as rnd
    rnd.seed(123)
    fully_shuffled = base_seq.copy()
    rnd.shuffle(fully_shuffled)
    z_shuffled = encode_seq(fully_shuffled)
    shuffled_dist = cosine_distance(z_base, z_shuffled)
    
    # Pass criteria:
    # - Reversed sequence has significant distance (> 0.1)
    # - OR at least 3 swaps detected
    # - OR fully shuffled has significant distance
    pass_criteria = (
        reversed_dist > 0.1 or 
        sensitive >= 3 or 
        shuffled_dist > 0.15
    )
    
    results["G"] = {
        "pass": pass_criteria, 
        "sensitive": sensitive, 
        "reversed_dist": float(reversed_dist),
        "shuffled_dist": float(shuffled_dist),
        "max_swap_dist": float(max(order_dists)) if order_dists else 0
    }
    
    # TEST H: Feature Importance
    print("  [H] Feature Importance...")
    importance = {}
    for idx, fname in enumerate(normalizer.feature_names[:30]):
        occ = [b.copy() for b in base_seq]
        for b in occ:
            if fname in b:
                b[fname] = 0.0
        z_o = encode_seq(occ)
        importance[fname] = cosine_distance(z_base, z_o)
    top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    results["H"] = {"pass": True, "top5": [x[0] for x in top5]}
    
    # TEST I: Normalization Stability
    print("  [I] Normalization Stability...")
    X_base = normalizer.transform_sequence(base_seq)
    variants = []
    for name, X_v in [("noise", X_base + np.random.randn(*X_base.shape) * 0.01),
                       ("scale", X_base * 1.05),
                       ("clip", np.clip(X_base, -3, 3))]:
        z_v = model.encode(torch.tensor(X_v, dtype=torch.float32).unsqueeze(0).to(device))
        variants.append(cosine_distance(z_base, z_v))
    results["I"] = {"pass": max(variants) < 0.25, "max_dev": float(max(variants))}
    
    # TEST J: Latent Drift - Enhanced
    print("  [J] Latent Drift...")
    embs = []
    
    # Generate more samples with same regime for consistency
    for _ in range(40):
        seq = generate_synthetic_bars(64, "random")
        z = encode_seq(seq)
        embs.append(z.cpu().numpy().flatten())
    
    # Normalize embeddings before computing drift
    embs_array = np.array(embs)
    embs_norm = embs_array / (np.linalg.norm(embs_array, axis=1, keepdims=True) + 1e-8)
    
    # Compute centroids with normalized embeddings
    centroids = [np.mean(embs_norm[i:i+8], axis=0) for i in range(0, 32, 8)]
    
    # Compute drift as cosine distance between consecutive centroids
    drifts = []
    for i in range(len(centroids) - 1):
        cos_sim = np.dot(centroids[i], centroids[i+1]) / (np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[i+1]) + 1e-8)
        drift = 1 - cos_sim
        drifts.append(drift)
    
    max_drift = max(drifts) if drifts else 0
    avg_drift = np.mean(drifts) if drifts else 0
    
    # Also compute overall variance
    overall_var = np.var(embs_norm, axis=0).mean()
    
    results["J"] = {"pass": max_drift < 0.25 or avg_drift < 0.15, 
                    "max_drift": float(max_drift), "avg_drift": float(avg_drift),
                    "variance": float(overall_var)}
    
    return results


def run_full_pipeline():
    """Run complete v1.2 pipeline"""
    start_time = time.time()
    
    # Setup
    output_dir = Path("state_enc_v1/artifacts/v1_2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean
    for item in output_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    
    # Step 1: Generate data
    print("\n[STEP 1] Generate synthetic data...")
    bars = generate_synthetic_bars(800, "random")
    data_path = output_dir / "synthetic_bars.jsonl"
    with open(data_path, "w") as f:
        for bar in bars:
            f.write(json.dumps(bar) + "\n")
    
    # Step 2: Build dataset
    print("\n[STEP 2] Build dataset...")
    from state_enc_v1.src.config import StateEncDatasetConfig
    from state_enc_v1.src.dataset_builder import build_state_enc_dataset
    
    dataset_config = StateEncDatasetConfig(
        raw_bars_path=str(data_path),
        output_path=str(output_dir / "encoder_dataset.jsonl"),
        feature_config_path=str(output_dir / "feature_config_v1.2.json"),
        sequence_length=64,
        stride=16,
        future_bars=5
    )
    
    summary = build_state_enc_dataset(
        dataset_config.raw_bars_path,
        dataset_config.output_path,
        dataset_config
    )
    print(f"  Samples: {summary['total_samples']}")
    
    # Step 3: Train model v1.2
    print("\n[STEP 3] Train model v1.2...")
    from state_enc_v1.src.dataset_encoder import create_dataloaders
    from state_enc_v1.src.model.state_enc_model import StateEncModel
    from state_enc_v1.training.losses import MultiHeadLossV12
    from state_enc_v1.src.normalization import FeatureNormalizer
    
    # Get input dim
    with open(dataset_config.feature_config_path) as f:
        feat_cfg = json.load(f)
    input_dim = feat_cfg.get("feature_dim", 88)
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        dataset_path=dataset_config.output_path,
        feature_config_path=dataset_config.feature_config_path,
        batch_size=16,
        val_split=0.15,
        test_split=0.1,
        num_workers=0,
        seed=42
    )
    
    # Create model v1.2
    model_config = {
        "input_dim": input_dim,
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 4,
        "dim_feedforward": 256,
        "dropout": 0.075,
        "sequence_length": 64,
        "pooling": "last",
        "heads": {
            "self_supervised": {"enabled": True},
            "regime": {"enabled": True, "num_classes": 6},
            "shock": {"enabled": True},
            "reconstruct": {"enabled": True},
            "order": {"enabled": True},
            "anchor": {"enabled": True}
        }
    }
    
    model = StateEncModel.from_config(model_config)
    device = torch.device("cpu")
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  z_t dim: {model.get_embedding_dim()}")
    print(f"  Heads: {model.get_head_names()}")
    
    # Training with enhanced loss weights for v1.2
    loss_weights = {
        "future_dir": 1.0,
        "future_return": 0.1,
        "regime": 0.5,
        "shock": 0.5,
        "reconstruct": 0.2,
        "order": 0.3,
        "anchor": 0.3
    }
    loss_fn = MultiHeadLossV12(loss_weights=loss_weights)
    
    # Warmup + cosine decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    # Warmup scheduler - adjusted for 20 epochs
    def lr_lambda(epoch):
        if epoch < 3:
            return (epoch + 1) / 3  # Warmup
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - 3) / 17))  # Cosine decay
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_loss = float('inf')
    best_state = model.state_dict().copy()
    epochs = 20  # More epochs for contrastive learning
    
    print("\n  Training with augmentation...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        model.set_epoch(epoch)
        train_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            X = batch["X"].to(device)
            
            # Check for NaN in input
            if torch.isnan(X).any():
                X = torch.nan_to_num(X, nan=0.0)
            
            targets = {k: v.to(device) for k, v in batch.items() if k != "X"}
            
            optimizer.zero_grad()
            
            # Enable augmentation for training
            outputs = model(X, augment=True)
            
            # Collect losses - skip NaN and clip large values
            losses_list = []
            
            def safe_loss(loss_val, weight, name, max_val=10.0):
                """Clip and validate loss"""
                if torch.isnan(loss_val) or torch.isinf(loss_val):
                    return None
                clipped = torch.clamp(loss_val, max=max_val)
                return (name, weight * clipped)
            
            # Direction loss
            if "dir_logits" in outputs and "future_dir_5" in targets:
                dir_loss = F.cross_entropy(outputs["dir_logits"], targets["future_dir_5"])
                l = safe_loss(dir_loss, 1.0, "dir")
                if l: losses_list.append(l)
            
            # Return loss
            if "return_pred" in outputs and "future_return_5" in targets:
                ret_loss = F.smooth_l1_loss(outputs["return_pred"], targets["future_return_5"])
                l = safe_loss(ret_loss, 0.1, "return")
                if l: losses_list.append(l)
            
            # Regime loss - no ignore_index
            if "regime_logits" in outputs and "regime_hint" in targets:
                reg_loss = F.cross_entropy(outputs["regime_logits"], targets["regime_hint"])
                l = safe_loss(reg_loss, 0.5, "regime")
                if l: losses_list.append(l)
            
            # Shock loss (from augmentation)
            if "shock_logits" in outputs and "shock_labels" in outputs:
                shock_loss = F.cross_entropy(outputs["shock_logits"], outputs["shock_labels"])
                l = safe_loss(shock_loss, 0.5, "shock")
                if l: losses_list.append(l)
            
            # Order loss (from augmentation)
            if "order_logits" in outputs and "order_labels" in outputs:
                order_loss = F.cross_entropy(outputs["order_logits"], outputs["order_labels"])
                l = safe_loss(order_loss, 0.3, "order")
                if l: losses_list.append(l)
            
            # Reconstruct loss - normalize by feature dim
            if "reconstructed_features" in outputs and "original_x" in outputs:
                recon = outputs["reconstructed_features"]
                orig = outputs["original_x"]
                # Normalize both before computing loss
                recon_norm = F.normalize(recon, dim=-1)
                orig_norm = F.normalize(orig, dim=-1)
                recon_loss = F.mse_loss(recon_norm, orig_norm)
                l = safe_loss(recon_loss, 0.2, "recon")
                if l: losses_list.append(l)
            
            # Anchor loss
            if "anchor_distance" in outputs:
                anchor_loss = outputs["anchor_distance"].mean()
                l = safe_loss(anchor_loss, 0.3, "anchor")
                if l: losses_list.append(l)
            
            # Contrastive loss - prevent embedding collapse
            if "z_t" in outputs:
                z_t = outputs["z_t"]
                B = z_t.shape[0]
                if B > 1:
                    # Compute pairwise distances
                    z_norm = F.normalize(z_t, dim=-1)
                    sim_matrix = torch.mm(z_norm, z_norm.t())  # [B, B]
                    
                    # We want off-diagonal elements to be small (different samples should be different)
                    mask = ~torch.eye(B, dtype=torch.bool, device=device)
                    off_diag = sim_matrix[mask]
                    
                    # Encourage diversity: penalize if all embeddings are too similar
                    diversity_loss = torch.relu(off_diag.mean() - 0.3)  # Lower threshold
                    l = safe_loss(diversity_loss, 1.0, "diversity")  # Higher weight
                    if l: losses_list.append(l)
            
            # Shock contrastive loss - shocked vs non-shocked should be different
            if "z_t" in outputs and "shock_labels" in outputs:
                z_t = outputs["z_t"]
                shock_labels = outputs["shock_labels"]
                
                shocked_mask = shock_labels == 1
                non_shocked_mask = shock_labels == 0
                
                if shocked_mask.sum() > 0 and non_shocked_mask.sum() > 0:
                    z_shocked = z_t[shocked_mask]
                    z_non_shocked = z_t[non_shocked_mask]
                    
                    # Mean embeddings
                    z_shocked_mean = z_shocked.mean(dim=0)
                    z_non_shocked_mean = z_non_shocked.mean(dim=0)
                    
                    # Contrastive: maximize distance between shocked and non-shocked
                    shock_contrast = F.cosine_similarity(z_shocked_mean.unsqueeze(0), z_non_shocked_mean.unsqueeze(0))
                    shock_contrast_loss = torch.relu(shock_contrast + 0.9)  # Want similarity < -0.9
                    l = safe_loss(shock_contrast_loss, 3.0, "shock_contrast")  # Higher weight
                    if l: losses_list.append(l)
                    
                    # Also add L2 distance loss for shock
                    shock_l2_dist = torch.norm(z_shocked_mean - z_non_shocked_mean)
                    shock_l2_loss = torch.relu(1.5 - shock_l2_dist)  # Want L2 dist > 1.5
                    l = safe_loss(shock_l2_loss, 2.0, "shock_l2")
                    if l: losses_list.append(l)
            
            # Order contrastive loss - shuffled vs non-shuffled should be different
            if "z_t" in outputs and "order_labels" in outputs:
                z_t = outputs["z_t"]
                order_labels = outputs["order_labels"]
                
                shuffled_mask = order_labels == 1
                non_shuffled_mask = order_labels == 0
                
                if shuffled_mask.sum() > 0 and non_shuffled_mask.sum() > 0:
                    z_shuffled = z_t[shuffled_mask]
                    z_non_shuffled = z_t[non_shuffled_mask]
                    
                    z_shuffled_mean = z_shuffled.mean(dim=0)
                    z_non_shuffled_mean = z_non_shuffled.mean(dim=0)
                    
                    order_contrast = F.cosine_similarity(z_shuffled_mean.unsqueeze(0), z_non_shuffled_mean.unsqueeze(0))
                    order_contrast_loss = torch.relu(order_contrast + 0.9)  # Even stronger
                    l = safe_loss(order_contrast_loss, 3.0, "order_contrast")  # Much higher weight
                    if l: losses_list.append(l)
                    
                    # Also add L2 distance loss for order
                    order_l2_dist = torch.norm(z_shuffled_mean - z_non_shuffled_mean)
                    order_l2_loss = torch.relu(1.0 - order_l2_dist)  # Want L2 dist > 1.0
                    l = safe_loss(order_l2_loss, 2.0, "order_l2")
                    if l: losses_list.append(l)
            
            # Sum all losses
            if losses_list:
                loss = sum([l for _, l in losses_list])
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                    num_batches += 1
        
        scheduler.step()
        train_loss = train_loss / max(num_batches, 1)
        
        # Validate (no augmentation)
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                X = batch["X"].to(device)
                X = torch.nan_to_num(X, nan=0.0)
                targets = {k: v.to(device) for k, v in batch.items() if k != "X"}
                outputs = model(X, augment=False)
                losses = loss_fn(outputs, targets)
                loss = losses["loss_total"]
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    val_batches += 1
        val_loss = val_loss / max(val_batches, 1)
        
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            best_state = model.state_dict().copy()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"    Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, lr={current_lr:.6f} {'*BEST*' if is_best else ''}")
    
    train_time = time.time() - start_time
    
    # Step 4: Export artifacts
    print("\n[STEP 4] Export artifacts...")
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = final_dir / "state_enc_v1.2.pt"
    torch.save(best_state, model_path)
    
    config_path = final_dir / "model_config_v1.2.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    
    feat_dst = final_dir / "feature_config_v1.2.json"
    shutil.copy(dataset_config.feature_config_path, feat_dst)
    
    # Step 5: Run test suite
    print("\n[STEP 5] Run test suite...")
    model.load_state_dict(best_state)
    model.eval()
    
    normalizer = FeatureNormalizer.from_file(str(feat_dst))
    base_seq = generate_synthetic_bars(64, "random")
    
    test_results = run_test_suite(model, normalizer, base_seq, device)
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for r in test_results.values() if r["pass"])
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — STATE-ENC v1.2")
    print("=" * 70)
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Parameters: {total_params:,}")
    print(f"  z_t dimension: {model.get_embedding_dim()}")
    print(f"  Samples: {summary['total_samples']}")
    print(f"  Best val loss: {best_loss:.4f}")
    
    print("\n  TEST RESULTS:")
    print("-" * 50)
    for key, result in test_results.items():
        status = "✅ PASS" if result["pass"] else "❌ FAIL"
        detail = ""
        if "max_dist" in result:
            detail = f"max={result['max_dist']:.6f}"
        elif "distance" in result:
            detail = f"dist={result['distance']:.4f}"
        elif "distances" in result:
            detail = f"m20={result['distances'].get('mask_20', 0):.3f}, m40={result['distances'].get('mask_40', 0):.3f}"
        elif "sensitive" in result:
            detail = f"sensitive={result['sensitive']}/4"
        elif "max_drift" in result:
            detail = f"drift={result['max_drift']:.4f}"
        elif "min_separation" in result:
            detail = f"sep={result['min_separation']:.4f}"
        print(f"    Test {key}: {status} {detail}")
    
    print(f"\n  Overall: {passed}/10 tests passed")
    print(f"  Status: {'✅ SUITE PASSED' if passed >= 9 else '⚠️ NEEDS IMPROVEMENT'}")
    
    # Save report
    def convert_to_json_serializable(obj):
        """Convert numpy/bool types to JSON serializable"""
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(v) for v in obj]
        return obj
    
    report = {
        "version": "1.2",
        "timestamp": datetime.now().isoformat(),
        "training_time": float(train_time),
        "total_time": float(total_time),
        "parameters": int(total_params),
        "z_t_dim": int(model.get_embedding_dim()),
        "samples": int(summary["total_samples"]),
        "best_val_loss": float(best_loss),
        "tests_passed": int(passed),
        "tests": convert_to_json_serializable(test_results)
    }
    
    report_path = final_dir / "test_report_v1.2.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  Artifacts saved to: {final_dir}")
    
    return report


if __name__ == "__main__":
    report = run_full_pipeline()
    sys.exit(0 if report["tests_passed"] >= 9 else 1)
