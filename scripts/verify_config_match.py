#!/usr/bin/env python3
"""Verify Live Gateway config matches Backtest config"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

print("=" * 70)
print("VERIFICATION: Live Gateway vs Backtest Config")
print("=" * 70)

# 1. Feature order check
print("\n1. FEATURE ORDER CHECK")
print("-" * 50)

with open(ROOT / "output/asm_dataset_v1/asm_dataset_v1_stats.json") as f:
    stats = json.load(f)
train_features = stats["features"]["feature_names"][:100]

from services.live_gateway.context_store import ASM_FEATURE_COLS

match = all(t == l for t, l in zip(train_features, ASM_FEATURE_COLS))
print(f"  Training features: {len(train_features)}")
print(f"  Live features: {len(ASM_FEATURE_COLS)}")
print(f"  Order match: {'YES' if match else 'NO'}")

# 2. Model config check
print("\n2. MODEL CONFIG CHECK")
print("-" * 50)

import torch
checkpoint = torch.load(ROOT / "output/asm_models_v1/ASM-GRU64-v1.0-C3.pt", map_location="cpu", weights_only=False)
model_config = checkpoint.get("config", {})
print(f"  Model: ASM-GRU64-v1.0-C3.pt")
print(f"  Hidden dim: {model_config.get('hidden_dim', 64)}")
print(f"  Num layers: {model_config.get('num_layers', 2)}")

for key in checkpoint["model_state_dict"].keys():
    if "gru.weight_ih_l0" in key:
        input_dim = checkpoint["model_state_dict"][key].shape[1]
        print(f"  Input dim (from weights): {input_dim}")
        break

# 3. Strategy config check
print("\n3. STRATEGY CONFIG CHECK")
print("-" * 50)

from services.live_gateway.app import P_SHIFT_THRESHOLD, STRATEGY_VERSION
from services.live_gateway.s4_engine import S4_CONFIG

print(f"  Strategy: {STRATEGY_VERSION}")
print(f"  P_SHIFT_THRESHOLD: {P_SHIFT_THRESHOLD}")
print(f"  Session filter: {S4_CONFIG['session']}")
print(f"  RR target: {S4_CONFIG['rr_target']}")

with open(ROOT / "backtests/s4_asm_lowshift_extval_new6w_v1.json") as f:
    backtest = json.load(f)

bt_meta = backtest["meta"]
print(f"\n  Backtest config:")
print(f"    Strategy: {bt_meta['strategy_id']}")
print(f"    Session: {bt_meta['session']}")
print(f"    RR target: {bt_meta['rr_target']}")
print(f"    ASM seq_len: {bt_meta['asm_seq_len']}")
print(f"    ASM feature_dim: {bt_meta['asm_feature_dim']}")

# 4. ASM context check
print("\n4. ASM CONTEXT CONFIG CHECK")
print("-" * 50)

from services.live_gateway.context_store import ASM_SEQ_LEN, ASM_FEATURE_DIM
print(f"  Live ASM_SEQ_LEN: {ASM_SEQ_LEN}")
print(f"  Live ASM_FEATURE_DIM: {ASM_FEATURE_DIM}")
print(f"  Backtest seq_len: {bt_meta['asm_seq_len']}")
print(f"  Backtest feature_dim: {bt_meta['asm_feature_dim']}")

seq_match = ASM_SEQ_LEN == bt_meta["asm_seq_len"]
dim_match = ASM_FEATURE_DIM == bt_meta["asm_feature_dim"]
print(f"  Seq len match: {'YES' if seq_match else 'NO'}")
print(f"  Feature dim match: {'YES' if dim_match else 'NO'}")

print("\n" + "=" * 70)
if match and seq_match and dim_match:
    print("ALL CONFIGS MATCH - READY FOR NINJA!")
else:
    print("CONFIG MISMATCH - PLEASE FIX BEFORE RUNNING")
print("=" * 70)
