# Restructure Plan - Simplified Folder Structure

## Current Structure (Too Many Folders ❌)
```
src/
├── layer1_ninjatrader/
│   ├── ExportRawData.cs
│   └── README.md
├── layer2_feature_engine/
│   ├── api_server.py
│   ├── core/
│   ├── smc/
│   ├── volume_profile/
│   ├── orderflow_l2/
│   └── utils/
└── layer3_model/
    ├── training/
    ├── inference/
    └── evaluation/
```

## New Structure (Clean & Simple ✅)
```
layer1_ninjatrader/
├── ExportRawData.cs
└── README.md

layer2_features/
├── api_server.py          # FastAPI server (port 5001)
├── schema.py              # Data structures
├── normalizer.py          # Feature normalization
├── context_manager.py     # Main orchestrator
├── smc_features.py        # SMC: Swing + BOS + Zones (MERGED)
├── volume_profile.py      # Volume Profile features
├── orderflow.py           # Level 2 orderflow
└── utils.py               # Time features + logging + config (MERGED)

layer3_model/
├── api_server.py          # FastAPI inference server (port 5002)
├── labeler.py             # Generate labeled dataset
├── train.py               # Training script
├── models.py              # Model architectures (Transformer + MLP)
└── metrics.py             # Evaluation metrics
```

## Benefits of New Structure

### ✅ Fewer Folders
- **Before**: 11 subfolders
- **After**: 3 main folders

### ✅ Merged Related Files
- SMC modules (swing + structure + zones) → `smc_features.py`
- Utils modules (time + logging + config) → `utils.py`
- Training modules → single folder

### ✅ Clearer Purpose
- `layer1_ninjatrader/` - NinjaTrader C# code only
- `layer2_features/` - ALL feature engineering
- `layer3_model/` - ALL ML model code

### ✅ Easier Navigation
- Want to change SMC logic? → `layer2_features/smc_features.py`
- Want to train model? → `layer3_model/train.py`
- Want to deploy? → `layer3_model/api_server.py`

## File Mapping

### Layer 2 Merges:
```
core/schema.py → schema.py
core/normalizer.py → normalizer.py
core/context_manager.py → context_manager.py

smc/swing.py + smc/structure.py + smc/zones.py → smc_features.py

volume_profile/vp_builder.py → volume_profile.py

orderflow_l2/l2_features.py → orderflow.py

utils/time_features.py + utils/logging_utils.py + utils/config_loader.py → utils.py
```

### Layer 3 Merges:
```
training/labeler.py → labeler.py
training/train_outcome_model.py → train.py (also extract models to models.py)
inference/server.py → api_server.py
evaluation/metrics.py → metrics.py
```

## Implementation Steps

1. Create new structure in parallel (don't delete old yet)
2. Merge files with clear section comments
3. Update imports in all files
4. Test that everything still works
5. Delete old structure
6. Update documentation

## Estimated Time: 15-20 minutes
