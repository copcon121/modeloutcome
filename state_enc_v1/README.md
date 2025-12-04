# STATE-ENC v1 — Market State Encoder

## Mục tiêu

Module này encode chuỗi N bar M1 liên tục thành embedding vector `z_t` đại diện cho **market state tại bar cuối**. Embedding này được sử dụng cho:
- **ASM v2**: Auction/Regime model
- **Meta S4_LDN**: Filter trades, điều chỉnh risk

## Data Flow

```
NinjaTrader 8 (raw OHLCV + vol/delta/tick)
        ↓
Python SMC Core (BOS/CHoCH, swing, OB/FVG, VA, session...)
        ↓
bars_enhanced.jsonl (bar-level features)
        ↓
[BUILD DATASET] → encoder_dataset.jsonl (sequence samples)
        ↓
[TRAIN] → state_enc_v1.pt
        ↓
[EXPORT] → artifacts/final/
```

## Cấu trúc thư mục

```
state_enc_v1/
├── README.md
├── configs/
│   ├── state_enc_dataset_v1.json
│   ├── state_enc_model_v1.json
│   └── state_enc_train_v1.json
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── features_spec.py
│   ├── normalization.py
│   ├── dataset_builder.py
│   ├── dataset_encoder.py
│   └── model/
│       ├── __init__.py
│       ├── encoder_backbone.py
│       ├── heads.py
│       └── state_enc_model.py
├── training/
│   ├── losses.py
│   ├── trainer.py
│   └── eval_metrics.py
├── scripts/
│   ├── build_state_enc_dataset.py
│   ├── train_state_enc.py
│   └── export_state_enc_artifacts.py
└── artifacts/
    ├── encoder_dataset.jsonl  (generated)
    └── runs/
        └── final/
            ├── state_enc_v1.pt
            ├── feature_config.json
            └── model_config.json
```

## Hướng dẫn sử dụng

### Bước 1: Cấu hình dataset

Chỉnh sửa `configs/state_enc_dataset_v1.json`:

```json
{
  "raw_bars_path": "data/bars_enhanced.jsonl",
  "output_path": "state_enc_v1/artifacts/encoder_dataset.jsonl",
  "sequence_length": 128,
  "stride": 4,
  "future_bars": 5,
  "future_dir_threshold_up": 0.0005,
  "future_dir_threshold_down": -0.0005,
  "tick_size": 0.25
}
```

### Bước 2: Build dataset

```bash
python state_enc_v1/scripts/build_state_enc_dataset.py --config state_enc_v1/configs/state_enc_dataset_v1.json
```

### Bước 3: Train model

Chỉnh sửa `configs/state_enc_train_v1.json`, sau đó:

```bash
python state_enc_v1/scripts/train_state_enc.py --config state_enc_v1/configs/state_enc_train_v1.json
```

### Bước 4: Export artifacts

```bash
python state_enc_v1/scripts/export_state_enc_artifacts.py
```

### Bước 5: Sử dụng encoder trong module khác

```python
import torch
import json
from state_enc_v1.src.model.state_enc_model import StateEncModel
from state_enc_v1.src.normalization import FeatureNormalizer
from state_enc_v1.src.features_spec import FEATURE_SPEC

# Load configs
with open("state_enc_v1/artifacts/final/model_config.json") as f:
    model_config = json.load(f)
with open("state_enc_v1/artifacts/final/feature_config.json") as f:
    feature_config = json.load(f)

# Initialize model
model = StateEncModel(
    input_dim=model_config["input_dim"],
    d_model=model_config["d_model"],
    num_layers=model_config["num_layers"],
    num_heads=model_config["num_heads"],
    sequence_length=model_config["sequence_length"]
)
model.load_state_dict(torch.load("state_enc_v1/artifacts/final/state_enc_v1.pt"))
model.eval()

# Initialize normalizer
normalizer = FeatureNormalizer()
normalizer.load(feature_config["normalization"])

# Prepare input: sequence of N bars
# bars = [bar_dict_0, bar_dict_1, ..., bar_dict_{N-1}]
# x = normalizer.transform_sequence(bars)  # [N, D]
# x = torch.tensor(x).unsqueeze(0)  # [1, N, D]

# Get embedding
with torch.no_grad():
    outputs = model(x)
    z_t = outputs["z_t"]  # [1, d_model] - market state embedding
    
# Use z_t for ASM v2 or Meta S4
```

## Feature Groups

1. **Core OHLCV & Shape**: o, h, l, c, hl_range, body, wicks, bar_type, volume, ATR...
2. **Delta & Tick Microstructure**: delta, tick_count, buy/sell volume, imbalance...
3. **SMC Structure**: BOS/CHoCH, swing, premium/discount, sweep, OB/FVG proximity...
4. **VA / Auction**: VAH, VAL, POC, session position...
5. **Regime Hint**: ASM v1 rule-based regime (optional)

## Model Architecture

- **Backbone**: Transformer Encoder (configurable layers)
- **Heads**:
  - `SelfSupervisedHead`: Predict future_dir_5, future_return_5
  - `RegimeHead`: Predict regime (6 classes)
  - `MetaS4Head`: Placeholder for future extension

## Output

- `z_t`: Market state embedding [B, d_model]
- `z_seq`: Full sequence embeddings [B, N, d_model]
- Head outputs: logits for classification/regression tasks
