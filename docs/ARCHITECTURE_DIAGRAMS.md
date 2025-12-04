# SMC Auto-Trading System — Architecture Diagrams

> Tài liệu này chứa các sơ đồ kiến trúc cho hệ thống auto-trading SMC.

---

## 1. High-Level System Architecture (Block Diagram)

```mermaid
flowchart TB
    subgraph NT8["🖥️ NinjaTrader 8"]
        IND[/"Indicator Exporter<br/>OHLCV + Vol/Delta/Tick"/]
    end

    subgraph EXPORT["📤 EXPORT-SMC-JSON"]
        RAW[/"raw_bars.jsonl<br/>(Bar-level JSONL)"/]
    end

    subgraph SMC["🔧 SMC Python Core"]
        CALC["SMC Feature Calculator<br/>BOS/CHoCH • Swing • OB/FVG<br/>VA • Session • Premium/Discount"]
        ENH[/"bars_enhanced.jsonl<br/>(88+ features per bar)"/]
    end

    subgraph STATEENC["🧠 STATE-ENC v1"]
        DSB["Dataset Builder<br/>Sliding Window N=128"]
        EDS[/"encoder_dataset.jsonl<br/>(Sequence samples)"/]
        ENC["Transformer Encoder<br/>+ Multi-Head"]
        ART[/"Artifacts<br/>state_enc_v1.pt<br/>feature_config.json"/]
    end

    subgraph ASM["📊 ASM v2"]
        ASMM["Auction State Model<br/>Regime Classification"]
        REG[/"regime_state<br/>auction_phase"/]
    end

    subgraph S4["⚡ S4_LDN Strategy"]
        META["Meta Layer<br/>Trade Filter + Risk"]
        SIG["Signal Generator<br/>FVG/OB Entry"]
        DEC{{"Trade Decision<br/>LONG / SHORT / SKIP"}}
    end

    NT8 -->|"Real-time export"| EXPORT
    RAW --> SMC
    CALC --> ENH
    ENH --> STATEENC
    DSB --> EDS
    EDS --> ENC
    ENC --> ART
    
    ART -->|"z_t embedding"| ASM
    ART -->|"z_t embedding"| S4
    
    ASMM --> REG
    REG -->|"regime info"| S4
    
    ENH -->|"SMC signals"| SIG
    SIG --> META
    META --> DEC

    style NT8 fill:#e1f5fe
    style STATEENC fill:#fff3e0
    style ASM fill:#f3e5f5
    style S4 fill:#e8f5e9
```

**Mô tả:** Sơ đồ tổng quan thể hiện luồng dữ liệu từ NinjaTrader 8 qua các module xử lý. Data flow chính: Raw OHLCV → SMC Features → State Encoding → Regime Detection → Trade Decision. STATE-ENC v1 đóng vai trò trung tâm, cung cấp embedding z_t cho cả ASM v2 và S4_LDN Strategy.

---

## 2. Data Pipeline Flow (Detailed)

```mermaid
flowchart LR
    subgraph INPUT["📥 Input Layer"]
        NT["NinjaTrader 8<br/>M1 Bars"]
        OHLCV["OHLCV"]
        VOL["Volume<br/>Delta<br/>Tick"]
    end

    subgraph PROCESS["⚙️ Processing Layer"]
        EXP["EXPORT-SMC-JSON"]
        SMC["SMC Python Core"]
    end

    subgraph FEATURES["📋 Feature Layer"]
        F1["OHLCV & Shape<br/>(15 features)"]
        F2["Delta & Tick<br/>(13 features)"]
        F3["SMC Structure<br/>(23 features)"]
        F4["Liquidity/Sweep<br/>(4 features)"]
        F5["OB/FVG Proximity<br/>(12 features)"]
        F6["VA/Session<br/>(20 features)"]
        F7["Regime Hint<br/>(1 feature)"]
    end

    subgraph ENCODE["🧠 Encoding Layer"]
        SEQ["Sequence Builder<br/>N=128, stride=4"]
        NORM["Normalizer<br/>Z-Score / Raw"]
        TRANS["Transformer<br/>Encoder"]
        POOL["Pooling<br/>(last token)"]
    end

    subgraph OUTPUT["📤 Output Layer"]
        ZT["z_t<br/>[B, 128]"]
        ZSEQ["z_seq<br/>[B, N, 128]"]
    end

    NT --> OHLCV & VOL
    OHLCV & VOL --> EXP
    EXP -->|"raw_bars.jsonl"| SMC
    
    SMC --> F1 & F2 & F3 & F4 & F5 & F6 & F7
    
    F1 & F2 & F3 & F4 & F5 & F6 & F7 -->|"88 features"| SEQ
    SEQ -->|"[N, 88]"| NORM
    NORM -->|"[N, 95]"| TRANS
    TRANS --> POOL
    POOL --> ZT
    TRANS --> ZSEQ

    style INPUT fill:#e3f2fd
    style PROCESS fill:#fff8e1
    style FEATURES fill:#f1f8e9
    style ENCODE fill:#fce4ec
    style OUTPUT fill:#e8eaf6
```

**Mô tả:** Pipeline chi tiết từ raw data đến embedding. NinjaTrader xuất OHLCV + microstructure → SMC Core tính 88 features (7 groups) → Sequence builder tạo sliding window → Transformer encoder → Output z_t (market state) và z_seq (full sequence).

---

## 3. STATE-ENC v1 Internal Architecture

```mermaid
flowchart TB
    subgraph INPUT["Input"]
        X["X: [B, N, D]<br/>B=batch, N=128, D=95"]
    end

    subgraph ENCODER["Encoder Backbone"]
        PROJ["Linear Projection<br/>D → d_model"]
        PE["Positional Encoding<br/>(Sinusoidal)"]
        TF1["TransformerEncoderLayer 1"]
        TF2["TransformerEncoderLayer 2"]
        TF3["TransformerEncoderLayer 3"]
        TF4["TransformerEncoderLayer 4"]
        LN["LayerNorm"]
        POOL["Pooling (last)"]
    end

    subgraph HEADS["Prediction Heads"]
        SSH["SelfSupervisedHead"]
        RH["RegimeHead"]
        MSH["MetaS4Head"]
    end

    subgraph OUTPUTS["Outputs"]
        ZSEQ["z_seq: [B, N, 128]"]
        ZT["z_t: [B, 128]"]
        DIR["dir_logits: [B, 3]"]
        RET["return_pred: [B]"]
        REG["regime_logits: [B, 6]"]
        META["meta_output: [B, 4]"]
    end

    X --> PROJ
    PROJ --> PE
    PE --> TF1 --> TF2 --> TF3 --> TF4
    TF4 --> LN
    LN --> ZSEQ
    LN --> POOL --> ZT

    ZT --> SSH
    ZT --> RH
    ZT --> MSH

    SSH --> DIR & RET
    RH --> REG
    MSH --> META

    style ENCODER fill:#fff3e0
    style HEADS fill:#e8f5e9
    style OUTPUTS fill:#e3f2fd
```

**Mô tả:** Kiến trúc nội bộ STATE-ENC v1. Input tensor [B, N, 95] đi qua Linear projection → Positional encoding → 4 Transformer layers → Pooling để lấy z_t. Ba heads dự đoán: (1) future direction/return, (2) regime classification, (3) meta output cho S4.

---

## 4. Training & Inference Pipeline

```mermaid
flowchart TB
    subgraph TRAIN["🎯 Training Phase"]
        direction TB
        TD[/"bars_enhanced.jsonl"/]
        TDB["Dataset Builder"]
        TDS[/"encoder_dataset.jsonl"/]
        TDL["DataLoader<br/>train/val/test"]
        TM["StateEncModel"]
        TL["MultiHeadLoss<br/>CE + MSE"]
        TO["AdamW Optimizer"]
        TC["Checkpoint"]
    end

    subgraph EXPORT["📦 Export Phase"]
        EX["export_artifacts.py"]
        PT[/"state_enc_v1.pt"/]
        FC[/"feature_config.json"/]
        MC[/"model_config.json"/]
    end

    subgraph INFER["⚡ Inference Phase"]
        direction TB
        IB[/"Live bars (N=128)"/]
        IN["Normalizer"]
        IM["StateEncModel<br/>(eval mode)"]
        IZ["z_t embedding"]
        IA["ASM v2"]
        IS["S4_LDN"]
    end

    TD --> TDB --> TDS --> TDL
    TDL --> TM
    TM --> TL --> TO
    TO -->|"backward"| TM
    TM -->|"best model"| TC

    TC --> EX
    EX --> PT & FC & MC

    PT & FC & MC --> IM
    IB --> IN --> IM --> IZ
    IZ --> IA & IS

    style TRAIN fill:#fff8e1
    style EXPORT fill:#e8f5e9
    style INFER fill:#e3f2fd
```

**Mô tả:** Pipeline hoàn chỉnh từ training đến inference. Training: build dataset → train với multi-head loss → save checkpoint. Export: extract best model + configs. Inference: load model → normalize live bars → encode → feed z_t vào ASM v2 và S4_LDN.

---

## 5. Module Dependency Graph

```mermaid
graph LR
    subgraph EXTERNAL["External"]
        NT8["NinjaTrader 8"]
    end

    subgraph CORE["Core Modules"]
        EXP["EXPORT-SMC-JSON"]
        SMC["SMC Python Core"]
    end

    subgraph ML["ML Modules"]
        SE["STATE-ENC v1"]
        ASM["ASM v2"]
    end

    subgraph STRATEGY["Strategy"]
        S4["S4_LDN"]
    end

    NT8 -->|"raw data"| EXP
    EXP -->|"JSONL"| SMC
    SMC -->|"features"| SE
    SE -->|"z_t"| ASM
    SE -->|"z_t"| S4
    ASM -->|"regime"| S4
    SMC -->|"SMC signals"| S4

    style NT8 fill:#ffcdd2
    style SE fill:#fff9c4
    style ASM fill:#c8e6c9
    style S4 fill:#bbdefb
```

**Mô tả:** Dependency graph thể hiện quan hệ giữa các module. STATE-ENC v1 là hub trung tâm, nhận input từ SMC Core và cung cấp embedding cho cả ASM v2 và S4_LDN. S4_LDN tổng hợp tất cả signals để ra quyết định trade.

---

## 6. Real-Time Inference Flow

```mermaid
sequenceDiagram
    participant NT as NinjaTrader 8
    participant EXP as Exporter
    participant SMC as SMC Core
    participant SE as STATE-ENC
    participant ASM as ASM v2
    participant S4 as S4_LDN
    participant EXEC as Execution

    Note over NT,EXEC: Every M1 Bar Close
    
    NT->>EXP: New bar data
    EXP->>SMC: raw_bar
    SMC->>SMC: Compute SMC features
    SMC->>SE: bar_enhanced
    
    SE->>SE: Update sliding window
    SE->>SE: Normalize & Encode
    SE-->>ASM: z_t embedding
    SE-->>S4: z_t embedding
    
    ASM->>ASM: Classify regime
    ASM-->>S4: regime_state
    
    SMC-->>S4: entry_signals (FVG/OB)
    
    S4->>S4: Meta filter
    S4->>S4: Risk assessment
    
    alt Valid Entry
        S4->>EXEC: LONG/SHORT + size
        EXEC->>NT: Place order
    else No Entry
        S4->>S4: SKIP
    end
```

**Mô tả:** Sequence diagram cho real-time inference. Mỗi khi bar M1 close: data flow qua SMC → STATE-ENC encode → ASM classify regime → S4 tổng hợp signals và quyết định trade. Toàn bộ pipeline cần hoàn thành trong vài giây trước bar tiếp theo.

---

## Quick Reference

| Module | Input | Output | Role |
|--------|-------|--------|------|
| EXPORT-SMC-JSON | NT8 indicator | raw_bars.jsonl | Data export |
| SMC Python Core | raw_bars.jsonl | bars_enhanced.jsonl | Feature engineering |
| STATE-ENC v1 | bars_enhanced.jsonl | z_t, z_seq | State encoding |
| ASM v2 | z_t + meta | regime, auction_state | Regime detection |
| S4_LDN | z_t + regime + signals | trade decision | Strategy execution |
