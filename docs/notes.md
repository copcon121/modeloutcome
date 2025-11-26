# Development Notes

## Project Status
- **Created**: 2025-01-26
- **Status**: Initial setup complete, ready for Phase 1 execution

## Key Design Decisions

### 1. Outcome-Based Labeling
- Using R-multiple (risk-reward) instead of binary up/down prediction
- Target: 2R, Stop: 1R (configurable)
- Label: long/short/skip based on which target is hit first
- Rationale: Aligns with real trading risk management

### 2. Feature Engineering
- **SMC**: Swing detection with lookback=2, simple BOS/CHoCH logic
- **Volume Profile**: 50 price bins, 70% value area
- **L2 Depth**: Placeholder for now, needs Rithmic integration
- **Time Features**: Sinusoidal encoding for cyclical patterns
- Rationale: Rich context from multiple domains

### 3. Model Architecture
- **Primary**: Transformer encoder (4 heads, 3 layers)
- **Fallback**: MLP for faster training
- Input: [60, ~60] (context_len × feature_dim)
- Output: 3-class softmax (long/short/skip)
- Rationale: Transformer captures temporal patterns well

### 4. Data Pipeline
- Layer 1 → Layer 2: HTTP POST with JSON (localhost, non-blocking)
- Layer 2 → Layer 3: HTTP POST with feature matrix
- Rationale: Decoupled services, easy to scale/debug

## TODO / Future Enhancements

### Phase 1 Extensions
- [ ] Add Rithmic API integration for real delta and L2 depth
- [ ] Multi-timeframe features (M1 + M5 aggregated)
- [ ] More sophisticated SMC detection (liquidity sweeps, multi-swing)

### Phase 2 Optimizations
- [ ] Ensemble models (Transformer + LightGBM)
- [ ] Feature selection / importance analysis
- [ ] Hyperparameter tuning (Optuna)
- [ ] Data augmentation techniques

### Phase 3 Production
- [ ] Docker Compose for multi-container orchestration
- [ ] Monitoring dashboard (Streamlit or Grafana)
- [ ] Auto-execution module with safety checks
- [ ] Cloud deployment (AWS/GCP with low latency)

## Known Limitations

1. **L2 Depth Features**: Currently placeholder values
   - Need Rithmic API subscription
   - Integration requires OnMarketDepth() implementation

2. **Normalization**: Z-score based on training data
   - May need periodic re-fitting as market regime changes
   - Consider online/incremental normalization

3. **Latency**: End-to-end ~200ms target
   - Current bottleneck: Feature extraction (~100ms)
   - Potential optimization: Cython for SMC calculations

4. **Label Imbalance**: Skip class typically 50-60%
   - May need class weighting or focal loss
   - Consider stratified sampling

## Performance Benchmarks (Target)

### Offline Metrics
- **Validation Accuracy**: >50% (baseline: 33%)
- **Long Precision**: >60%
- **Short Precision**: >60%
- **Skip Recall**: >70%

### Live Metrics
- **Feature Extraction**: <100ms per bar
- **Model Inference**: <50ms per prediction
- **End-to-End Latency**: <200ms (NinjaTrader → Decision)

### Trading Metrics
- **Expected R**: >0.5R per trade
- **Win Rate**: >45% on long/short trades
- **Max Drawdown**: <10R over 100 trades

## Troubleshooting Log

### Common Issues

**Issue**: Import errors with layer2_feature_engine modules
- **Cause**: Python path not set correctly
- **Fix**: Add project root to PYTHONPATH or use `sys.path.append()`

**Issue**: Model server returns 503
- **Cause**: Model file not found
- **Fix**: Train model first with `python src/layer3_model/training/train_outcome_model.py`

**Issue**: Feature extraction crashes with NaN values
- **Cause**: ATR or normalization divide-by-zero
- **Fix**: Added epsilon checks in normalizer and feature extraction

**Issue**: NinjaTrader HTTP POST fails
- **Cause**: Feature Engine server not running
- **Fix**: Start server with `uvicorn src.layer2_feature_engine.api_server:app --port 5001`

## Testing Checklist

### Unit Tests (TODO)
- [ ] Swing detection accuracy
- [ ] BOS/CHoCH logic correctness
- [ ] Volume Profile VAH/VAL calculation
- [ ] Outcome labeling logic
- [ ] Feature normalization

### Integration Tests (TODO)
- [ ] Layer 1 → Layer 2 communication
- [ ] Layer 2 → Layer 3 communication
- [ ] End-to-end pipeline with sample data

### Performance Tests (TODO)
- [ ] Latency benchmarks for each layer
- [ ] Memory usage under sustained load
- [ ] Model inference throughput

## References

- NinjaTrader 8 API: https://ninjatrader.com/support/helpGuides/nt8/
- Smart Money Concepts: https://www.tradingview.com/scripts/smartmoneyconcepts/
- PyTorch Transformer: https://pytorch.org/docs/stable/nn.html#transformer
- FastAPI: https://fastapi.tiangolo.com/

---

**Last Updated**: 2025-01-26
