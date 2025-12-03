# Deployment Guide

**S4_LDN_ASM_LowShift_0.2_v1.1** - Complete deployment guide from fresh install to production

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Fresh Installation](#fresh-installation)
3. [Configuration](#configuration)
4. [Testing & Validation](#testing--validation)
5. [Production Deployment](#production-deployment)
6. [Monitoring & Maintenance](#monitoring--maintenance)

---

## System Requirements

### Hardware
- **CPU**: 4+ cores (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space (SSD recommended)
- **Network**: Stable internet connection for NinjaTrader integration

### Software
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 10.15+
- **Python**: 3.9+ (3.11 recommended)
- **NinjaTrader**: 8.0+ (for live trading integration)

### Dependencies
```
Python packages:
- torch>=2.0.0
- numpy>=1.24.0
- pandas>=2.0.0
- fastapi>=0.104.0
- uvicorn[standard]>=0.24.0
- pydantic>=2.0.0
- requests>=2.31.0
```

---

## Fresh Installation

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd modeloutcome
```

### Step 2: Python Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Verify ASM Model
```bash
# Check model file exists
ls output/asm_models_v1/ASM-GRU64-v1.0-C3.pt

# Expected: File should exist (~2MB)
```

### Step 4: Test Installation
```bash
# Run quick pipeline test
python scripts/test_full_pipeline_v1.py --quick

# Expected: All 5 components PASS
```

---

## Configuration

### Model Configuration
File: `config/model_config.yaml`
```yaml
asm_model:
  path: output/asm_models_v1/ASM-GRU64-v1.0-C3.pt
  context_length: 60
  feature_count: 100
  device: cpu  # or cuda

s4_config:
  session: London
  rr_target: 2.0
  p_shift_threshold: 0.2
```

### Data Paths
File: `config/data_paths.yaml`
```yaml
raw_data: data/raw/new_data/
processed_data: data/processed/
logs: logs/
```

### Live Gateway Settings
Edit `services/live_gateway/app.py`:
```python
STRATEGY_VERSION = "S4_LDN_ASM_LowShift_0.2_v1.1"
P_SHIFT_THRESHOLD = 0.2  # ASM filter threshold
```

---

## Testing & Validation

### 1. Quick Pipeline Test
```bash
python scripts/test_full_pipeline_v1.py --quick
```
Tests: Feature Generation, ASM Model, S4 Engine, Live Gateway, Integration

### 2. Full Pipeline Test
```bash
python scripts/test_full_pipeline_v1.py
```
Uses all 6 weeks of data for comprehensive validation.

### 3. Shadow Replay Validation
```bash
# Start gateway (optional - can run in-process)
python services/live_gateway/run_server.py

# Run replay validation
python scripts/simulate_live_gateway_from_jsonl.py
```

Expected results:
- Trades: ~258 per 6 weeks
- Winrate: ~61.6%
- Expectancy: ~+0.85R
- MaxDD: ~26R

### 4. Health Check
```bash
python scripts/health_check.py
```

---

## Production Deployment

### Option A: Direct Python
```bash
# Start server
python services/live_gateway/run_server.py --port 8000

# Or with uvicorn directly
uvicorn services.live_gateway.app:app --host 0.0.0.0 --port 8000
```

### Option B: Docker
```bash
# Build image
docker build -t live-gateway -f deployment/Dockerfile .

# Run container
docker run -d -p 8000:8000 --name live-gateway live-gateway
```

### Option C: Docker Compose
```bash
docker-compose -f deployment/docker-compose.yml up -d
```

### NinjaTrader Integration
1. Configure SMC_Exporter_Pro_v3 indicator
2. Set endpoint: `http://localhost:8000/live_bar`
3. Enable export on bar close
4. Start in shadow mode (log only)

---

## Production Checklist

### Pre-Deployment
- [ ] All pipeline tests pass
- [ ] Shadow replay matches backtest results (±1%)
- [ ] ASM model file verified
- [ ] Log directory writable
- [ ] Network connectivity tested

### Deployment
- [ ] Server started successfully
- [ ] Health endpoint returns OK
- [ ] NinjaTrader connection established
- [ ] First bar processed without error

### Post-Deployment
- [ ] Monitor logs for errors
- [ ] Verify signal generation
- [ ] Check resource usage (CPU, RAM)
- [ ] Review daily signal count

---

## Monitoring & Maintenance

### Log Files
- `logs/live_signals_s4_asm_v1.jsonl` - All signals
- `logs/gateway.log` - Server logs (if configured)

### Health Monitoring
```bash
# Check health
curl http://localhost:8000/health

# Check stats
curl http://localhost:8000/stats
```

### Performance Metrics
- Response time: <100ms per bar
- Memory usage: <500MB
- CPU usage: <10% idle, <50% during inference

### Maintenance Tasks
- Weekly: Review signal logs
- Monthly: Retrain ASM model if needed
- Quarterly: Full pipeline validation

---

## Troubleshooting

See `docs/TROUBLESHOOTING.md` for common issues and solutions.

---

**Version**: 1.0  
**Last Updated**: 2025-12-03
