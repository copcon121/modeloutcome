# Troubleshooting Guide

**S4_LDN_ASM_LowShift_0.2_v1.1** - Common issues and solutions

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Model Loading Issues](#model-loading-issues)
3. [API Issues](#api-issues)
4. [Performance Issues](#performance-issues)
5. [NinjaTrader Integration](#ninjatrader-integration)
6. [Debug Procedures](#debug-procedures)
7. [Monitoring & Logging](#monitoring--logging)

---

## Installation Issues

### Python Version Mismatch
**Symptom:** Import errors or syntax errors
```
SyntaxError: invalid syntax
```

**Solution:**
```bash
# Check Python version
python --version

# Requires Python 3.9+
# If using older version, upgrade or use pyenv
```

### Missing Dependencies
**Symptom:** ModuleNotFoundError
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
pip install -r requirements.txt

# Or install individually
pip install torch numpy pandas fastapi uvicorn pydantic
```

### CUDA/GPU Issues
**Symptom:** CUDA errors or slow inference
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Force CPU mode in asm_inference.py
device = torch.device("cpu")

# Or set environment variable
export CUDA_VISIBLE_DEVICES=""
```

---

## Model Loading Issues

### Model File Not Found
**Symptom:**
```
FileNotFoundError: ASM model not found at output/asm_models_v1/ASM-GRU64-v1.0-C3.pt
```

**Solution:**
1. Check file exists:
   ```bash
   ls output/asm_models_v1/ASM-GRU64-v1.0-C3.pt
   ```
2. If missing, retrain or restore from backup:
   ```bash
   python scripts/train_asm_v1.py
   ```

### Model Architecture Mismatch
**Symptom:**
```
RuntimeError: Error(s) in loading state_dict
```

**Solution:**
- Ensure model architecture matches saved weights
- Check `ASM_MODEL_CONFIG` in training script
- Retrain model if architecture changed

### Model Loading Timeout
**Symptom:** Server hangs on startup

**Solution:**
```python
# Add timeout to model loading
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Model loading timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout
```

---

## API Issues

### 422 Validation Error
**Symptom:**
```json
{"detail":[{"loc":["body","bar","volume"],"msg":"field required"}]}
```

**Solution:**
- Check request JSON matches schema exactly
- All fields in `BarData` and `TickFeatures` are required
- Use correct data types (float, not string)

### 500 Internal Server Error
**Symptom:**
```json
{"detail": "Internal server error"}
```

**Debug:**
```bash
# Check server logs
tail -f logs/gateway.log

# Run with debug mode
uvicorn services.live_gateway.app:app --reload --log-level debug
```

### Connection Refused
**Symptom:**
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solution:**
1. Check server is running:
   ```bash
   curl http://localhost:8000/health
   ```
2. Check port is not in use:
   ```bash
   netstat -an | grep 8000
   ```
3. Start server:
   ```bash
   python services/live_gateway/run_server.py
   ```

### CORS Errors
**Symptom:** Browser shows CORS error

**Solution:**
CORS is already enabled in `app.py`. If issues persist:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Performance Issues

### Slow Response Time
**Symptom:** Response time >100ms

**Diagnosis:**
```python
import time

start = time.time()
# ... inference code ...
print(f"Inference time: {(time.time() - start) * 1000:.2f}ms")
```

**Solutions:**
1. Use CPU instead of GPU for small batches
2. Reduce context window if possible
3. Profile code to find bottlenecks

### High Memory Usage
**Symptom:** Memory >1GB or OOM errors

**Solutions:**
1. Limit context store size:
   ```python
   MAX_CONTEXTS = 10  # Limit active symbols
   ```
2. Clear old contexts periodically
3. Use smaller batch sizes

### Context Store Growing
**Symptom:** Memory increases over time

**Solution:**
```python
# Add cleanup in context_store.py
def cleanup_old_contexts(self, max_age_hours=24):
    """Remove contexts not updated recently"""
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    for key in list(self.contexts.keys()):
        if self.contexts[key].last_update < cutoff:
            del self.contexts[key]
```

---

## NinjaTrader Integration

### No Bars Received
**Symptom:** Gateway receives no requests

**Checklist:**
1. SMC_Exporter_Pro_v3 enabled on chart
2. Endpoint URL correct: `http://localhost:8000/live_bar`
3. Export on bar close enabled
4. Firewall not blocking connection

### Timestamp Parsing Error
**Symptom:**
```
ValueError: Invalid isoformat string
```

**Solution:**
Ensure timestamp format matches:
```python
# Expected format
"2025-11-17T23:01:00"

# Or with timezone
"2025-11-17T23:01:00.0000000"
```

### Session Detection Wrong
**Symptom:** Signals generated outside London session

**Debug:**
```python
# Check session detection
from services.live_gateway.s4_engine import detect_session

for hour in range(24):
    print(f"Hour {hour}: {detect_session(hour)}")

# Expected: London = hours 8-13 UTC
```

---

## Debug Procedures

### 1. Health Check
```bash
# Basic health
curl http://localhost:8000/health

# Expected response
{"status":"ok","model_loaded":true,"contexts_active":1}
```

### 2. Test Single Bar
```bash
python -c "
import requests
import json

bar = {
    'symbol': 'GC 12-25',
    'timeframe': 'M1',
    'timestamp': '2025-11-17T10:01:00',
    'bar_index': 1,
    'bar': {'o':4045,'h':4046,'l':4044,'c':4045,'volume':100,'delta':0,'buy_volume':50,'sell_volume':50,'best_bid':4045,'best_ask':4046,'vwap_daily':4045},
    'tick_features': {'tick_speed':50,'aggr_buy_speed':50,'aggr_sell_speed':50,'price_speed':1}
}

r = requests.post('http://localhost:8000/live_bar', json=bar)
print(json.dumps(r.json(), indent=2))
"
```

### 3. Run Pipeline Test
```bash
python scripts/test_full_pipeline_v1.py --quick
```

### 4. Check Logs
```bash
# Signal log
tail -f logs/live_signals_s4_asm_v1.jsonl

# Parse and analyze
python -c "
import json
with open('logs/live_signals_s4_asm_v1.jsonl') as f:
    signals = [json.loads(l) for l in f]
print(f'Total signals: {len(signals)}')
print(f'Filter pass: {sum(1 for s in signals if s.get(\"filter_pass\"))}')
"
```

### 5. Validate Against Backtest
```bash
python scripts/simulate_live_gateway_from_jsonl.py
```

---

## Monitoring & Logging

### Enable Detailed Logging
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/gateway.log'),
        logging.StreamHandler()
    ]
)
```

### Key Metrics to Monitor
1. **Response time** - Should be <100ms
2. **Memory usage** - Should be <500MB
3. **Signal count** - Compare with expected rate
4. **Error rate** - Should be <1%

### Alerting
Set up alerts for:
- Server down (health check fails)
- High error rate
- Memory usage >80%
- No signals for extended period

### Log Rotation
```bash
# Add to crontab for daily rotation
0 0 * * * mv logs/live_signals_s4_asm_v1.jsonl logs/live_signals_$(date +%Y%m%d).jsonl
```

---

## Performance Optimization

### 1. Use In-Process Calls
For replay/backtest, avoid HTTP overhead:
```python
from services.live_gateway.app import process_live_bar
from services.live_gateway.models import LiveBarEvent

# Direct call (faster)
response = await process_live_bar(LiveBarEvent(**bar_data))
```

### 2. Batch Processing
For historical data, process in batches:
```python
# Process multiple bars without HTTP
for bar in bars:
    feature_bar, feature_dict = context_store.update(symbol, timeframe, raw_bar)
    # ... rest of logic
```

### 3. Model Optimization
```python
# Use torch.no_grad() for inference
with torch.no_grad():
    output = model(input_tensor)

# Use half precision if supported
model = model.half()
```

---

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `ASM model not loaded` | Model file missing | Check path, retrain |
| `Context not found` | First bar for symbol | Normal, context builds up |
| `Invalid timestamp` | Wrong format | Use ISO 8601 format |
| `NaN in features` | Missing data | Check input completeness |
| `Session filter failed` | Outside London hours | Normal, no signal expected |

---

## Getting Help

1. Check this troubleshooting guide
2. Review logs for error details
3. Run pipeline test to isolate issue
4. Check GitHub issues (if applicable)

---

**Version**: 1.0  
**Last Updated**: 2025-12-03
