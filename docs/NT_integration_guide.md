# NinjaTrader Integration Guide - Quality Model API

## Overview

This guide shows how to integrate NinjaTrader 8 with the Quality Model API for trade filtering.

**Two Modes**:
1. **SHADOW**: Log predictions only, no real trades
2. **LIVE**: Execute trades based on ML filter

---

## API Endpoint

**URL**: `http://localhost:8000/predict_quality`  
**Method**: POST  
**Content-Type**: application/json

---

## Integration Architecture

```
NinjaTrader Strategy
        ↓
   P2 Event Detected (SMC signal)
        ↓
   Build Features [60×66]
        ↓
   HTTP POST → /predict_quality
        ↓
   If shadow_only=true → Log only
   If shadow_only=false AND keep=true → Enter trade
```

---

## C# Pseudo-Code

### 1. Configuration

```csharp
// Strategy parameters
[Display(Name = "API Mode", Description = "SHADOW or LIVE")]
public string APIMode { get; set; } = "SHADOW";

[Display(Name = "API URL", Description = "Quality Model API endpoint")]
public string APIURL { get; set; } = "http://localhost:8000/predict_quality";

[Display(Name = "Model Type", Description = "seq_v1 or tabular_v1")]
public string ModelType { get; set; } = "seq_v1";

[Display(Name = "Model Mode", Description = "seq_conservative, seq_balanced, etc.")]
public string ModelMode { get; set; } = "seq_conservative";
```

### 2. Feature Preparation

```csharp
private double[][] BuildFeatureContext()
{
    // Build [60, 66] feature matrix
    double[][] features = new double[60][];
    
    for (int i = 0; i < 60; i++)
    {
        features[i] = new double[66];
        
        int barIndex = CurrentBar - 59 + i;
        
        // OHLCV (indices 0-4)
        features[i][0] = Opens[barIndex];
        features[i][1] = Highs[barIndex];
        features[i][2] = Lows[barIndex];
        features[i][3] = Closes[barIndex];
        features[i][4] = Volumes[barIndex];
        
        // Delta & Tick features (5-12)
        features[i][5] = GetDelta(barIndex);
        features[i][6] = GetBuyVolume(barIndex);
        // ... populate all 66 features
        
        // SMC features (13-40)
        features[i][13] = GetIntOBBull(barIndex);
        // ... etc
        
        // Volume Profile (41-50)
        features[i][41] = GetVAH(barIndex);
        // ... etc
        
        // Additional features up to index 65
    }
    
    return features;
}
```

### 3. API Call

```csharp
using System.Net.Http;
using System.Text;
using Newtonsoft.Json;

private async Task<PredictionResult> CallQualityAPI(double[][] features, int side)
{
    using (HttpClient client = new HttpClient())
    {
        // Build request
        var request = new
        {
            X = features,
            side = side,
            model_type = ModelType,
            mode = ModelMode,
            shadow_only = (APIMode == "SHADOW"),
            meta = new
            {
                symbol = Instrument.MasterInstrument.Name,
                timeframe = BarsPeriod.ToString(),
                event_time = Time[0].ToString("yyyy-MM-ddTHH:mm:ss"),
                bar_index = CurrentBar,
                session = GetSession()
            }
        };
        
        string json = JsonConvert.SerializeObject(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        
        // Send request
        HttpResponseMessage response = await client.PostAsync(APIURL, content);
        
        if (response.IsSuccessStatusCode)
        {
            string responseBody = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<PredictionResult>(responseBody);
        }
        else
        {
            Print($"API Error: {response.StatusCode}");
            return null;
        }
    }
}

// Response class
public class PredictionResult
{
    public double p_keep { get; set; }
    public bool keep { get; set; }
    public string model_type { get; set; }
    public string mode { get; set; }
    public double threshold { get; set; }
    public int side { get; set; }
    public bool shadow_only { get; set; }
    public string timestamp { get; set; }
}
```

### 4. Trade Execution Logic

```csharp
protected override void OnBarUpdate()
{
    // Check for P2 event + SMC signal
    if (!IsP2Event()) return;
    
    string signalSide = GetSMCSignal(); // "long" or "short"
    if (signalSide == "none") return;
    
    // Build features
    double[][] features = BuildFeatureContext();
    int side = (signalSide == "long") ? 1 : -1;
    
    // Call API
    var result = CallQualityAPI(features, side).Result;
    
    if (result == null)
    {
        Print("API call failed");
        return;
    }
    
    // Log prediction
    LogPrediction(result);
    
    // Execution decision
    if (APIMode == "SHADOW")
    {
        // SHADOW MODE: Log only, no trade
        Print($"[SHADOW] Signal: {signalSide}, p_keep: {result.p_keep:F3}, keep: {result.keep}");
        // Save to shadow log file
        SaveShadowLog(result, signalSide);
    }
    else if (APIMode == "LIVE")
    {
        // LIVE MODE: Execute if keep=true
        if (result.keep)
        {
            Print($"[LIVE] ML KEEP - Entering {signalSide}");
            
            // Calculate SL/TP from SMC rule
            double entryPrice = Close[0];
            double stopLoss = CalculateStopLoss(signalSide);
            double takeProfit = CalculateTakeProfit(signalSide);
            
            if (signalSide == "long")
            {
                EnterLong(1, "ML_Long");
                SetStopLoss("ML_Long", CalculationMode.Price, stopLoss, false);
                SetProfitTarget("ML_Long", CalculationMode.Price, takeProfit);
            }
            else
            {
                EnterShort(1, "ML_Short");
                SetStopLoss("ML_Short", CalculationMode.Price, stopLoss, false);
                SetProfitTarget("ML_Short", CalculationMode.Price, takeProfit);
            }
        }
        else
        {
            Print($"[LIVE] ML DROP - Skipping {signalSide}");
        }
    }
}
```

### 5. Shadow Logging

```csharp
private void SaveShadowLog(PredictionResult result, string signalSide)
{
    string logPath = @"C:\NinjaTrader\ShadowLog\shadow_log.jsonl";
    
    var logEntry = new
    {
        timestamp = DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ss.fff"),
        model_type = result.model_type,
        mode = result.mode,
        threshold = result.threshold,
        shadow_only = result.shadow_only,
        p_keep = result.p_keep,
        keep = result.keep,
        signal_side = signalSide,
        meta = new
        {
            symbol = Instrument.MasterInstrument.Name,
            bar_time = Time[0].ToString("yyyy-MM-ddTHH:mm:ss"),
            price = Close[0]
        }
    };
    
    string json = JsonConvert.SerializeObject(logEntry);
    File.AppendAllText(logPath, json + "\n");
}
```

---

## Deployment Workflow

### Step 1: Shadow Testing (1-2 weeks)

```csharp
// In strategy parameters
APIMode = "SHADOW"
ModelType = "seq_v1"
ModelMode = "seq_conservative"
```

**What happens**:
- API called with `shadow_only=true`
- Predictions logged
- NO real trades executed
- Collect data for validation

**Monitor**:
- Check shadow log daily
- After 1-2 weeks, analyze results
- Compare to backtest expectations

### Step 2: Go-Live Decision

Run analysis:
```bash
python -m phase7_shadow.analyze_shadow_vs_baseline --shadow-log C:\NinjaTrader\ShadowLog\shadow_log.jsonl
```

**Criteria to proceed**:
- [ ] Expectancy ≥ 80% of backtest (+0.95R minimum)
- [ ] Max DD acceptable (≤ 10R target)
- [ ] Stable across sessions
- [ ] No API errors

### Step 3: Micro-Lot Live

```csharp
// Change to live mode
APIMode = "LIVE"

// Use 1 micro-lot
DefaultQuantity = 1
```

**Monitor closely**:
- First week: Manual verification of each trade
- Compare live vs shadow predictions
- Track actual vs expected performance

### Step 4: Scale Up

Once micro-lot proves stable:
- Increase position size gradually
- Consider risk management rules
- Maintain shadow logging for analysis

---

## Error Handling

```csharp
private PredictionResult CallQualityAPIWithRetry(double[][] features, int side, int maxRetries = 3)
{
    for (int attempt = 1; attempt <= maxRetries; attempt++)
    {
        try
        {
            var result = CallQualityAPI(features, side).Result;
            if (result != null) return result;
        }
        catch (Exception ex)
        {
            Print($"API call attempt {attempt} failed: {ex.Message}");
            if (attempt < maxRetries)
            {
                Thread.Sleep(1000); // Wait 1s before retry
            }
        }
    }
    
    Print("API call failed after all retries - SKIPPING TRADE");
    return null;
}
```

---

## Configuration Summary

| Mode | shadow_only | Trades | Purpose |
|------|-------------|--------|---------|
| SHADOW | true | NO | Validation, data collection |
| LIVE | false | YES | Production trading |

**Recommended Timeline**:
1. Week 0: Start shadow period
2. Week 1-2: Collect shadow data
3. Week 3: Analyze & make go-live decision
4. Week 3+: Micro-lot if approved

---

## Support

**API Health Check**: `GET http://localhost:8000/health`  
**Model Info**: `GET http://localhost:8000/models`  
**Swagger Docs**: `http://localhost:8000/docs`

**Issues**:
- Check API server is running
- Verify features match Phase 2 exactly
- Review shadow logs for patterns
- Test with history replay first

---

**Last Updated**: 2025-11-28  
**Primary Model**: QUALITY_SEQ_V1  
**Recommended Mode**: seq_conservative (t=0.8)
