# 🧪 PHASE 1 TESTING GUIDE - NinjaTrader Data Export

**Date**: 2025-01-26
**Status**: ✅ Ready for Testing
**Phase**: Phase 1 - NinjaTrader Adapter

---

## 📋 OVERVIEW

Phase 1 tests the data export from NinjaTrader to our Python Feature Engine. We've created a simplified test server that validates the data format without requiring the full ML model.

**What we're testing:**
- ✅ NinjaTrader C# strategy compiles correctly
- ✅ HTTP POST from NinjaTrader to Python server works
- ✅ JSON data format is correct
- ✅ OHLCV data is valid and complete
- ✅ Timestamps are in correct format
- ✅ Bars are in chronological order

---

## 🎯 TESTING STEPS

### Step 1: Prepare Test Environment

#### 1.1 Start the Test Server

Open PowerShell/Command Prompt in the project folder:

```bash
cd c:\Users\Administrator\Desktop\modeloutcome

# Activate virtual environment
venv\Scripts\activate

# Start the Phase 1 test server
python tests\test_phase1_ninjatrader.py
```

**Expected output:**
```
================================================================================
🚀 PHASE 1 TEST SERVER - NinjaTrader Data Validation
================================================================================

This server will:
  1. Listen on http://localhost:5001/raw
  2. Receive data from NinjaTrader ExportRawData strategy
  3. Validate the data structure and format
  4. Print detailed validation results
...
INFO:     Uvicorn running on http://0.0.0.0:5001 (Press CTRL+C to quit)
```

✅ **Leave this terminal open!** The server must keep running while you test NinjaTrader.

---

### Step 2: Setup NinjaTrader Strategy

#### 2.1 Locate the Strategy File

Navigate to:
```
c:\Users\Administrator\Desktop\modeloutcome\src\layer1_ninjatrader\ExportRawData.cs
```

#### 2.2 Install in NinjaTrader

**Option A: Manual Copy** (Recommended for first time)

1. Open NinjaTrader 8
2. Click **Tools → Edit NinjaScript → Strategy**
3. In the NinjaScript Editor, click **File → Open Folder**
4. This opens: `C:\Users\Administrator\Documents\NinjaTrader 8\bin\Custom\Strategies\`
5. Copy `ExportRawData.cs` to this folder
6. In NinjaScript Editor, click **File → Refresh**
7. You should see `ExportRawData` appear in the list

**Option B: Import via UI**

1. In NinjaTrader, click **Tools → Import → NinjaScript Add-On**
2. Browse to the `.cs` file and import
3. Click **Compile** when prompted

#### 2.3 Compile the Strategy

1. In NinjaScript Editor, click **Compile** (or press F5)
2. Check the **Output** window for errors

**Expected output:**
```
Compile successful: 0 errors, 0 warnings
```

**If you see compilation errors:**
- ❌ **"Newtonsoft.Json not found"**: Install via NuGet (see Troubleshooting below)
- ❌ **Syntax errors**: Double-check the `.cs` file wasn't corrupted during copy

---

### Step 3: Attach Strategy to Chart

#### 3.1 Open a Chart

1. In NinjaTrader, open a **Control Center**
2. Click **New → Chart**
3. Configure chart:
   - **Instrument**: ES 03-25 (or any futures contract)
   - **Type**: Minute
   - **Value**: 1 (1-minute bars)
   - **Data Series**: Last 100 days or "Days to Load: 5"
4. Click **OK**

#### 3.2 Attach the Strategy

1. Right-click on the chart → **Strategies**
2. In the **Strategies** tab, select `ExportRawData` from the dropdown
3. Click **New**
4. Configure parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Endpoint URL** | `http://localhost:5001/raw` | Test server address |
| **Bars To Export** | 100 | Number of bars to send |
| **Export Interval Bars** | 1 | Send every bar |
| **Include Delta** | ☐ Unchecked | Not implemented yet |
| **Include L2 Depth** | ☐ Unchecked | Not implemented yet |

5. Click **OK**
6. The strategy should now be **enabled** (green checkbox)

#### 3.3 Verify Strategy is Running

Look for:
- ✅ Green checkmark next to strategy name
- ✅ No error messages in NinjaTrader Output window
- ✅ Strategy appears in "Enabled" state

---

### Step 4: Monitor Test Results

#### 4.1 Watch the Test Server Console

Go back to the PowerShell window running the test server.

**Within 1-2 minutes**, you should see output like this:

```
================================================================================
✅ RECEIVED DATA FROM NINJATRADER
================================================================================
Symbol: ES 03-25
Timeframe: 1Minute
Timestamp: 2025-01-26T14:32:15.123
Number of bars: 100

First Bar:
  Timestamp: 2025-01-26T13:52:00
  OHLCV: O=17502.75, H=17508.00, L=17501.50, C=17506.25, V=1380

Last Bar:
  Timestamp: 2025-01-26T14:31:00
  OHLCV: O=17510.50, H=17512.25, L=17509.75, C=17511.00, V=1520

📊 VALIDATION CHECKS:
  ✅ All timestamps are valid ISO format
  ✅ All bars have valid OHLCV data (all positive)
  ✅ All bars have High >= Low
  ✅ All bars have Close within [Low, High]
  ✅ Bars are in chronological order

📈 VALIDATION SUMMARY: 5/5 checks passed
🎉 ALL CHECKS PASSED! Data format is correct.
================================================================================
```

#### 4.2 Interpret Results

**✅ SUCCESS**: All 5 checks passed
- The data format is correct!
- NinjaTrader → Python communication works!
- Ready to proceed to Phase 2

**⚠️ PARTIAL**: 4/5 checks passed
- Minor issue detected (e.g., timestamp formatting)
- Review the specific failed check
- May need minor fix in ExportRawData.cs

**❌ FAILURE**: Less than 4 checks passed
- Major data format issue
- See Troubleshooting section below
- Do NOT proceed to Phase 2 until fixed

---

## ✅ SUCCESS CRITERIA

Phase 1 is **COMPLETE** when:
- [x] Test server receives data from NinjaTrader
- [x] All 5 validation checks pass
- [x] Data is received consistently (at least 3 successful exports in a row)
- [x] No errors in NinjaTrader Output window
- [x] No exceptions in Python test server console

---

## 🔧 TROUBLESHOOTING

### Issue 1: "HTTP POST failed with status 404"

**Symptom**: NinjaTrader Output window shows:
```
ExportRawData: HTTP POST failed with status 404
```

**Solution**:
- ❌ Test server is not running → Go back to Step 1.1
- ❌ Wrong endpoint URL → Check strategy parameters (should be `http://localhost:5001/raw`)
- ❌ Firewall blocking → Temporarily disable Windows Firewall for testing

---

### Issue 2: "Newtonsoft.Json not found" Compilation Error

**Symptom**: NinjaScript Editor shows:
```
The type or namespace name 'Newtonsoft' could not be found
```

**Solution**:
1. In NinjaScript Editor, click **Tools → References**
2. Click **Add** → Browse to:
   ```
   C:\Program Files\NinjaTrader 8\bin\Newtonsoft.Json.dll
   ```
   (Or search for `Newtonsoft.Json.dll` in NinjaTrader installation folder)
3. Click **OK** and recompile

**Alternative**: Use System.Text.Json instead
- Replace `using Newtonsoft.Json;` with `using System.Text.Json;`
- Replace `JsonConvert.SerializeObject(...)` with `JsonSerializer.Serialize(...)`

---

### Issue 3: No Data Received (Timeout)

**Symptom**: Test server shows no output after 2+ minutes

**Possible causes**:
1. **Strategy not attached to chart**
   - Check chart → Right-click → Strategies → Verify `ExportRawData` is enabled

2. **Not enough bars**
   - Strategy waits for 100 bars before first export
   - Solution: Reduce `BarsToExport` to 20-30 in strategy parameters

3. **Export interval too large**
   - Check `ExportIntervalBars` parameter (should be 1 for testing)

4. **Strategy crashed**
   - Check NinjaTrader Output window for errors
   - Look for red error messages

---

### Issue 4: Invalid Timestamp Format

**Symptom**: Validation check fails:
```
❌ Invalid timestamp format: ...
```

**Solution**:
- Check line 172 in `ExportRawData.cs`:
  ```csharp
  { "ts", Time[barsAgo].ToString("yyyy-MM-ddTHH:mm:ss") }
  ```
- Ensure the format string is exactly as shown (capital Y, M, D, H, m, s)

---

### Issue 5: High < Low Error

**Symptom**: Validation check fails:
```
❌ Some bars have High < Low (invalid)
```

**Solution**:
- This indicates a bug in data collection logic
- Check lines 173-175 in `ExportRawData.cs`:
  ```csharp
  { "high", High[barsAgo] },
  { "low", Low[barsAgo] },
  ```
- Ensure `High[barsAgo]` and `Low[barsAgo]` are not swapped

---

### Issue 6: Python Server Crashes

**Symptom**: Test server window shows exception and exits

**Solution**:
1. Check Python error message (last few lines before crash)
2. Common issues:
   - **Missing dependencies**: Run `pip install -r requirements.txt`
   - **Port 5001 already in use**: Kill the other process or change port
   - **Import errors**: Ensure virtual environment is activated

**Check port 5001**:
```bash
netstat -ano | findstr :5001
```
If port is in use, kill the process:
```bash
taskkill /PID <PID> /F
```

---

## 📊 VIEWING TEST RESULTS

### Option 1: Console Output
- Watch the test server console in real-time
- Each request is logged with full validation details

### Option 2: HTTP API
Open browser or use `curl`:
```bash
# Get summary of all received requests
curl http://localhost:5001/received
```

Response:
```json
{
  "total_requests": 5,
  "requests": [
    {
      "timestamp": "2025-01-26T14:32:15.123",
      "symbol": "ES 03-25",
      "num_bars": 100,
      "checks_passed": 5,
      "checks_total": 5
    },
    ...
  ]
}
```

### Option 3: Health Check
```bash
curl http://localhost:5001/health
```

Response:
```json
{
  "status": "ok",
  "received_requests": 5
}
```

---

## 🎯 WHAT'S NEXT?

### After Phase 1 is Complete:

1. **Stop the test server** (Ctrl+C in the console)

2. **Update ROADMAP.md**:
   - Mark Phase 1 as ✅ DONE
   - Add sign-off with date and notes

3. **Proceed to Phase 2**: Feature Engine
   - Build SMC, Volume Profile, L2 features
   - Test feature extraction pipeline
   - Validate feature outputs

### Optional: Phase 1B (Add Orderflow)

If you want to implement delta volume:
1. Read [docs/NINJATRADER_DATA_SPEC.md](NINJATRADER_DATA_SPEC.md) → Phase 1B section
2. Implement `OnMarketData()` handler in `ExportRawData.cs`
3. Enable `IncludeDelta` parameter in strategy
4. Retest with test server

---

## 📝 TESTING CHECKLIST

Use this checklist to track your progress:

### Pre-Testing
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Test server started (`python tests\test_phase1_ninjatrader.py`)
- [ ] Test server shows "running on http://0.0.0.0:5001"

### NinjaTrader Setup
- [ ] `ExportRawData.cs` copied to NinjaScript folder
- [ ] Strategy compiled successfully (no errors)
- [ ] Chart opened with 1-minute bars
- [ ] Strategy attached to chart
- [ ] Parameters configured (endpoint, bars to export, etc.)
- [ ] Strategy enabled (green checkmark)

### Testing
- [ ] Data received by test server within 2 minutes
- [ ] All 5 validation checks passed
- [ ] At least 3 successful exports in a row
- [ ] No errors in NinjaTrader Output window
- [ ] No exceptions in Python console

### Validation
- [ ] Timestamps are valid ISO format
- [ ] OHLCV data is positive and non-zero
- [ ] High >= Low for all bars
- [ ] Close within [Low, High] for all bars
- [ ] Bars in chronological order

### Documentation
- [ ] Recorded test results (date, time, checks passed)
- [ ] Updated ROADMAP.md with Phase 1 completion
- [ ] Noted any issues or workarounds used

---

## 📈 PERFORMANCE METRICS

Track these metrics during testing:

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| **HTTP Response Time** | <100ms | ___ ms | ___ |
| **Data Export Frequency** | Every 1 min | ___ | ___ |
| **Bars per Request** | 100 | ___ | ___ |
| **Validation Pass Rate** | 100% (5/5) | ___/5 | ___ |
| **Consecutive Successes** | ≥3 | ___ | ___ |

---

## 🐛 KNOWN ISSUES

### Issue: "Fire-and-forget" async may not wait for completion
**Impact**: Minor - may miss some POST responses
**Workaround**: None needed for Phase 1
**Fix**: In production, use proper async/await pattern

### Issue: Mock delta values (0.5 * volume)
**Impact**: Delta/buy_volume/sell_volume are not real
**Workaround**: This is expected for Phase 1A
**Fix**: Implement Phase 1B (OnMarketData) for real delta

---

## 📞 NEED HELP?

### Check These First:
1. [ROADMAP.md](../ROADMAP.md) → Phase 1 section
2. [docs/notes.md](notes.md) → Known issues
3. This document → Troubleshooting section

### Common Commands:

**Restart test server:**
```bash
# Ctrl+C to stop
python tests\test_phase1_ninjatrader.py
```

**Check if port 5001 is in use:**
```bash
netstat -ano | findstr :5001
```

**Kill process on port 5001:**
```bash
# Get PID from netstat output, then:
taskkill /PID <PID> /F
```

**Test server manually:**
```bash
curl http://localhost:5001/health
```

---

## ✅ SIGN-OFF

**Phase 1 Testing Complete**

- **Date**: ___________
- **Tester**: ___________
- **Validation Checks Passed**: ___/5
- **Issues Encountered**: ___________
- **Ready for Phase 2**: ☐ Yes  ☐ No (reason: _________)

**Notes:**
___________________________________________________________________________
___________________________________________________________________________

---

**Version**: 1.0
**Last Updated**: 2025-01-26
**Status**: ✅ Ready for Use

**Good luck testing! 🚀**
