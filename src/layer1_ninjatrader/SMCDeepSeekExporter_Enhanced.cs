// SMC_DeepSeek_Exporter_Enhanced v2.0.0
// COMPREHENSIVE VERSION với:
// - Full technical indicators (RSI, MACD, EMA, ATR, ADX, Stochastic, BB)
// - Multi-Timeframe structure (M5, M15)
// - Advanced volume analysis
// - Market regime detection
// - Price action patterns
// - Session context
// - Round number proximity

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Globalization;
using System.IO;
using System.Text;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class SMC_DeepSeek_Exporter_Enhanced : Indicator
    {
        private SMC_Structure_OB_Only_v12_FVG_CHOCHFlags smc;
        // NOTE: No longer need separate smcM5/smcM15 instances!
        // M5/M15 data now comes from main SMC indicator via public properties

        // Volume Delta indicator (Volumdelta.cs)
        private Volumdelta _volumdelta;
        private bool _vdfInitialized = false;
        private ATR _atr14;
        private ADX _adx14;
        private EMA _m5Ema20;
        private EMA _m5Ema50;
        private EMA _m15Ema20;
        private EMA _m15Ema50;

        // Index of added data series
        private int _m5Index = -1;
        private int _m15Index = -1;

        // Simple DI+ / DI- buffers (manual calc to avoid missing indicator definitions)
        private const int DiPeriod = 14;
        private List<double> _trBuffer = new List<double>();
        private List<double> _dmpBuffer = new List<double>();
        private List<double> _dmmBuffer = new List<double>();
        private double _diPlusValue = 0.0;
        private double _diMinusValue = 0.0;

        private string exportFolder;
        private string currentDate;
        private string currentFilePath;
        private int _lastFvgBarIndex = -1;
        private int debugVolStatsPrints = 0;

        #region Parameters

        [NinjaScriptProperty]
        [Display(Name = "OnlySignalBars", Order = 1, GroupName = "Exporter")]
        public bool OnlySignalBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "UseVolumeDelta", Order = 3, GroupName = "Exporter")]
        public bool UseVolumeDelta { get; set; }

        [NinjaScriptProperty]
        [Range(0, int.MaxValue)]
        [Display(Name = "MinDeltaSize", Order = 4, GroupName = "Exporter")]
        public int MinDeltaSize { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "LogRetestSignals", Order = 5, GroupName = "Exporter")]
        public bool LogRetestSignals { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EnableM5Structure", Order = 6, GroupName = "Exporter")]
        public bool EnableM5Structure { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EnableM15Structure", Order = 7, GroupName = "Exporter")]
        public bool EnableM15Structure { get; set; }

        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "SMC_DeepSeek_Exporter_Enhanced";
                Description = "Enhanced exporter với full technical indicators & MTF";

                Calculate = Calculate.OnBarClose;
                IsOverlay = false;
                DisplayInDataBox = false;
                DrawOnPricePanel = false;
                IsSuspendedWhileInactive = true;
                MaximumBarsLookBack = MaximumBarsLookBack.Infinite;

                OnlySignalBars = false;
                UseVolumeDelta = true;
                MinDeltaSize = 0;
                LogRetestSignals = true;
                EnableM5Structure = true;
                EnableM15Structure = true;
            }
            else if (State == State.Configure)
            {
                // IMPORTANT: Add Tick series FIRST if using volume delta
                // This ensures VolumeDeltaFeed_v2 can find it at the expected index
                if (UseVolumeDelta)
                {
                    try { AddDataSeries(BarsPeriodType.Tick, 1); }
                    catch { }
                }

                // Add M5 data series if enabled
                if (EnableM5Structure)
                {
                    AddDataSeries(BarsPeriodType.Minute, 5);
                }

                // Add M15 data series if enabled
                if (EnableM15Structure)
                {
                    AddDataSeries(BarsPeriodType.Minute, 15);
                }
            }
            else if (State == State.DataLoaded)
            {
                _lastFvgBarIndex = -1;
                // Compute indices of added series (order: primary M1 = 0, optional tick, then M5, then M15)
                _m5Index = -1;
                _m15Index = -1;
                int idx = 1; // primary = 0
                if (UseVolumeDelta)
                    idx++; // tick series at 1
                if (EnableM5Structure)
                {
                    _m5Index = idx;
                    idx++;
                }
                if (EnableM15Structure)
                {
                    _m15Index = idx;
                    idx++;
                }

                // Primary M1 SMC Structure with M5/M15 multi-timeframe support
                // NOTE: The SMC indicator now handles M5/M15 internally!
                try
                {
                    smc = SMC_Structure_OB_Only_v12_FVG_CHOCHFlags(
                        1, 2, true, false, 50, true, false, true, 20,
                        5, true, false, 40, true, 40, true,
                        true, 1, true, 20, 1.0, 0, false,
                        30, 1, true, true, 48, 2, true,
                        true, true, 2.0, 0, 160, 2,
                        System.Windows.Media.Colors.Lime, System.Windows.Media.Colors.Red, 1,
                        false, System.Windows.Media.Colors.Red,
                        System.Windows.Media.Colors.Gray, System.Windows.Media.Colors.Green, 80,
                        EnableM5Structure,    // Enable M5 processing
                        EnableM15Structure);  // Enable M15 processing

                    if (smc != null)
                    {
                        Print("[Enhanced Exporter] SMC indicator loaded successfully");
                        if (EnableM5Structure)
                            Print("[Enhanced Exporter] M5 structure ENABLED in SMC indicator");
                        if (EnableM15Structure)
                            Print("[Enhanced Exporter] M15 structure ENABLED in SMC indicator");
                    }
                }
                catch (Exception ex)
                {
                    smc = null;
                    Print(string.Format("[Enhanced Exporter] ERROR loading SMC: {0}", ex.Message));
                }

                // Minimal ATR for export fields
                try { _atr14 = ATR(BarsArray[0], 14); } catch { _atr14 = null; }
                try { _adx14 = ADX(BarsArray[0], 14); } catch { _adx14 = null; }

                // HTF EMAs if series available
                try
                {
                    if (_m5Index >= 0)
                    {
                        _m5Ema20 = EMA(BarsArray[_m5Index], 20);
                        _m5Ema50 = EMA(BarsArray[_m5Index], 50);
                    }
                    if (_m15Index >= 0)
                    {
                        _m15Ema20 = EMA(BarsArray[_m15Index], 20);
                        _m15Ema50 = EMA(BarsArray[_m15Index], 50);
                    }
                }
                catch { _m5Ema20 = _m5Ema50 = _m15Ema20 = _m15Ema50 = null; }

                // NOTE: Volumdelta will be initialized later in OnBarUpdate
                // when all data series are ready
                _volumdelta = null;
                _vdfInitialized = false;

                try
                {
                    string doc = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
                    exportFolder = Path.Combine(doc, "NinjaTrader 8", "smc_exports_enhanced");
                    if (!Directory.Exists(exportFolder))
                        Directory.CreateDirectory(exportFolder);
                }
                catch { exportFolder = null; }

                currentDate = string.Empty;
                currentFilePath = string.Empty;
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0) return;

            // Initialize Volumdelta on first bar update when all data series are ready
            if (!_vdfInitialized && UseVolumeDelta && CurrentBar >= 0)
            {
                try
                {
                    int barsArrayLen = BarsArray != null ? BarsArray.Length : 0;
                    Print(string.Format("[Enhanced Exporter] Initializing Volumdelta: BarsArray.Length={0}, CurrentBar={1}", barsArrayLen, CurrentBar));

                    _volumdelta = Volumdelta(System.Windows.Media.Brushes.Red, System.Windows.Media.Brushes.LimeGreen, System.Windows.Media.Brushes.Black, 1, false, MinDeltaSize, false);
                    if (_volumdelta != null)
                        Print("[Enhanced Exporter] Volumdelta loaded successfully");
                    else
                        Print("[Enhanced Exporter] WARNING: Volumdelta is null");
                }
                catch (Exception ex)
                {
                    _volumdelta = null;
                    Print(string.Format("[Enhanced Exporter] ERROR loading Volumdelta: {0}", ex.Message));
                }
                _vdfInitialized = true;
            }

            if (CurrentBar > 0)
                UpdateDirectionalMovement();

            if (smc == null) return;
            if (exportFolder == null || exportFolder == string.Empty) return;
            if (CurrentBar < 5) return;

            string barDate = Time[0].ToString("yyyyMMdd");
            if (barDate != currentDate)
            {
                currentDate = barDate;
                currentFilePath = BuildFilePath(barDate);
            }

            bool obLong = GetSeriesBool(smc.ObExtRetestBull, 0);
            bool obShort = GetSeriesBool(smc.ObExtRetestBear, 0);
            // FVG retest removed from C#
            bool fvgLong = false;
            bool fvgShort = false;

            if (LogRetestSignals && (obLong || obShort || fvgLong || fvgShort))
            {
                var tags = new List<string>();
                if (obLong) tags.Add("OB_EXT_RETEST_BULL");
                if (obShort) tags.Add("OB_EXT_RETEST_BEAR");
                if (fvgLong) tags.Add("FVG_RETEST_BULL");
                if (fvgShort) tags.Add("FVG_RETEST_BEAR");
                Print(string.Format("[Enhanced Exporter] {0:yyyy-MM-dd HH:mm:ss} signal(s): {1} @ Close={2}",
                    Time[0], string.Join(", ", tags), Close[0]));
            }

            // Print M5 BOS/CHOCH events
            if (EnableM5Structure && smc != null)
            {
                bool m5BosUp = GetSeriesBool(smc.M5_BosUpPulseSeries, 0);
                bool m5BosDown = GetSeriesBool(smc.M5_BosDownPulseSeries, 0);
                bool m5ChochUp = GetSeriesBool(smc.M5_ChochUpPulseSeries, 0);
                bool m5ChochDown = GetSeriesBool(smc.M5_ChochDownPulseSeries, 0);

                if (m5BosUp || m5BosDown || m5ChochUp || m5ChochDown)
                {
                    var m5Events = new List<string>();
                    if (m5BosUp) m5Events.Add("BOS_UP");
                    if (m5BosDown) m5Events.Add("BOS_DOWN");
                    if (m5ChochUp) m5Events.Add("CHOCH_UP");
                    if (m5ChochDown) m5Events.Add("CHOCH_DOWN");
                    Print(string.Format("[M5 Structure] {0:yyyy-MM-dd HH:mm:ss} {1} @ Close={2}",
                        Time[0], string.Join(", ", m5Events), Close[0]));
                }
            }

            // Print M15 BOS/CHOCH events
            if (EnableM15Structure && smc != null)
            {
                bool m15BosUp = smc.M15_BosUpPulse;
                bool m15BosDown = smc.M15_BosDownPulse;
                bool m15ChochUp = smc.M15_ChochUpPulse;
                bool m15ChochDown = smc.M15_ChochDownPulse;

                if (m15BosUp || m15BosDown || m15ChochUp || m15ChochDown)
                {
                    var m15Events = new List<string>();
                    if (m15BosUp) m15Events.Add("BOS_UP");
                    if (m15BosDown) m15Events.Add("BOS_DOWN");
                    if (m15ChochUp) m15Events.Add("CHOCH_UP");
                    if (m15ChochDown) m15Events.Add("CHOCH_DOWN");
                    Print(string.Format("[M15 Structure] {0:yyyy-MM-dd HH:mm:ss} {1} @ Close={2}",
                        Time[0], string.Join(", ", m15Events), Close[0]));
                }
            }

            // REMOVED: Signal decision logic (OB/FVG retest priority)
            // The exporter now outputs RAW data only. Downstream Python modules will determine signals.

            string jsonLine = BuildJsonLine(obLong, obShort, fvgLong, fvgShort);
            AppendLineSafe(currentFilePath, jsonLine);
        }

        #region Volume Delta Helpers

        private void AppendVolumeStats(StringBuilder sb, int barsAgo)
        {
            sb.Append("{");

            if (_volumdelta != null && UseVolumeDelta)
            {
                try
                {
                    double deltaClose = _volumdelta.DeltasClose[barsAgo];
                    double cumDelta = _volumdelta.DeltasClosecum[barsAgo];

                    // Calculate derived metrics from delta
                    // Note: Volumdelta doesn't expose buy/sell volumes separately
                    // We estimate: if deltaClose > 0, more buying; if < 0, more selling
                    double totalVol = Volumes[0][barsAgo]; // Use bar volume from primary series
                    if (totalVol < 0) totalVol = 0.0;
                    double clampedDelta = deltaClose;
                    if (totalVol > 0)
                        clampedDelta = Math.Max(-totalVol, Math.Min(deltaClose, totalVol));

                    // Estimate buy/sell split from delta (clamped)
                    double buyVol = (totalVol + clampedDelta) / 2.0;
                    double sellVol = totalVol - buyVol;
                    if (buyVol < 0) buyVol = 0.0;
                    if (sellVol < 0) sellVol = 0.0;

                    double deltaRatio = totalVol > 0 ? deltaClose / totalVol : 0.0;
                    double buyPct = totalVol > 0 ? (buyVol / totalVol) * 100.0 : 0.0;
                    double sellPct = totalVol > 0 ? (sellVol / totalVol) * 100.0 : 0.0;

                    // Debug first few calls
                    if (debugVolStatsPrints < 5)
                    {
                        Print(string.Format("[Volumdelta Data] barsAgo={0} totalVol={1} deltaClose={2} ratio={3} buyPct={4} sellPct={5} cumDelta={6}",
                            barsAgo, totalVol, deltaClose, deltaRatio, buyPct, sellPct, cumDelta));
                        debugVolStatsPrints++;
                    }

                    AppendProp(sb, "total_volume", totalVol, false, false); sb.Append(",");
                    AppendProp(sb, "delta_close", deltaClose, false, false); sb.Append(",");
                    AppendProp(sb, "delta_ratio", FormatRatio(deltaRatio), false, false); sb.Append(",");
                    AppendProp(sb, "buy_percent", FormatPercent(buyPct), false, false); sb.Append(",");
                    AppendProp(sb, "sell_percent", FormatPercent(sellPct), false, false); sb.Append(",");
                    AppendProp(sb, "cum_delta", cumDelta, false, false);
                }
                catch (Exception ex)
                {
                    if (debugVolStatsPrints < 3)
                    {
                        Print(string.Format("[Volumdelta Data ERROR] {0}", ex.Message));
                        debugVolStatsPrints++;
                    }
                    AppendProp(sb, "total_volume", 0, false, false); sb.Append(",");
                    AppendProp(sb, "delta_close", 0, false, false); sb.Append(",");
                    AppendProp(sb, "delta_ratio", 0, false, false); sb.Append(",");
                    AppendProp(sb, "buy_percent", 0, false, false); sb.Append(",");
                    AppendProp(sb, "sell_percent", 0, false, false); sb.Append(",");
                    AppendProp(sb, "cum_delta", 0, false, false);
                }
            }
            else
            {
                if (debugVolStatsPrints < 3)
                {
                    Print(string.Format("[Volumdelta Data] _volumdelta is null or UseVolumeDelta=false (_volumdelta={0}, UseVolumeDelta={1})", _volumdelta != null, UseVolumeDelta));
                    debugVolStatsPrints++;
                }
                AppendProp(sb, "total_volume", 0, false, false); sb.Append(",");
                AppendProp(sb, "delta_close", 0, false, false); sb.Append(",");
                AppendProp(sb, "delta_ratio", 0, false, false); sb.Append(",");
                AppendProp(sb, "buy_percent", 0, false, false); sb.Append(",");
                AppendProp(sb, "sell_percent", 0, false, false); sb.Append(",");
                AppendProp(sb, "cum_delta", 0, false, false);
            }
            sb.Append("}");
        }

        private void AppendVolumeStatsForMTF(StringBuilder sb, int barsArrayIndex)
        {
            sb.Append("{");

            if (_volumdelta != null && UseVolumeDelta)
            {
                try
                {
                    // Determine how many M1 bars to aggregate
                    int m1BarsToAggregate = barsArrayIndex == 2 ? 5 : 15; // M5=5 bars, M15=15 bars

                    // Aggregate recent M1 bars
                    double totalVolume = 0;
                    double totalDelta = 0;
                    double cumDelta = 0;

                    // Sum up the last N M1 bars (0 = current, 1 = previous, etc.)
                    int barsProcessed = 0;
                    for (int i = 0; i < m1BarsToAggregate && i <= CurrentBar; i++)
                    {
                        try
                        {
                            double vol = Volumes[0][i];
                            double delta = _volumdelta.DeltasClose[i];

                            totalVolume += vol;
                            totalDelta += delta;
                            barsProcessed++;
                        }
                        catch { }
                    }

                    // Get cumulative delta from most recent M1 bar
                    try { cumDelta = _volumdelta.DeltasClosecum[0]; } catch { cumDelta = 0; }

                    // Calculate derived metrics
                    double buyVol = (totalVolume + totalDelta) / 2.0;
                    double sellVol = (totalVolume - totalDelta) / 2.0;
                    double deltaRatio = totalVolume > 0 ? totalDelta / totalVolume : 0.0;
                    double buyPct = totalVolume > 0 ? (buyVol / totalVolume) * 100.0 : 0.0;
                    double sellPct = totalVolume > 0 ? (sellVol / totalVolume) * 100.0 : 0.0;

                    AppendProp(sb, "total_volume", totalVolume, false, false); sb.Append(",");
                    AppendProp(sb, "delta_close", totalDelta, false, false); sb.Append(",");
                    AppendProp(sb, "delta_ratio", FormatRatio(deltaRatio), false, false); sb.Append(",");
                    AppendProp(sb, "buy_percent", FormatPercent(buyPct), false, false); sb.Append(",");
                    AppendProp(sb, "sell_percent", FormatPercent(sellPct), false, false); sb.Append(",");
                    AppendProp(sb, "cum_delta", cumDelta, false, false);
                }
                catch
                {
                    // Fallback to zero values
                    AppendProp(sb, "total_volume", 0, false, false); sb.Append(",");
                    AppendProp(sb, "delta_close", 0, false, false); sb.Append(",");
                    AppendProp(sb, "delta_ratio", 0, false, false); sb.Append(",");
                    AppendProp(sb, "buy_percent", 0, false, false); sb.Append(",");
                    AppendProp(sb, "sell_percent", 0, false, false); sb.Append(",");
                    AppendProp(sb, "cum_delta", 0, false, false);
                }
            }
            else
            {
                // No volume delta available
                double volume = 0;
                if (BarsArray != null && BarsArray.Length > barsArrayIndex)
                {
                    try { volume = Volumes[barsArrayIndex][0]; } catch { volume = 0; }
                }

                AppendProp(sb, "total_volume", volume, false, false); sb.Append(",");
                AppendProp(sb, "delta_close", 0, false, false); sb.Append(",");
                AppendProp(sb, "delta_ratio", 0, false, false); sb.Append(",");
                AppendProp(sb, "buy_percent", 0, false, false); sb.Append(",");
                AppendProp(sb, "sell_percent", 0, false, false); sb.Append(",");
                AppendProp(sb, "cum_delta", 0, false, false);
            }

            sb.Append("}");
        }

        private void UpdateDirectionalMovement()
        {
            if (CurrentBar == 0) return;

            double prevClose = Close[1];
            double tr = Math.Max(High[0] - Low[0], Math.Max(Math.Abs(High[0] - prevClose), Math.Abs(Low[0] - prevClose)));
            double upMove = High[0] - High[1];
            double downMove = Low[1] - Low[0];
            double dmPlus = (upMove > downMove && upMove > 0) ? upMove : 0.0;
            double dmMinus = (downMove > upMove && downMove > 0) ? downMove : 0.0;

            _trBuffer.Add(tr);
            _dmpBuffer.Add(dmPlus);
            _dmmBuffer.Add(dmMinus);
            if (_trBuffer.Count > DiPeriod) _trBuffer.RemoveAt(0);
            if (_dmpBuffer.Count > DiPeriod) _dmpBuffer.RemoveAt(0);
            if (_dmmBuffer.Count > DiPeriod) _dmmBuffer.RemoveAt(0);

            double sumTr = 0.0, sumDmp = 0.0, sumDmm = 0.0;
            for (int i = 0; i < _trBuffer.Count; i++) sumTr += _trBuffer[i];
            for (int i = 0; i < _dmpBuffer.Count; i++) sumDmp += _dmpBuffer[i];
            for (int i = 0; i < _dmmBuffer.Count; i++) sumDmm += _dmmBuffer[i];

            if (sumTr > 0)
            {
                _diPlusValue = 100.0 * sumDmp / sumTr;
                _diMinusValue = 100.0 * sumDmm / sumTr;
            }
            else
            {
                _diPlusValue = 0.0;
                _diMinusValue = 0.0;
            }
        }


        #endregion

        #region JSON Building

        private string BuildJsonLine(bool obLong, bool obShort, bool fvgLong, bool fvgShort)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("{");

            // Meta
            AppendProp(sb, "id", BuildId(), true, true); sb.Append(",");
            AppendProp(sb, "time", Time[0].ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ"), true, true); sb.Append(",");
            AppendProp(sb, "symbol", Instrument != null ? Instrument.FullName : "UNKNOWN", true, true); sb.Append(",");
            AppendProp(sb, "tf", GetTimeframeLabel(), true, true); sb.Append(",");
            AppendProp(sb, "bar_index", CurrentBar, false, false); sb.Append(",");
            AppendProp(sb, "session", GetSessionTag(), true, false); sb.Append(",");
            // Removed "signal" field - raw data only
            AppendProp(sb, "signal", "none", true, false); sb.Append(",");

            // Base bar fields (top-level)
            AppendProp(sb, "timestamp", Time[0].ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ"), true, false); sb.Append(",");
            AppendProp(sb, "open", Open[0], false, false); sb.Append(",");
            AppendProp(sb, "high", High[0], false, false); sb.Append(",");
            AppendProp(sb, "low", Low[0], false, false); sb.Append(",");
            AppendProp(sb, "close", Close[0], false, false); sb.Append(",");

            // Volume & delta
            double totalVol = 0.0;
            try { totalVol = Volumes[0][0]; } catch { totalVol = 0.0; }
            double deltaClose = 0.0;
            double cumDelta = 0.0;
            if (UseVolumeDelta && _volumdelta != null)
            {
                try { deltaClose = _volumdelta.DeltasClose[0]; } catch { deltaClose = 0.0; }
                try { cumDelta = _volumdelta.DeltasClosecum[0]; } catch { cumDelta = 0.0; }
            }
            if (totalVol < 0) totalVol = 0.0;
            double clampedDelta = deltaClose;
            if (totalVol > 0)
            {
                clampedDelta = Math.Max(-totalVol, Math.Min(deltaClose, totalVol));
            }
            double buyVol = (totalVol + clampedDelta) / 2.0;
            double sellVol = totalVol - buyVol;
            if (buyVol < 0) buyVol = 0.0;
            if (sellVol < 0) sellVol = 0.0;

            AppendProp(sb, "atr_14", GetIndicatorValue(_atr14, 0), false, false);

            double entry = Close[0];
            // Removed SL/TP calculation logic - this belongs in Python modules

            sb.Append(",");
            // signal_type is now "raw" or "none"
            AppendProp(sb, "signal_type", "raw", true, false); sb.Append(",");
            AppendProp(sb, "entry", entry, false, false); sb.Append(",");
            AppendProp(sb, "sl", 0.0, false, false); sb.Append(",");
            AppendProp(sb, "tp", 0.0, false, false);

            // FVG/OB details (top-level)
            int fvgDir = 0;
            double fvgTop = double.NaN, fvgBottom = double.NaN;
            int fvgBarIndex = -1;
            if (smc != null)
            {
                if (smc.ActiveFvgDirection != null && HasSeriesValue(smc.ActiveFvgDirection, 0))
                    fvgDir = smc.ActiveFvgDirection[0];
                if (smc.ActiveFvgTop != null && HasSeriesValue(smc.ActiveFvgTop, 0))
                    fvgTop = smc.ActiveFvgTop[0];
                if (smc.ActiveFvgBottom != null && HasSeriesValue(smc.ActiveFvgBottom, 0))
                    fvgBottom = smc.ActiveFvgBottom[0];
                if (smc.ActiveFvgBarIndex != null && HasSeriesValue(smc.ActiveFvgBarIndex, 0))
                    fvgBarIndex = smc.ActiveFvgBarIndex[0];
            }
            bool fvgDetected = fvgDir != 0 && !double.IsNaN(fvgTop) && !double.IsNaN(fvgBottom);
            // Pulse only when a NEW FVG is created; keep active flag separately
            bool fvgActive = fvgDetected;
            bool fvgNew = fvgDetected && fvgBarIndex >= 0 && fvgBarIndex != _lastFvgBarIndex;
            string fvgType = fvgDir == 1 ? "bullish" : (fvgDir == -1 ? "bearish" : "none");
            double fvgGap = (!double.IsNaN(fvgTop) && !double.IsNaN(fvgBottom)) ? Math.Abs(fvgTop - fvgBottom) : double.NaN;
            double fvgCreationHigh = High[0];
            double fvgCreationLow = Low[0];

            sb.Append(","); AppendProp(sb, "fvg_detected", fvgNew, false, false);
            sb.Append(","); AppendProp(sb, "fvg_active", fvgActive, false, false);
            sb.Append(","); AppendPropNullableString(sb, "fvg_type", fvgType, false);
            sb.Append(","); AppendProp(sb, "fvg_top", fvgTop, false, false);
            sb.Append(","); AppendProp(sb, "fvg_bottom", fvgBottom, false, false);
            sb.Append(","); AppendProp(sb, "fvg_bar_index", fvgBarIndex, false, false);
            sb.Append(","); AppendProp(sb, "fvg_gap_size", fvgGap, false, false);
            sb.Append(","); AppendProp(sb, "fvg_filled", false, false, false);
            sb.Append(","); AppendProp(sb, "fvg_fill_percentage", 0.0, false, false);
            sb.Append(","); AppendProp(sb, "fvg_creation_volume", double.IsNaN(totalVol) ? 0.0 : totalVol, false, false);
            sb.Append(","); AppendProp(sb, "fvg_creation_delta", double.IsNaN(deltaClose) ? 0.0 : deltaClose, false, false);
            sb.Append(","); AppendProp(sb, "fvg_creation_high", fvgCreationHigh, false, false);
            sb.Append(","); AppendProp(sb, "fvg_creation_low", fvgCreationLow, false, false);

            int obDir = 0;
            double obTop = double.NaN, obBottom = double.NaN;
            int obBarIndex = -1;
            if (smc != null)
            {
                if (smc.ActiveObDirection != null && HasSeriesValue(smc.ActiveObDirection, 0))
                    obDir = smc.ActiveObDirection[0];
                if (smc.ActiveObTop != null && HasSeriesValue(smc.ActiveObTop, 0))
                    obTop = smc.ActiveObTop[0];
                if (smc.ActiveObBottom != null && HasSeriesValue(smc.ActiveObBottom, 0))
                    obBottom = smc.ActiveObBottom[0];
                if (smc.ActiveObBarIndex != null && HasSeriesValue(smc.ActiveObBarIndex, 0))
                    obBarIndex = smc.ActiveObBarIndex[0];
            }
            bool obDetected = obDir != 0 && !double.IsNaN(obTop) && !double.IsNaN(obBottom);
            string obType = obDir == 1 ? "bullish" : (obDir == -1 ? "bearish" : "none");

            // Default zeros instead of nulls for downstream validators
            if (double.IsNaN(obTop)) obTop = 0.0;
            if (double.IsNaN(obBottom)) obBottom = 0.0;
            if (double.IsNaN(fvgTop)) fvgTop = 0.0;
            if (double.IsNaN(fvgBottom)) fvgBottom = 0.0;
            if (double.IsNaN(fvgGap)) fvgGap = 0.0;
            if (obBarIndex < 0) obBarIndex = 0;
            if (fvgBarIndex < 0) fvgBarIndex = 0;

            sb.Append(","); AppendProp(sb, "ob_detected", obDetected, false, false);
            sb.Append(","); AppendPropNullableString(sb, "ob_type", obType, false);
            sb.Append(","); AppendProp(sb, "ob_top", obTop, false, false);
            sb.Append(","); AppendProp(sb, "ob_bottom", obBottom, false, false);
            sb.Append(","); AppendProp(sb, "ob_bar_index", obBarIndex, false, false);
            sb.Append(","); AppendProp(sb, "nearest_ob_top", obTop, false, false);
            sb.Append(","); AppendProp(sb, "nearest_ob_bottom", obBottom, false, false);
            sb.Append(","); AppendPropNullableString(sb, "nearest_ob_type", obType, false);

            // Structure context (simplified)
            bool chochUp = smc != null && GetSeriesBool(smc.ExtChochUpPulse, 0);
            bool chochDown = smc != null && GetSeriesBool(smc.ExtChochDownPulse, 0);
            bool bosUp = smc != null && GetSeriesBool(smc.ExtBosUpPulse, 0);
            bool bosDown = smc != null && GetSeriesBool(smc.ExtBosDownPulse, 0);
            bool sweepPrevHigh = smc != null && GetSeriesBool(smc.SweepPrevHighPulse, 0);
            bool sweepPrevLow = smc != null && GetSeriesBool(smc.SweepPrevLowPulse, 0);
            // HTF structure pulses (e.g., M5/M15) if available
            bool htfBosUp = smc != null && GetSeriesBool(smc.M5_BosUpPulseSeries, 0);
            bool htfBosDown = smc != null && GetSeriesBool(smc.M5_BosDownPulseSeries, 0);
            bool htfChochUp = smc != null && GetSeriesBool(smc.M5_ChochUpPulseSeries, 0);
            bool htfChochDown = smc != null && GetSeriesBool(smc.M5_ChochDownPulseSeries, 0);

            bool chochDetected = chochUp || chochDown;
            bool bosDetected = bosUp || bosDown;
            string chochType = chochUp ? "bullish" : (chochDown ? "bearish" : "none");
            string bosType = bosUp ? "bullish" : (bosDown ? "bearish" : "none");
            string lastStructureBreak = bosDetected
                ? (bosUp ? "bos_bullish" : "bos_bearish")
                : (chochDetected ? (chochUp ? "choch_bullish" : "choch_bearish") : "none");

            sb.Append(","); AppendProp(sb, "choch_detected", chochDetected, false, false);
            sb.Append(","); AppendPropNullableString(sb, "choch_type", chochType, false);
            sb.Append(","); AppendProp(sb, "choch_bars_ago", chochDetected ? 0 : -1, false, false);
            sb.Append(","); AppendProp(sb, "bos_detected", bosDetected, false, false);
            sb.Append(","); AppendPropNullableString(sb, "bos_type", bosType, false);
            sb.Append(","); AppendProp(sb, "bos_bars_ago", bosDetected ? 0 : -1, false, false);
            sb.Append(","); AppendPropNullableString(sb, "last_structure_break", lastStructureBreak, false);
            sb.Append(","); AppendProp(sb, "sweep_prev_high", sweepPrevHigh, false, false);
            sb.Append(","); AppendProp(sb, "sweep_prev_low", sweepPrevLow, false, false);
            sb.Append(","); AppendProp(sb, "htf_bos_type", htfBosUp ? "bullish" : (htfBosDown ? "bearish" : "none"), true, true);
            sb.Append(","); AppendProp(sb, "htf_choch_type", htfChochUp ? "bullish" : (htfChochDown ? "bearish" : "none"), true, true);
            sb.Append(","); AppendProp(sb, "htf_bos_bars_ago", htfBosUp || htfBosDown ? 0 : -1, false, false);
            sb.Append(","); AppendProp(sb, "htf_choch_bars_ago", htfChochUp || htfChochDown ? 0 : -1, false, false);

            int currentTrend = smc != null && smc.ExtStructureDir != null && HasSeriesValue(smc.ExtStructureDir, 0)
                ? smc.ExtStructureDir[0]
                : 0;
            sb.Append(","); AppendProp(sb, "current_trend", currentTrend, false, false);
            double lastSwingHigh = GetSeriesDouble(smc != null ? smc.ExtLastSwingHigh : null, 0);
            double lastSwingLow = GetSeriesDouble(smc != null ? smc.ExtLastSwingLow : null, 0);
            double recentSwingHigh = GetSeriesDouble(smc != null ? smc.ExtPrevSwingHigh : null, 0);
            double recentSwingLow = GetSeriesDouble(smc != null ? smc.ExtPrevSwingLow : null, 0);
            if (double.IsNaN(lastSwingHigh)) lastSwingHigh = 0.0;
            if (double.IsNaN(lastSwingLow)) lastSwingLow = 0.0;
            if (double.IsNaN(recentSwingHigh)) recentSwingHigh = 0.0;
            if (double.IsNaN(recentSwingLow)) recentSwingLow = 0.0;

            sb.Append(","); AppendProp(sb, "last_swing_high", lastSwingHigh, false, false);
            sb.Append(","); AppendProp(sb, "last_swing_low", lastSwingLow, false, false);
            sb.Append(","); AppendProp(sb, "recent_swing_high", recentSwingHigh, false, false);
            sb.Append(","); AppendProp(sb, "recent_swing_low", recentSwingLow, false, false);

            // Market condition ADX/DI
            double adx = GetIndicatorValue(_adx14, 0);
            double diPlus = _diPlusValue;
            double diMinus = _diMinusValue;
            sb.Append(","); AppendProp(sb, "adx_14", adx, false, false);
            sb.Append(","); AppendProp(sb, "di_plus_14", diPlus, false, false);
            sb.Append(","); AppendProp(sb, "di_minus_14", diMinus, false, false);

            // Volume divergence swing flags (reuse ext swing pattern; fallback to simple local swing check)
            bool isSwingHigh = smc != null && smc.ExtSwingPattern != null && HasSeriesValue(smc.ExtSwingPattern, 0) && smc.ExtSwingPattern[0] == 1;
            bool isSwingLow = smc != null && smc.ExtSwingPattern != null && HasSeriesValue(smc.ExtSwingPattern, 0) && smc.ExtSwingPattern[0] == -1;
            if (!isSwingHigh && !isSwingLow)
            {
                int swingLookback = 3;
                isSwingHigh = IsLocalSwingHigh(swingLookback);
                isSwingLow = IsLocalSwingLow(swingLookback);
            }
            sb.Append(","); AppendProp(sb, "is_swing_high", isSwingHigh, false, false);
            sb.Append(","); AppendProp(sb, "is_swing_low", isSwingLow, false, false);

            // HTF data (prefer M15, fallback M5)
            double htfHigh = 0.0, htfLow = 0.0, htfClose = 0.0, htfEma20 = 0.0, htfEma50 = 0.0;
            if (_m15Index >= 0)
            {
                htfHigh = GetSeriesValueSafe(Highs[_m15Index], 0);
                htfLow = GetSeriesValueSafe(Lows[_m15Index], 0);
                htfClose = GetSeriesValueSafe(Closes[_m15Index], 0);
                htfEma20 = GetIndicatorValue(_m15Ema20, 0);
                htfEma50 = GetIndicatorValue(_m15Ema50, 0);
            }
            else if (_m5Index >= 0)
            {
                htfHigh = GetSeriesValueSafe(Highs[_m5Index], 0);
                htfLow = GetSeriesValueSafe(Lows[_m5Index], 0);
                htfClose = GetSeriesValueSafe(Closes[_m5Index], 0);
                htfEma20 = GetIndicatorValue(_m5Ema20, 0);
                htfEma50 = GetIndicatorValue(_m5Ema50, 0);
            }
            if (double.IsNaN(htfHigh)) htfHigh = 0.0;
            if (double.IsNaN(htfLow)) htfLow = 0.0;
            if (double.IsNaN(htfClose)) htfClose = 0.0;
            if (double.IsNaN(htfEma20)) htfEma20 = 0.0;
            if (double.IsNaN(htfEma50)) htfEma50 = 0.0;

            sb.Append(","); AppendProp(sb, "htf_high", htfHigh, false, false);
            sb.Append(","); AppendProp(sb, "htf_low", htfLow, false, false);
            sb.Append(","); AppendProp(sb, "htf_close", htfClose, false, false);
            sb.Append(","); AppendProp(sb, "htf_ema_20", htfEma20, false, false);
            sb.Append(","); AppendProp(sb, "htf_ema_50", htfEma50, false, false);
            sb.Append(","); AppendProp(sb, "htf_is_swing_high", false, false, false);
            sb.Append(","); AppendProp(sb, "htf_is_swing_low", false, false, false);

            // Liquidity: fall back to recent swings if no dedicated map
            double nearestLiquidityHigh = lastSwingHigh;
            double nearestLiquidityLow = lastSwingLow;
            string liquidityHighType = nearestLiquidityHigh != 0.0 ? "swing_high" : "none";
            string liquidityLowType = nearestLiquidityLow != 0.0 ? "swing_low" : "none";
            sb.Append(","); AppendProp(sb, "nearest_liquidity_high", nearestLiquidityHigh, false, false);
            sb.Append(","); AppendProp(sb, "nearest_liquidity_low", nearestLiquidityLow, false, false);
            sb.Append(","); AppendPropNullableString(sb, "liquidity_high_type", liquidityHighType, false);
            sb.Append(","); AppendPropNullableString(sb, "liquidity_low_type", liquidityLowType, false);

            // Reason - now exporting raw event flags
            sb.Append(",\"reason\":[");
            bool firstReason = true;
            if (obLong) AppendReason(sb, "ob_ext_retest_bull_event", ref firstReason);
            if (obShort) AppendReason(sb, "ob_ext_retest_bear_event", ref firstReason);
            if (fvgLong) AppendReason(sb, "fvg_retest_bull_event", ref firstReason);
            if (fvgShort) AppendReason(sb, "fvg_retest_bear_event", ref firstReason);
            sb.Append("]");

            // Current bar data (Python script will build context window from sequential bars)
            sb.Append(",\"bar\":");
            AppendBarContext(sb, 0);

            // MTF Context
            sb.Append(",\"mtf_context\":{");
            AppendMTFContext(sb);
            sb.Append("}");

            sb.Append("}");
            return sb.ToString();
        }

        private void AppendBarContext(StringBuilder sb, int barsAgo)
        {
            if (barsAgo < 0) barsAgo = 0;
            if (barsAgo > CurrentBar) barsAgo = CurrentBar;

            // Local volume/delta for this barsAgo
            double totalVol = 0.0;
            try { totalVol = Volumes[0][barsAgo]; } catch { totalVol = 0.0; }
            double deltaClose = 0.0;
            double cumDelta = 0.0;
            if (UseVolumeDelta && _volumdelta != null)
            {
                try { deltaClose = _volumdelta.DeltasClose[barsAgo]; } catch { deltaClose = 0.0; }
                try { cumDelta = _volumdelta.DeltasClosecum[barsAgo]; } catch { cumDelta = 0.0; }
            }

            sb.Append("{");

            // OHLC
            AppendProp(sb, "o", Open[barsAgo], false, false); sb.Append(",");
            AppendProp(sb, "h", High[barsAgo], false, false); sb.Append(",");
            AppendProp(sb, "l", Low[barsAgo], false, false); sb.Append(",");
            AppendProp(sb, "c", Close[barsAgo], false, false);

            // Structure
            int extDir = GetSeriesInt(smc.ExtStructureDir, barsAgo);
            int intDir = GetSeriesInt(smc.IntStructureDir, barsAgo);
            sb.Append(","); AppendProp(sb, "ext_dir", extDir, false, false);
            sb.Append(","); AppendProp(sb, "int_dir", intDir, false, false);

            // Pulses
            sb.Append(","); AppendProp(sb, "ext_bos_up", GetSeriesBool(smc.ExtBosUpPulse, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "ext_bos_down", GetSeriesBool(smc.ExtBosDownPulse, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "ext_choch_up", GetSeriesBool(smc.ExtChochUpPulse, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "ext_choch_down", GetSeriesBool(smc.ExtChochDownPulse, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "int_bos_up", GetSeriesBool(smc.IntBosUpPulse, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "int_bos_down", GetSeriesBool(smc.IntBosDownPulse, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "int_choch_up", GetSeriesBool(smc.IntChochUpPulse, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "int_choch_down", GetSeriesBool(smc.IntChochDownPulse, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "sweep_prev_high", GetSeriesBool(smc.SweepPrevHighPulse, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "sweep_prev_low", GetSeriesBool(smc.SweepPrevLowPulse, barsAgo), false, false);

            // OB/FVG
            sb.Append(","); AppendProp(sb, "has_ob_ext_bull", GetSeriesBool(smc.HasActiveObExtBull, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "has_ob_ext_bear", GetSeriesBool(smc.HasActiveObExtBear, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "has_fvg_bull", GetSeriesBool(smc.HasActiveFvgBull, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "has_fvg_bear", GetSeriesBool(smc.HasActiveFvgBear, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "in_premium", GetSeriesBool(smc.InPremiumZone, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "in_discount", GetSeriesBool(smc.InDiscountZone, barsAgo), false, false);

            // Swing Levels (M1)
            sb.Append(","); AppendProp(sb, "last_swing_high", GetSeriesDouble(smc.ExtLastSwingHigh, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "last_swing_low", GetSeriesDouble(smc.ExtLastSwingLow, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "prev_swing_high", GetSeriesDouble(smc.ExtPrevSwingHigh, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "prev_swing_low", GetSeriesDouble(smc.ExtPrevSwingLow, barsAgo), false, false);
            sb.Append(","); AppendProp(sb, "swing_pattern", GetSeriesInt(smc.ExtSwingPattern, barsAgo), false, false);

            // ATR (minimal)
            sb.Append(","); AppendProp(sb, "atr_14", GetIndicatorValue(_atr14, barsAgo), false, false);

            // FVG details (active/nearest)
            int fvgDir = 0;
            double fvgTop = double.NaN, fvgBottom = double.NaN;
            int fvgBarIndex = -1;
            if (smc.ActiveFvgDirection != null && HasSeriesValue(smc.ActiveFvgDirection, barsAgo))
                fvgDir = smc.ActiveFvgDirection[barsAgo];
            if (smc.ActiveFvgTop != null && HasSeriesValue(smc.ActiveFvgTop, barsAgo))
                fvgTop = smc.ActiveFvgTop[barsAgo];
            if (smc.ActiveFvgBottom != null && HasSeriesValue(smc.ActiveFvgBottom, barsAgo))
                fvgBottom = smc.ActiveFvgBottom[barsAgo];
            if (smc.ActiveFvgBarIndex != null && HasSeriesValue(smc.ActiveFvgBarIndex, barsAgo))
                fvgBarIndex = smc.ActiveFvgBarIndex[barsAgo];

            bool fvgDetected = fvgDir != 0 && !double.IsNaN(fvgTop) && !double.IsNaN(fvgBottom);
            // Only mark fvg_detected when bar index changes (creation pulse); keep active state separately
            bool fvgActive = fvgDetected;
            bool fvgNew = fvgDetected && fvgBarIndex >= 0 && fvgBarIndex != _lastFvgBarIndex;
            string fvgType = fvgDir == 1 ? "bullish" : (fvgDir == -1 ? "bearish" : null);
            double fvgGap = (!double.IsNaN(fvgTop) && !double.IsNaN(fvgBottom)) ? Math.Abs(fvgTop - fvgBottom) : double.NaN;

            sb.Append(","); AppendProp(sb, "fvg_detected", fvgNew, false, false);
            sb.Append(","); AppendProp(sb, "fvg_active", fvgActive, false, false);
            sb.Append(","); AppendPropNullableString(sb, "fvg_type", fvgType, false);
            sb.Append(","); AppendProp(sb, "fvg_top", fvgTop, false, false);
            sb.Append(","); AppendProp(sb, "fvg_bottom", fvgBottom, false, false);
            sb.Append(","); AppendProp(sb, "fvg_bar_index", fvgBarIndex, false, false);
            sb.Append(","); AppendProp(sb, "fvg_gap_size", fvgGap, false, false);
            sb.Append(","); AppendProp(sb, "fvg_filled", false, false, false); // not tracked, default false
            sb.Append(","); AppendProp(sb, "fvg_fill_percentage", 0.0, false, false);
            sb.Append(","); AppendProp(sb, "fvg_creation_volume", totalVol, false, false);
            sb.Append(","); AppendProp(sb, "fvg_creation_delta", deltaClose, false, false);

            if (fvgNew)
                _lastFvgBarIndex = fvgBarIndex;

            // OB details (nearest/active)
            int obDir = 0;
            double obTop = double.NaN, obBottom = double.NaN;
            int obBarIndex = -1;
            if (smc.ActiveObDirection != null && HasSeriesValue(smc.ActiveObDirection, barsAgo))
                obDir = smc.ActiveObDirection[barsAgo];
            if (smc.ActiveObTop != null && HasSeriesValue(smc.ActiveObTop, barsAgo))
                obTop = smc.ActiveObTop[barsAgo];
            if (smc.ActiveObBottom != null && HasSeriesValue(smc.ActiveObBottom, barsAgo))
                obBottom = smc.ActiveObBottom[barsAgo];
            if (smc.ActiveObBarIndex != null && HasSeriesValue(smc.ActiveObBarIndex, barsAgo))
                obBarIndex = smc.ActiveObBarIndex[barsAgo];

            bool obDetected = obDir != 0 && !double.IsNaN(obTop) && !double.IsNaN(obBottom);
            string obType = obDir == 1 ? "bullish" : (obDir == -1 ? "bearish" : null);

            sb.Append(","); AppendProp(sb, "ob_detected", obDetected, false, false);
            sb.Append(","); AppendPropNullableString(sb, "ob_type", obType, false);
            sb.Append(","); AppendProp(sb, "ob_top", obTop, false, false);
            sb.Append(","); AppendProp(sb, "ob_bottom", obBottom, false, false);
            sb.Append(","); AppendProp(sb, "ob_bar_index", obBarIndex, false, false);
            // Using same active OB as "nearest" for now
            sb.Append(","); AppendProp(sb, "nearest_ob_top", obTop, false, false);
            sb.Append(","); AppendProp(sb, "nearest_ob_bottom", obBottom, false, false);
            sb.Append(","); AppendPropNullableString(sb, "nearest_ob_type", obType, false);

            // Price Action Analysis
            sb.Append(",\"price_action\":");
            AppendPriceAction(sb, barsAgo);

            // Volume Delta Stats for M1
            if (UseVolumeDelta)
            {
                sb.Append(",\"volume_stats\":");
                AppendVolumeStats(sb, barsAgo);
            }

            sb.Append("}");
        }

        private void AppendPriceAction(StringBuilder sb, int barsAgo)
        {
            double o = Open[barsAgo];
            double h = High[barsAgo];
            double l = Low[barsAgo];
            double c = Close[barsAgo];

            double range = h - l;
            double body = Math.Abs(c - o);
            double bodyRatio = range > 0 ? body / range : 0;
            double upperWick = h - Math.Max(o, c);
            double lowerWick = Math.Min(o, c) - l;
            double upperWickRatio = range > 0 ? upperWick / range : 0;
            double lowerWickRatio = range > 0 ? lowerWick / range : 0;

            bool isBullish = c > o;
            bool isDoji = bodyRatio < 0.1;
            bool isHammer = lowerWickRatio > 0.6 && upperWickRatio < 0.1;
            bool isShootingStar = upperWickRatio > 0.6 && lowerWickRatio < 0.1;
            bool isEngulfing = false;

            if (barsAgo < CurrentBar)
            {
                double prevO = Open[barsAgo + 1];
                double prevC = Close[barsAgo + 1];
                bool prevBullish = prevC > prevO;
                if (isBullish && !prevBullish && o < prevC && c > prevO) isEngulfing = true;
                if (!isBullish && prevBullish && o > prevC && c < prevO) isEngulfing = true;
            }

            sb.Append("{");
            AppendProp(sb, "body_ratio", FormatRatio(bodyRatio), false, false); sb.Append(",");
            AppendProp(sb, "upper_wick_ratio", FormatRatio(upperWickRatio), false, false); sb.Append(",");
            AppendProp(sb, "lower_wick_ratio", FormatRatio(lowerWickRatio), false, false); sb.Append(",");
            AppendProp(sb, "is_bullish", isBullish, false, false); sb.Append(",");
            AppendProp(sb, "is_doji", isDoji, false, false); sb.Append(",");
            AppendProp(sb, "is_hammer", isHammer, false, false); sb.Append(",");
            AppendProp(sb, "is_shooting_star", isShootingStar, false, false); sb.Append(",");
            AppendProp(sb, "is_engulfing", isEngulfing, false, false);
            sb.Append("}");
        }

        private void AppendMTFContext(StringBuilder sb)
        {
            bool firstMTF = true;

            // M5 Structure (from main SMC indicator's public properties)
            if (EnableM5Structure && smc != null && smc.HasM5Data)
            {
                if (!firstMTF) sb.Append(",");
                firstMTF = false;
                sb.Append("\"m5\":{");

                // Structure Direction (persistent - every bar)
                AppendProp(sb, "structure_dir", GetSeriesInt(smc.M5_StructureDirSeries, 0), false, false); sb.Append(",");

                // Swing Levels (persistent - every bar)
                AppendProp(sb, "last_swing_high", smc.M5_LastSwingHigh, false, false); sb.Append(",");
                AppendProp(sb, "last_swing_low", smc.M5_LastSwingLow, false, false); sb.Append(",");
                AppendProp(sb, "bars_in_swing", smc.M5_BarsInCurrentSwing, false, false); sb.Append(",");

                // BOS/CHoCH Pulses (one-time signals)
                AppendProp(sb, "bos_up_pulse", GetSeriesBool(smc.M5_BosUpPulseSeries, 0), false, false); sb.Append(",");
                AppendProp(sb, "bos_down_pulse", GetSeriesBool(smc.M5_BosDownPulseSeries, 0), false, false); sb.Append(",");
                AppendProp(sb, "choch_up_pulse", GetSeriesBool(smc.M5_ChochUpPulseSeries, 0), false, false); sb.Append(",");
                AppendProp(sb, "choch_down_pulse", GetSeriesBool(smc.M5_ChochDownPulseSeries, 0), false, false); sb.Append(",");

                // OrderBlock States (persistent - every bar)
                AppendProp(sb, "has_active_bull_ob", smc.M5_HasActiveBullOB, false, false); sb.Append(",");
                AppendProp(sb, "has_active_bear_ob", smc.M5_HasActiveBearOB, false, false); sb.Append(",");

                // OB Retest Signals (pulses)
                AppendProp(sb, "ob_retest_bull_pulse", smc.M5_OBRetestBullPulse, false, false); sb.Append(",");
                AppendProp(sb, "ob_retest_bear_pulse", smc.M5_OBRetestBearPulse, false, false); sb.Append(",");
                AppendProp(sb, "ob_retest_bull_sl", smc.M5_OBRetestBullSL, false, false); sb.Append(",");
                AppendProp(sb, "ob_retest_bear_sl", smc.M5_OBRetestBearSL, false, false); sb.Append(",");

                // FVG States (persistent - every bar)
                AppendProp(sb, "has_active_bull_fvg", smc.M5_HasActiveBullFVG, false, false); sb.Append(",");
                AppendProp(sb, "has_active_bear_fvg", smc.M5_HasActiveBearFVG, false, false); sb.Append(",");

                // Premium/Discount Context (persistent - every bar)
                AppendProp(sb, "in_premium_zone", smc.M5_InPremiumZone, false, false); sb.Append(",");
                AppendProp(sb, "in_discount_zone", smc.M5_InDiscountZone, false, false); sb.Append(",");

                // M5 Volume Stats
                sb.Append("\"volume_stats\":");
                int m5Index = UseVolumeDelta ? 2 : 1; // M5 at index 2 if tick series exists, otherwise 1
                AppendVolumeStatsForMTF(sb, m5Index);

                sb.Append("}");
            }

            // M15 Structure (from main SMC indicator's public properties)
            if (EnableM15Structure && smc != null && smc.HasM15Data)
            {
                if (!firstMTF) sb.Append(",");
                firstMTF = false;
                sb.Append("\"m15\":{");

                // Structure Direction (persistent - every bar)
                AppendProp(sb, "structure_dir", smc.M15_StructureDir, false, false); sb.Append(",");

                // Swing Levels (persistent - every bar)
                AppendProp(sb, "last_swing_high", smc.M15_LastSwingHigh, false, false); sb.Append(",");
                AppendProp(sb, "last_swing_low", smc.M15_LastSwingLow, false, false); sb.Append(",");
                AppendProp(sb, "bars_in_swing", smc.M15_BarsInCurrentSwing, false, false); sb.Append(",");

                // BOS/CHoCH Pulses (one-time signals)
                AppendProp(sb, "bos_up_pulse", smc.M15_BosUpPulse, false, false); sb.Append(",");
                AppendProp(sb, "bos_down_pulse", smc.M15_BosDownPulse, false, false); sb.Append(",");
                AppendProp(sb, "choch_up_pulse", smc.M15_ChochUpPulse, false, false); sb.Append(",");
                AppendProp(sb, "choch_down_pulse", smc.M15_ChochDownPulse, false, false); sb.Append(",");

                // OrderBlock States (persistent - every bar)
                AppendProp(sb, "has_active_bull_ob", smc.M15_HasActiveBullOB, false, false); sb.Append(",");
                AppendProp(sb, "has_active_bear_ob", smc.M15_HasActiveBearOB, false, false); sb.Append(",");

                // OB Retest Signals (pulses)
                AppendProp(sb, "ob_retest_bull_pulse", smc.M15_OBRetestBullPulse, false, false); sb.Append(",");
                AppendProp(sb, "ob_retest_bear_pulse", smc.M15_OBRetestBearPulse, false, false); sb.Append(",");
                AppendProp(sb, "ob_retest_bull_sl", smc.M15_OBRetestBullSL, false, false); sb.Append(",");
                AppendProp(sb, "ob_retest_bear_sl", smc.M15_OBRetestBearSL, false, false); sb.Append(",");

                // FVG States (persistent - every bar)
                AppendProp(sb, "has_active_bull_fvg", smc.M15_HasActiveBullFVG, false, false); sb.Append(",");
                AppendProp(sb, "has_active_bear_fvg", smc.M15_HasActiveBearFVG, false, false); sb.Append(",");

                // Premium/Discount Context (persistent - every bar)
                AppendProp(sb, "in_premium_zone", smc.M15_InPremiumZone, false, false); sb.Append(",");
                AppendProp(sb, "in_discount_zone", smc.M15_InDiscountZone, false, false); sb.Append(",");

                // M15 Volume Stats
                // Calculate correct index based on what's enabled
                int m15Index;
                if (UseVolumeDelta && EnableM5Structure)
                    m15Index = 3; // Tick(1) + M5(2) + M15(3)
                else if (UseVolumeDelta || EnableM5Structure)
                    m15Index = 2; // One of them at index 1, M15 at 2
                else
                    m15Index = 1; // M15 is first additional series

                sb.Append("\"volume_stats\":");
                AppendVolumeStatsForMTF(sb, m15Index);

                sb.Append("}");
            }
        }

        private double GetIndicatorValue(ISeries<double> indicator, int barsAgo)
        {
            if (indicator == null) return 0.0;
            if (indicator.Count <= barsAgo) return 0.0;
            try { return indicator[barsAgo]; }
            catch { return 0.0; }
        }

        private double GetIndicatorValueMTF(Series<double> series)
        {
            if (series == null) return 0.0;
            try
            {
                if (series.Count > 0)
                    return series[0];
            }
            catch { }
            return 0.0;
        }

        private double GetMTFVolume(int barsInProgress)
        {
            try
            {
                if (BarsArray == null || BarsArray.Length <= barsInProgress) return 0.0;
                if (CurrentBars == null || CurrentBars.Length <= barsInProgress) return 0.0;
                if (CurrentBars[barsInProgress] < 0) return 0.0;
                if (Volumes[barsInProgress] == null || Volumes[barsInProgress].Count == 0) return 0.0;
                return Volumes[barsInProgress][0];
            }
            catch { return 0.0; }
        }

        private string FormatRatio(double value)
        {
            double rounded = Math.Round(value, 3, MidpointRounding.AwayFromZero);
            return rounded.ToString("0.###", CultureInfo.InvariantCulture);
        }

        private string FormatPercent(double value)
        {
            double rounded = Math.Round(value, 1, MidpointRounding.AwayFromZero);
            return rounded.ToString("0.#", CultureInfo.InvariantCulture);
        }

        // Simple local swing helpers for fallback swing detection
        private bool IsLocalSwingHigh(int lookback)
        {
            if (CurrentBar < lookback) return false;
            double currentHigh = High[0];
            for (int i = 1; i <= lookback; i++)
            {
                if (High[i] > currentHigh) return false;
            }
            return true;
        }

        private bool IsLocalSwingLow(int lookback)
        {
            if (CurrentBar < lookback) return false;
            double currentLow = Low[0];
            for (int i = 1; i <= lookback; i++)
            {
                if (Low[i] < currentLow) return false;
            }
            return true;
        }

        #endregion

        #region Helpers

        private string BuildFilePath(string yyyymmdd)
        {
            string inst = Instrument != null ? Instrument.FullName : "UNKNOWN";
            string tf = GetTimeframeLabel();
            string fileName = "deepseek_enhanced_" + inst + "_" + tf + "_" + yyyymmdd + ".jsonl";
            return Path.Combine(exportFolder, SanitizeFileName(fileName));
        }

        private string GetTimeframeLabel()
        {
            if (BarsPeriod == null) return "TF";
            switch (BarsPeriod.BarsPeriodType)
            {
                case BarsPeriodType.Minute: return "M" + BarsPeriod.Value.ToString();
                case BarsPeriodType.Second: return "S" + BarsPeriod.Value.ToString();
                case BarsPeriodType.Tick: return "T" + BarsPeriod.Value.ToString();
                case BarsPeriodType.Day: return "D" + BarsPeriod.Value.ToString();
            }
            return BarsPeriod.BarsPeriodType.ToString();
        }

        private string BuildId()
        {
            string inst = Instrument != null ? Instrument.FullName : "UNKNOWN";
            string tf = GetTimeframeLabel();
            return inst + "_" + tf + "_" + Time[0].ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ") + "_" + CurrentBar.ToString();
        }

        private void AppendProp(StringBuilder sb, string name, string value, bool quote, bool first)
        {
            sb.Append("\""); sb.Append(name); sb.Append("\":");
            if (quote) { sb.Append("\""); sb.Append(EscapeJson(value)); sb.Append("\""); }
            else sb.Append(value);
        }

        private void AppendPropNullableString(StringBuilder sb, string name, string value, bool first)
        {
            sb.Append("\""); sb.Append(name); sb.Append("\":");
            if (string.IsNullOrEmpty(value)) sb.Append("null");
            else
            {
                sb.Append("\"");
                sb.Append(EscapeJson(value));
                sb.Append("\"");
            }
        }

        private void AppendProp(StringBuilder sb, string name, double value, bool quote, bool first)
        {
            sb.Append("\""); sb.Append(name); sb.Append("\":");
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                sb.Append("null");
            }
            else
            {
                double v = Math.Round(value, 4);
                sb.Append(v.ToString(CultureInfo.InvariantCulture));
            }
        }

        private void AppendProp(StringBuilder sb, string name, int value, bool quote, bool first)
        {
            sb.Append("\""); sb.Append(name); sb.Append("\":");
            sb.Append(value);
        }

        private void AppendProp(StringBuilder sb, string name, bool value, bool quote, bool first)
        {
            sb.Append("\""); sb.Append(name); sb.Append("\":");
            sb.Append(value ? "true" : "false");
        }

        private void AppendReason(StringBuilder sb, string reason, ref bool first)
        {
            if (!first) sb.Append(",");
            first = false;
            sb.Append("\""); sb.Append(reason); sb.Append("\"");
        }

        private string EscapeJson(string s)
        {
            if (string.IsNullOrEmpty(s)) return s;
            return s.Replace("\\", "\\\\").Replace("\"", "\\\"");
        }

        private string SanitizeFileName(string name)
        {
            char[] invalid = Path.GetInvalidFileNameChars();
            for (int i = 0; i < invalid.Length; i++)
                name = name.Replace(invalid[i], '_');
            return name;
        }

        private void AppendLineSafe(string path, string line)
        {
            try
            {
                using (StreamWriter sw = new StreamWriter(path, true, new UTF8Encoding(false)))
                    sw.WriteLine(line);
            }
            catch { }
        }

        private bool HasSeriesValue<T>(Series<T> series, int barsAgo) where T : struct
        {
            if (series == null) return false;
            if (barsAgo < 0) return false;
            if (CurrentBar - barsAgo < 0) return false;
            if (series.Count <= barsAgo) return false;
            return true;
        }

        private bool GetSeriesBool(Series<bool> s, int barsAgo)
        {
            if (!HasSeriesValue(s, barsAgo)) return false;
            return s[barsAgo];
        }

        private int GetSeriesInt(Series<int> s, int barsAgo)
        {
            if (!HasSeriesValue(s, barsAgo)) return 0;
            return s[barsAgo];
        }

        private double GetSeriesDouble(Series<double> s, int barsAgo)
        {
            if (!HasSeriesValue(s, barsAgo)) return 0.0;
            return s[barsAgo];
        }

        private double GetSeriesValueSafe(ISeries<double> s, int barsAgo)
        {
            if (s == null) return 0.0;
            if (barsAgo < 0 || CurrentBar - barsAgo < 0) return 0.0;
            try
            {
                double v = s[barsAgo];
                if (double.IsNaN(v) || double.IsInfinity(v)) return 0.0;
                return v;
            }
            catch { return 0.0; }
        }

        private string GetSessionTag()
        {
            int h = Time[0].ToUniversalTime().Hour;
            if (h >= 0 && h < 7) return "Asia";
            if (h >= 7 && h < 13) return "London";
            if (h >= 13 && h < 22) return "NY";
            return "Other";
        }

        #endregion
    }
}


#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private SMC_DeepSeek_Exporter_Enhanced[] cacheSMC_DeepSeek_Exporter_Enhanced;
		public SMC_DeepSeek_Exporter_Enhanced SMC_DeepSeek_Exporter_Enhanced(bool onlySignalBars, bool useVolumeDelta, int minDeltaSize, bool logRetestSignals, bool enableM5Structure, bool enableM15Structure)
		{
			return SMC_DeepSeek_Exporter_Enhanced(Input, onlySignalBars, useVolumeDelta, minDeltaSize, logRetestSignals, enableM5Structure, enableM15Structure);
		}

		public SMC_DeepSeek_Exporter_Enhanced SMC_DeepSeek_Exporter_Enhanced(ISeries<double> input, bool onlySignalBars, bool useVolumeDelta, int minDeltaSize, bool logRetestSignals, bool enableM5Structure, bool enableM15Structure)
		{
			if (cacheSMC_DeepSeek_Exporter_Enhanced != null)
				for (int idx = 0; idx < cacheSMC_DeepSeek_Exporter_Enhanced.Length; idx++)
					if (cacheSMC_DeepSeek_Exporter_Enhanced[idx] != null && cacheSMC_DeepSeek_Exporter_Enhanced[idx].OnlySignalBars == onlySignalBars && cacheSMC_DeepSeek_Exporter_Enhanced[idx].UseVolumeDelta == useVolumeDelta && cacheSMC_DeepSeek_Exporter_Enhanced[idx].MinDeltaSize == minDeltaSize && cacheSMC_DeepSeek_Exporter_Enhanced[idx].LogRetestSignals == logRetestSignals && cacheSMC_DeepSeek_Exporter_Enhanced[idx].EnableM5Structure == enableM5Structure && cacheSMC_DeepSeek_Exporter_Enhanced[idx].EnableM15Structure == enableM15Structure && cacheSMC_DeepSeek_Exporter_Enhanced[idx].EqualsInput(input))
						return cacheSMC_DeepSeek_Exporter_Enhanced[idx];
			return CacheIndicator<SMC_DeepSeek_Exporter_Enhanced>(new SMC_DeepSeek_Exporter_Enhanced(){ OnlySignalBars = onlySignalBars, UseVolumeDelta = useVolumeDelta, MinDeltaSize = minDeltaSize, LogRetestSignals = logRetestSignals, EnableM5Structure = enableM5Structure, EnableM15Structure = enableM15Structure }, input, ref cacheSMC_DeepSeek_Exporter_Enhanced);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.SMC_DeepSeek_Exporter_Enhanced SMC_DeepSeek_Exporter_Enhanced(bool onlySignalBars, bool useVolumeDelta, int minDeltaSize, bool logRetestSignals, bool enableM5Structure, bool enableM15Structure)
		{
			return indicator.SMC_DeepSeek_Exporter_Enhanced(Input, onlySignalBars, useVolumeDelta, minDeltaSize, logRetestSignals, enableM5Structure, enableM15Structure);
		}

		public Indicators.SMC_DeepSeek_Exporter_Enhanced SMC_DeepSeek_Exporter_Enhanced(ISeries<double> input , bool onlySignalBars, bool useVolumeDelta, int minDeltaSize, bool logRetestSignals, bool enableM5Structure, bool enableM15Structure)
		{
			return indicator.SMC_DeepSeek_Exporter_Enhanced(input, onlySignalBars, useVolumeDelta, minDeltaSize, logRetestSignals, enableM5Structure, enableM15Structure);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.SMC_DeepSeek_Exporter_Enhanced SMC_DeepSeek_Exporter_Enhanced(bool onlySignalBars, bool useVolumeDelta, int minDeltaSize, bool logRetestSignals, bool enableM5Structure, bool enableM15Structure)
		{
			return indicator.SMC_DeepSeek_Exporter_Enhanced(Input, onlySignalBars, useVolumeDelta, minDeltaSize, logRetestSignals, enableM5Structure, enableM15Structure);
		}

		public Indicators.SMC_DeepSeek_Exporter_Enhanced SMC_DeepSeek_Exporter_Enhanced(ISeries<double> input , bool onlySignalBars, bool useVolumeDelta, int minDeltaSize, bool logRetestSignals, bool enableM5Structure, bool enableM15Structure)
		{
			return indicator.SMC_DeepSeek_Exporter_Enhanced(input, onlySignalBars, useVolumeDelta, minDeltaSize, logRetestSignals, enableM5Structure, enableM15Structure);
		}
	}
}

#endregion
