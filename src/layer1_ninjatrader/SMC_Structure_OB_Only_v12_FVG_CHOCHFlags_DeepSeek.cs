#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Collections.Generic;
using System.Windows.Media;
using System.Windows;
using System.Xml.Serialization;
using System.Reflection;
using NinjaTrader.Cbi;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.Tools;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.DrawingTools;
using NinjaTrader.Data;
#endregion

// NT 8.0.28 - C#5 - ASCII only
// Structure (External/Internal) + BOS/CHoCH + OrderBlocks + Fair Value Gaps (FVG).
// Version: 1.6.0-DeepSeek
// - Giá»¯ nguyÃªn behavior v1.5.1
// - ThÃªm public pulses & states cho ML/Exporter
// - ThÃªm FVG first-retest signal
// - ThÃªm External OB first-mitigation (retest) signal
// - KhÃ´ng dÃ¹ng future leak; táº¥t cáº£ dá»±a trÃªn thÃ´ng tin Ä‘Ã£ xáº£y ra táº¡i bar hiá»‡n táº¡i.

namespace NinjaTrader.NinjaScript.Indicators
{
    public class SMC_Structure_OB_Only_v12_FVG_CHOCHFlags : Indicator
    {
        #region Parameters
        // ===== SHARED SETTINGS =====
        [NinjaScriptProperty]
        [Range(0, 10)]
        [Display(Name = "BOS Buffer (ticks)", GroupName = "00. Shared Settings", Order = 1)]
        public int BosBufferTicks { get; set; }

        [NinjaScriptProperty]
        [Range(0, 20)]
        [Display(Name = "Label Offset (ticks)", GroupName = "00. Shared Settings", Order = 2)]
        public int LabelOffsetTicks { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Confirm On Close (BOS buffer)", GroupName = "00. Shared Settings", Order = 3)]
        public bool ConfirmOnClose { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable Logs", GroupName = "00. Shared Settings", Order = 4)]
        public bool EnableLogs { get; set; }

        // ===== EXTERNAL SWING =====
        [NinjaScriptProperty]
        [Range(5, 500)]
        [Display(Name = "Window Size", GroupName = "01. External Swing", Order = 10)]
        public int WinExt { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show External", GroupName = "01. External Swing", Order = 11)]
        public bool ShowExternal { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Labels", GroupName = "01. External Swing", Order = 12)]
        public bool ExtLabels { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable OrderBlocks", GroupName = "01. External Swing", Order = 13)]
        public bool EnableSwingOrderBlocks { get; set; }

        [NinjaScriptProperty]
        [Range(0, 500)]
        [Display(Name = "Max OrderBlocks (0 = unlimited)", GroupName = "01. External Swing", Order = 14)]
        public int MaxExtOBs { get; set; }

        // ===== INTERNAL SWING =====
        [NinjaScriptProperty]
        [Range(2, 50)]
        [Display(Name = "Window Size", GroupName = "02. Internal Swing", Order = 20)]
        public int IntWindow { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Internal", GroupName = "02. Internal Swing", Order = 21)]
        public bool ShowInternal { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Labels", GroupName = "02. Internal Swing", Order = 22)]
        public bool IntLabels { get; set; }

        [NinjaScriptProperty]
        [Range(0, 200)]
        [Display(Name = "Max Labels", GroupName = "02. Internal Swing", Order = 23)]
        public int MaxInternalLabels { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable OrderBlocks", GroupName = "02. Internal Swing", Order = 24)]
        public bool EnableInternalOrderBlocks { get; set; }

        [NinjaScriptProperty]
        [Range(0, 500)]
        [Display(Name = "Max OrderBlocks (0 = unlimited)", GroupName = "02. Internal Swing", Order = 25)]
        public int MaxIntOBs { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "OB: Remove When Fully Filled", GroupName = "02. Internal Swing", Order = 26)]
        public bool OBRemoveInternalOnFullFill { get; set; }

        // --- Internal CHoCH Filters ---
        [NinjaScriptProperty]
        [Display(Name = "CHoCH: Body Close Break", GroupName = "02. Internal Swing", Order = 30)]
        public bool IntChochBreakByBodyClose { get; set; }

        [NinjaScriptProperty]
        [Range(0, 10)]
        [Display(Name = "CHoCH: Break Buffer Ticks", GroupName = "02. Internal Swing", Order = 31)]
        public int IntChochBreakBufferTicks { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "CHoCH: Require Displacement", GroupName = "02. Internal Swing", Order = 32)]
        public bool IntChochRequireDisplacement { get; set; }

        [NinjaScriptProperty]
        [Range(2, 200)]
        [Display(Name = "CHoCH: Displacement ATR Period", GroupName = "02. Internal Swing", Order = 33)]
        public int IntChochDisplacementATRPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 10.0)]
        [Display(Name = "CHoCH: Displacement ATR Mult", GroupName = "02. Internal Swing", Order = 34)]
        public double IntChochDisplacementATRMult { get; set; }

        [NinjaScriptProperty]
        [Range(0, 10)]
        [Display(Name = "CHoCH: Min Break Ticks", GroupName = "02. Internal Swing", Order = 35)]
        public int IntChochMinBreakTicks { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "CHoCH: Strict Affects Display", GroupName = "02. Internal Swing", Order = 36)]
        public bool IntChochStrictAffectsDisplay { get; set; }

        // ===== ORDERBLOCK SHARED SETTINGS =====
        [NinjaScriptProperty]
        [Range(1, 200)]
        [Display(Name = "Lookback Bars", GroupName = "03. OrderBlock Settings", Order = 40)]
        public int OBLookbackBars { get; set; }

        [NinjaScriptProperty]
        [Range(0, 50)]
        [Display(Name = "Buffer (ticks)", GroupName = "03. OrderBlock Settings", Order = 41)]
        public int OBBufferTicks { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Labels", GroupName = "03. OrderBlock Settings", Order = 42)]
        public bool OBShowLabels { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Use Full Candle", GroupName = "03. OrderBlock Settings", Order = 43)]
        public bool OBUseFullCandle { get; set; }

        [NinjaScriptProperty]
        [Range(0, 255)]
        [Display(Name = "Fill Opacity (0-255)", GroupName = "03. OrderBlock Settings", Order = 44)]
        public int OBFillOpacity { get; set; }

        [NinjaScriptProperty]
        [Range(1, 5)]
        [Display(Name = "Outline Width", GroupName = "03. OrderBlock Settings", Order = 45)]
        public int OBOutlineWidth { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Full Fill Uses Wicks", GroupName = "03. OrderBlock Settings", Order = 46)]
        public bool OBFullFillUseWicks { get; set; }

        // ===== FVG (FAIR VALUE GAPS) =====
        [NinjaScriptProperty]
        [Display(Name = "Show FVG", GroupName = "04. Fair Value Gaps", Order = 50)]
        public bool ShowFVG { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Auto Threshold", GroupName = "04. Fair Value Gaps", Order = 51)]
        public bool FVGAutoThreshold { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 10.0)]
        [Display(Name = "Threshold Multiplier", GroupName = "04. Fair Value Gaps", Order = 52)]
        public double FVGThresholdMultiplier { get; set; }

        [NinjaScriptProperty]
        [Range(0, 200)]
        [Display(Name = "Extend Bars", GroupName = "04. Fair Value Gaps", Order = 53)]
        public int FVGExtendBars { get; set; }

        [NinjaScriptProperty]
        [Range(0, 255)]
        [Display(Name = "Fill Opacity (0-255)", GroupName = "04. Fair Value Gaps", Order = 54)]
        public int FVGFillOpacity { get; set; }

        [NinjaScriptProperty]
        [Range(1, 5)]
        [Display(Name = "Outline Width", GroupName = "04. Fair Value Gaps", Order = 55)]
        public int FVGOutlineWidth { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Bull Color", GroupName = "04. Fair Value Gaps", Order = 56)]
        public System.Windows.Media.Color FVGBullColor { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Bear Color", GroupName = "04. Fair Value Gaps", Order = 57)]
        public System.Windows.Media.Color FVGBearColor { get; set; }

        [NinjaScriptProperty]
        [Range(0, 20)]
        [Display(Name = "Retest Ticks Into Zone", GroupName = "04. Fair Value Gaps", Order = 58)]
        public int FVGRetestTicksIntoZone { get; set; }

        // ===== PREMIUM/DISCOUNT ZONES =====
        [NinjaScriptProperty]
        [Display(Name = "Show Premium/Discount Zones", GroupName = "05. Premium & Discount Zones", Order = 60)]
        public bool ShowPremiumDiscount { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Premium Color", GroupName = "05. Premium & Discount Zones", Order = 61)]
        public System.Windows.Media.Color PremiumColor { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Equilibrium Color", GroupName = "05. Premium & Discount Zones", Order = 62)]
        public System.Windows.Media.Color EquilibriumColor { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Discount Color", GroupName = "05. Premium & Discount Zones", Order = 63)]
        public System.Windows.Media.Color DiscountColor { get; set; }

        [NinjaScriptProperty]
        [Range(0, 255)]
        [Display(Name = "Zone Opacity (0-255)", GroupName = "05. Premium & Discount Zones", Order = 64)]
        public int ZoneOpacity { get; set; }

        // ===== MULTI-TIMEFRAME (M5, M15) =====
        [NinjaScriptProperty]
        [Display(Name = "Enable M5 Structure", GroupName = "06. Multi-Timeframe", Order = 70)]
        public bool EnableM5Structure { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable M15 Structure", GroupName = "06. Multi-Timeframe", Order = 71)]
        public bool EnableM15Structure { get; set; }
        #endregion

        #region Version
        private const string VersionTag = "SMC_Structure_OB_Only_v12_FVG_CHOCHFlags 1.6.0-DeepSeek";
        public override string DisplayName { get { return "SMC Structure"; } }
        #endregion

        #region State (core)
        private const int BULL = 1, BEAR = -1;
        private const int BULLISH_LEG = 1, BEARISH_LEG = 0;

        private int lastLegExt = -1;
        private double extHighCurr = double.NaN, extHighLast = double.NaN;
        private double extLowCurr = double.NaN, extLowLast = double.NaN;
        private int extHighBar = -1, extLowBar = -1;
        private bool extHighCrossed = false, extLowCrossed = false;
        private int extBias = 0;
        private double extLowPrev = double.NaN;

        private int lastLegInt = -1;
        private double intHighCurr = double.NaN, intHighLast = double.NaN;
        private double intLowCurr = double.NaN, intLowLast = double.NaN;
        private int intHighBar = -1, intLowBar = -1;
        private bool intHighCrossed = false, intLowCrossed = false;
        private int intTrendBias = 0;
        private double intLowPrev = double.NaN;

        private Queue<string> _intLabelQueue = new Queue<string>();

        // Premium/Discount Zones tracking
        private double trailUp = double.NaN;
        private double trailDn = double.NaN;
        private int trailUpBar = -1;
        private int trailDnBar = -1;

        private class OBZ
        {
            internal bool IsInternal;
            internal bool Bull;
            internal int SourceBar;
            internal double Lo;
            internal double Hi;
            internal bool Mitigated;
            internal bool Invalidated;
            internal string RectTag;
            internal string LblTag;
            internal bool HitTop;
            internal bool HitBottom;
        }

        private List<OBZ> _obZones = new List<OBZ>();
        private Brush _bullFill; private Brush _bullOutline; private Brush _bearFill; private Brush _bearOutline; private Brush _mitigatedOutline; private Brush _invalidOutline; private Brush _textBrush;

        // ===== FVG =====
        private class FVGZ
        {
            internal bool Bull;
            internal int SourceBar;
            internal int Age;
            internal double Top;
            internal double Bottom;
            internal bool Removed;
            internal string RectTag;
            internal string LblTag;
            internal bool Retested; // DeepSeek: mark first retest
        }

        private List<FVGZ> _fvgs = new List<FVGZ>();
        private Brush _fvgBullFill; private Brush _fvgBullOutline; private Brush _fvgBearFill; private Brush _fvgBearOutline; private Brush _fvgTextBrush;
        private double _cumAbsDeltaPct = 0.0; private int _cumCount = 0;

        private NinjaTrader.NinjaScript.Indicators.ATR _atr200;
        private NinjaTrader.NinjaScript.Indicators.ATR _atrDisp;
        #endregion

        #region DeepSeek: pulses & states
        // Structure pulses
        private Series<bool> _extBosUpPulse;
        private Series<bool> _extBosDownPulse;
        private Series<bool> _extChochUpPulse;
        private Series<bool> _extChochDownPulse;

        private Series<bool> _intBosUpPulse;
        private Series<bool> _intBosDownPulse;
        private Series<bool> _intChochUpPulse;
        private Series<bool> _intChochDownPulse;

        // Liquidity sweeps
        private Series<bool> _sweepPrevHighPulse;
        private Series<bool> _sweepPrevLowPulse;

        // OB/FVG retest pulses
        private Series<bool> _obExtRetestBull;
        private Series<bool> _obExtRetestBear;
        private Series<bool> _fvgRetestBull;
        private Series<bool> _fvgRetestBear;
        private Series<double> _obExtRetestBullSL;
        private Series<double> _obExtRetestBearSL;
        private Series<double> _fvgRetestBullSL;
        private Series<double> _fvgRetestBearSL;

        // Context states
        private Series<bool> _hasActiveObExtBull;
        private Series<bool> _hasActiveObExtBear;
        private Series<bool> _hasActiveFvgBull;
        private Series<bool> _hasActiveFvgBear;
        private Series<bool> _inPremiumZone;
        private Series<bool> _inDiscountZone;

        // Structure direction (1,0,-1)
        private Series<int> _extStructureDir;
        private Series<int> _intStructureDir;

        // MTF export (M5 -> primary) for exporter/ML
        private Series<bool> _m5ExtBosUpPulsePrimary;
        private Series<bool> _m5ExtBosDownPulsePrimary;
        private Series<bool> _m5ExtChochUpPulsePrimary;
        private Series<bool> _m5ExtChochDownPulsePrimary;
        private Series<int> _m5ExtStructureDirPrimary;

        // Swing levels (External)
        private Series<double> _extLastSwingHigh;
        private Series<double> _extLastSwingLow;
        private Series<double> _extPrevSwingHigh;
        private Series<double> _extPrevSwingLow;
        private Series<int> _extSwingBar; // Bar index of last swing
        private Series<int> _extSwingPattern; // 1=HH, 2=HL, -1=LL, -2=LH, 0=undefined

        // Swing levels (Internal)
        private Series<double> _intLastSwingHigh;
        private Series<double> _intLastSwingLow;
        private Series<double> _intPrevSwingHigh;
        private Series<double> _intPrevSwingLow;
        private Series<int> _intSwingBar;
        private Series<int> _intSwingPattern;

        // ===== Multi-Timeframe Data (M5, M15) =====
        // Helper class to store MTF structure data - FULL PROCESSING
        private class MTFStructureData
        {
            // Swing/Structure state
            public int LastLegExt = -1;
            public double ExtHighLast = double.NaN;
            public double ExtLowLast = double.NaN;
            public double ExtHighCurr = double.NaN;
            public double ExtLowCurr = double.NaN;
            public int ExtHighBar = -1;
            public int ExtLowBar = -1;
            public bool ExtHighCrossed = false;
            public bool ExtLowCrossed = false;
            public int ExtBias = 0;
            public double ExtLowPrev = double.NaN;

            // Premium/Discount zones
            public double TrailUp = double.NaN;
            public double TrailDn = double.NaN;
            public int TrailUpBar = -1;
            public int TrailDnBar = -1;

            // BOS/CHoCH one-time signals (pulses)
            public bool BosUpPulse = false;
            public bool BosDownPulse = false;
            public bool ChochUpPulse = false;
            public bool ChochDownPulse = false;

            // Structure direction (persistent state)
            public int StructureDir = 0; // 1=bullish, -1=bearish, 0=undefined

            // OrderBlocks tracking
            public List<OBZ> OrderBlocks = new List<OBZ>();

            // OB Retest signals (pulses)
            public bool OBRetestBullPulse = false;
            public bool OBRetestBearPulse = false;
            public double OBRetestBullSL = double.NaN;
            public double OBRetestBearSL = double.NaN;

            // OB Context (persistent states)
            public bool HasActiveObExtBull = false;
            public bool HasActiveObExtBear = false;

            // FVG tracking
            public List<FVGZ> FVGs = new List<FVGZ>();

            // FVG Retest signals (pulses)
            public bool FVGRetestBullPulse = false;
            public bool FVGRetestBearPulse = false;
            public double FVGRetestBullSL = double.NaN;
            public double FVGRetestBearSL = double.NaN;

            // FVG Context (persistent states)
            public bool HasActiveFvgBull = false;
            public bool HasActiveFvgBear = false;

            // Premium/Discount Context (persistent states)
            public bool InPremiumZone = false;
            public bool InDiscountZone = false;

            // Swing tracking for ML (persistent states)
            public double LastSwingHigh = double.NaN;
            public double LastSwingLow = double.NaN;
            public int BarsInCurrentSwing = 0;
        }

        private MTFStructureData _m5Data = null;
        private MTFStructureData _m15Data = null;
        private int _m5BarsIndex = -1;
        private int _m15BarsIndex = -1;

        // Pending MTF exports (captured on M5 bar, flushed on next primary bar)
        private bool _pendingM5BosUpPulse = false;
        private bool _pendingM5BosDownPulse = false;
        private bool _pendingM5ChochUpPulse = false;
        private bool _pendingM5ChochDownPulse = false;
        private int _pendingM5StructureDir = 0;
        private DateTime _pendingM5Time = DateTime.MinValue;
        private bool _hasPendingM5Update = false;
        #endregion

        #region Lifecycle
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "SMC_Structure_OB_Only_v12_FVG_CHOCHFlags";
                Calculate = Calculate.OnBarClose;
                IsOverlay = true;

                WinExt = 50; IntWindow = 5; BosBufferTicks = 1; LabelOffsetTicks = 2;
                ShowExternal = true; ShowInternal = true;
                ExtLabels = true; IntLabels = false; MaxInternalLabels = 40; EnableLogs = true;
                ConfirmOnClose = true;

                IntChochBreakByBodyClose = true;
                IntChochBreakBufferTicks = 1;
                IntChochRequireDisplacement = true;
                IntChochDisplacementATRPeriod = 20;
                IntChochDisplacementATRMult = 1.0;
                IntChochMinBreakTicks = 0;
                IntChochStrictAffectsDisplay = false;

                EnableSwingOrderBlocks = true;
                EnableInternalOrderBlocks = true;
                OBLookbackBars = 30; OBBufferTicks = 1;
                OBShowLabels = true; OBUseFullCandle = true;
                OBFillOpacity = 48; OBOutlineWidth = 2;
                OBRemoveInternalOnFullFill = true;
                OBFullFillUseWicks = true;
                MaxExtOBs = 20; MaxIntOBs = 40;

                ShowFVG = true; FVGAutoThreshold = true; FVGThresholdMultiplier = 2.0;
                FVGExtendBars = 0; FVGFillOpacity = 160; FVGOutlineWidth = 2;
                FVGBullColor = Color.FromArgb(200, 0, 255, 104);
                FVGBearColor = Color.FromArgb(200, 255, 0, 8);
                FVGRetestTicksIntoZone = 1;

                ShowPremiumDiscount = false;
                PremiumColor = Color.FromArgb(255, 242, 54, 69);
                EquilibriumColor = Color.FromArgb(255, 178, 181, 190);
                DiscountColor = Color.FromArgb(255, 8, 153, 129);
                ZoneOpacity = 80;

                // Multi-Timeframe defaults
                EnableM5Structure = false;
                EnableM15Structure = false;
            }
            else if (State == State.Configure)
            {
                // Add Multi-Timeframe data series
                // IMPORTANT: M5/M15 data is LIMITED by PRIMARY chart's "Days to load" setting!
                // To get more M5/M15 bars: Right-click chart → Properties → Data Series → Set "Days to load" = 5-7 days
                if (EnableM5Structure && Instrument != null)
                {
                    try
                    {
                        string instrumentName = Instrument.FullName;
                        BarsPeriod m5Period = new BarsPeriod { BarsPeriodType = BarsPeriodType.Minute, Value = 5 };
                        AddDataSeries(instrumentName, m5Period);
                        if (EnableLogs) Print("[SMC MTF] Added M5 data series");
                    }
                    catch (Exception ex)
                    {
                        if (EnableLogs) Print("[SMC MTF] ERROR adding M5 series: " + ex.Message);
                    }
                }

                if (EnableM15Structure && Instrument != null)
                {
                    try
                    {
                        string instrumentName = Instrument.FullName;
                        BarsPeriod m15Period = new BarsPeriod { BarsPeriodType = BarsPeriodType.Minute, Value = 15 };
                        AddDataSeries(instrumentName, m15Period);
                        if (EnableLogs) Print("[SMC MTF] Added M15 data series");
                    }
                    catch (Exception ex)
                    {
                        if (EnableLogs) Print("[SMC MTF] ERROR adding M15 series: " + ex.Message);
                    }
                }

                try
                {
                    var bf = new SolidColorBrush(Color.FromArgb((byte)Math.Max(0, Math.Min(255, OBFillOpacity)), 70, 130, 180)); bf.Freeze(); _bullFill = bf;
                    var bo = new SolidColorBrush(Color.FromArgb(255, 30, 144, 255)); bo.Freeze(); _bullOutline = bo;
                    var sf = new SolidColorBrush(Color.FromArgb((byte)Math.Max(0, Math.Min(255, OBFillOpacity)), 220, 20, 60)); sf.Freeze(); _bearFill = sf;
                    var so = new SolidColorBrush(Color.FromArgb(255, 178, 34, 34)); so.Freeze(); _bearOutline = so;
                    var mo = new SolidColorBrush(Color.FromArgb(255, 72, 61, 139)); mo.Freeze(); _mitigatedOutline = mo;
                    var io = new SolidColorBrush(Color.FromArgb(255, 128, 128, 128)); io.Freeze(); _invalidOutline = io;
                    var tb = new SolidColorBrush(Color.FromArgb(255, 25, 25, 25)); tb.Freeze(); _textBrush = tb;

                    var fbg = new SolidColorBrush(Color.FromArgb((byte)Math.Max(0, Math.Min(255, FVGFillOpacity)), FVGBullColor.R, FVGBullColor.G, FVGBullColor.B)); fbg.Freeze(); _fvgBullFill = fbg;
                    var fbo = new SolidColorBrush(Color.FromArgb(0, 0, 0, 0)); fbo.Freeze(); _fvgBullOutline = fbo;
                    var frg = new SolidColorBrush(Color.FromArgb((byte)Math.Max(0, Math.Min(255, FVGFillOpacity)), FVGBearColor.R, FVGBearColor.G, FVGBearColor.B)); frg.Freeze(); _fvgBearFill = frg;
                    var fro = new SolidColorBrush(Color.FromArgb(0, 0, 0, 0)); fro.Freeze(); _fvgBearOutline = fro;
                    var ftxt = new SolidColorBrush(Color.FromArgb(255, 25, 25, 25)); ftxt.Freeze(); _fvgTextBrush = ftxt;
                }
                catch
                {
                    _bullFill = Brushes.SteelBlue; _bullOutline = Brushes.DodgerBlue;
                    _bearFill = Brushes.IndianRed; _bearOutline = Brushes.Red;
                    _mitigatedOutline = Brushes.DarkSlateBlue; _invalidOutline = Brushes.Gray; _textBrush = Brushes.Black;
                    _fvgBullFill = Brushes.LightGreen; _fvgBullOutline = Brushes.Transparent;
                    _fvgBearFill = Brushes.LightCoral; _fvgBearOutline = Brushes.Transparent; _fvgTextBrush = Brushes.Black;
                }
            }
            else if (State == State.DataLoaded)
            {
                try { _atr200 = ATR(200); } catch { }
                try { _atrDisp = ATR(Math.Max(2, IntChochDisplacementATRPeriod)); } catch { }

                // Allocate DeepSeek series
                _extBosUpPulse = new Series<bool>(this);
                _extBosDownPulse = new Series<bool>(this);
                _extChochUpPulse = new Series<bool>(this);
                _extChochDownPulse = new Series<bool>(this);

                _intBosUpPulse = new Series<bool>(this);
                _intBosDownPulse = new Series<bool>(this);
                _intChochUpPulse = new Series<bool>(this);
                _intChochDownPulse = new Series<bool>(this);

                _sweepPrevHighPulse = new Series<bool>(this);
                _sweepPrevLowPulse = new Series<bool>(this);

                _obExtRetestBull = new Series<bool>(this);
                _obExtRetestBear = new Series<bool>(this);
                _fvgRetestBull = new Series<bool>(this);
                _fvgRetestBear = new Series<bool>(this);
                _obExtRetestBullSL = new Series<double>(this);
                _obExtRetestBearSL = new Series<double>(this);
                _fvgRetestBullSL = new Series<double>(this);
                _fvgRetestBearSL = new Series<double>(this);

                _hasActiveObExtBull = new Series<bool>(this);
                _hasActiveObExtBear = new Series<bool>(this);
                _hasActiveFvgBull = new Series<bool>(this);
                _hasActiveFvgBear = new Series<bool>(this);
                _inPremiumZone = new Series<bool>(this);
                _inDiscountZone = new Series<bool>(this);

                _extStructureDir = new Series<int>(this);
                _intStructureDir = new Series<int>(this);

                // Swing levels
                _extLastSwingHigh = new Series<double>(this);
                _extLastSwingLow = new Series<double>(this);
                _extPrevSwingHigh = new Series<double>(this);
                _extPrevSwingLow = new Series<double>(this);
                _extSwingBar = new Series<int>(this);
                _extSwingPattern = new Series<int>(this);

                _intLastSwingHigh = new Series<double>(this);
                _intLastSwingLow = new Series<double>(this);
                _intPrevSwingHigh = new Series<double>(this);
                _intPrevSwingLow = new Series<double>(this);
                _intSwingBar = new Series<int>(this);
                _intSwingPattern = new Series<int>(this);

                // Primary-series mirrors of M5 signals for exporter
                _m5ExtBosUpPulsePrimary = new Series<bool>(this);
                _m5ExtBosDownPulsePrimary = new Series<bool>(this);
                _m5ExtChochUpPulsePrimary = new Series<bool>(this);
                _m5ExtChochDownPulsePrimary = new Series<bool>(this);
                _m5ExtStructureDirPrimary = new Series<int>(this);

                // Initialize Multi-Timeframe data
                if (EnableM5Structure || EnableM15Structure)
                {
                    // Determine BarsArray indices
                    // BarsArray[0] = Primary chart
                    // BarsArray[1] = M5 (if enabled)
                    // BarsArray[2] = M15 (if both enabled) or M15 (if only M15 enabled)
                    int currentIndex = 1;

                    if (EnableLogs)
                    {
                        Print(string.Format("[SMC MTF] PRIMARY M1 chart has {0} bars (CurrentBar={1})",
                            BarsArray[0].Count, CurrentBars[0]));
                        if (CurrentBars[0] >= 0 && BarsArray[0].Count > 0)
                        {
                            int idx = Math.Min(CurrentBars[0], BarsArray[0].Count - 1);
                            Print(string.Format("[SMC MTF] PRIMARY M1 time range: {0} to {1}",
                                Times[0][idx],
                                Times[0][0]));
                        }
                        else
                        {
                            Print("[SMC MTF] PRIMARY M1 time range: waiting for bars...");
                        }
                    }

                    if (EnableM5Structure)
                    {
                        _m5BarsIndex = currentIndex;
                        _m5Data = new MTFStructureData();
                        currentIndex++;
                        if (EnableLogs) Print(string.Format("[SMC MTF] M5 initialized at BarsArray[{0}]", _m5BarsIndex));
                    }

                    if (EnableM15Structure)
                    {
                        _m15BarsIndex = currentIndex;
                        _m15Data = new MTFStructureData();
                        if (EnableLogs) Print(string.Format("[SMC MTF] M15 initialized at BarsArray[{0}]", _m15BarsIndex));
                    }
                }
            }
        }

        #region Multi-Timeframe Processing - FULL LOGIC (NO DRAWING)

        private void ProcessMTFBar_Full(int bip, MTFStructureData data)
        {
            // FULL processing for M5/M15 - ALL LOGIC, NO DRAWING
            try
            {
                int win = WinExt;
                int currentBar = CurrentBars[bip]; // Use CurrentBars instead of BarsArray.Count - 1
                string tfLabel = (bip == _m5BarsIndex) ? "M5" : "M15";

                // Check if we have valid bar index
                if (currentBar < 0)
                {
                    if (EnableLogs) Print(string.Format("[SMC MTF {0}] CurrentBars[{1}] < 0, waiting...", tfLabel, bip));
                    return;
                }

                // Debug: Print bar info on first few bars
                if (EnableLogs && currentBar <= 5)
                {
                    Print(string.Format("[SMC MTF {0}] Bar #{1}, CurrentBars[{2}]={3}, BarsArray.Count={4}, WinExt={5}",
                        tfLabel, currentBar, bip, CurrentBars[bip], BarsArray[bip].Count, win));
                }

                if (currentBar < win + 3)
                {
                    if (EnableLogs && (currentBar == win + 2 || currentBar % 10 == 0))
                        Print(string.Format("[SMC MTF {0}] Waiting for data... currentBar={1}, need={2}, WinExt={3}",
                            tfLabel, currentBar, win + 3, win));
                    return;
                }

                var highs = Highs[bip];
                var lows = Lows[bip];
                var opens = Opens[bip];
                var closes = Closes[bip];

                if (highs == null || lows == null || opens == null || closes == null)
                {
                    if (EnableLogs) Print(string.Format("[SMC MTF {0}] ERROR: OHLC data is null!", tfLabel));
                    return;
                }

                // 1. Reset pulses
                ResetPulses_MTF(data);

                // 2. Update structure (swing detection)
                UpdateStructure_MTF(bip, data, win, highs, lows);

                // 3. Check BOS/CHoCH
                CheckBosChoCh_MTF(bip, data, highs, lows, closes, currentBar);

                // 4. Update trailing high/low for premium/discount
                UpdateTrailingHL_MTF(data);

                // 5. Detect & maintain OrderBlocks
                if (EnableSwingOrderBlocks)
                {
                    DetectOrderBlocks_MTF(bip, data, highs, lows, opens, closes, currentBar);
                    MaintainOrderBlocks_MTF(bip, data, highs, lows, currentBar);
                }

                // 6. Detect & maintain FVGs
                if (ShowFVG)
                {
                    TryDetectFVG_MTF(bip, data, highs, lows, currentBar);
                    MaintainFVGs_MTF(bip, data, highs, lows, currentBar);
                }

                // 7. Update context states
                UpdateContext_MTF(bip, data, highs, lows, closes, currentBar);

                // 8. Capture outputs for exporter (M5 only)
                CaptureMtfExports(bip, data);

                // 9. Update swing tracking
                data.BarsInCurrentSwing++;
            }
            catch (Exception ex)
            {
                if (EnableLogs) Print("[SMC MTF] ProcessMTFBar_Full error: " + ex.Message);
            }
        }

        private void CaptureMtfExports(int bip, MTFStructureData data)
        {
            if (bip != _m5BarsIndex || data == null)
                return;

            bool anyPulse =
                data.BosUpPulse || data.BosDownPulse || data.ChochUpPulse || data.ChochDownPulse;

            _pendingM5BosUpPulse |= data.BosUpPulse;
            _pendingM5BosDownPulse |= data.BosDownPulse;
            _pendingM5ChochUpPulse |= data.ChochUpPulse;
            _pendingM5ChochDownPulse |= data.ChochDownPulse;
            _pendingM5StructureDir = data.StructureDir;
            _pendingM5Time = Times[bip][0];
            _hasPendingM5Update = true;

            if (EnableLogs && anyPulse)
            {
                Print(string.Format("[SMC MTF M5->Primary] Captured pulses @ {0}: BOSup={1}, BOSdn={2}, CHOCHup={3}, CHOCHdn={4}, Dir={5}",
                    _pendingM5Time,
                    data.BosUpPulse, data.BosDownPulse, data.ChochUpPulse, data.ChochDownPulse, data.StructureDir));
            }
        }

        private void ResetPulses_MTF(MTFStructureData data)
        {
            data.BosUpPulse = false;
            data.BosDownPulse = false;
            data.ChochUpPulse = false;
            data.ChochDownPulse = false;
            data.OBRetestBullPulse = false;
            data.OBRetestBearPulse = false;
            data.OBRetestBullSL = double.NaN;
            data.OBRetestBearSL = double.NaN;
            data.FVGRetestBullPulse = false;
            data.FVGRetestBearPulse = false;
            data.FVGRetestBullSL = double.NaN;
            data.FVGRetestBearSL = double.NaN;
        }

        private void UpdateStructure_MTF(int bip, MTFStructureData data, int win, ISeries<double> highs, ISeries<double> lows)
        {
            int currentBar = CurrentBars[bip];
            int idx = win;
            if (currentBar <= idx + 2) return;

            string tfLabel = (bip == _m5BarsIndex) ? "M5" : "M15";

            // Mirror primary timeframe logic using MAX/MIN
            var maxSeries = MAX(highs, win);
            var minSeries = MIN(lows, win);
            if (maxSeries == null || minSeries == null) return;

            bool newLegHigh = highs[idx] > maxSeries[0];
            bool newLegLow = lows[idx] < minSeries[0];

            int legNow = data.LastLegExt;
            if (newLegHigh) legNow = BEARISH_LEG;
            else if (newLegLow) legNow = BULLISH_LEG;

            if (data.LastLegExt == -1 && (newLegHigh || newLegLow))
            {
                data.LastLegExt = legNow;
                if (EnableLogs) Print(string.Format("[SMC MTF {0}] First swing detected: {1}", tfLabel, legNow == BULLISH_LEG ? "BULLISH" : "BEARISH"));
                return;
            }

            bool startOfNewLeg = (data.LastLegExt != -1) && (legNow != data.LastLegExt);
            if (!startOfNewLeg) return;

            if (legNow == BULLISH_LEG)
            {
                data.ExtLowCurr = lows[idx];
                data.ExtLowBar = currentBar - idx;
                data.ExtLowCrossed = false;
                data.LastLegExt = legNow;
                data.ExtLowPrev = data.ExtLowLast;
                data.ExtLowLast = data.ExtLowCurr;
                data.TrailDn = data.ExtLowCurr;
                data.TrailDnBar = data.ExtLowBar;
                data.LastSwingLow = data.ExtLowCurr;
                data.BarsInCurrentSwing = 0;

                if (EnableLogs) Print(string.Format("[SMC MTF {0}] New BULLISH swing @ {1} (low={2})", tfLabel, currentBar, data.ExtLowCurr));
            }
            else
            {
                data.ExtHighCurr = highs[idx];
                data.ExtHighBar = currentBar - idx;
                data.ExtHighCrossed = false;
                data.LastLegExt = legNow;
                data.ExtHighLast = data.ExtHighCurr;
                data.TrailUp = data.ExtHighCurr;
                data.TrailUpBar = data.ExtHighBar;
                data.LastSwingHigh = data.ExtHighCurr;
                data.BarsInCurrentSwing = 0;

                if (EnableLogs) Print(string.Format("[SMC MTF {0}] New BEARISH swing @ {1} (high={2})", tfLabel, currentBar, data.ExtHighCurr));
            }
        }

        private void CheckBosChoCh_MTF(int bip, MTFStructureData data, ISeries<double> highs, ISeries<double> lows, ISeries<double> closes, int currentBar)
        {
            // Check for breaks similar to primary timeframe
            double tickSize = Instrument.MasterInstrument.TickSize;
            double buffer = BosBufferTicks * tickSize;

            string tfLabel = (bip == _m5BarsIndex) ? "M5" : "M15";

            // Check bullish break (breaking previous high)
            if (!double.IsNaN(data.ExtHighLast) && !data.ExtHighCrossed)
            {
                double breakLevel = data.ExtHighLast + buffer;
                bool broken = ConfirmOnClose ? (closes[0] > breakLevel) : (highs[0] > breakLevel);

                if (broken)
                {
                    data.ExtHighCrossed = true;
                    if (data.ExtBias == BULL)
                    {
                        data.BosUpPulse = true;
                        data.StructureDir = BULL;
                        if (EnableLogs) Print(string.Format("[SMC MTF {0}] BOS UP @ {1} (broke {2})", tfLabel, closes[0], breakLevel));
                    }
                    else
                    {
                        data.ChochUpPulse = true;
                        data.StructureDir = BULL;
                        if (EnableLogs) Print(string.Format("[SMC MTF {0}] CHoCH UP @ {1} (broke {2})", tfLabel, closes[0], breakLevel));
                    }
                    data.ExtBias = BULL;
                }
            }

            // Check bearish break (breaking previous low)
            if (!double.IsNaN(data.ExtLowLast) && !data.ExtLowCrossed)
            {
                double breakLevel = data.ExtLowLast - buffer;
                bool broken = ConfirmOnClose ? (closes[0] < breakLevel) : (lows[0] < breakLevel);

                if (broken)
                {
                    data.ExtLowCrossed = true;
                    if (data.ExtBias == BEAR)
                    {
                        data.BosDownPulse = true;
                        data.StructureDir = BEAR;
                        if (EnableLogs) Print(string.Format("[SMC MTF {0}] BOS DOWN @ {1} (broke {2})", tfLabel, closes[0], breakLevel));
                    }
                    else
                    {
                        data.ChochDownPulse = true;
                        data.StructureDir = BEAR;
                        if (EnableLogs) Print(string.Format("[SMC MTF {0}] CHoCH DOWN @ {1} (broke {2})", tfLabel, closes[0], breakLevel));
                    }
                    data.ExtBias = BEAR;
                }
            }
        }

        private void UpdateTrailingHL_MTF(MTFStructureData data)
        {
            // Update trailing high/low for premium/discount calculation
            // (Already done in UpdateStructure_MTF)
        }

        private void DetectOrderBlocks_MTF(int bip, MTFStructureData data, ISeries<double> highs, ISeries<double> lows,
                                            ISeries<double> opens, ISeries<double> closes, int currentBar)
        {
            // Simplified OB detection for MTF (similar to primary timeframe)
            if (currentBar < OBLookbackBars + 1) return;

            // Detect bull OB (after bearish move then bullish break)
            if (data.BosUpPulse || data.ChochUpPulse)
            {
                for (int i = 1; i <= OBLookbackBars; i++)
                {
                    bool isBearish = closes[i] < opens[i];
                    if (isBearish)
                    {
                        double obHi = OBUseFullCandle ? highs[i] : Math.Max(opens[i], closes[i]);
                        double obLo = OBUseFullCandle ? lows[i] : Math.Min(opens[i], closes[i]);

                        var ob = new OBZ
                        {
                            IsInternal = false,
                            Bull = true,
                            SourceBar = currentBar - i,
                            Lo = obLo,
                            Hi = obHi,
                            Mitigated = false,
                            Invalidated = false,
                            RectTag = "MTFOB_B_" + bip + "_" + (currentBar - i),
                            LblTag = "MTFOBL_B_" + bip + "_" + (currentBar - i),
                            HitTop = false,
                            HitBottom = false
                        };
                        data.OrderBlocks.Add(ob);
                        break;
                    }
                }
            }

            // Detect bear OB
            if (data.BosDownPulse || data.ChochDownPulse)
            {
                for (int i = 1; i <= OBLookbackBars; i++)
                {
                    bool isBullish = closes[i] > opens[i];
                    if (isBullish)
                    {
                        double obHi = OBUseFullCandle ? highs[i] : Math.Max(opens[i], closes[i]);
                        double obLo = OBUseFullCandle ? lows[i] : Math.Min(opens[i], closes[i]);

                        var ob = new OBZ
                        {
                            IsInternal = false,
                            Bull = false,
                            SourceBar = currentBar - i,
                            Lo = obLo,
                            Hi = obHi,
                            Mitigated = false,
                            Invalidated = false,
                            RectTag = "MTFOB_S_" + bip + "_" + (currentBar - i),
                            LblTag = "MTFOBL_S_" + bip + "_" + (currentBar - i),
                            HitTop = false,
                            HitBottom = false
                        };
                        data.OrderBlocks.Add(ob);
                        break;
                    }
                }
            }

            // Limit OBs
            if (MaxExtOBs > 0)
            {
                while (data.OrderBlocks.Count > MaxExtOBs)
                    data.OrderBlocks.RemoveAt(0);
            }
        }

        private void MaintainOrderBlocks_MTF(int bip, MTFStructureData data, ISeries<double> highs, ISeries<double> lows, int currentBar)
        {
            double tickSize = Instrument.MasterInstrument.TickSize;
            double buffer = OBBufferTicks * tickSize;

            data.HasActiveObExtBull = false;
            data.HasActiveObExtBear = false;

            for (int i = data.OrderBlocks.Count - 1; i >= 0; i--)
            {
                var z = data.OrderBlocks[i];
                if (z.Invalidated) continue;

                double hi = z.Hi + buffer;
                double lo = z.Lo - buffer;

                // Check mitigation
                if (!z.Mitigated)
                {
                    if (z.Bull && lows[0] <= (OBFullFillUseWicks ? lo : z.Lo))
                    {
                        z.HitBottom = true;
                        if (z.HitTop || !OBFullFillUseWicks)
                        {
                            z.Mitigated = true;
                            data.OBRetestBullPulse = true;
                            data.OBRetestBullSL = lo;
                        }
                    }
                    else if (!z.Bull && highs[0] >= (OBFullFillUseWicks ? hi : z.Hi))
                    {
                        z.HitTop = true;
                        if (z.HitBottom || !OBFullFillUseWicks)
                        {
                            z.Mitigated = true;
                            data.OBRetestBearPulse = true;
                            data.OBRetestBearSL = hi;
                        }
                    }

                    // Check if still active
                    if (!z.Mitigated)
                    {
                        if (z.Bull) data.HasActiveObExtBull = true;
                        else data.HasActiveObExtBear = true;
                    }
                }
            }
        }

        private void TryDetectFVG_MTF(int bip, MTFStructureData data, ISeries<double> highs, ISeries<double> lows, int currentBar)
        {
            // Simplified FVG detection
            if (currentBar < 3) return;

            double gap = 0;
            bool isBullFVG = false;

            // Bull FVG: low[0] > high[2]
            if (lows[0] > highs[2])
            {
                gap = lows[0] - highs[2];
                isBullFVG = true;
            }
            // Bear FVG: high[0] < low[2]
            else if (highs[0] < lows[2])
            {
                gap = lows[2] - highs[0];
                isBullFVG = false;
            }
            else
            {
                return;
            }

            // Check threshold
            double threshold = 0;
            if (FVGAutoThreshold && _atr200 != null)
            {
                try { threshold = _atr200[0] * FVGThresholdMultiplier; }
                catch { }
            }
            if (gap < threshold) return;

            // Create FVG
            var fvg = new FVGZ
            {
                Bull = isBullFVG,
                SourceBar = currentBar - 1,
                Age = 0,
                Top = isBullFVG ? lows[0] : lows[2],
                Bottom = isBullFVG ? highs[2] : highs[0],
                Removed = false,
                RectTag = "MTFFVG_" + (isBullFVG ? "B" : "S") + "_" + bip + "_" + (currentBar - 1),
                LblTag = "MTFFVGL_" + (isBullFVG ? "B" : "S") + "_" + bip + "_" + (currentBar - 1),
                Retested = false
            };
            data.FVGs.Add(fvg);
        }

        private void MaintainFVGs_MTF(int bip, MTFStructureData data, ISeries<double> highs, ISeries<double> lows, int currentBar)
        {
            data.HasActiveFvgBull = false;
            data.HasActiveFvgBear = false;

            for (int i = data.FVGs.Count - 1; i >= 0; i--)
            {
                var z = data.FVGs[i];
                z.Age++;

                if (z.Age > FVGExtendBars)
                {
                    z.Removed = true;
                    continue;
                }

                // Check retest
                if (!z.Retested)
                {
                    bool filled = false;
                    if (z.Bull && lows[0] <= z.Top)
                    {
                        filled = true;
                        z.Retested = true;
                        data.FVGRetestBullPulse = true;
                        data.FVGRetestBullSL = z.Bottom;
                    }
                    else if (!z.Bull && highs[0] >= z.Bottom)
                    {
                        filled = true;
                        z.Retested = true;
                        data.FVGRetestBearPulse = true;
                        data.FVGRetestBearSL = z.Top;
                    }

                    if (!filled)
                    {
                        if (z.Bull) data.HasActiveFvgBull = true;
                        else data.HasActiveFvgBear = true;
                    }
                }
            }

            // Clean up old FVGs
            data.FVGs.RemoveAll(z => z.Removed);
        }

        private void UpdateContext_MTF(int bip, MTFStructureData data, ISeries<double> highs, ISeries<double> lows,
                                        ISeries<double> closes, int currentBar)
        {
            // Update premium/discount zones
            if (!double.IsNaN(data.TrailUp) && !double.IsNaN(data.TrailDn))
            {
                double avg = (data.TrailUp + data.TrailDn) / 2.0;
                double premiumBottom = 0.95 * data.TrailUp + 0.05 * data.TrailDn;
                double discountTop = 0.95 * data.TrailDn + 0.05 * data.TrailUp;

                data.InPremiumZone = (closes[0] > premiumBottom);
                data.InDiscountZone = (closes[0] < discountTop);
            }
        }

        #endregion

        protected override void OnBarUpdate()
        {
            // Handle Multi-Timeframe bars first
            if (BarsInProgress > 0)
            {
                if (BarsInProgress == _m5BarsIndex && _m5Data != null)
                {
                    ProcessMTFBar_Full(BarsInProgress, _m5Data);
                }
                else if (BarsInProgress == _m15BarsIndex && _m15Data != null)
                {
                    ProcessMTFBar_Full(BarsInProgress, _m15Data);
                }
                return; // Don't process non-primary bars further
            }

            // Primary timeframe processing (original code)
            if (BarsInProgress != 0) return;
            if (CurrentBar < Math.Max(WinExt, IntWindow) + 3) return;
            if (High == null || Low == null || Open == null || Close == null) return;
            if (CurrentBar < 0 || CurrentBar >= Bars.Count) return;

            ResetDeepSeekPulses();
            ResetMtfExportSeries();
            FlushPendingM5Exports();

            if (ShowExternal)
                UpdateStructure(WinExt, false);
            if (ShowInternal)
                UpdateStructure(IntWindow, true);

            CheckBosChExt();
            if (ShowInternal)
                CheckBosChInt();

            DetectSweeps();           // liquidity sweeps (simple, non-intrusive)
            UpdateTrailingHighLow();

            MaintainOrderBlocks();    // includes OB retest pulses
            if (ShowFVG)
            {
                TryDetectFVG();
                MaintainFVGs();       // includes FVG retest pulses
            }

            if (ShowPremiumDiscount)
                DrawPremiumDiscountZones();

            UpdateDeepSeekStates();
        }
        #endregion

        #region DeepSeek helpers

        private void ResetDeepSeekPulses()
        {
            if (_extBosUpPulse != null) _extBosUpPulse[0] = false;
            if (_extBosDownPulse != null) _extBosDownPulse[0] = false;
            if (_extChochUpPulse != null) _extChochUpPulse[0] = false;
            if (_extChochDownPulse != null) _extChochDownPulse[0] = false;

            if (_intBosUpPulse != null) _intBosUpPulse[0] = false;
            if (_intBosDownPulse != null) _intBosDownPulse[0] = false;
            if (_intChochUpPulse != null) _intChochUpPulse[0] = false;
            if (_intChochDownPulse != null) _intChochDownPulse[0] = false;

            if (_sweepPrevHighPulse != null) _sweepPrevHighPulse[0] = false;
            if (_sweepPrevLowPulse != null) _sweepPrevLowPulse[0] = false;

            if (_obExtRetestBull != null) _obExtRetestBull[0] = false;
            if (_obExtRetestBear != null) _obExtRetestBear[0] = false;
            if (_fvgRetestBull != null) _fvgRetestBull[0] = false;
            if (_fvgRetestBear != null) _fvgRetestBear[0] = false;
            if (_obExtRetestBullSL != null) _obExtRetestBullSL[0] = double.NaN;
            if (_obExtRetestBearSL != null) _obExtRetestBearSL[0] = double.NaN;
            if (_fvgRetestBullSL != null) _fvgRetestBullSL[0] = double.NaN;
            if (_fvgRetestBearSL != null) _fvgRetestBearSL[0] = double.NaN;

            // Propagate swing levels from previous bar (unless updated by UpdateExtSwingHigh/Low)
            if (CurrentBar > 0)
            {
                if (_extLastSwingHigh != null) _extLastSwingHigh[0] = _extLastSwingHigh[1];
                if (_extLastSwingLow != null) _extLastSwingLow[0] = _extLastSwingLow[1];
                if (_extPrevSwingHigh != null) _extPrevSwingHigh[0] = _extPrevSwingHigh[1];
                if (_extPrevSwingLow != null) _extPrevSwingLow[0] = _extPrevSwingLow[1];
                if (_extSwingBar != null) _extSwingBar[0] = _extSwingBar[1];
                if (_extSwingPattern != null) _extSwingPattern[0] = _extSwingPattern[1];

                if (_intLastSwingHigh != null) _intLastSwingHigh[0] = _intLastSwingHigh[1];
                if (_intLastSwingLow != null) _intLastSwingLow[0] = _intLastSwingLow[1];
                if (_intPrevSwingHigh != null) _intPrevSwingHigh[0] = _intPrevSwingHigh[1];
                if (_intPrevSwingLow != null) _intPrevSwingLow[0] = _intPrevSwingLow[1];
                if (_intSwingBar != null) _intSwingBar[0] = _intSwingBar[1];
                if (_intSwingPattern != null) _intSwingPattern[0] = _intSwingPattern[1];
            }
        }

        private void ResetMtfExportSeries()
        {
            if (_m5ExtBosUpPulsePrimary != null) _m5ExtBosUpPulsePrimary[0] = false;
            if (_m5ExtBosDownPulsePrimary != null) _m5ExtBosDownPulsePrimary[0] = false;
            if (_m5ExtChochUpPulsePrimary != null) _m5ExtChochUpPulsePrimary[0] = false;
            if (_m5ExtChochDownPulsePrimary != null) _m5ExtChochDownPulsePrimary[0] = false;

            if (CurrentBar > 0)
            {
                if (_m5ExtStructureDirPrimary != null) _m5ExtStructureDirPrimary[0] = _m5ExtStructureDirPrimary[1];
            }
        }

        private void FlushPendingM5Exports()
        {
            if (!_hasPendingM5Update)
                return;

            if (Time[0] < _pendingM5Time)
                return; // avoid future leak: only publish when primary time >= M5 time

            if (_m5ExtStructureDirPrimary != null)
                _m5ExtStructureDirPrimary[0] = _pendingM5StructureDir;

            if (_pendingM5BosUpPulse && _m5ExtBosUpPulsePrimary != null)
                _m5ExtBosUpPulsePrimary[0] = true;
            if (_pendingM5BosDownPulse && _m5ExtBosDownPulsePrimary != null)
                _m5ExtBosDownPulsePrimary[0] = true;
            if (_pendingM5ChochUpPulse && _m5ExtChochUpPulsePrimary != null)
                _m5ExtChochUpPulsePrimary[0] = true;
            if (_pendingM5ChochDownPulse && _m5ExtChochDownPulsePrimary != null)
                _m5ExtChochDownPulsePrimary[0] = true;

            if (EnableLogs)
            {
                Print(string.Format("[SMC MTF M5->Primary] Flushed to primary @ {0}: BOSup={1}, BOSdn={2}, CHOCHup={3}, CHOCHdn={4}, Dir={5}",
                    Time[0],
                    _pendingM5BosUpPulse, _pendingM5BosDownPulse, _pendingM5ChochUpPulse, _pendingM5ChochDownPulse, _pendingM5StructureDir));
            }

            _pendingM5BosUpPulse = false;
            _pendingM5BosDownPulse = false;
            _pendingM5ChochUpPulse = false;
            _pendingM5ChochDownPulse = false;
            _hasPendingM5Update = false;
        }

        private void MarkExtBreakPulses(bool isUp, bool isChoch)
        {
            if (isUp)
            {
                if (isChoch && _extChochUpPulse != null) _extChochUpPulse[0] = true;
                else if (!isChoch && _extBosUpPulse != null) _extBosUpPulse[0] = true;
            }
            else
            {
                if (isChoch && _extChochDownPulse != null) _extChochDownPulse[0] = true;
                else if (!isChoch && _extBosDownPulse != null) _extBosDownPulse[0] = true;
            }
        }

        private void MarkIntBreakPulses(bool isUp, bool isChoch)
        {
            if (isUp)
            {
                if (isChoch && _intChochUpPulse != null) _intChochUpPulse[0] = true;
                else if (!isChoch && _intBosUpPulse != null) _intBosUpPulse[0] = true;
            }
            else
            {
                if (isChoch && _intChochDownPulse != null) _intChochDownPulse[0] = true;
                else if (!isChoch && _intBosDownPulse != null) _intBosDownPulse[0] = true;
            }
        }

        // Simple sweep: láº¥y strong swing external lÃ m má»‘c
        private void DetectSweeps()
        {
            if (!double.IsNaN(extHighCurr))
            {
                if (High[0] > extHighCurr && Close[0] < extHighCurr)
                    if (_sweepPrevHighPulse != null) _sweepPrevHighPulse[0] = true;
            }
            if (!double.IsNaN(extLowCurr))
            {
                if (Low[0] < extLowCurr && Close[0] > extLowCurr)
                    if (_sweepPrevLowPulse != null) _sweepPrevLowPulse[0] = true;
            }
        }

        /// <summary>
        /// Keeps the premium/discount anchors aligned with the latest confirmed external swings.
        /// </summary>
        private void UpdateTrailingHighLow()
        {
            if (!double.IsNaN(extHighCurr))
            {
                if (trailUpBar != extHighBar || double.IsNaN(trailUp))
                {
                    trailUp = extHighCurr;
                    trailUpBar = extHighBar;
                }
            }
            else if (double.IsNaN(trailUp) && !double.IsNaN(extHighLast))
            {
                trailUp = extHighLast;
            }

            if (!double.IsNaN(extLowCurr))
            {
                if (trailDnBar != extLowBar || double.IsNaN(trailDn))
                {
                    trailDn = extLowCurr;
                    trailDnBar = extLowBar;
                }
            }
            else if (double.IsNaN(trailDn) && !double.IsNaN(extLowLast))
            {
                trailDn = extLowLast;
            }
        }

        private void UpdateDeepSeekStates()
        {
            // Structure dir from biases
            if (_extStructureDir != null)
            {
                if (extBias == BULL) _extStructureDir[0] = 1;
                else if (extBias == BEAR) _extStructureDir[0] = -1;
                else _extStructureDir[0] = 0;
            }

            if (_intStructureDir != null)
            {
                if (intTrendBias == BULL) _intStructureDir[0] = 1;
                else if (intTrendBias == BEAR) _intStructureDir[0] = -1;
                else _intStructureDir[0] = 0;
            }

            bool hasObBull = false, hasObBear = false;
            if (_obZones != null && _obZones.Count > 0)
            {
                for (int i = 0; i < _obZones.Count; i++)
                {
                    OBZ z = _obZones[i];
                    if (z == null || z.Invalidated) continue;
                    if (!z.IsInternal)
                    {
                        if (z.Bull) hasObBull = true;
                        else hasObBear = true;
                    }
                }
            }
            if (_hasActiveObExtBull != null) _hasActiveObExtBull[0] = hasObBull;
            if (_hasActiveObExtBear != null) _hasActiveObExtBear[0] = hasObBear;

            bool hasFvgBull = false, hasFvgBear = false;
            if (_fvgs != null && _fvgs.Count > 0)
            {
                for (int i = 0; i < _fvgs.Count; i++)
                {
                    FVGZ z = _fvgs[i];
                    if (z == null || z.Removed) continue;
                    if (z.Bull) hasFvgBull = true;
                    else hasFvgBear = true;
                }
            }
            if (_hasActiveFvgBull != null) _hasActiveFvgBull[0] = hasFvgBull;
            if (_hasActiveFvgBear != null) _hasActiveFvgBear[0] = hasFvgBear;

            // Premium / Discount (dá»±a trÃªn trailing strong high/low)
            bool inPrem = false, inDisc = false;
            if (!double.IsNaN(trailUp) && !double.IsNaN(trailDn) && trailUp > trailDn)
            {
                double mid = 0.5 * (trailUp + trailDn);
                if (Close[0] > mid) inPrem = true;
                if (Close[0] < mid) inDisc = true;
            }
            if (_inPremiumZone != null) _inPremiumZone[0] = inPrem;
            if (_inDiscountZone != null) _inDiscountZone[0] = inDisc;
        }

        #endregion

        #region Structure & Breaks
        private void UpdateStructure(int win, bool isInternal)
        {
            int idx = win;
            if (CurrentBar <= idx + 2) return;

            var maxSeries = MAX(High, win);
            var minSeries = MIN(Low, win);
            if (maxSeries == null || minSeries == null) return;

            bool newLegHigh = High[idx] > maxSeries[0];
            bool newLegLow = Low[idx] < minSeries[0];

            int legNow = isInternal ? lastLegInt : lastLegExt;
            if (newLegHigh) legNow = BEARISH_LEG;
            else if (newLegLow) legNow = BULLISH_LEG;

            if ((isInternal ? lastLegInt : lastLegExt) == -1 && (newLegHigh || newLegLow))
            {
                if (isInternal) lastLegInt = legNow;
                else lastLegExt = legNow;
                return;
            }

            bool startOfNewLeg =
                ((isInternal ? lastLegInt : lastLegExt) != -1) &&
                (legNow != (isInternal ? lastLegInt : lastLegExt));
            if (!startOfNewLeg) return;

            if (legNow == BULLISH_LEG)
            {
                if (isInternal)
                {
                    intLowCurr = Low[idx]; intLowBar = CurrentBar - idx; intLowCrossed = false; lastLegInt = legNow;
                    if (IntLabels && ShowInternal)
                        SafeDrawText("INT_L_" + intLowBar.ToString(),
                            (double.IsNaN(intLowLast) ? "L" : (intLowCurr > intLowLast ? "HL" : "LL")),
                            intLowBar, intLowCurr, Brushes.Gray);
                    intLowPrev = intLowLast;
                    intLowLast = intLowCurr;
                }
                else
                {
                    extLowCurr = Low[idx]; extLowBar = CurrentBar - idx; extLowCrossed = false; lastLegExt = legNow;
                    if (ExtLabels && ShowExternal)
                    {
                        string lblL = double.IsNaN(extLowLast) ? "L" : (extLowCurr > extLowLast ? "HL" : "LL");
                        SafeDrawText("EXT_L_" + extLowBar.ToString(), lblL, extLowBar, extLowCurr, Brushes.DimGray);
                    }
                    extLowPrev = extLowLast;
                    extLowLast = extLowCurr;
                    trailDn = extLowCurr;
                    trailDnBar = extLowBar;

                    // Update swing tracking (DeepSeek: for ML export)
                    UpdateExtSwingLow();
                }
            }
            else
            {
                if (isInternal)
                {
                    intHighCurr = High[idx]; intHighBar = CurrentBar - idx; intHighCrossed = false; lastLegInt = legNow;
                    if (IntLabels && ShowInternal)
                        SafeDrawText("INT_H_" + intHighBar.ToString(),
                            (double.IsNaN(intHighLast) ? "H" : (intHighCurr > intHighLast ? "HH" : "LH")),
                            intHighBar, intHighCurr, Brushes.Gray);
                    intHighLast = intHighCurr;
                }
                else
                {
                    extHighCurr = High[idx]; extHighBar = CurrentBar - idx; extHighCrossed = false; lastLegExt = legNow;
                    if (ExtLabels && ShowExternal)
                    {
                        string lblH = double.IsNaN(extHighLast) ? "H" : (extHighCurr > extHighLast ? "HH" : "LH");
                        SafeDrawText("EXT_H_" + extHighBar.ToString(), lblH, extHighBar, extHighCurr, Brushes.DimGray);
                    }
                    extHighLast = extHighCurr;
                    trailUp = extHighCurr;
                    trailUpBar = extHighBar;

                    // Update swing tracking (DeepSeek: for ML export)
                    UpdateExtSwingHigh();
                }
            }
        }

        private void CheckBosChExt()
        {
            double buffer = Instrument.MasterInstrument.TickSize *
                            (ConfirmOnClose ? BosBufferTicks : Math.Max(BosBufferTicks, 0));

            if (!double.IsNaN(extHighCurr) && !extHighCrossed && Close[0] > extHighCurr + buffer)
            {
                string tagText = (extBias == BEAR ? "CHoCH" : "BOS");
                bool isChoch = tagText != "BOS";
                extBias = BULL; extHighCrossed = true;

                SafeDrawBosStyled(tagText, extHighBar, extHighCurr, Brushes.LimeGreen, false, true);
                BroadcastBreak(false, true, tagText, extHighBar, extHighCurr);
                MarkExtBreakPulses(true, isChoch);
            }

            if (!double.IsNaN(extLowCurr) && !extLowCrossed && Close[0] < extLowCurr - buffer)
            {
                string tagText = (extBias == BULL ? "CHoCH" : "BOS");
                bool isChoch = tagText != "BOS";
                extBias = BEAR; extLowCrossed = true;

                SafeDrawBosStyled(tagText, extLowBar, extLowCurr, Brushes.OrangeRed, false, false);
                BroadcastBreak(false, false, tagText, extLowBar, extLowCurr);
                MarkExtBreakPulses(false, isChoch);
            }
        }

        private void CheckBosChInt()
        {
            double buffer = Instrument.MasterInstrument.TickSize *
                            (ConfirmOnClose ? BosBufferTicks : Math.Max(BosBufferTicks, 0));

            if (!double.IsNaN(intHighCurr) && !intHighCrossed && Close[0] > intHighCurr + buffer)
            {
                string tagText = (intTrendBias == BEAR ? "CH" : "BOS");
                bool isChoch = (tagText != "BOS");
                intTrendBias = BULL; intHighCrossed = true;

                bool valid = true;
                if (IntChochBreakByBodyClose)
                {
                    double need1 = Math.Max(0, IntChochBreakBufferTicks) * TickSize;
                    if (Close[0] <= intHighCurr + need1) valid = false;
                }
                if (valid && IntChochMinBreakTicks > 0)
                {
                    double need2 = IntChochMinBreakTicks * TickSize;
                    if (Close[0] < intHighCurr + need2) valid = false;
                }
                if (valid && IntChochRequireDisplacement)
                {
                    double rng = High[0] - Low[0];
                    double atrv = (_atrDisp != null) ? _atrDisp[0] : (High[0] - Low[0]);
                    bool strongRange = rng >= IntChochDisplacementATRMult * atrv;
                    bool closeHighQ = (High[0] > Low[0])
                        ? (Close[0] >= Low[0] + 0.75 * (High[0] - Low[0]))
                        : true;
                    if (!strongRange && !closeHighQ) valid = false;
                }

                if (!(IntChochStrictAffectsDisplay && isChoch && !valid))
                    if (ShowInternal)
                        SafeDrawBosStyled(tagText, intHighBar, intHighCurr, Brushes.DarkGreen, true, true);

                BroadcastBreak(true, true, tagText, intHighBar, intHighCurr);
                MarkIntBreakPulses(true, isChoch);
            }

            if (!double.IsNaN(intLowCurr) && !intLowCrossed && Close[0] < intLowCurr - buffer)
            {
                string tagText = (intTrendBias == BULL ? "CH" : "BOS");
                bool isChoch = (tagText != "BOS");
                intTrendBias = BEAR; intLowCrossed = true;

                if (ShowInternal)
                    SafeDrawBosStyled(tagText, intLowBar, intLowCurr, Brushes.IndianRed, true, false);

                BroadcastBreak(true, false, tagText, intLowBar, intLowCurr);
                MarkIntBreakPulses(false, isChoch);
            }
        }

        private class BreakInfo
        {
            internal bool IsInternal;
            internal bool UpBreak;
            internal string Tag;
            internal int PivotBar;
            internal double Level;
        }

        private void BroadcastBreak(bool isInternal, bool upBreak, string tag, int pivotBar, double level)
        {
            BreakInfo b = new BreakInfo();
            b.IsInternal = isInternal;
            b.UpBreak = upBreak;
            b.Tag = tag;
            b.PivotBar = pivotBar;
            b.Level = level;
            TryCreateOrderBlockFromBreak(b);
        }
        #endregion

        #region Drawing
        private string GetTemplateName(bool isInternal, bool upBreak)
        {
            if (!isInternal) return upBreak ? "SMC_Ext_Up" : "SMC_Ext_Dn";
            return upBreak ? "SMC_Int_Dashed_Up" : "SMC_Int_Dashed_Dn";
        }

        private void SafeDrawBosStyled(string tagText, int pivotBarIndex, double level,
            Brush labelBrush, bool isInternal, bool upBreak)
        {
            // Skip drawing for multi-timeframe bars (M5/M15)
            if (BarsInProgress != 0) return;

            try
            {
                if ((!isInternal && !ShowExternal) || (isInternal && !ShowInternal)) return;

                string lineTag = "BOSLINE_" + tagText + "_" + pivotBarIndex.ToString();
                int startBarsAgo = Math.Max(0, CurrentBar - pivotBarIndex);
                string tpl = GetTemplateName(isInternal, upBreak);
                Draw.Line(this, lineTag, false, startBarsAgo, level, 0, level, false, tpl);

                double y = level + (upBreak ? 1 : -1) * LabelOffsetTicks * Instrument.MasterInstrument.TickSize;
                string lblTag = "BOSLBL_" + tagText + "_" + pivotBarIndex.ToString();
                int midBarsAgo = startBarsAgo / 2;

                try
                {
                    var sfSmall = new SimpleFont("Arial", 9);
                    Draw.Text(this, lblTag, false, tagText, midBarsAgo, y, 0,
                              labelBrush, sfSmall,
                              System.Windows.TextAlignment.Center,
                              null, null, 0);
                }
                catch
                {
                    Draw.Text(this, lblTag, tagText, midBarsAgo, y, labelBrush);
                }

                if (isInternal) TrackInternalLabel(lblTag);
            }
            catch { }
        }

        private void SafeDrawText(string tag, string text,
                                  int pivotAbsBarIndex, double price, Brush brush)
        {
            // Skip drawing for multi-timeframe bars (M5/M15)
            if (BarsInProgress != 0) return;

            try
            {
                int barsAgo = Math.Max(0, CurrentBar - pivotAbsBarIndex);
                Draw.Text(this, tag, text, barsAgo, price, brush);
                if (tag != null && tag.Length >= 4 && tag.Substring(0, 4) == "INT_")
                    TrackInternalLabel(tag);
            }
            catch { }
        }

        private void TrackInternalLabel(string tag)
        {
            try
            {
                _intLabelQueue.Enqueue(tag);
                if (MaxInternalLabels <= 0) return;
                while (_intLabelQueue.Count > MaxInternalLabels)
                {
                    string oldTag = _intLabelQueue.Dequeue();
                    try { RemoveDrawObject(oldTag); } catch { }
                }
            }
            catch { }
        }
        #endregion

        #region OrderBlocks
        private struct ParsedHL
        {
            internal double ParsedHigh;
            internal double ParsedLow;
            internal double TrueHigh;
            internal double TrueLow;
        }

        private ParsedHL GetParsedAtBarsAgo(int barsAgo)
        {
            ParsedHL p = new ParsedHL();
            double h = High[barsAgo];
            double l = Low[barsAgo];
            double atr = (_atr200 == null) ? double.NaN : _atr200[barsAgo];
            bool highVol = (!double.IsNaN(atr)) ? ((h - l) >= 2.0 * atr) : false;
            p.ParsedHigh = highVol ? l : h;
            p.ParsedLow = highVol ? h : l;
            p.TrueHigh = h;
            p.TrueLow = l;
            return p;
        }

        private void TryCreateOrderBlockFromBreak(BreakInfo b)
        {
            if ((b.IsInternal && !EnableInternalOrderBlocks) ||
                (!b.IsInternal && !EnableSwingOrderBlocks))
                return;

            if (b.Tag != "BOS" && b.Tag != "CH" && b.Tag != "CHoCH")
                return;

            int startAbs = Math.Max(0, b.PivotBar);
            int endAbs = CurrentBar;
            int bestAbs = -1;
            double bestVal = b.UpBreak ? double.PositiveInfinity : double.NegativeInfinity;

            for (int i = startAbs; i <= endAbs; i++)
            {
                int ba = Math.Max(0, CurrentBar - i);
                ParsedHL p = GetParsedAtBarsAgo(ba);
                if (b.UpBreak)
                {
                    if (p.ParsedLow < bestVal) { bestVal = p.ParsedLow; bestAbs = i; }
                }
                else
                {
                    if (p.ParsedHigh > bestVal) { bestVal = p.ParsedHigh; bestAbs = i; }
                }
            }

            if (bestAbs < 0)
            {
                int lb = Math.Min(OBLookbackBars, CurrentBar);
                startAbs = Math.Max(0, endAbs - lb);
                for (int i = endAbs; i >= startAbs; i--)
                {
                    int ba = Math.Max(0, CurrentBar - i);
                    ParsedHL p = GetParsedAtBarsAgo(ba);
                    if (b.UpBreak)
                    {
                        if (p.ParsedLow < bestVal) { bestVal = p.ParsedLow; bestAbs = i; }
                    }
                    else
                    {
                        if (p.ParsedHigh > bestVal) { bestVal = p.ParsedHigh; bestAbs = i; }
                    }
                }
            }
            if (bestAbs < 0) return;

            int selBA = Math.Max(0, CurrentBar - bestAbs);
            ParsedHL sel = GetParsedAtBarsAgo(selBA);

            double hi = OBUseFullCandle
                ? sel.TrueHigh
                : (b.UpBreak ? Math.Max(sel.ParsedHigh, sel.ParsedLow) : sel.ParsedHigh);
            double lo = OBUseFullCandle
                ? sel.TrueLow
                : (b.UpBreak ? sel.ParsedLow : Math.Min(sel.ParsedHigh, sel.ParsedLow));

            double buf = Instrument.MasterInstrument.TickSize * OBBufferTicks;
            double pLo = Math.Min(lo, hi) - buf;
            double pHi = Math.Max(lo, hi) + buf;

            OBZ ob = new OBZ();
            ob.IsInternal = b.IsInternal;
            ob.Bull = b.UpBreak;
            ob.SourceBar = bestAbs;
            ob.Lo = pLo;
            ob.Hi = pHi;
            ob.Mitigated = false;
            ob.Invalidated = false;
            ob.HitTop = false;
            ob.HitBottom = false;

            string kind = b.IsInternal ? "INT" : "EXT";
            ob.RectTag = "OBRECT_" + kind + "_" + (ob.Bull ? "B" : "S") + "_" + bestAbs.ToString();
            ob.LblTag = "OBLBL_" + kind + "_" + (ob.Bull ? "B" : "S") + "_" + bestAbs.ToString();

            _obZones.Add(ob);
            DrawOb(ob);
            EnforceObLimits();
        }

        private void EnforceObLimits()
        {
            try
            {
                if (_obZones == null || _obZones.Count == 0) return;
                int maxExt = MaxExtOBs;
                int maxInt = MaxIntOBs;
                if (maxExt <= 0 && maxInt <= 0) return;

                int cntExt = 0, cntInt = 0;
                int i;
                for (i = 0; i < _obZones.Count; i++)
                {
                    if (_obZones[i].IsInternal) cntInt++;
                    else cntExt++;
                }

                if (maxExt > 0 && cntExt > maxExt)
                {
                    while (cntExt > maxExt)
                    {
                        int idxOld = -1;
                        int minBar = int.MaxValue;
                        for (i = 0; i < _obZones.Count; i++)
                        {
                            if (!_obZones[i].IsInternal &&
                                _obZones[i].SourceBar < minBar)
                            {
                                minBar = _obZones[i].SourceBar;
                                idxOld = i;
                            }
                        }
                        if (idxOld >= 0)
                        {
                            RemoveOb(_obZones[idxOld]);
                            _obZones.RemoveAt(idxOld);
                            cntExt--;
                        }
                        else break;
                    }
                }

                if (maxInt > 0 && cntInt > maxInt)
                {
                    while (cntInt > maxInt)
                    {
                        int idxOld = -1;
                        int minBar = int.MaxValue;
                        for (i = 0; i < _obZones.Count; i++)
                        {
                            if (_obZones[i].IsInternal &&
                                _obZones[i].SourceBar < minBar)
                            {
                                minBar = _obZones[i].SourceBar;
                                idxOld = i;
                            }
                        }
                        if (idxOld >= 0)
                        {
                            RemoveOb(_obZones[idxOld]);
                            _obZones.RemoveAt(idxOld);
                            cntInt--;
                        }
                        else break;
                    }
                }
            }
            catch { }
        }

        private void MaintainOrderBlocks()
        {
            if (_obZones == null || _obZones.Count == 0) return;

            double obTick = (Instrument != null && Instrument.MasterInstrument != null)
                ? Instrument.MasterInstrument.TickSize
                : 0.01;
            if (obTick <= 0.0) obTick = 0.01;

            for (int i = 0; i < _obZones.Count; i++)
            {
                OBZ z = _obZones[i];
                if (z.Invalidated) continue;

                bool touch = false;
                bool invalid = false;

                double hiNow = High[0];
                double loNow = Low[0];

                if (!OBFullFillUseWicks)
                {
                    double bHi = Math.Max(Open[0], Close[0]);
                    double bLo = Math.Min(Open[0], Close[0]);
                    hiNow = bHi;
                    loNow = bLo;
                }

                double retestBuffer = obTick; // cho phép lệch tối đa 1 tick quanh mép

                if (z.Bull)
                {
                    bool hitTop = loNow <= z.Hi;
                    bool nearTop = Close[0] <= z.Hi + retestBuffer;
                    bool stayedAboveBottom = loNow > z.Lo;

                    if ((hitTop || nearTop) && stayedAboveBottom)
                        touch = true;

                    if (Close[0] < z.Lo) invalid = true;
                }
                else
                {
                    bool hitBottom = hiNow >= z.Lo;
                    bool nearBottom = Close[0] >= z.Lo - retestBuffer;
                    bool stayedBelowTop = hiNow < z.Hi;

                    if ((hitBottom || nearBottom) && stayedBelowTop)
                        touch = true;

                    if (Close[0] > z.Hi) invalid = true;
                }

                if (hiNow >= z.Hi) z.HitTop = true;
                if (loNow <= z.Lo) z.HitBottom = true;

                if (OBRemoveInternalOnFullFill && z.IsInternal && z.HitTop && z.HitBottom)
                {
                    RemoveOb(z);
                    _obZones.RemoveAt(i);
                    i--;
                    continue;
                }

                if (invalid)
                {
                    z.Invalidated = true;
                    RedrawOb(z);
                    continue;
                }

                // First mitigation = first valid retest => External OB retest pulse
                if (!z.Mitigated && touch)
                {
                    if (!z.IsInternal)
                    {
                        if (z.Bull)
                        {
                            if (_obExtRetestBull != null)
                                _obExtRetestBull[0] = true;
                            if (_obExtRetestBullSL != null)
                                _obExtRetestBullSL[0] = z.Lo - 2.0 * obTick;
                        }
                        else
                        {
                            if (_obExtRetestBear != null)
                                _obExtRetestBear[0] = true;
                            if (_obExtRetestBearSL != null)
                                _obExtRetestBearSL[0] = z.Hi + 2.0 * obTick;
                        }
                    }
                    z.Mitigated = true;
                    RedrawOb(z);
                }
            }
        }

        private void DrawOb(OBZ z)
        {
            // Skip drawing for multi-timeframe bars (M5/M15)
            if (BarsInProgress != 0) return;

            try
            {
                int startAgo = Math.Max(0, CurrentBar - z.SourceBar);
                Brush area = z.Bull ? _bullFill : _bearFill;
                Brush outline = z.Bull ? _bullOutline : _bearOutline;
                Draw.Rectangle(this, z.RectTag, false, startAgo, z.Hi, 0, z.Lo, area, outline, OBOutlineWidth);
                if (OBShowLabels)
                {
                    string text = z.Bull ? (z.IsInternal ? "OB^i" : "OB^") : (z.IsInternal ? "OBvi" : "OBv");
                    try
                    {
                        var sf = new SimpleFont("Arial", 9);
                        Draw.Text(this, z.LblTag, false, text,
                                  startAgo / 2, (z.Hi + z.Lo) * 0.5, 0,
                                  _textBrush, sf,
                                  System.Windows.TextAlignment.Center,
                                  null, null, 0);
                    }
                    catch
                    {
                        Draw.Text(this, z.LblTag, text, startAgo / 2, (z.Hi + z.Lo) * 0.5, _textBrush);
                    }
                }
            }
            catch { }
        }

        private void RedrawOb(OBZ z)
        {
            // Skip drawing for multi-timeframe bars (M5/M15)
            if (BarsInProgress != 0) return;

            try
            {
                Brush outline = z.Invalidated
                    ? _invalidOutline
                    : (z.Mitigated ? _mitigatedOutline : (z.Bull ? _bullOutline : _bearOutline));
                Brush area = z.Bull ? _bullFill : _bearFill;
                int startAgo = Math.Max(0, CurrentBar - z.SourceBar);
                RemoveDrawObject(z.RectTag);
                Draw.Rectangle(this, z.RectTag, false, startAgo, z.Hi, 0, z.Lo, area, outline, OBOutlineWidth);
            }
            catch { }
        }

        private void RemoveOb(OBZ z)
        {
            try { RemoveDrawObject(z.RectTag); RemoveDrawObject(z.LblTag); } catch { }
        }
        #endregion

        #region FVG
        private void TryDetectFVG()
        {
            if (CurrentBar < 3) return;

            double lastClose = Close[1];
            double lastOpen = Open[1];
            double last2High = High[2];
            double last2Low = Low[2];
            double currentHigh = High[0];
            double currentLow = Low[0];

            double barDeltaPct = 0.0;
            if (lastOpen != 0)
                barDeltaPct = (lastClose - lastOpen) / (lastOpen * 100.0);

            _cumAbsDeltaPct += Math.Abs(barDeltaPct);
            _cumCount = _cumCount + 1;

            double threshold = FVGAutoThreshold
                ? ((_cumCount > 0 ? (_cumAbsDeltaPct / _cumCount) : 0.0) * FVGThresholdMultiplier)
                : 0.0;

            bool bullishFVG = (currentLow > last2High) && (lastClose > last2High) && (barDeltaPct > threshold);
            bool bearishFVG = (currentHigh < last2Low) && (lastClose < last2Low) && (-barDeltaPct > threshold);

            if (bullishFVG)
                CreateFVG(true, Math.Max(currentLow, last2High), last2High);

            if (bearishFVG)
                CreateFVG(false, Math.Min(currentHigh, last2Low), last2Low);
        }

        private void CreateFVG(bool bull, double top, double bottom)
        {
            try
            {
                FVGZ up = new FVGZ();
                up.Bull = bull;
                up.SourceBar = CurrentBar - 1;
                up.Age = 0;
                up.Removed = false;
                up.Retested = false;
                up.Top = Math.Max(top, bottom);
                up.Bottom = Math.Min(top, bottom);
                up.RectTag = "FVGRECT_" + (bull ? "B" : "S") + "_" + up.SourceBar.ToString();
                up.LblTag = "FVGLBL_" + (bull ? "B" : "S") + "_" + up.SourceBar.ToString();
                _fvgs.Add(up);
                DrawFVG(up, true);
            }
            catch { }
        }

        private void MaintainFVGs()
        {
            if (_fvgs == null || _fvgs.Count == 0) return;

            double tickSize = (Instrument != null && Instrument.MasterInstrument != null)
                ? Instrument.MasterInstrument.TickSize
                : 0.01;
            if (tickSize <= 0.0) tickSize = 0.01;

            for (int i = _fvgs.Count - 1; i >= 0; i--)
            {
                FVGZ z = _fvgs[i];
                if (z.Removed) continue;

                z.Age = z.Age + 1;

                // Invalidation if completely filled beyond opposite edge
                bool invalidate = z.Bull
                    ? (Low[0] <= z.Bottom)
                    : (High[0] >= z.Top);

                if (invalidate)
                {
                    RemoveFVG(z);
                    z.Removed = true;
                    continue;
                }

                // First retest logic:
                if (!z.Retested)
                {
                    double intoPx = tickSize * Math.Max(0, FVGRetestTicksIntoZone);

                    if (z.Bull)
                    {
                        // Cho phép retest khi giá đóng cửa cách cạnh trên tối đa intoPx tick và gap vẫn còn mở
                        double upperBuffer = z.Top + intoPx;
                        bool nearTop = Close[0] <= upperBuffer;
                        bool gapStillOpen = Low[0] > z.Bottom;

                        if (nearTop && gapStillOpen)
                        {
                            if (_fvgRetestBull != null)
                                _fvgRetestBull[0] = true;
                            if (_fvgRetestBullSL != null)
                                _fvgRetestBullSL[0] = z.Bottom - 2.0 * tickSize;
                            z.Retested = true;
                        }
                    }
                    else
                    {
                        // Ngược lại cho FVG bearish: chỉ cần đóng nến trong vòng intoPx tick phía dưới cạnh dưới
                        double lowerBuffer = z.Bottom - intoPx;
                        bool nearBottom = Close[0] >= lowerBuffer;
                        bool gapStillOpen = High[0] < z.Top;

                        if (nearBottom && gapStillOpen)
                        {
                            if (_fvgRetestBear != null)
                                _fvgRetestBear[0] = true;
                            if (_fvgRetestBearSL != null)
                                _fvgRetestBearSL[0] = z.Top + 2.0 * tickSize;
                            z.Retested = true;
                        }
                    }
                }
                if (z.Age <= FVGExtendBars)
                    DrawFVG(z, true);
                else
                    DrawFVG(z, false);
            }
        }

        private void DrawFVG(FVGZ z, bool liveExtend)
        {
            // Skip drawing for multi-timeframe bars (M5/M15)
            if (BarsInProgress != 0) return;

            try
            {
                int startAgo = Math.Max(0, CurrentBar - z.SourceBar);
                int endAgo = liveExtend ? 0 : Math.Max(0, startAgo - FVGExtendBars);
                Brush area = z.Bull ? _fvgBullFill : _fvgBearFill;
                RemoveDrawObject(z.RectTag);
                Draw.Rectangle(this, z.RectTag, false, startAgo, z.Top,
                               endAgo, z.Bottom, area, Brushes.Transparent, FVGOutlineWidth);
            }
            catch { }
        }

        private void RemoveFVG(FVGZ z)
        {
            try
            {
                RemoveDrawObject(z.RectTag);
                RemoveDrawObject(z.LblTag);
            }
            catch { }
        }
        #endregion

        #region Premium/Discount Zones (visuals only; states Ä‘Ã£ tÃ­nh riÃªng)
        private void DrawPremiumDiscountZones()
        {
            // Skip drawing for multi-timeframe bars (M5/M15)
            if (BarsInProgress != 0) return;

            if (double.IsNaN(trailUp) || double.IsNaN(trailDn)) return;
            if (CurrentBar < 1) return;

            double avg = (trailUp + trailDn) / 2.0;

            double premiumBottom = 0.95 * trailUp + 0.05 * trailDn;
            double premiumTop = trailUp;

            double eqTop = 0.525 * trailUp + 0.475 * trailDn;
            double eqBottom = 0.525 * trailDn + 0.475 * trailUp;

            double discountTop = 0.95 * trailDn + 0.05 * trailUp;
            double discountBottom = trailDn;

            int startBar = Math.Max(extHighBar, extLowBar);
            if (startBar < 0) return;

            int startBarsAgo = Math.Max(0, CurrentBar - startBar);

            try
            {
                Color premiumColorWithAlpha = Color.FromArgb((byte)ZoneOpacity, PremiumColor.R, PremiumColor.G, PremiumColor.B);
                Color eqColorWithAlpha = Color.FromArgb((byte)ZoneOpacity, EquilibriumColor.R, EquilibriumColor.G, EquilibriumColor.B);
                Color discountColorWithAlpha = Color.FromArgb((byte)ZoneOpacity, DiscountColor.R, DiscountColor.G, DiscountColor.B);

                Brush premiumBrush = new SolidColorBrush(premiumColorWithAlpha); premiumBrush.Freeze();
                Brush eqBrush = new SolidColorBrush(eqColorWithAlpha); eqBrush.Freeze();
                Brush discountBrush = new SolidColorBrush(discountColorWithAlpha); discountBrush.Freeze();

                Draw.Rectangle(this, "PremiumZone", false, startBarsAgo, premiumTop, 0, premiumBottom,
                               premiumBrush, premiumBrush, 0);
                Draw.Rectangle(this, "EquilibriumZone", false, startBarsAgo, eqTop, 0, eqBottom,
                               eqBrush, eqBrush, 0);
                Draw.Rectangle(this, "DiscountZone", false, startBarsAgo, discountTop, 0, discountBottom,
                               discountBrush, discountBrush, 0);

                Color premiumTextColor = Color.FromArgb(255, PremiumColor.R, PremiumColor.G, PremiumColor.B);
                Color eqTextColor = Color.FromArgb(255, EquilibriumColor.R, EquilibriumColor.G, EquilibriumColor.B);
                Color discountTextColor = Color.FromArgb(255, DiscountColor.R, DiscountColor.G, DiscountColor.B);

                Brush premiumTextBrush = new SolidColorBrush(premiumTextColor); premiumTextBrush.Freeze();
                Brush eqTextBrush = new SolidColorBrush(eqTextColor); eqTextBrush.Freeze();
                Brush discountTextBrush = new SolidColorBrush(discountTextColor); discountTextBrush.Freeze();

                try
                {
                    var sf = new SimpleFont("Arial", 10);
                    Draw.Text(this, "PremiumLabel", false, "Premium", 0, premiumTop, 0,
                              premiumTextBrush, sf,
                              System.Windows.TextAlignment.Left,
                              Brushes.Transparent, Brushes.Transparent, 0);

                    Draw.Text(this, "EquilibriumLabel", false, "Equilibrium", 0, avg, 0,
                              eqTextBrush, sf,
                              System.Windows.TextAlignment.Left,
                              Brushes.Transparent, Brushes.Transparent, 0);

                    Draw.Text(this, "DiscountLabel", false, "Discount", 0, discountBottom, 0,
                              discountTextBrush, sf,
                              System.Windows.TextAlignment.Left,
                              Brushes.Transparent, Brushes.Transparent, 0);
                }
                catch
                {
                    Draw.Text(this, "PremiumLabel", "Premium", 0, premiumTop);
                    Draw.Text(this, "EquilibriumLabel", "Equilibrium", 0, avg);
                    Draw.Text(this, "DiscountLabel", "Discount", 0, discountBottom);
                }
            }
            catch { }
        }
        #endregion

        #region DeepSeek public outputs

        [Browsable(false), XmlIgnore]
        public Series<bool> ExtBosUpPulse { get { return _extBosUpPulse; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> ExtBosDownPulse { get { return _extBosDownPulse; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> ExtChochUpPulse { get { return _extChochUpPulse; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> ExtChochDownPulse { get { return _extChochDownPulse; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> IntBosUpPulse { get { return _intBosUpPulse; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> IntBosDownPulse { get { return _intBosDownPulse; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> IntChochUpPulse { get { return _intChochUpPulse; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> IntChochDownPulse { get { return _intChochDownPulse; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> SweepPrevHighPulse { get { return _sweepPrevHighPulse; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> SweepPrevLowPulse { get { return _sweepPrevLowPulse; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> ObExtRetestBull { get { return _obExtRetestBull; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> ObExtRetestBear { get { return _obExtRetestBear; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> FvgRetestBull { get { return _fvgRetestBull; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> FvgRetestBear { get { return _fvgRetestBear; } }

        [Browsable(false), XmlIgnore]
        public Series<double> ObExtRetestBullSL { get { return _obExtRetestBullSL; } }

        [Browsable(false), XmlIgnore]
        public Series<double> ObExtRetestBearSL { get { return _obExtRetestBearSL; } }

        [Browsable(false), XmlIgnore]
        public Series<double> FvgRetestBullSL { get { return _fvgRetestBullSL; } }

        [Browsable(false), XmlIgnore]
        public Series<double> FvgRetestBearSL { get { return _fvgRetestBearSL; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> HasActiveObExtBull { get { return _hasActiveObExtBull; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> HasActiveObExtBear { get { return _hasActiveObExtBear; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> HasActiveFvgBull { get { return _hasActiveFvgBull; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> HasActiveFvgBear { get { return _hasActiveFvgBear; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> InPremiumZone { get { return _inPremiumZone; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> InDiscountZone { get { return _inDiscountZone; } }

        [Browsable(false), XmlIgnore]
        public Series<int> ExtStructureDir { get { return _extStructureDir; } }

        [Browsable(false), XmlIgnore]
        public Series<int> IntStructureDir { get { return _intStructureDir; } }

        // External Swing Levels
        [Browsable(false), XmlIgnore]
        public Series<double> ExtLastSwingHigh { get { return _extLastSwingHigh; } }

        [Browsable(false), XmlIgnore]
        public Series<double> ExtLastSwingLow { get { return _extLastSwingLow; } }

        [Browsable(false), XmlIgnore]
        public Series<double> ExtPrevSwingHigh { get { return _extPrevSwingHigh; } }

        [Browsable(false), XmlIgnore]
        public Series<double> ExtPrevSwingLow { get { return _extPrevSwingLow; } }

        [Browsable(false), XmlIgnore]
        public Series<int> ExtSwingBar { get { return _extSwingBar; } }

        [Browsable(false), XmlIgnore]
        public Series<int> ExtSwingPattern { get { return _extSwingPattern; } }

        // Internal Swing Levels
        [Browsable(false), XmlIgnore]
        public Series<double> IntLastSwingHigh { get { return _intLastSwingHigh; } }

        [Browsable(false), XmlIgnore]
        public Series<double> IntLastSwingLow { get { return _intLastSwingLow; } }

        [Browsable(false), XmlIgnore]
        public Series<double> IntPrevSwingHigh { get { return _intPrevSwingHigh; } }

        [Browsable(false), XmlIgnore]
        public Series<double> IntPrevSwingLow { get { return _intPrevSwingLow; } }

        [Browsable(false), XmlIgnore]
        public Series<int> IntSwingBar { get { return _intSwingBar; } }

        [Browsable(false), XmlIgnore]
        public Series<int> IntSwingPattern { get { return _intSwingPattern; } }

        // ===== Multi-Timeframe Data Access (M5, M15) - COMPREHENSIVE =====

        [Browsable(false), XmlIgnore]
        public bool HasM5Data { get { return _m5Data != null; } }

        [Browsable(false), XmlIgnore]
        public bool HasM15Data { get { return _m15Data != null; } }

        // ===== M5 STRUCTURE DATA =====

        // Structure Direction (persistent state - every bar)
        [Browsable(false), XmlIgnore]
        public int M5_StructureDir { get { return _m5Data != null ? _m5Data.StructureDir : 0; } }

        // BOS/CHoCH Pulses (one-time signals - only TRUE on break bar)
        [Browsable(false), XmlIgnore]
        public bool M5_BosUpPulse { get { return _m5Data != null && _m5Data.BosUpPulse; } }

        [Browsable(false), XmlIgnore]
        public bool M5_BosDownPulse { get { return _m5Data != null && _m5Data.BosDownPulse; } }

        [Browsable(false), XmlIgnore]
        public bool M5_ChochUpPulse { get { return _m5Data != null && _m5Data.ChochUpPulse; } }

        [Browsable(false), XmlIgnore]
        public bool M5_ChochDownPulse { get { return _m5Data != null && _m5Data.ChochDownPulse; } }

        // Swing Levels (persistent - every bar)
        [Browsable(false), XmlIgnore]
        public double M5_LastSwingHigh { get { return _m5Data != null ? _m5Data.LastSwingHigh : double.NaN; } }

        [Browsable(false), XmlIgnore]
        public double M5_LastSwingLow { get { return _m5Data != null ? _m5Data.LastSwingLow : double.NaN; } }

        [Browsable(false), XmlIgnore]
        public int M5_BarsInCurrentSwing { get { return _m5Data != null ? _m5Data.BarsInCurrentSwing : 0; } }

        // OrderBlock States (persistent - every bar)
        [Browsable(false), XmlIgnore]
        public bool M5_HasActiveBullOB { get { return _m5Data != null && _m5Data.HasActiveObExtBull; } }

        [Browsable(false), XmlIgnore]
        public bool M5_HasActiveBearOB { get { return _m5Data != null && _m5Data.HasActiveObExtBear; } }

        // OB Retest Pulses (one-time signals)
        [Browsable(false), XmlIgnore]
        public bool M5_OBRetestBullPulse { get { return _m5Data != null && _m5Data.OBRetestBullPulse; } }

        [Browsable(false), XmlIgnore]
        public bool M5_OBRetestBearPulse { get { return _m5Data != null && _m5Data.OBRetestBearPulse; } }

        [Browsable(false), XmlIgnore]
        public double M5_OBRetestBullSL { get { return _m5Data != null ? _m5Data.OBRetestBullSL : double.NaN; } }

        [Browsable(false), XmlIgnore]
        public double M5_OBRetestBearSL { get { return _m5Data != null ? _m5Data.OBRetestBearSL : double.NaN; } }

        // FVG States (persistent - every bar)
        [Browsable(false), XmlIgnore]
        public bool M5_HasActiveBullFVG { get { return _m5Data != null && _m5Data.HasActiveFvgBull; } }

        [Browsable(false), XmlIgnore]
        public bool M5_HasActiveBearFVG { get { return _m5Data != null && _m5Data.HasActiveFvgBear; } }

        // FVG Retest Pulses (one-time signals)
        [Browsable(false), XmlIgnore]
        public bool M5_FVGRetestBullPulse { get { return _m5Data != null && _m5Data.FVGRetestBullPulse; } }

        [Browsable(false), XmlIgnore]
        public bool M5_FVGRetestBearPulse { get { return _m5Data != null && _m5Data.FVGRetestBearPulse; } }

        [Browsable(false), XmlIgnore]
        public double M5_FVGRetestBullSL { get { return _m5Data != null ? _m5Data.FVGRetestBullSL : double.NaN; } }

        [Browsable(false), XmlIgnore]
        public double M5_FVGRetestBearSL { get { return _m5Data != null ? _m5Data.FVGRetestBearSL : double.NaN; } }

        // M5 exports mapped to primary bars (for exporter/ML on lower TF charts)
        [Browsable(false), XmlIgnore]
        public Series<bool> M5_BosUpPulseSeries { get { return _m5ExtBosUpPulsePrimary; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> M5_BosDownPulseSeries { get { return _m5ExtBosDownPulsePrimary; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> M5_ChochUpPulseSeries { get { return _m5ExtChochUpPulsePrimary; } }

        [Browsable(false), XmlIgnore]
        public Series<bool> M5_ChochDownPulseSeries { get { return _m5ExtChochDownPulsePrimary; } }

        [Browsable(false), XmlIgnore]
        public Series<int> M5_StructureDirSeries { get { return _m5ExtStructureDirPrimary; } }

        // Premium/Discount Context (persistent - every bar)
        [Browsable(false), XmlIgnore]
        public bool M5_InPremiumZone { get { return _m5Data != null && _m5Data.InPremiumZone; } }

        [Browsable(false), XmlIgnore]
        public bool M5_InDiscountZone { get { return _m5Data != null && _m5Data.InDiscountZone; } }

        // ===== M15 STRUCTURE DATA =====

        // Structure Direction (persistent state - every bar)
        [Browsable(false), XmlIgnore]
        public int M15_StructureDir { get { return _m15Data != null ? _m15Data.StructureDir : 0; } }

        // BOS/CHoCH Pulses (one-time signals - only TRUE on break bar)
        [Browsable(false), XmlIgnore]
        public bool M15_BosUpPulse { get { return _m15Data != null && _m15Data.BosUpPulse; } }

        [Browsable(false), XmlIgnore]
        public bool M15_BosDownPulse { get { return _m15Data != null && _m15Data.BosDownPulse; } }

        [Browsable(false), XmlIgnore]
        public bool M15_ChochUpPulse { get { return _m15Data != null && _m15Data.ChochUpPulse; } }

        [Browsable(false), XmlIgnore]
        public bool M15_ChochDownPulse { get { return _m15Data != null && _m15Data.ChochDownPulse; } }

        // Swing Levels (persistent - every bar)
        [Browsable(false), XmlIgnore]
        public double M15_LastSwingHigh { get { return _m15Data != null ? _m15Data.LastSwingHigh : double.NaN; } }

        [Browsable(false), XmlIgnore]
        public double M15_LastSwingLow { get { return _m15Data != null ? _m15Data.LastSwingLow : double.NaN; } }

        [Browsable(false), XmlIgnore]
        public int M15_BarsInCurrentSwing { get { return _m15Data != null ? _m15Data.BarsInCurrentSwing : 0; } }

        // OrderBlock States (persistent - every bar)
        [Browsable(false), XmlIgnore]
        public bool M15_HasActiveBullOB { get { return _m15Data != null && _m15Data.HasActiveObExtBull; } }

        [Browsable(false), XmlIgnore]
        public bool M15_HasActiveBearOB { get { return _m15Data != null && _m15Data.HasActiveObExtBear; } }

        // OB Retest Pulses (one-time signals)
        [Browsable(false), XmlIgnore]
        public bool M15_OBRetestBullPulse { get { return _m15Data != null && _m15Data.OBRetestBullPulse; } }

        [Browsable(false), XmlIgnore]
        public bool M15_OBRetestBearPulse { get { return _m15Data != null && _m15Data.OBRetestBearPulse; } }

        [Browsable(false), XmlIgnore]
        public double M15_OBRetestBullSL { get { return _m15Data != null ? _m15Data.OBRetestBullSL : double.NaN; } }

        [Browsable(false), XmlIgnore]
        public double M15_OBRetestBearSL { get { return _m15Data != null ? _m15Data.OBRetestBearSL : double.NaN; } }

        // FVG States (persistent - every bar)
        [Browsable(false), XmlIgnore]
        public bool M15_HasActiveBullFVG { get { return _m15Data != null && _m15Data.HasActiveFvgBull; } }

        [Browsable(false), XmlIgnore]
        public bool M15_HasActiveBearFVG { get { return _m15Data != null && _m15Data.HasActiveFvgBear; } }

        // FVG Retest Pulses (one-time signals)
        [Browsable(false), XmlIgnore]
        public bool M15_FVGRetestBullPulse { get { return _m15Data != null && _m15Data.FVGRetestBullPulse; } }

        [Browsable(false), XmlIgnore]
        public bool M15_FVGRetestBearPulse { get { return _m15Data != null && _m15Data.FVGRetestBearPulse; } }

        [Browsable(false), XmlIgnore]
        public double M15_FVGRetestBullSL { get { return _m15Data != null ? _m15Data.FVGRetestBullSL : double.NaN; } }

        [Browsable(false), XmlIgnore]
        public double M15_FVGRetestBearSL { get { return _m15Data != null ? _m15Data.FVGRetestBearSL : double.NaN; } }

        // Premium/Discount Context (persistent - every bar)
        [Browsable(false), XmlIgnore]
        public bool M15_InPremiumZone { get { return _m15Data != null && _m15Data.InPremiumZone; } }

        [Browsable(false), XmlIgnore]
        public bool M15_InDiscountZone { get { return _m15Data != null && _m15Data.InDiscountZone; } }

        #endregion

        #region Swing Tracking Helpers (DeepSeek ML export)

        private void UpdateExtSwingHigh()
        {
            // Update previous swing high
            if (!double.IsNaN(extHighLast) && extHighLast != extHighCurr)
                _extPrevSwingHigh[0] = extHighLast;
            else if (CurrentBar > 0)
                _extPrevSwingHigh[0] = _extPrevSwingHigh[1];

            // Update current swing high
            _extLastSwingHigh[0] = extHighCurr;
            _extSwingBar[0] = extHighBar;

            // Classify pattern: HH=1, LH=-2
            if (!double.IsNaN(extHighLast))
            {
                if (extHighCurr > extHighLast)
                    _extSwingPattern[0] = 1; // HH (Higher High)
                else
                    _extSwingPattern[0] = -2; // LH (Lower High)
            }
            else
            {
                _extSwingPattern[0] = 0; // First swing, undefined
            }
        }

        private void UpdateExtSwingLow()
        {
            // Update previous swing low
            if (!double.IsNaN(extLowLast) && extLowLast != extLowCurr)
                _extPrevSwingLow[0] = extLowLast;
            else if (CurrentBar > 0)
                _extPrevSwingLow[0] = _extPrevSwingLow[1];

            // Update current swing low
            _extLastSwingLow[0] = extLowCurr;
            _extSwingBar[0] = extLowBar;

            // Classify pattern: LL=-1, HL=2
            if (!double.IsNaN(extLowLast))
            {
                if (extLowCurr < extLowLast)
                    _extSwingPattern[0] = -1; // LL (Lower Low)
                else
                    _extSwingPattern[0] = 2; // HL (Higher Low)
            }
            else
            {
                _extSwingPattern[0] = 0; // First swing, undefined
            }
        }

        #endregion
    }
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private SMC_Structure_OB_Only_v12_FVG_CHOCHFlags[] cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags;
		public SMC_Structure_OB_Only_v12_FVG_CHOCHFlags SMC_Structure_OB_Only_v12_FVG_CHOCHFlags(int bosBufferTicks, int labelOffsetTicks, bool confirmOnClose, bool enableLogs, int winExt, bool showExternal, bool extLabels, bool enableSwingOrderBlocks, int maxExtOBs, int intWindow, bool showInternal, bool intLabels, int maxInternalLabels, bool enableInternalOrderBlocks, int maxIntOBs, bool oBRemoveInternalOnFullFill, bool intChochBreakByBodyClose, int intChochBreakBufferTicks, bool intChochRequireDisplacement, int intChochDisplacementATRPeriod, double intChochDisplacementATRMult, int intChochMinBreakTicks, bool intChochStrictAffectsDisplay, int oBLookbackBars, int oBBufferTicks, bool oBShowLabels, bool oBUseFullCandle, int oBFillOpacity, int oBOutlineWidth, bool oBFullFillUseWicks, bool showFVG, bool fVGAutoThreshold, double fVGThresholdMultiplier, int fVGExtendBars, int fVGFillOpacity, int fVGOutlineWidth, System.Windows.Media.Color fVGBullColor, System.Windows.Media.Color fVGBearColor, int fVGRetestTicksIntoZone, bool showPremiumDiscount, System.Windows.Media.Color premiumColor, System.Windows.Media.Color equilibriumColor, System.Windows.Media.Color discountColor, int zoneOpacity, bool enableM5Structure, bool enableM15Structure)
		{
			return SMC_Structure_OB_Only_v12_FVG_CHOCHFlags(Input, bosBufferTicks, labelOffsetTicks, confirmOnClose, enableLogs, winExt, showExternal, extLabels, enableSwingOrderBlocks, maxExtOBs, intWindow, showInternal, intLabels, maxInternalLabels, enableInternalOrderBlocks, maxIntOBs, oBRemoveInternalOnFullFill, intChochBreakByBodyClose, intChochBreakBufferTicks, intChochRequireDisplacement, intChochDisplacementATRPeriod, intChochDisplacementATRMult, intChochMinBreakTicks, intChochStrictAffectsDisplay, oBLookbackBars, oBBufferTicks, oBShowLabels, oBUseFullCandle, oBFillOpacity, oBOutlineWidth, oBFullFillUseWicks, showFVG, fVGAutoThreshold, fVGThresholdMultiplier, fVGExtendBars, fVGFillOpacity, fVGOutlineWidth, fVGBullColor, fVGBearColor, fVGRetestTicksIntoZone, showPremiumDiscount, premiumColor, equilibriumColor, discountColor, zoneOpacity, enableM5Structure, enableM15Structure);
		}

		public SMC_Structure_OB_Only_v12_FVG_CHOCHFlags SMC_Structure_OB_Only_v12_FVG_CHOCHFlags(ISeries<double> input, int bosBufferTicks, int labelOffsetTicks, bool confirmOnClose, bool enableLogs, int winExt, bool showExternal, bool extLabels, bool enableSwingOrderBlocks, int maxExtOBs, int intWindow, bool showInternal, bool intLabels, int maxInternalLabels, bool enableInternalOrderBlocks, int maxIntOBs, bool oBRemoveInternalOnFullFill, bool intChochBreakByBodyClose, int intChochBreakBufferTicks, bool intChochRequireDisplacement, int intChochDisplacementATRPeriod, double intChochDisplacementATRMult, int intChochMinBreakTicks, bool intChochStrictAffectsDisplay, int oBLookbackBars, int oBBufferTicks, bool oBShowLabels, bool oBUseFullCandle, int oBFillOpacity, int oBOutlineWidth, bool oBFullFillUseWicks, bool showFVG, bool fVGAutoThreshold, double fVGThresholdMultiplier, int fVGExtendBars, int fVGFillOpacity, int fVGOutlineWidth, System.Windows.Media.Color fVGBullColor, System.Windows.Media.Color fVGBearColor, int fVGRetestTicksIntoZone, bool showPremiumDiscount, System.Windows.Media.Color premiumColor, System.Windows.Media.Color equilibriumColor, System.Windows.Media.Color discountColor, int zoneOpacity, bool enableM5Structure, bool enableM15Structure)
		{
			if (cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags != null)
				for (int idx = 0; idx < cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags.Length; idx++)
					if (cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx] != null && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].BosBufferTicks == bosBufferTicks && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].LabelOffsetTicks == labelOffsetTicks && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].ConfirmOnClose == confirmOnClose && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].EnableLogs == enableLogs && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].WinExt == winExt && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].ShowExternal == showExternal && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].ExtLabels == extLabels && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].EnableSwingOrderBlocks == enableSwingOrderBlocks && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].MaxExtOBs == maxExtOBs && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].IntWindow == intWindow && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].ShowInternal == showInternal && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].IntLabels == intLabels && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].MaxInternalLabels == maxInternalLabels && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].EnableInternalOrderBlocks == enableInternalOrderBlocks && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].MaxIntOBs == maxIntOBs && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].OBRemoveInternalOnFullFill == oBRemoveInternalOnFullFill && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].IntChochBreakByBodyClose == intChochBreakByBodyClose && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].IntChochBreakBufferTicks == intChochBreakBufferTicks && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].IntChochRequireDisplacement == intChochRequireDisplacement && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].IntChochDisplacementATRPeriod == intChochDisplacementATRPeriod && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].IntChochDisplacementATRMult == intChochDisplacementATRMult && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].IntChochMinBreakTicks == intChochMinBreakTicks && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].IntChochStrictAffectsDisplay == intChochStrictAffectsDisplay && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].OBLookbackBars == oBLookbackBars && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].OBBufferTicks == oBBufferTicks && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].OBShowLabels == oBShowLabels && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].OBUseFullCandle == oBUseFullCandle && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].OBFillOpacity == oBFillOpacity && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].OBOutlineWidth == oBOutlineWidth && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].OBFullFillUseWicks == oBFullFillUseWicks && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].ShowFVG == showFVG && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].FVGAutoThreshold == fVGAutoThreshold && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].FVGThresholdMultiplier == fVGThresholdMultiplier && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].FVGExtendBars == fVGExtendBars && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].FVGFillOpacity == fVGFillOpacity && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].FVGOutlineWidth == fVGOutlineWidth && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].FVGBullColor == fVGBullColor && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].FVGBearColor == fVGBearColor && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].FVGRetestTicksIntoZone == fVGRetestTicksIntoZone && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].ShowPremiumDiscount == showPremiumDiscount && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].PremiumColor == premiumColor && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].EquilibriumColor == equilibriumColor && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].DiscountColor == discountColor && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].ZoneOpacity == zoneOpacity && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].EnableM5Structure == enableM5Structure && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].EnableM15Structure == enableM15Structure && cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx].EqualsInput(input))
						return cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags[idx];
			return CacheIndicator<SMC_Structure_OB_Only_v12_FVG_CHOCHFlags>(new SMC_Structure_OB_Only_v12_FVG_CHOCHFlags(){ BosBufferTicks = bosBufferTicks, LabelOffsetTicks = labelOffsetTicks, ConfirmOnClose = confirmOnClose, EnableLogs = enableLogs, WinExt = winExt, ShowExternal = showExternal, ExtLabels = extLabels, EnableSwingOrderBlocks = enableSwingOrderBlocks, MaxExtOBs = maxExtOBs, IntWindow = intWindow, ShowInternal = showInternal, IntLabels = intLabels, MaxInternalLabels = maxInternalLabels, EnableInternalOrderBlocks = enableInternalOrderBlocks, MaxIntOBs = maxIntOBs, OBRemoveInternalOnFullFill = oBRemoveInternalOnFullFill, IntChochBreakByBodyClose = intChochBreakByBodyClose, IntChochBreakBufferTicks = intChochBreakBufferTicks, IntChochRequireDisplacement = intChochRequireDisplacement, IntChochDisplacementATRPeriod = intChochDisplacementATRPeriod, IntChochDisplacementATRMult = intChochDisplacementATRMult, IntChochMinBreakTicks = intChochMinBreakTicks, IntChochStrictAffectsDisplay = intChochStrictAffectsDisplay, OBLookbackBars = oBLookbackBars, OBBufferTicks = oBBufferTicks, OBShowLabels = oBShowLabels, OBUseFullCandle = oBUseFullCandle, OBFillOpacity = oBFillOpacity, OBOutlineWidth = oBOutlineWidth, OBFullFillUseWicks = oBFullFillUseWicks, ShowFVG = showFVG, FVGAutoThreshold = fVGAutoThreshold, FVGThresholdMultiplier = fVGThresholdMultiplier, FVGExtendBars = fVGExtendBars, FVGFillOpacity = fVGFillOpacity, FVGOutlineWidth = fVGOutlineWidth, FVGBullColor = fVGBullColor, FVGBearColor = fVGBearColor, FVGRetestTicksIntoZone = fVGRetestTicksIntoZone, ShowPremiumDiscount = showPremiumDiscount, PremiumColor = premiumColor, EquilibriumColor = equilibriumColor, DiscountColor = discountColor, ZoneOpacity = zoneOpacity, EnableM5Structure = enableM5Structure, EnableM15Structure = enableM15Structure }, input, ref cacheSMC_Structure_OB_Only_v12_FVG_CHOCHFlags);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.SMC_Structure_OB_Only_v12_FVG_CHOCHFlags SMC_Structure_OB_Only_v12_FVG_CHOCHFlags(int bosBufferTicks, int labelOffsetTicks, bool confirmOnClose, bool enableLogs, int winExt, bool showExternal, bool extLabels, bool enableSwingOrderBlocks, int maxExtOBs, int intWindow, bool showInternal, bool intLabels, int maxInternalLabels, bool enableInternalOrderBlocks, int maxIntOBs, bool oBRemoveInternalOnFullFill, bool intChochBreakByBodyClose, int intChochBreakBufferTicks, bool intChochRequireDisplacement, int intChochDisplacementATRPeriod, double intChochDisplacementATRMult, int intChochMinBreakTicks, bool intChochStrictAffectsDisplay, int oBLookbackBars, int oBBufferTicks, bool oBShowLabels, bool oBUseFullCandle, int oBFillOpacity, int oBOutlineWidth, bool oBFullFillUseWicks, bool showFVG, bool fVGAutoThreshold, double fVGThresholdMultiplier, int fVGExtendBars, int fVGFillOpacity, int fVGOutlineWidth, System.Windows.Media.Color fVGBullColor, System.Windows.Media.Color fVGBearColor, int fVGRetestTicksIntoZone, bool showPremiumDiscount, System.Windows.Media.Color premiumColor, System.Windows.Media.Color equilibriumColor, System.Windows.Media.Color discountColor, int zoneOpacity, bool enableM5Structure, bool enableM15Structure)
		{
			return indicator.SMC_Structure_OB_Only_v12_FVG_CHOCHFlags(Input, bosBufferTicks, labelOffsetTicks, confirmOnClose, enableLogs, winExt, showExternal, extLabels, enableSwingOrderBlocks, maxExtOBs, intWindow, showInternal, intLabels, maxInternalLabels, enableInternalOrderBlocks, maxIntOBs, oBRemoveInternalOnFullFill, intChochBreakByBodyClose, intChochBreakBufferTicks, intChochRequireDisplacement, intChochDisplacementATRPeriod, intChochDisplacementATRMult, intChochMinBreakTicks, intChochStrictAffectsDisplay, oBLookbackBars, oBBufferTicks, oBShowLabels, oBUseFullCandle, oBFillOpacity, oBOutlineWidth, oBFullFillUseWicks, showFVG, fVGAutoThreshold, fVGThresholdMultiplier, fVGExtendBars, fVGFillOpacity, fVGOutlineWidth, fVGBullColor, fVGBearColor, fVGRetestTicksIntoZone, showPremiumDiscount, premiumColor, equilibriumColor, discountColor, zoneOpacity, enableM5Structure, enableM15Structure);
		}

		public Indicators.SMC_Structure_OB_Only_v12_FVG_CHOCHFlags SMC_Structure_OB_Only_v12_FVG_CHOCHFlags(ISeries<double> input , int bosBufferTicks, int labelOffsetTicks, bool confirmOnClose, bool enableLogs, int winExt, bool showExternal, bool extLabels, bool enableSwingOrderBlocks, int maxExtOBs, int intWindow, bool showInternal, bool intLabels, int maxInternalLabels, bool enableInternalOrderBlocks, int maxIntOBs, bool oBRemoveInternalOnFullFill, bool intChochBreakByBodyClose, int intChochBreakBufferTicks, bool intChochRequireDisplacement, int intChochDisplacementATRPeriod, double intChochDisplacementATRMult, int intChochMinBreakTicks, bool intChochStrictAffectsDisplay, int oBLookbackBars, int oBBufferTicks, bool oBShowLabels, bool oBUseFullCandle, int oBFillOpacity, int oBOutlineWidth, bool oBFullFillUseWicks, bool showFVG, bool fVGAutoThreshold, double fVGThresholdMultiplier, int fVGExtendBars, int fVGFillOpacity, int fVGOutlineWidth, System.Windows.Media.Color fVGBullColor, System.Windows.Media.Color fVGBearColor, int fVGRetestTicksIntoZone, bool showPremiumDiscount, System.Windows.Media.Color premiumColor, System.Windows.Media.Color equilibriumColor, System.Windows.Media.Color discountColor, int zoneOpacity, bool enableM5Structure, bool enableM15Structure)
		{
			return indicator.SMC_Structure_OB_Only_v12_FVG_CHOCHFlags(input, bosBufferTicks, labelOffsetTicks, confirmOnClose, enableLogs, winExt, showExternal, extLabels, enableSwingOrderBlocks, maxExtOBs, intWindow, showInternal, intLabels, maxInternalLabels, enableInternalOrderBlocks, maxIntOBs, oBRemoveInternalOnFullFill, intChochBreakByBodyClose, intChochBreakBufferTicks, intChochRequireDisplacement, intChochDisplacementATRPeriod, intChochDisplacementATRMult, intChochMinBreakTicks, intChochStrictAffectsDisplay, oBLookbackBars, oBBufferTicks, oBShowLabels, oBUseFullCandle, oBFillOpacity, oBOutlineWidth, oBFullFillUseWicks, showFVG, fVGAutoThreshold, fVGThresholdMultiplier, fVGExtendBars, fVGFillOpacity, fVGOutlineWidth, fVGBullColor, fVGBearColor, fVGRetestTicksIntoZone, showPremiumDiscount, premiumColor, equilibriumColor, discountColor, zoneOpacity, enableM5Structure, enableM15Structure);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.SMC_Structure_OB_Only_v12_FVG_CHOCHFlags SMC_Structure_OB_Only_v12_FVG_CHOCHFlags(int bosBufferTicks, int labelOffsetTicks, bool confirmOnClose, bool enableLogs, int winExt, bool showExternal, bool extLabels, bool enableSwingOrderBlocks, int maxExtOBs, int intWindow, bool showInternal, bool intLabels, int maxInternalLabels, bool enableInternalOrderBlocks, int maxIntOBs, bool oBRemoveInternalOnFullFill, bool intChochBreakByBodyClose, int intChochBreakBufferTicks, bool intChochRequireDisplacement, int intChochDisplacementATRPeriod, double intChochDisplacementATRMult, int intChochMinBreakTicks, bool intChochStrictAffectsDisplay, int oBLookbackBars, int oBBufferTicks, bool oBShowLabels, bool oBUseFullCandle, int oBFillOpacity, int oBOutlineWidth, bool oBFullFillUseWicks, bool showFVG, bool fVGAutoThreshold, double fVGThresholdMultiplier, int fVGExtendBars, int fVGFillOpacity, int fVGOutlineWidth, System.Windows.Media.Color fVGBullColor, System.Windows.Media.Color fVGBearColor, int fVGRetestTicksIntoZone, bool showPremiumDiscount, System.Windows.Media.Color premiumColor, System.Windows.Media.Color equilibriumColor, System.Windows.Media.Color discountColor, int zoneOpacity, bool enableM5Structure, bool enableM15Structure)
		{
			return indicator.SMC_Structure_OB_Only_v12_FVG_CHOCHFlags(Input, bosBufferTicks, labelOffsetTicks, confirmOnClose, enableLogs, winExt, showExternal, extLabels, enableSwingOrderBlocks, maxExtOBs, intWindow, showInternal, intLabels, maxInternalLabels, enableInternalOrderBlocks, maxIntOBs, oBRemoveInternalOnFullFill, intChochBreakByBodyClose, intChochBreakBufferTicks, intChochRequireDisplacement, intChochDisplacementATRPeriod, intChochDisplacementATRMult, intChochMinBreakTicks, intChochStrictAffectsDisplay, oBLookbackBars, oBBufferTicks, oBShowLabels, oBUseFullCandle, oBFillOpacity, oBOutlineWidth, oBFullFillUseWicks, showFVG, fVGAutoThreshold, fVGThresholdMultiplier, fVGExtendBars, fVGFillOpacity, fVGOutlineWidth, fVGBullColor, fVGBearColor, fVGRetestTicksIntoZone, showPremiumDiscount, premiumColor, equilibriumColor, discountColor, zoneOpacity, enableM5Structure, enableM15Structure);
		}

		public Indicators.SMC_Structure_OB_Only_v12_FVG_CHOCHFlags SMC_Structure_OB_Only_v12_FVG_CHOCHFlags(ISeries<double> input , int bosBufferTicks, int labelOffsetTicks, bool confirmOnClose, bool enableLogs, int winExt, bool showExternal, bool extLabels, bool enableSwingOrderBlocks, int maxExtOBs, int intWindow, bool showInternal, bool intLabels, int maxInternalLabels, bool enableInternalOrderBlocks, int maxIntOBs, bool oBRemoveInternalOnFullFill, bool intChochBreakByBodyClose, int intChochBreakBufferTicks, bool intChochRequireDisplacement, int intChochDisplacementATRPeriod, double intChochDisplacementATRMult, int intChochMinBreakTicks, bool intChochStrictAffectsDisplay, int oBLookbackBars, int oBBufferTicks, bool oBShowLabels, bool oBUseFullCandle, int oBFillOpacity, int oBOutlineWidth, bool oBFullFillUseWicks, bool showFVG, bool fVGAutoThreshold, double fVGThresholdMultiplier, int fVGExtendBars, int fVGFillOpacity, int fVGOutlineWidth, System.Windows.Media.Color fVGBullColor, System.Windows.Media.Color fVGBearColor, int fVGRetestTicksIntoZone, bool showPremiumDiscount, System.Windows.Media.Color premiumColor, System.Windows.Media.Color equilibriumColor, System.Windows.Media.Color discountColor, int zoneOpacity, bool enableM5Structure, bool enableM15Structure)
		{
			return indicator.SMC_Structure_OB_Only_v12_FVG_CHOCHFlags(input, bosBufferTicks, labelOffsetTicks, confirmOnClose, enableLogs, winExt, showExternal, extLabels, enableSwingOrderBlocks, maxExtOBs, intWindow, showInternal, intLabels, maxInternalLabels, enableInternalOrderBlocks, maxIntOBs, oBRemoveInternalOnFullFill, intChochBreakByBodyClose, intChochBreakBufferTicks, intChochRequireDisplacement, intChochDisplacementATRPeriod, intChochDisplacementATRMult, intChochMinBreakTicks, intChochStrictAffectsDisplay, oBLookbackBars, oBBufferTicks, oBShowLabels, oBUseFullCandle, oBFillOpacity, oBOutlineWidth, oBFullFillUseWicks, showFVG, fVGAutoThreshold, fVGThresholdMultiplier, fVGExtendBars, fVGFillOpacity, fVGOutlineWidth, fVGBullColor, fVGBearColor, fVGRetestTicksIntoZone, showPremiumDiscount, premiumColor, equilibriumColor, discountColor, zoneOpacity, enableM5Structure, enableM15Structure);
		}
	}
}

#endregion
