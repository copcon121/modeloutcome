#region Using declarations
using System;
using System.Text;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Xml.Serialization;
using System.Windows.Media;
using System.IO;
using System.Globalization;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class SMC_Exporter_Pro_v3 : Indicator
    {
        // -------- cấu hình xuất file ----------
        [NinjaScriptProperty]
        [Display(Name = "FileName", Order = 1, GroupName = "SMC Export")]
        public string FileName { get; set; }

        private string filePath;
        private string exportDirectory;
        private string fileBaseName;
        private string fileExtension;
        private DateTime currentWeekStartDate;

        // Tick count cho BAR hiện tại
        private int  curTickCount;

        // Snapshot cho BAR đã đóng
        private int    prevTickCount;
        private double prevBuyVol;
        private double prevSellVol;
        private double prevPriceRange;
        private double prevBid;
        private double prevAsk;

        private bool   hasPrevBar;

        // Daily VWAP tracking
        private double sessionPvSum;
        private double sessionVolume;

        // Volumdelta “chuẩn” để lấy delta bar
        private Volumdelta volDelta;
        private bool loggedVolDeltaError;
        private bool loggedFilePathError;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name        = "SMC_Exporter_Pro_v3";
                Description = "Export OHLCV + delta từ Volumdelta + tick-features; visual + JSONL.";
                Calculate   = Calculate.OnEachTick;
                IsOverlay   = false;

                AddPlot(Brushes.Gold,  "TickSpeed");
                AddPlot(Brushes.Lime,  "AggBuySpeed");
                AddPlot(Brushes.Red,   "AggSellSpeed");
                AddPlot(Brushes.White, "PriceSpeed");

                FileName = "smc_export_gc_m1_v3.jsonl";
            }
            else if (State == State.Configure)
            {
                // Ensure tick series is added up front so nested Volumdelta (which requires a 1-tick series) can load
                AddDataSeries(BarsPeriodType.Tick, 1);

                try
                {
                    string dir = NinjaTrader.Core.Globals.UserDataDir + "SMC_Exports\\";
                    if (!Directory.Exists(dir))
                        Directory.CreateDirectory(dir);

                    if (string.IsNullOrEmpty(FileName))
                        FileName = "smc_export_gc_m1_v3.jsonl";

                    exportDirectory = dir;
                    fileBaseName    = Path.GetFileNameWithoutExtension(FileName);
                    fileExtension   = Path.GetExtension(FileName);

                    if (string.IsNullOrEmpty(fileBaseName))
                        fileBaseName = "smc_export_gc_m1_v3";
                    if (string.IsNullOrEmpty(fileExtension))
                        fileExtension = ".jsonl";

                    filePath             = Path.Combine(dir, fileBaseName + fileExtension);
                    currentWeekStartDate = DateTime.MinValue;
                }
                catch
                {
                    filePath = null;
                }
            }
            else if (State == State.DataLoaded)
            {
                try
                {
                    // Gắn Volumdelta vào cùng Input (M1) với exporter
                    volDelta = Volumdelta(Brushes.Red, Brushes.LimeGreen, Brushes.Black, 1, false, 1, false);
                }
                catch
                {
                    volDelta = null;
                }
            }
        }

        // ======================= TICK FLOW =======================
        protected override void OnMarketData(MarketDataEventArgs e)
        {
            try
            {
                if (e.MarketDataType == MarketDataType.Last)
                    curTickCount++;
            }
            catch
            {
            }
        }

        // ======================= BAR FLOW =======================
        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0)
                return;

            if (CurrentBar < 1)
                return;

            if (IsFirstTickOfBar)
            {
                bool isNewSession = Bars.IsFirstBarOfSession;

                if (hasPrevBar)
                {
                    // 1) Tick-speed for the bar that just closed
                    int tickCountClosedBar = curTickCount;

                    // 2) Volume & delta for the closed bar (barsAgo = 1)
                    double barVol   = Volume[1];
                    double barDelta = 0.0;

                    try
                    {
                        if (volDelta != null)
                        {
                            // Đảm bảo Volumdelta cập nhật trước khi lấy giá trị
                            volDelta.Update();
                            barDelta = volDelta.DeltasClose[1];
                        }
                        else if (!loggedVolDeltaError)
                        {
                            Print("SMCExporterProv2: Volumdelta not available; delta will be 0.");
                            loggedVolDeltaError = true;
                        }
                    }
                    catch (Exception ex)
                    {
                        if (!loggedVolDeltaError)
                        {
                            Print("SMCExporterProv2: Volumdelta error -> " + ex.Message);
                            loggedVolDeltaError = true;
                        }
                        barDelta = 0.0;
                    }

                    // 3) Infer buy/sell volume from volume + delta
                    double buyVol  = 0.0;
                    double sellVol = 0.0;

                    try
                    {
                        buyVol  = (barVol + barDelta) / 2.0;
                        sellVol = barVol - buyVol;
                    }
                    catch
                    {
                        buyVol  = 0.0;
                        sellVol = 0.0;
                    }

                    // 4) Save snapshot for plots + JSON
                    prevTickCount  = tickCountClosedBar;
                    prevBuyVol     = buyVol;
                    prevSellVol    = sellVol;
                    prevPriceRange = Math.Abs(High[1] - Low[1]);

                    // best_bid / best_ask: use close as stub
                    prevBid = Close[1];
                    prevAsk = Close[1];
                    DateTime barTime = Time[1];

                    // 4b) VWAP daily (session cumulative)
                    double typicalPrice = (High[1] + Low[1] + Close[1]) / 3.0;
                    sessionPvSum   += typicalPrice * barVol;
                    sessionVolume  += barVol;
                    double barDailyVwap = sessionVolume > 0.0 ? sessionPvSum / sessionVolume : 0.0;

                    // 5) Build + write JSON for the closed bar
                    if (!IsWeekend(barTime))
                    {
                        try
                        {
                            EnsureWeeklyFile(barTime);
                            string json = BuildJsonForClosedBar(barVol, barDelta, buyVol, sellVol, barDailyVwap);
                            Print(json);          // debug
                            WriteJsonLine(json);  // write .jsonl
                        }
                        catch
                        {
                        }
                    }
                }

                // Reset VWAP accumulators at start of a new session/day
                if (isNewSession)
                {
                    sessionPvSum  = 0.0;
                    sessionVolume = 0.0;
                }

                // Reset tick count for new bar
                curTickCount = 0;
                hasPrevBar   = true;
            }

            // 6) Visual features from most recent closed bar
            TickSpeed[0]    = prevTickCount;
            AggBuySpeed[0]  = prevBuyVol;
            AggSellSpeed[0] = prevSellVol;
            PriceSpeed[0]   = prevPriceRange;
        }

        private void EnsureWeeklyFile(DateTime barTime)
        {
            if (string.IsNullOrEmpty(exportDirectory) || string.IsNullOrEmpty(fileBaseName))
                return;

            DateTime weekStart = GetWeekStart(barTime);
            if (weekStart != currentWeekStartDate)
            {
                string ext = string.IsNullOrEmpty(fileExtension) ? ".jsonl" : fileExtension;
                currentWeekStartDate = weekStart;
                filePath             = Path.Combine(exportDirectory, fileBaseName + "_" + weekStart.ToString("yyyyMMdd") + ext);
            }
        }

        private DateTime GetWeekStart(DateTime time)
        {
            int diff = ((int)time.DayOfWeek - (int)DayOfWeek.Monday + 7) % 7;
            return time.Date.AddDays(-diff);
        }

        private bool IsWeekend(DateTime time)
        {
            return time.DayOfWeek == DayOfWeek.Saturday || time.DayOfWeek == DayOfWeek.Sunday;
        }

        // =============== WRITE JSON LINE TO FILE ===============
        private void WriteJsonLine(string json)
        {
            EnsureFilePathIfMissing();
            if (string.IsNullOrEmpty(filePath))
            {
                if (!loggedFilePathError)
                {
                    Print("SMCExporterProv2: filePath is null/empty; cannot write JSON.");
                    loggedFilePathError = true;
                }
                return;
            }

            try
            {
                string dir = Path.GetDirectoryName(filePath);
                if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                    Directory.CreateDirectory(dir);

                using (StreamWriter sw = new StreamWriter(filePath, true))
                    sw.WriteLine(json);
            }
            catch (Exception ex)
            {
                Print("SMCExporterProv2: failed to write JSON -> " + ex.Message);
            }
        }

        private void EnsureFilePathIfMissing()
        {
            if (!string.IsNullOrEmpty(filePath))
                return;

            try
            {
                string dir = exportDirectory;
                if (string.IsNullOrEmpty(dir))
                    dir = NinjaTrader.Core.Globals.UserDataDir + "SMC_Exports\\";

                if (!Directory.Exists(dir))
                    Directory.CreateDirectory(dir);

                string baseName = string.IsNullOrEmpty(fileBaseName) ? "smc_export_gc_m1_v3" : fileBaseName;
                string ext      = string.IsNullOrEmpty(fileExtension) ? ".jsonl" : fileExtension;

                filePath = Path.Combine(dir, baseName + ext);
            }
            catch (Exception ex)
            {
                if (!loggedFilePathError)
                {
                    Print("SMCExporterProv2: failed to rebuild filePath -> " + ex.Message);
                    loggedFilePathError = true;
                }
            }
        }

        // =============== BUILD JSON (bar closed, barsAgo=1) ===============
        private string BuildJsonForClosedBar(double barVol, double barDelta, double buyVol, double sellVol, double barDailyVwap)
        {
            int barsAgo = 1;

            double o   = Open[barsAgo];
            double h   = High[barsAgo];
            double l   = Low[barsAgo];
            double c   = Close[barsAgo];

            double bid = prevBid;
            double ask = prevAsk;

            StringBuilder sb = new StringBuilder();
            sb.Append("{");

            sb.Append("\"symbol\":\"" + Instrument.FullName + "\",");
            sb.Append("\"timeframe\":\"M1\",");
            sb.Append("\"timestamp\":\"" + Time[barsAgo].ToString("o") + "\",");
            sb.Append("\"bar_index\":" + (CurrentBar - 1) + ",");

            sb.Append("\"bar\":{");
            sb.Append("\"o\":" + F(o) + ",");
            sb.Append("\"h\":" + F(h) + ",");
            sb.Append("\"l\":" + F(l) + ",");
            sb.Append("\"c\":" + F(c) + ",");
            sb.Append("\"volume\":" + F(barVol) + ",");
            sb.Append("\"delta\":" + F(barDelta) + ",");
            sb.Append("\"buy_volume\":" + F(buyVol) + ",");
            sb.Append("\"sell_volume\":" + F(sellVol) + ",");
            sb.Append("\"best_bid\":" + F(bid) + ",");
            sb.Append("\"best_ask\":" + F(ask) + ",");
            sb.Append("\"vwap_daily\":" + F(barDailyVwap));
            sb.Append("},");

            sb.Append("\"tick_features\":{");
            sb.Append("\"tick_speed\":" + prevTickCount + ",");
            sb.Append("\"aggr_buy_speed\":" + F(prevBuyVol) + ",");
            sb.Append("\"aggr_sell_speed\":" + F(prevSellVol) + ",");
            sb.Append("\"price_speed\":" + F(prevPriceRange));
            sb.Append("}");

            sb.Append("}");
            return sb.ToString();
        }

        // Format double with trimmed trailing zeros, up to 8 decimals
        private string F(double value)
        {
            return value.ToString("0.####", CultureInfo.InvariantCulture);
        }

        // ======================= PLOTS =======================
        [Browsable(false)]
        [XmlIgnore]
        public Series<double> TickSpeed
        {
            get { return Values[0]; }
        }

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> AggBuySpeed
        {
            get { return Values[1]; }
        }

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> AggSellSpeed
        {
            get { return Values[2]; }
        }

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> PriceSpeed
        {
            get { return Values[3]; }
        }
    }
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private SMC_Exporter_Pro_v3[] cacheSMC_Exporter_Pro_v3;
		public SMC_Exporter_Pro_v3 SMC_Exporter_Pro_v3(string fileName)
		{
			return SMC_Exporter_Pro_v3(Input, fileName);
		}

		public SMC_Exporter_Pro_v3 SMC_Exporter_Pro_v3(ISeries<double> input, string fileName)
		{
			if (cacheSMC_Exporter_Pro_v3 != null)
				for (int idx = 0; idx < cacheSMC_Exporter_Pro_v3.Length; idx++)
					if (cacheSMC_Exporter_Pro_v3[idx] != null && cacheSMC_Exporter_Pro_v3[idx].FileName == fileName && cacheSMC_Exporter_Pro_v3[idx].EqualsInput(input))
						return cacheSMC_Exporter_Pro_v3[idx];
			return CacheIndicator<SMC_Exporter_Pro_v3>(new SMC_Exporter_Pro_v3(){ FileName = fileName }, input, ref cacheSMC_Exporter_Pro_v3);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.SMC_Exporter_Pro_v3 SMC_Exporter_Pro_v3(string fileName)
		{
			return indicator.SMC_Exporter_Pro_v3(Input, fileName);
		}

		public Indicators.SMC_Exporter_Pro_v3 SMC_Exporter_Pro_v3(ISeries<double> input , string fileName)
		{
			return indicator.SMC_Exporter_Pro_v3(input, fileName);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.SMC_Exporter_Pro_v3 SMC_Exporter_Pro_v3(string fileName)
		{
			return indicator.SMC_Exporter_Pro_v3(Input, fileName);
		}

		public Indicators.SMC_Exporter_Pro_v3 SMC_Exporter_Pro_v3(ISeries<double> input , string fileName)
		{
			return indicator.SMC_Exporter_Pro_v3(input, fileName);
		}
	}
}

#endregion
