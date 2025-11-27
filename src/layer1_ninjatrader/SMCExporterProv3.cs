#region Using declarations
using System;
using System.Text;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Xml.Serialization;
using System.Windows.Media;
using System.IO;
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

        // Volumdelta “chuẩn” để lấy delta bar
        private Volumdelta volDelta;

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

                    filePath = Path.Combine(dir, FileName);
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
            // Process only primary series; tick series (BarsInProgress > 0) is for Volumdelta internals
            if (BarsInProgress != 0)
                return;

            if (CurrentBar < 1)
                return;

            if (IsFirstTickOfBar)
            {
                if (hasPrevBar)
                {
                    // 1) Tick-speed cho bar vừa xong
                    int tickCountClosedBar = curTickCount;

                    // 2) Volume & delta bar vừa đóng (barsAgo = 1)
                    double barVol   = Volume[1];
                    double barDelta = 0.0;

                    try
                    {
                        if (volDelta != null && CurrentBar > 1)
                        {
                            // DeltasClose[1] = delta của bar đã đóng
                            barDelta = volDelta.DeltasClose[1];
                        }
                    }
                    catch
                    {
                        barDelta = 0.0;
                    }

                    // 3) Suy ra buy/sell volume từ volume + delta
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

                    // 4) Lưu snapshot cho plots + JSON
                    prevTickCount  = tickCountClosedBar;
                    prevBuyVol     = buyVol;
                    prevSellVol    = sellVol;
                    prevPriceRange = Math.Abs(High[1] - Low[1]);

                    // best_bid / best_ask: dùng close của bar làm stub
                    prevBid = Close[1];
                    prevAsk = Close[1];

                    // 5) Build + ghi JSON cho bar đã đóng
                    try
                    {
                        string json = BuildJsonForClosedBar(barVol, barDelta, buyVol, sellVol);
                        Print(json);          // debug
                        WriteJsonLine(json);  // ghi file .jsonl
                    }
                    catch
                    {
                    }
                }

                // Reset tick count cho bar mới
                curTickCount = 0;
                hasPrevBar   = true;
            }

            // 6) Visual 4 features của bar đã đóng gần nhất
            TickSpeed[0]    = prevTickCount;
            AggBuySpeed[0]  = prevBuyVol;
            AggSellSpeed[0] = prevSellVol;
            PriceSpeed[0]   = prevPriceRange;
        }

        // =============== WRITE JSON LINE TO FILE ===============
        private void WriteJsonLine(string json)
        {
            if (string.IsNullOrEmpty(filePath))
                return;

            try
            {
                using (StreamWriter sw = new StreamWriter(filePath, true))
                {
                    sw.WriteLine(json);
                }
            }
            catch
            {
            }
        }

        // =============== BUILD JSON (bar closed, barsAgo=1) ===============
        private string BuildJsonForClosedBar(double barVol, double barDelta, double buyVol, double sellVol)
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
            sb.Append("\"o\":" + o + ",");
            sb.Append("\"h\":" + h + ",");
            sb.Append("\"l\":" + l + ",");
            sb.Append("\"c\":" + c + ",");
            sb.Append("\"volume\":" + barVol + ",");
            sb.Append("\"delta\":" + barDelta + ",");
            sb.Append("\"buy_volume\":" + buyVol + ",");
            sb.Append("\"sell_volume\":" + sellVol + ",");
            sb.Append("\"best_bid\":" + bid + ",");
            sb.Append("\"best_ask\":" + ask);
            sb.Append("},");

            sb.Append("\"tick_features\":{");
            sb.Append("\"tick_speed\":" + prevTickCount + ",");
            sb.Append("\"aggr_buy_speed\":" + prevBuyVol + ",");
            sb.Append("\"aggr_sell_speed\":" + prevSellVol + ",");
            sb.Append("\"price_speed\":" + prevPriceRange);
            sb.Append("}");

            sb.Append("}");
            return sb.ToString();
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
