#region Using declarations
using System;
using System.IO;
using System.Net;
using System.Text;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Windows.Media;
using System.Xml.Serialization;
using System.Globalization;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class NT_Gateway_S4_ASM_Live : Strategy
    {
        // ================== USER CONFIG ==================
        [NinjaScriptProperty]
        [Display(Name = "ApiUrl", Order = 1, GroupName = "Gateway")]
        public string ApiUrl { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "ApiKey (optional)", Order = 2, GroupName = "Gateway")]
        public string ApiKey { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable HTTP Send", Order = 3, GroupName = "Gateway")]
        public bool EnableHttp { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Execute Trades", Order = 4, GroupName = "Trading")]
        public bool ExecuteTrades { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Order Quantity", Order = 5, GroupName = "Trading")]
        public int OrderQuantity { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Debug Print JSON", Order = 6, GroupName = "Debug")]
        public bool DebugPrintJson { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Debug Print Response", Order = 7, GroupName = "Debug")]
        public bool DebugPrintResponse { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Send History", Order = 8, GroupName = "Gateway")]
        public bool SendHistory { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Delay HTTP until Realtime", Order = 9, GroupName = "Gateway")]
        public bool DelayHttpUntilRealtime { get; set; }

        // ============ INTERNAL STATE (copy từ Exporter) ============
        private int    curTickCount;
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

        // Volumdelta cho delta bar
        private Indicators.Volumdelta volDelta;
        private bool loggedVolDeltaError;

        // Simple throttling for HTTP errors
        private int httpErrorCount;
        private int httpErrorMax;
        
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name        = "NT_Gateway_S4_ASM_Live";
                Description = "Gateway gửi OHLCV+delta+tick-features tới FastAPI (S4+ASM live).";

                Calculate   = Calculate.OnEachTick;
                IsOverlay   = false;
                EnableHttp  = true;
                ExecuteTrades = false;
                OrderQuantity = 1;
                SendHistory   = true;
                DelayHttpUntilRealtime = true;

                ApiUrl   = "http://127.0.0.1:8000/live_decision";
                ApiKey   = string.Empty;

                DebugPrintJson      = false;
                DebugPrintResponse  = true;

                httpErrorCount = 0;
                httpErrorMax   = 5;
            }
            else if (State == State.Configure)
            {
                // Tick series cho Volumdelta
                AddDataSeries(BarsPeriodType.Tick, 1);
            }
            else if (State == State.DataLoaded)
            {
                try
                {
                    // Giống exporter: Volumdelta trên input M1
                    volDelta = Volumdelta(Brushes.Red, Brushes.LimeGreen, Brushes.Black, 1, false, 1, false);
                }
                catch
                {
                    volDelta = null;
                }
            }
            else if (State == State.Realtime)
            {
                httpErrorCount = 0;
                Print("NT_Gateway: State changed to Realtime. Resetting error count.");
            }
        }

        // ======================= TICK FLOW =======================
        protected override void OnMarketData(MarketDataEventArgs e)
        {
            try
            {
                if (e.MarketDataType == MarketDataType.Last)
                {
                    curTickCount++;
                }
            }
            catch
            {
            }
        }

        // ======================= BAR FLOW =======================
        protected override void OnBarUpdate()
        {
            // Chỉ chạy trên series chính (M1)
            if (BarsInProgress != 0)
                return;

            if (CurrentBar < 1)
                return;

            // Xử lý khi BAR MỚI bắt đầu -> BAR trước đó (barsAgo=1) vừa đóng
            if (IsFirstTickOfBar)
            {
                bool isNewSession = Bars.IsFirstBarOfSession;

                if (hasPrevBar)
                {
                    // === 1) Tick-speed cho bar vừa đóng ===
                    int tickCountClosedBar = curTickCount;

                    // === 2) Volume & Delta cho bar vừa đóng (barsAgo = 1) ===
                    double barVol   = Volume[1];
                    double barDelta = 0.0;

                    try
                    {
                        if (volDelta != null)
                        {
                            volDelta.Update();
                            barDelta = volDelta.DeltasClose[1];
                        }
                        else if (!loggedVolDeltaError)
                        {
                            Print("NT_Gateway_S4_ASM_Live: Volumdelta not available; delta=0.");
                            loggedVolDeltaError = true;
                        }
                    }
                    catch (Exception ex)
                    {
                        if (!loggedVolDeltaError)
                        {
                            Print("NT_Gateway_S4_ASM_Live: Volumdelta error -> " + ex.Message);
                            loggedVolDeltaError = true;
                        }
                        barDelta = 0.0;
                    }

                    // === 3) Suy buy/sell volume từ volume + delta ===
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

                    // === 4) Snapshot cho tick_features & bar meta ===
                    prevTickCount  = tickCountClosedBar;
                    prevBuyVol     = buyVol;
                    prevSellVol    = sellVol;
                    prevPriceRange = Math.Abs(High[1] - Low[1]);

                    prevBid = Close[1];
                    prevAsk = Close[1];

                    DateTime barTime = Time[1];

                    // === 5) VWAP daily (session cumulative) ===
                    double typicalPrice = (High[1] + Low[1] + Close[1]) / 3.0;
                    sessionPvSum  += typicalPrice * barVol;
                    sessionVolume += barVol;
                    double barDailyVwap = sessionVolume > 0.0 ? sessionPvSum / sessionVolume : 0.0;

                    // === 6) Build JSON & gửi HTTP ===
                    bool isHistory = (State == State.Historical);
                    bool isWeekend = IsWeekend(barTime);
                    bool isPlayback = Connection.PlaybackConnection != null
                                      && Connection.PlaybackConnection.Status == ConnectionStatus.Connected;
                    if (isPlayback)
                        isHistory = false; // allow playback to flow like realtime
                    bool skipStartupHistory = DelayHttpUntilRealtime && isHistory && !isPlayback;
                    
                    if (!isWeekend && EnableHttp && httpErrorCount < httpErrorMax && (!isHistory || SendHistory) && !skipStartupHistory)
                    {
                        string json = BuildJsonForClosedBar(barVol, barDelta, buyVol, sellVol, barDailyVwap);
                        if (DebugPrintJson)
                            Print("NT_Gateway JSON: " + json);

                        string response = SendHttp(json);

                        if (!string.IsNullOrEmpty(response))
                        {
                            if (DebugPrintResponse)
                                Print("NT_Gateway RESP: " + response);

                            ProcessDecision(response);
                        }
                    }
                }

                // Reset VWAP khi ngày/phiên mới
                if (isNewSession)
                {
                    sessionPvSum  = 0.0;
                    sessionVolume = 0.0;
                }

                // Reset tick count cho bar mới
                curTickCount = 0;
                hasPrevBar   = true;
            }
        }

        // =============== BUILD JSON (bar closed, barsAgo=1) ===============
        private string BuildJsonForClosedBar(double barVol, double barDelta, double buyVol, double sellVol, double barDailyVwap)
        {
            int barsAgo = 1;

            double o = Open[barsAgo];
            double h = High[barsAgo];
            double l = Low[barsAgo];
            double c = Close[barsAgo];

            double bid = prevBid;
            double ask = prevAsk;

            StringBuilder sb = new StringBuilder();
            sb.Append("{");

            // Meta
            sb.Append("\"symbol\":\"");
            sb.Append(Instrument.FullName);
            sb.Append("\",");

            sb.Append("\"timeframe\":\"M1\",");
            sb.Append("\"timestamp\":\"");
            sb.Append(Time[barsAgo].ToString("o"));
            sb.Append("\",");

            sb.Append("\"bar_index\":");
            sb.Append(CurrentBar - 1);
            sb.Append(",");

            // Bar object
            sb.Append("\"bar\":{");
            sb.Append("\"o\":");
            sb.Append(F(o));
            sb.Append(",");

            sb.Append("\"h\":");
            sb.Append(F(h));
            sb.Append(",");

            sb.Append("\"l\":");
            sb.Append(F(l));
            sb.Append(",");

            sb.Append("\"c\":");
            sb.Append(F(c));
            sb.Append(",");

            sb.Append("\"volume\":");
            sb.Append(F(barVol));
            sb.Append(",");

            sb.Append("\"delta\":");
            sb.Append(F(barDelta));
            sb.Append(",");

            sb.Append("\"buy_volume\":");
            sb.Append(F(buyVol));
            sb.Append(",");

            sb.Append("\"sell_volume\":");
            sb.Append(F(sellVol));
            sb.Append(",");

            sb.Append("\"best_bid\":");
            sb.Append(F(bid));
            sb.Append(",");

            sb.Append("\"best_ask\":");
            sb.Append(F(ask));
            sb.Append(",");

            sb.Append("\"vwap_daily\":");
            sb.Append(F(barDailyVwap));
            sb.Append("},");

            // Tick features
            sb.Append("\"tick_features\":{");
            sb.Append("\"tick_speed\":");
            sb.Append(prevTickCount);
            sb.Append(",");

            sb.Append("\"aggr_buy_speed\":");
            sb.Append(F(prevBuyVol));
            sb.Append(",");

            sb.Append("\"aggr_sell_speed\":");
            sb.Append(F(prevSellVol));
            sb.Append(",");

            sb.Append("\"price_speed\":");
            sb.Append(F(prevPriceRange));
            sb.Append("}");

            sb.Append("}");
            return sb.ToString();
        }

        // Format double, giống exporter
        private string F(double value)
        {
            return value.ToString("0.####", CultureInfo.InvariantCulture);
        }

        private bool IsWeekend(DateTime time)
        {
            return time.DayOfWeek == DayOfWeek.Saturday || time.DayOfWeek == DayOfWeek.Sunday;
        }

        // ================= HTTP CLIENT (sync) =================
        private string SendHttp(string json)
        {
            if (string.IsNullOrEmpty(ApiUrl))
                return null;

            try
            {
                HttpWebRequest request = (HttpWebRequest)WebRequest.Create(ApiUrl);
                request.Method      = "POST";
                request.ContentType = "application/json";

                if (!string.IsNullOrEmpty(ApiKey))
                    request.Headers["Authorization"] = "Bearer " + ApiKey;

                byte[] data = Encoding.UTF8.GetBytes(json);
                request.ContentLength = data.Length;

                using (Stream reqStream = request.GetRequestStream())
                {
                    reqStream.Write(data, 0, data.Length);
                }

                using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
                {
                    using (Stream respStream = response.GetResponseStream())
                    {
                        if (respStream == null)
                            return null;
                        using (StreamReader reader = new StreamReader(respStream))
                        {
                            string respText = reader.ReadToEnd();
                            httpErrorCount = 0;
                            return respText;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                httpErrorCount++;
                Print("NT_Gateway HTTP error (" + httpErrorCount + "): " + ex.Message);
                if (httpErrorCount >= httpErrorMax)
                    Print("NT_Gateway: Max HTTP errors reached. Stopping requests.");
                return null;
            }
        }

        // ================== RESPONSE PARSING ==================
        private void ProcessDecision(string json)
        {
            if (!ExecuteTrades)
                return;

            if (Position.MarketPosition != MarketPosition.Flat)
                return;

            string action = ExtractJsonString(json, "action");
            if (string.IsNullOrEmpty(action))
                return;

            action = action.ToLower();

            double defaultEntry = Close[0];
            double entry = ExtractJsonDouble(json, "entry", defaultEntry);
            double sl    = ExtractJsonDouble(json, "sl", double.NaN);
            double tp    = ExtractJsonDouble(json, "tp", double.NaN);

            int qty = OrderQuantity;
            if (qty <= 0)
                qty = 1;

            // Đặt SL/TP theo signal name
            if (action == "long")
            {
                string signalName = "LLM_LONG";
                if (!double.IsNaN(sl))
                    SetStopLoss(signalName, CalculationMode.Price, sl, false);
                if (!double.IsNaN(tp))
                    SetProfitTarget(signalName, CalculationMode.Price, tp);

                EnterLongLimit(qty, entry, signalName);
            }
            else if (action == "short")
            {
                string signalName = "LLM_SHORT";
                if (!double.IsNaN(sl))
                    SetStopLoss(signalName, CalculationMode.Price, sl, false);
                if (!double.IsNaN(tp))
                    SetProfitTarget(signalName, CalculationMode.Price, tp);

                EnterShortLimit(qty, entry, signalName);
            }
            else
            {
                // flat / hold -> no trade
            }
        }

        private string ExtractJsonString(string json, string field)
        {
            try
            {
                string pattern = "\"" + field + "\":";
                int idx = json.IndexOf(pattern, StringComparison.OrdinalIgnoreCase);
                if (idx < 0)
                    return null;

                idx += pattern.Length;

                // skip whitespace
                while (idx < json.Length && (json[idx] == ' ' || json[idx] == '\t'))
                    idx++;

                if (idx >= json.Length)
                    return null;

                if (json[idx] == '"')
                {
                    idx++;
                    int end = json.IndexOf('"', idx);
                    if (end > idx)
                        return json.Substring(idx, end - idx);
                    return null;
                }
                else
                {
                    int end = idx;
                    while (end < json.Length &&
                           json[end] != ',' &&
                           json[end] != '}' &&
                           json[end] != ' ' &&
                           json[end] != '\t' &&
                           json[end] != '\r' &&
                           json[end] != '\n')
                    {
                        end++;
                    }
                    return json.Substring(idx, end - idx);
                }
            }
            catch
            {
                return null;
            }
        }

        private double ExtractJsonDouble(string json, string field, double defaultValue)
        {
            string raw = ExtractJsonString(json, field);
            if (string.IsNullOrEmpty(raw))
                return defaultValue;

            double value;
            if (double.TryParse(raw, NumberStyles.Any, CultureInfo.InvariantCulture, out value))
                return value;

            return defaultValue;
        }
    }
}
