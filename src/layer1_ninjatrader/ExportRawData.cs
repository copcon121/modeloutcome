#region Using declarations
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.Strategies;
using Newtonsoft.Json;
#endregion

//This namespace holds Strategies in this folder and is required. Do not change it.
namespace NinjaTrader.NinjaScript.Strategies
{
	/// <summary>
	/// ExportRawData - Strategy to export raw OHLCV bars from NinjaTrader to Python Feature Engine
	/// Sends HTTP POST requests with JSON payload containing bar data
	/// </summary>
	public class ExportRawData : Strategy
	{
		#region Variables

		private HttpClient httpClient;
		private string endpointUrl = "http://localhost:5001/raw";
		private int barsToExport = 100; // Number of bars to include in each export
		private int exportIntervalBars = 1; // Export every N bars (1 = every bar)
		private int barCounter = 0;

		// Feature flags for future enhancement
		private bool includeDelta = false; // TODO: Implement with Rithmic API
		private bool includeL2Depth = false; // TODO: Implement with Rithmic API

		#endregion

		#region Properties

		[NinjaScriptProperty]
		[Display(Name="Endpoint URL", Description="URL of the Python Feature Engine", Order=1, GroupName="Parameters")]
		public string EndpointUrl
		{
			get { return endpointUrl; }
			set { endpointUrl = value; }
		}

		[NinjaScriptProperty]
		[Range(10, 200)]
		[Display(Name="Bars To Export", Description="Number of historical bars to include in each export", Order=2, GroupName="Parameters")]
		public int BarsToExport
		{
			get { return barsToExport; }
			set { barsToExport = value; }
		}

		[NinjaScriptProperty]
		[Range(1, 100)]
		[Display(Name="Export Interval Bars", Description="Export data every N bars (1 = every bar)", Order=3, GroupName="Parameters")]
		public int ExportIntervalBars
		{
			get { return exportIntervalBars; }
			set { exportIntervalBars = value; }
		}

		[NinjaScriptProperty]
		[Display(Name="Include Delta", Description="Include delta volume (requires Rithmic API)", Order=4, GroupName="Features")]
		public bool IncludeDelta
		{
			get { return includeDelta; }
			set { includeDelta = value; }
		}

		[NinjaScriptProperty]
		[Display(Name="Include L2 Depth", Description="Include Level 2 market depth (requires Rithmic API)", Order=5, GroupName="Features")]
		public bool IncludeL2Depth
		{
			get { return includeL2Depth; }
			set { includeL2Depth = value; }
		}

		#endregion

		#region Lifecycle Methods

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description									= @"Export raw OHLCV bar data to Python Feature Engine via HTTP POST";
				Name										= "ExportRawData";
				Calculate									= Calculate.OnBarClose;
				EntriesPerDirection							= 1;
				EntryHandling								= EntryHandling.AllEntries;
				IsExitOnSessionCloseStrategy				= true;
				ExitOnSessionCloseSeconds					= 30;
				IsFillLimitOnTouch							= false;
				MaximumBarsLookBack							= MaximumBarsLookBack.TwoHundredFiftySix;
				OrderFillResolution							= OrderFillResolution.Standard;
				Slippage									= 0;
				StartBehavior								= StartBehavior.WaitUntilFlat;
				TimeInForce									= TimeInForce.Gtc;
				TraceOrders									= false;
				RealtimeErrorHandling						= RealtimeErrorHandling.StopCancelClose;
				StopTargetHandling							= StopTargetHandling.PerEntryExecution;
				BarsRequiredToTrade							= 20;
				IsInstantiatedOnEachOptimizationIteration	= true;
			}
			else if (State == State.Configure)
			{
				// Initialize HTTP client
				httpClient = new HttpClient();
				httpClient.Timeout = TimeSpan.FromSeconds(5); // 5 second timeout to avoid blocking
			}
			else if (State == State.Terminated)
			{
				// Clean up HTTP client
				if (httpClient != null)
				{
					httpClient.Dispose();
					httpClient = null;
				}
			}
		}

		protected override void OnBarUpdate()
		{
			// Wait until we have enough bars
			if (CurrentBar < BarsToExport)
				return;

			// Export every N bars based on exportIntervalBars setting
			barCounter++;
			if (barCounter % ExportIntervalBars != 0)
				return;

			// Collect bar data
			var barData = CollectBarData();

			// Serialize to JSON
			string jsonPayload = JsonConvert.SerializeObject(barData, Formatting.None);

			// Send HTTP POST (fire-and-forget, non-blocking)
			_ = SendDataAsync(jsonPayload);
		}

		#endregion

		#region Helper Methods

		/// <summary>
		/// Collects the last N bars of OHLCV data
		/// </summary>
		private Dictionary<string, object> CollectBarData()
		{
			var bars = new List<Dictionary<string, object>>();

			int startBar = Math.Max(0, CurrentBar - BarsToExport + 1);
			int barsToCollect = CurrentBar - startBar + 1;

			for (int i = 0; i < barsToCollect; i++)
			{
				int barsAgo = barsToCollect - 1 - i;

				var barDict = new Dictionary<string, object>
				{
					{ "ts", Time[barsAgo].ToString("yyyy-MM-ddTHH:mm:ss") },
					{ "open", Open[barsAgo] },
					{ "high", High[barsAgo] },
					{ "low", Low[barsAgo] },
					{ "close", Close[barsAgo] },
					{ "volume", Volume[barsAgo] }
				};

				// TODO: Add delta volume if available
				if (IncludeDelta)
				{
					// Placeholder - requires Rithmic API integration
					// Example: barDict["delta"] = GetDelta(barsAgo);
					barDict["delta"] = 0.0; // Mock value
					barDict["buy_volume"] = Volume[barsAgo] * 0.5; // Mock: assume 50/50
					barDict["sell_volume"] = Volume[barsAgo] * 0.5;
				}

				// TODO: Add L2 depth data if available
				if (IncludeL2Depth)
				{
					// Placeholder - requires Rithmic API integration
					// Example: barDict["l2_bid_depth"] = GetL2BidDepth(barsAgo);
					barDict["l2_bid_depth"] = new List<double>(); // Mock empty
					barDict["l2_ask_depth"] = new List<double>();
				}

				bars.Add(barDict);
			}

			var payload = new Dictionary<string, object>
			{
				{ "symbol", Instrument.FullName },
				{ "timeframe", BarsPeriod.Value.ToString() + BarsPeriod.BarsPeriodType.ToString() },
				{ "timestamp", DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ss.fff") },
				{ "bars", bars }
			};

			return payload;
		}

		/// <summary>
		/// Sends JSON data to the Feature Engine via HTTP POST (async, non-blocking)
		/// </summary>
		private async Task SendDataAsync(string jsonPayload)
		{
			try
			{
				var content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");

				// Fire-and-forget: don't await the response to avoid blocking chart rendering
				var response = await httpClient.PostAsync(EndpointUrl, content);

				if (!response.IsSuccessStatusCode)
				{
					Print($"ExportRawData: HTTP POST failed with status {response.StatusCode}");
				}
				else
				{
					// Optionally log success (can be verbose)
					// Print($"ExportRawData: Successfully sent {BarsToExport} bars to {EndpointUrl}");
				}
			}
			catch (Exception ex)
			{
				// Log error but don't crash the strategy
				Print($"ExportRawData ERROR: {ex.Message}");
			}
		}

		#endregion
	}
}
