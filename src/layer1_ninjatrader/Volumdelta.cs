// #############################################################
// #														   #
// #            Adapted from Volume Delta by GillRymhes  	   #
// #						03.03.2023 in					   #
// #  https://ninjatraderecosystem.com/user-app-share-download/delta-volume/										   #
// #														   #
// #############################################################

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Controls;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.DrawingTools;
using System.Windows.Controls;
using NinjaTrader.Gui.Chart;


#endregion

//This namespace holds Indicators in this folder and is required. Do not change it. 
namespace NinjaTrader.NinjaScript.Indicators
{
	public class Volumdelta : Indicator
	{
		private double		buys 	= 0;
		private double 		sells 	= 0;
		private double		buyscum 	= 0;
		private double 		sellscum 	= 0;
		
		private double		cdHigh 	= 0;
		private double 		cdLow 	= 0;
		private double		cdOpen 	= 0;
		private double 		cdClose	= 0;
		private double		cdHighcum 	= 0;
		private double 		cdLowcum 	= 0;
		private double		cdOpencum 	= 0;
		private double 		cdClosecum	= 0;
		
		private int										barPaintWidth;
		private Dictionary<string, DXMediaMap>			dxmBrushes;
		private SharpDX.RectangleF						reuseRect;
		private SharpDX.Vector2							reuseVector1, reuseVector2;
		private double									tmpMax, tmpMin, tmpPlotVal;
		private int										x, y1, y2, y3, y4;
		private Series<Double> delta_open;
		private Series<Double> delta_close;
		private Series<Double> delta_high;
		private Series<Double> delta_low;	
		private Series<Double> delta_opencum;
		private Series<Double> delta_closecum;
		private Series<Double> delta_highcum;
		private Series<Double> delta_lowcum;

		
		private bool	isReset;

		private int 	lastBar;
		private bool 	lastInTransition;
		
		private Brush	divergeCandleup   = Brushes.Purple;  // Color body for Divergence Candle
		private Brush	divergeCandledown   = Brushes.Pink;  // Color body for Divergence Candle
		
		
		
		
		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description									= @"Volumdelta";
				Name										= "Volumdelta";
				Calculate									= Calculate.OnEachTick;
				IsOverlay									= false;
				DisplayInDataBox							= true;
				DrawOnPricePanel							= true;
				DrawHorizontalGridLines						= true;
				DrawVerticalGridLines						= true;
				PaintPriceMarkers							= true;
				ScaleJustification							= NinjaTrader.Gui.Chart.ScaleJustification.Right;
				IsSuspendedWhileInactive					= false;
	
				
				MaximumBarsLookBack = MaximumBarsLookBack.Infinite;
				
				dxmBrushes	= new Dictionary<string, DXMediaMap>();
				foreach (string brushName in new string[] { "barColorDown", "barColorUp", "shadowColor" })
					dxmBrushes.Add(brushName, new DXMediaMap());
				BarColorDown								= Brushes.Red;
				BarColorUp									= Brushes.LimeGreen;
				ShadowColor									= Brushes.Black;
				ShadowWidth									= 1;
				int MinSize 								= 0;
				CumulativeDelta								= false;
				ShowDivs 									= false;
				
				AddPlot(new Stroke(Brushes.Transparent),PlotStyle.PriceBox,"DeltaOpen");
				AddPlot(new Stroke(Brushes.Transparent),PlotStyle.PriceBox,"DeltaHigh");
				AddPlot(new Stroke(Brushes.Transparent),PlotStyle.PriceBox,"DeltaLow");
				AddPlot(new Stroke(Brushes.Orange),PlotStyle.PriceBox,"DeltaClose");
				
			}
			else if (State == State.Configure)
			{
				AddDataSeries(BarsPeriodType.Tick, 1);
			}
			
			else if (State == State.DataLoaded)
			{
				delta_open = new Series<double>(this);
				delta_close = new Series<double>(this);
				delta_high = new Series<double>(this);
				delta_low = new Series<double>(this);
				delta_opencum = new Series<double>(this);
				delta_closecum = new Series<double>(this);
				delta_highcum = new Series<double>(this);
				delta_lowcum = new Series<double>(this);
			}		
		}
		
		protected override void OnBarUpdate()
		{
			if (CurrentBars[0] < 5 || CurrentBars[1] < 5)
				return;
			if (BarsInProgress == 0)
			{
				
				int indexOffset = BarsArray[1].Count - 1 - CurrentBars[1];
				
								
				if (IsFirstTickOfBar && Calculate != Calculate.OnBarClose && (State == State.Realtime || BarsArray[0].IsTickReplay))
				{
					
					if (CurrentBars[0] > 0)
						SetValues(1);					
					
					if (BarsArray[0].IsTickReplay || State == State.Realtime && indexOffset == 0)
						ResetValues(false,cdClose,cdClosecum);
				}
				
				
				SetValues(0);
				
			
				if (Calculate == Calculate.OnBarClose || (lastBar != CurrentBars[0] && (State == State.Historical || State == State.Realtime && indexOffset > 0)))
					ResetValues(false,cdClose,cdClosecum);
				
				lastBar = CurrentBars[0];
				if (CumulativeDelta)
				{
					if (delta_close[0] > delta_close[1]) PlotBrushes[3][0] = (Brush) Brushes.LimeGreen;
					else if (delta_close[0] < delta_close[1]) PlotBrushes[3][0] = (Brush) Brushes.Red;
					else PlotBrushes[3][0] = (Brush) Brushes.Orange;
				}
				else
				{
					if (delta_close[0] > 0) PlotBrushes[3][0] = (Brush) Brushes.LimeGreen;
					else if (delta_close[0] < 0) PlotBrushes[3][0] = (Brush) Brushes.Red;
					else PlotBrushes[3][0] = (Brush) Brushes.Orange;
				}
				
				if (IsFirstTickOfBar && ShowDivs)
				{
				if (BarsPeriod.BarsPeriodType != BarsPeriodType.Tick && BarsPeriod.BarsPeriodType != BarsPeriodType.Day && BarsPeriod.BarsPeriodType != BarsPeriodType.Minute && BarsPeriod.BarsPeriodType != BarsPeriodType.Second && BarsPeriod.BarsPeriodType != BarsPeriodType.Week
				&& BarsPeriod.BarsPeriodType != BarsPeriodType.HeikenAshi && BarsPeriod.BarsPeriodType != BarsPeriodType.Kagi && BarsPeriod.BarsPeriodType != BarsPeriodType.Tick && BarsPeriod.BarsPeriodType != BarsPeriodType.Volume && BarsPeriod.BarsPeriodType != BarsPeriodType.Volumetric && BarsPeriod.BarsPeriodType != BarsPeriodType.Month
				&& BarsPeriod.BarsPeriodType != BarsPeriodType.LineBreak && BarsPeriod.BarsPeriodType != BarsPeriodType.Year)
				{
				if((delta_closecum[1] > delta_closecum[2]) && Close[1]==High[1] && Low[1] <= Low[2] && Low[1] <= Low[3] && (Stochastics(3, 14, 3).K[1] <= 20 || Stochastics(3, 14, 3).K[2] <= 20))
				{
				
					Draw.TriangleUp(this,CurrentBar.ToString(), true, 1, Low[1] - 2*TickSize, divergeCandleup);
					Alert("MyAlertsDiv" + CurrentBar.ToString(), Priority.High, "Deltadiv long", "", 10, Brushes.Black, Brushes.Lime);
				}		
				if((delta_closecum[1] < delta_closecum[2]) && Close[1]==Low[1] && High[1] >= High[2] && High[1] >= High[3] && (Stochastics(3, 14, 3).K[1] >= 80 || Stochastics(3, 14, 3).K[2] >= 80))
				{
				
					Draw.TriangleDown(this,CurrentBar.ToString(), true, 1, High[1] + 2*TickSize, divergeCandledown);
					Alert("MyAlertsDiv" + CurrentBar.ToString(), Priority.High, "Deltadiv short", "", 10, Brushes.Black, Brushes.Lime);
				}
				}
				else
				{
					if((delta_closecum[1] > delta_closecum[2]) && Low[1] <= Low[2] && Low[1] <= Low[3] && (Stochastics(3, 14, 3).K[1] <= 20 || Stochastics(3, 14, 3).K[2] <= 20))
				{
				
					Draw.TriangleUp(this,CurrentBar.ToString(), true, 1, Low[1] - 2*TickSize, divergeCandleup);
					Alert("MyAlertsDiv" + CurrentBar.ToString(), Priority.High, "Deltadiv long", "", 10, Brushes.Black, Brushes.Lime);
				}		
				if((delta_closecum[1] < delta_closecum[2]) && High[1] >= High[2] && High[1] >= High[3] && (Stochastics(3, 14, 3).K[1] >= 80 || Stochastics(3, 14, 3).K[2] >= 80))
				{
				
					Draw.TriangleDown(this,CurrentBar.ToString(), true, 1, High[1] + 2*TickSize, divergeCandledown);
					Alert("MyAlertsDiv" + CurrentBar.ToString(), Priority.High, "Deltadiv short", "", 10, Brushes.Black, Brushes.Lime);
				}
				}
				}
				
			}
			else if (BarsInProgress == 1)
			{
			
				if (BarsArray[1].IsFirstBarOfSession)
					ResetValues(true,cdClose,cdClosecum);
			
				CalculateValues(false);
				
				
			}
		}
		
				
		private void CalculateValues(bool forceCurrentBar)
		{
			
			int 	indexOffset 	= BarsArray[1].Count - 1 - CurrentBars[1];
			bool 	inTransition 	= State == State.Realtime && indexOffset > 1;
			if (!inTransition && lastInTransition && !forceCurrentBar && Calculate == Calculate.OnBarClose)
				CalculateValues(true);
			
			bool 	useCurrentBar 	= State == State.Historical || inTransition || Calculate != Calculate.OnBarClose || forceCurrentBar;
			int 	whatBar 		= useCurrentBar ? CurrentBars[1] : Math.Min(CurrentBars[1] + 1, BarsArray[1].Count - 1);
		
			double 	volume 			= BarsArray[1].GetVolume(whatBar);
			double	price			= BarsArray[1].GetClose(whatBar);
			
			if (price >= BarsArray[1].GetAsk(whatBar) && volume>=MinSize)
			{
				buys += volume;
				buyscum += volume;
			}
			else if (price <= BarsArray[1].GetBid(whatBar) && volume>=MinSize)
			{
				sells += volume;
				sellscum += volume;
			}
			
			cdClose = buys - sells;
			cdClosecum = buyscum - sellscum;
	
			if (cdClose > cdHigh)
					cdHigh = cdClose;
			if (cdClosecum > cdHighcum)
					cdHighcum = cdClosecum;
	
			if (cdClose < cdLow)
					cdLow = cdClose;
			if (cdClosecum < cdLowcum)
					cdLowcum = cdClosecum;
	
			
			lastInTransition 	= inTransition;
		}
		
		private void SetValues(int barsAgo)
		{
		
		
			
			Values[0][barsAgo] = delta_open[barsAgo] = cdOpen;
			Values[1][barsAgo] = delta_high[barsAgo] = cdHigh;
			Values[2][barsAgo] = delta_low[barsAgo] = cdLow;
			Values[3][barsAgo] = delta_close[barsAgo] = cdClose;
			delta_opencum[barsAgo] = cdOpencum;
			delta_highcum[barsAgo] = cdHighcum;
			delta_lowcum[barsAgo] = cdLowcum;
			delta_closecum[barsAgo] = cdClosecum;
			
	
		}
		
		private void ResetValues(bool isNewSession, double openlevel, double openlevelcum)
		{
		
		
			if (CumulativeDelta)
				{
				cdOpen = cdClose = cdHigh = cdLow = openlevel;
				}
			if (!CumulativeDelta)
				{
				cdOpen = cdClose = cdHigh = cdLow = buys = sells = 0;
				}
			if (ShowDivs)
				{
				cdOpencum = cdClosecum = cdHighcum = cdLowcum = openlevelcum;
				}
				
			if (isNewSession)
			{
				cdOpen = cdClose = cdHigh = cdLow = buys = sells = 0;
				cdOpencum = cdClosecum = cdHighcum = cdLowcum = buyscum = sellscum = 0;
			}
			isReset = true;
		}
		
		public override string DisplayName
		{
		  get { return "Volumdelta"; }
		}
		
		#region Miscellaneous
	
		protected override void OnRender(ChartControl chartControl, ChartScale chartScale)
		{
			base.OnRender(chartControl, chartScale);

			barPaintWidth = Math.Max(3, 1 + 2 * ((int)ChartBars.Properties.ChartStyle.BarWidth - 1) + 2 * ShadowWidth);
	

            for (int idx = ChartBars.FromIndex; idx <= ChartBars.ToIndex; idx++)
            {
                if (idx - Displacement < 0 || idx - Displacement >= BarsArray[0].Count || (idx - Displacement < BarsRequiredToPlot))
                    continue;

                x					= ChartControl.GetXByBarIndex(ChartBars, idx);
                y1					= chartScale.GetYByValue(delta_open.GetValueAt(idx));
                y2					= chartScale.GetYByValue(delta_high.GetValueAt(idx));
                y3					= chartScale.GetYByValue(delta_low.GetValueAt(idx));
                y4					= chartScale.GetYByValue(delta_close.GetValueAt(idx));

				reuseVector1.X		= x;
				reuseVector1.Y		= y2;
				reuseVector2.X		= x;
				reuseVector2.Y		= y3;

				RenderTarget.DrawLine(reuseVector1, reuseVector2, dxmBrushes["shadowColor"].DxBrush);

				if (y4 == y1)
				{
					reuseVector1.X	= (x - barPaintWidth / 2);
					reuseVector1.Y	= y1;
					reuseVector2.X	= (x + barPaintWidth / 2);
					reuseVector2.Y	= y1;

					RenderTarget.DrawLine(reuseVector1, reuseVector2, dxmBrushes["shadowColor"].DxBrush);
				}
				else
				{
					if (y4 > y1)
					{
						UpdateRect(ref reuseRect, (x - barPaintWidth / 2), y1, barPaintWidth, (y4 - y1));
						RenderTarget.FillRectangle(reuseRect, dxmBrushes["barColorDown"].DxBrush);
					}
					else
					{
						UpdateRect(ref reuseRect, (x - barPaintWidth / 2), y4, barPaintWidth, (y1 - y4));
						RenderTarget.FillRectangle(reuseRect, dxmBrushes["barColorUp"].DxBrush);
					}

					UpdateRect(ref reuseRect, ((x - barPaintWidth / 2) + (ShadowWidth / 2)), Math.Min(y4, y1), (barPaintWidth - ShadowWidth + 2), Math.Abs(y4 - y1));
					RenderTarget.DrawRectangle(reuseRect, dxmBrushes["shadowColor"].DxBrush);
				}
            }
		}
		public override void OnRenderTargetChanged()
		{		
			try
			{
				foreach (KeyValuePair<string, DXMediaMap> item in dxmBrushes)
				{
					if (item.Value.DxBrush != null)
						item.Value.DxBrush.Dispose();

					if (RenderTarget != null)
						item.Value.DxBrush = item.Value.MediaBrush.ToDxBrush(RenderTarget);					
				}
			}
			catch (Exception exception)
			{
			}
		}

		private void UpdateRect(ref SharpDX.RectangleF updateRectangle, float x, float y, float width, float height)
		{
			updateRectangle.X		= x;
			updateRectangle.Y		= y;
			updateRectangle.Width	= width;
			updateRectangle.Height	= height;
		}

		private void UpdateRect(ref SharpDX.RectangleF rectangle, int x, int y, int width, int height)
		{
			UpdateRect(ref rectangle, (float)x, (float)y, (float)width, (float)height);
		}
		#endregion
		
		#region Properties
		[Browsable(false)]
		public class DXMediaMap
		{
			public SharpDX.Direct2D1.Brush		DxBrush;
			public System.Windows.Media.Brush	MediaBrush;
		}
		
		[NinjaScriptProperty]
		[XmlIgnore]
		[Display(Name="BarColorDown", Order=4, GroupName= "Optics")]
		public Brush BarColorDown
		{
			get { return dxmBrushes["barColorDown"].MediaBrush; }
			set { dxmBrushes["barColorDown"].MediaBrush = value; }
		}

		[Browsable(false)]
		public string BarColorDownSerializable
		{
			get { return Serialize.BrushToString(BarColorDown); }
			set { BarColorDown = Serialize.StringToBrush(value); }
		}

		[NinjaScriptProperty]
		[XmlIgnore]
		[Display(Name="BarColorUp", Order=5, GroupName= "Optics")]
		public Brush BarColorUp
		{
			get { return dxmBrushes["barColorUp"].MediaBrush; }
			set { dxmBrushes["barColorUp"].MediaBrush = value; }
		}

		[Browsable(false)]
		public string BarColorUpSerializable
		{
			get { return Serialize.BrushToString(BarColorUp); }
			set { BarColorUp = Serialize.StringToBrush(value); }
		}
		
		[NinjaScriptProperty]
		[XmlIgnore]
		[Display(Name="ShadowColor", Order=6, GroupName="Optics")]
		public Brush ShadowColor
		{
			get { return dxmBrushes["shadowColor"].MediaBrush; }
			set { dxmBrushes["shadowColor"].MediaBrush = value; }
		}

		[Browsable(false)]
		public string ShadowColorSerializable
		{
			get { return Serialize.BrushToString(ShadowColor); }
			set { ShadowColor = Serialize.StringToBrush(value); }
		}

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name="ShadowWidth", Order=7, GroupName= "Optics")]
		public int ShadowWidth
		{ get; set; }
		

		[Browsable(false)]
		[XmlIgnore]
		public Series<double> DeltaOpen
		{
			get { return Values[0]; }
		}
		
		[Browsable(false)]
		[XmlIgnore]
		public Series<double> DeltaHigh
		{
			get { return Values[1]; }
		}
		
		[Browsable(false)]
		[XmlIgnore]
		public Series<double> DeltaLow
		{
			get { return Values[2]; }
		}
		
				
		[Browsable(false)]
		[XmlIgnore]
		public Series<double> DeltaClose
		{
			get { return Values[3]; }
		}
		
		[NinjaScriptProperty]
		[Display(Name="Cumulative Delta", Description="Enable cumulative delta", Order=1, GroupName="Parameters")]
		public bool CumulativeDelta
		{ get; set; }
		
		[Range(0, int.MaxValue)]
		[NinjaScriptProperty]
		[Display(Name="Size Filter", Description="Size filtering", Order=2, GroupName="Parameters")]
		public int MinSize
		{ get; set; }
		
		[NinjaScriptProperty]
		[Display(Name="Show Delta Divergences", Description="Enable to show volume cumulative delta divergences", Order=3, GroupName="Parameters")]
/// <summary>
///   [Cumulative Delta must be enabled] era antes
/// </summary>
		public bool ShowDivs
		{ get; set; }
		
		
		[Browsable(false)]	// this line prevents the data series from being displayed in the indicator properties dialog, do not remove
        [XmlIgnore()]		// this line ensures that the indicator can be saved/recovered as part of a chart template, do not remove
        public Series<double> DeltasOpen
        {
            get { return delta_open; }
        }	
		
		[Browsable(false)]	// this line prevents the data series from being displayed in the indicator properties dialog, do not remove
        [XmlIgnore()]		// this line ensures that the indicator can be saved/recovered as part of a chart template, do not remove
        public Series<double> DeltasHigh
        {
            get { return delta_high; }
        }	
		
		[Browsable(false)]	// this line prevents the data series from being displayed in the indicator properties dialog, do not remove
        [XmlIgnore()]		// this line ensures that the indicator can be saved/recovered as part of a chart template, do not remove
        public Series<double> DeltasClose
        {
            get { return delta_close; }
        }	
		
		[Browsable(false)]	// this line prevents the data series from being displayed in the indicator properties dialog, do not remove
        [XmlIgnore()]		// this line ensures that the indicator can be saved/recovered as part of a chart template, do not remove
        public Series<double> DeltasLow
        {
            get { return delta_low; }
        }	

/// Nuevo para mostrar divergencias condelta normal 23.05.2024
		/// 
		[Browsable(false)]	// this line prevents the data series from being displayed in the indicator properties dialog, do not remove
        [XmlIgnore()]		// this line ensures that the indicator can be saved/recovered as part of a chart template, do not remove
        public Series<double> DeltasOpencum
        {
            get { return delta_opencum; }
        }	
		
		[Browsable(false)]	// this line prevents the data series from being displayed in the indicator properties dialog, do not remove
        [XmlIgnore()]		// this line ensures that the indicator can be saved/recovered as part of a chart template, do not remove
        public Series<double> DeltasHighcum
        {
            get { return delta_highcum; }
        }	
		
		[Browsable(false)]	// this line prevents the data series from being displayed in the indicator properties dialog, do not remove
        [XmlIgnore()]		// this line ensures that the indicator can be saved/recovered as part of a chart template, do not remove
        public Series<double> DeltasClosecum
        {
            get { return delta_closecum; }
        }	
		
		[Browsable(false)]	// this line prevents the data series from being displayed in the indicator properties dialog, do not remove
        [XmlIgnore()]		// this line ensures that the indicator can be saved/recovered as part of a chart template, do not remove
        public Series<double> DeltasLowcum
        {
            get { return delta_lowcum; }
        }	
		
		#endregion
		
	}
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private Volumdelta[] cacheVolumdelta;
		public Volumdelta Volumdelta(Brush barColorDown, Brush barColorUp, Brush shadowColor, int shadowWidth, bool cumulativeDelta, int minSize, bool showDivs)
		{
			return Volumdelta(Input, barColorDown, barColorUp, shadowColor, shadowWidth, cumulativeDelta, minSize, showDivs);
		}

		public Volumdelta Volumdelta(ISeries<double> input, Brush barColorDown, Brush barColorUp, Brush shadowColor, int shadowWidth, bool cumulativeDelta, int minSize, bool showDivs)
		{
			if (cacheVolumdelta != null)
				for (int idx = 0; idx < cacheVolumdelta.Length; idx++)
					if (cacheVolumdelta[idx] != null && cacheVolumdelta[idx].BarColorDown == barColorDown && cacheVolumdelta[idx].BarColorUp == barColorUp && cacheVolumdelta[idx].ShadowColor == shadowColor && cacheVolumdelta[idx].ShadowWidth == shadowWidth && cacheVolumdelta[idx].CumulativeDelta == cumulativeDelta && cacheVolumdelta[idx].MinSize == minSize && cacheVolumdelta[idx].ShowDivs == showDivs && cacheVolumdelta[idx].EqualsInput(input))
						return cacheVolumdelta[idx];
			return CacheIndicator<Volumdelta>(new Volumdelta(){ BarColorDown = barColorDown, BarColorUp = barColorUp, ShadowColor = shadowColor, ShadowWidth = shadowWidth, CumulativeDelta = cumulativeDelta, MinSize = minSize, ShowDivs = showDivs }, input, ref cacheVolumdelta);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.Volumdelta Volumdelta(Brush barColorDown, Brush barColorUp, Brush shadowColor, int shadowWidth, bool cumulativeDelta, int minSize, bool showDivs)
		{
			return indicator.Volumdelta(Input, barColorDown, barColorUp, shadowColor, shadowWidth, cumulativeDelta, minSize, showDivs);
		}

		public Indicators.Volumdelta Volumdelta(ISeries<double> input , Brush barColorDown, Brush barColorUp, Brush shadowColor, int shadowWidth, bool cumulativeDelta, int minSize, bool showDivs)
		{
			return indicator.Volumdelta(input, barColorDown, barColorUp, shadowColor, shadowWidth, cumulativeDelta, minSize, showDivs);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.Volumdelta Volumdelta(Brush barColorDown, Brush barColorUp, Brush shadowColor, int shadowWidth, bool cumulativeDelta, int minSize, bool showDivs)
		{
			return indicator.Volumdelta(Input, barColorDown, barColorUp, shadowColor, shadowWidth, cumulativeDelta, minSize, showDivs);
		}

		public Indicators.Volumdelta Volumdelta(ISeries<double> input , Brush barColorDown, Brush barColorUp, Brush shadowColor, int shadowWidth, bool cumulativeDelta, int minSize, bool showDivs)
		{
			return indicator.Volumdelta(input, barColorDown, barColorUp, shadowColor, shadowWidth, cumulativeDelta, minSize, showDivs);
		}
	}
}

#endregion
