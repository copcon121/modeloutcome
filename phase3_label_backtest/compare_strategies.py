"""
Strategy Comparison: Internal vs External OB
Analyzes signal counts for 3 different OB strategies
"""

import pandas as pd
import numpy as np

print("="*80)
print("SMC STRATEGY COMPARISON - SIGNAL COUNTS")
print("="*80)

# Load P2-filtered features (already filtered, 20k events)
features_path = "output/production_10weeks/features_p2_filtered_10weeks.csv"
print(f"\nLoading: {features_path}")

df = pd.read_csv(features_path)
print(f"  Loaded {len(df):,} P2 events")

# Convert boolean fields (read as float from CSV)
bool_cols = ['in_bull_fvg', 'in_bear_fvg', 'int_in_bull_ob', 'int_in_bear_ob', 
             'ext_in_bull_ob', 'ext_in_bear_ob']
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(bool)

# Verify OB fields
print(f"\nOB Fields:")
print(f"  int_in_bull_ob: {'int_in_bull_ob' in df.columns}")
print(f"  ext_in_bull_ob: {'ext_in_bull_ob' in df.columns}")

# Zone occurrence counts
print("\n" + "="*80)
print("ZONE OCCURRENCE ANALYSIS")
print("="*80)

bull_fvg = df['in_bull_fvg'].sum()
bear_fvg = df['in_bear_fvg'].sum()
int_bull_ob = df['int_in_bull_ob'].sum()
int_bear_ob = df['int_in_bear_ob'].sum()
ext_bull_ob = df['ext_in_bull_ob'].sum()
ext_bear_ob = df['ext_in_bear_ob'].sum()

print(f"\nFVG Zones:")
print(f"  Bullish: {bull_fvg:,} ({bull_fvg/len(df)*100:.1f}%)")
print(f"  Bearish: {bear_fvg:,} ({bear_fvg/len(df)*100:.1f}%)")

print(f"\nInternal OB (Wave 5):")
print(f"  Bullish: {int_bull_ob:,} ({int_bull_ob/len(df)*100:.1f}%)")
print(f"  Bearish: {int_bear_ob:,} ({int_bear_ob/len(df)*100:.1f}%)")
print(f"  Total: {int_bull_ob + int_bear_ob:,} ({(int_bull_ob + int_bear_ob)/len(df)*100:.1f}%)")

print(f"\nExternal OB (Wave 50):")
print(f"  Bullish: {ext_bull_ob:,} ({ext_bull_ob/len(df)*100:.1f}%)")
print(f"  Bearish: {ext_bear_ob:,} ({ext_bear_ob/len(df)*100:.1f}%)")
print(f"  Total: {ext_bull_ob + ext_bear_ob:,} ({(ext_bull_ob + ext_bear_ob)/len(df)*100:.1f}%)")

# Strategy simulation
print("\n" + "="*80)
print("STRATEGY SIMULATION (Trend + Zone Alignment)")
print("="*80)

strategies = {}

# Strategy 1: External OB Only (Current Baseline)
ext_long = ((df['ext_trend_dir'] > 0) & 
            (df['in_bull_fvg'] | df['ext_in_bull_ob'])).sum()
ext_short = ((df['ext_trend_dir'] < 0) & 
             (df['in_bear_fvg'] | df['ext_in_bear_ob'])).sum()

strategies['ext_only'] = {
    'name': 'External OB Only (Baseline)',
    'long': ext_long,
    'short': ext_short,
    'total': ext_long + ext_short
}

# Strategy 2: Internal OB Only
int_long = ((df['ext_trend_dir'] > 0) & 
            (df['in_bull_fvg'] | df['int_in_bull_ob'])).sum()
int_short = ((df['ext_trend_dir'] < 0) & 
             (df['in_bear_fvg'] | df['int_in_bear_ob'])).sum()

strategies['int_only'] = {
    'name': 'Internal OB Only (Scalp)',
    'long': int_long,
    'short': int_short,
    'total': int_long + int_short
}

# Strategy 3: Combined (Int + Ext)
comb_long = ((df['ext_trend_dir'] > 0) & 
             (df['in_bull_fvg'] | df['int_in_bull_ob'] | df['ext_in_bull_ob'])).sum()
comb_short = ((df['ext_trend_dir'] < 0) & 
              (df['in_bear_fvg'] | df['int_in_bear_ob'] | df['ext_in_bear_ob'])).sum()

strategies['combined'] = {
    'name': 'Combined (Int + Ext)',
    'long': comb_long,
    'short': comb_short,
    'total': comb_long + comb_short
}

# Print comparison
for key, strat in strategies.items():
    print(f"\n{strat['name']}:")
    print(f"  Long:  {strat['long']:,} ({strat['long']/len(df)*100:.1f}%)")
    print(f"  Short: {strat['short']:,} ({strat['short']/len(df)*100:.1f}%)")
    print(f"  TOTAL: {strat['total']:,} ({strat['total']/len(df)*100:.1f}%)")

# Analysis
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

ext_total = strategies['ext_only']['total']
int_total = strategies['int_only']['total']
comb_total = strategies['combined']['total']

print(f"\n1. Signal Counts:")
print(f"   External OB:  {ext_total:,}")
print(f"   Internal OB:  {int_total:,} ({int_total/ext_total*100 if ext_total > 0 else 0:.1f}% of External)")
print(f"   Combined:     {comb_total:,} ({comb_total/ext_total*100 if ext_total > 0 else 0:.1f}% of External)")

added = comb_total - ext_total
print(f"\n2. Internal OB contribution:")
print(f"   Adds {added:,} NEW signals (+{added/ext_total*100 if ext_total > 0 else 0:.1f}%)")

print(f"\n3. Recommendation:")
if added < ext_total * 0.1:
    print(f"   - Internal OB adds VERY FEW signals - Stick with External")
    print(f"   - Proceed to Phase 4 with External OB baseline")
elif added > ext_total * 0.3:
    print(f"   - Internal OB adds MANY signals - Test quality via backtest")
    print(f"   - Run risk sweep on all 3 strategies before Phase 4")
else:
    print(f"   - Internal OB adds MODERATE signals - Worth testing")
    print(f"   - Quick backtest recommended to validate quality")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
Based on results above:

If Internal OB adds <10% signals:
  - Skip Internal OB testing
  - Use External OB baseline for Phase 4
  
If Internal OB adds 10-30% signals:
  - Run quick backtest on Int vs Ext
  - Choose best for Phase 4

If Internal OB adds >30% signals:
  - Full risk sweep on all 3 strategies
  - Compare expectancy/winrate/drawdown
  - Choose optimal config for Phase 4
""")
