"""
Load Shadow Trading Logs

Loads and normalizes shadow_trading_log.jsonl for analysis.
"""

from pathlib import Path
import json
import pandas as pd
from datetime import datetime


def load_shadow_log(log_path):
    """
    Load shadow trading log from JSONL
    
    Args:
        log_path: Path to shadow_trading_log.jsonl
    
    Returns:
        pd.DataFrame with normalized columns
    """
    if not Path(log_path).exists():
        raise FileNotFoundError(f"Shadow log not found: {log_path}")
    
    records = []
    with open(log_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if not records:
        return pd.DataFrame()
    
    # Normalize
    df = pd.DataFrame(records)
    
    # Extract meta fields if present
    if 'meta' in df.columns:
        meta_df = pd.json_normalize(df['meta'])
        # Prefix meta columns
        meta_df.columns = ['meta_' + col for col in meta_df.columns]
        df = pd.concat([df.drop('meta', axis=1), meta_df], axis=1)
    
    # Rename for clarity
    column_mapping = {
        'timestamp_server': 'timestamp',
        'meta_symbol': 'symbol',
        'meta_timeframe': 'timeframe',
        'meta_event_time': 'event_time',
        'meta_bar_index': 'bar_index'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    print(f"[load_shadow_log] Loaded {len(df):,} shadow entries")
    
    return df


if __name__ == "__main__":
    # Test
    from pathlib import Path
    
    root = Path(__file__).parent.parent
    log_path = root / "output/phase5_quality/shadow_trading_log.jsonl"
    
    if log_path.exists():
        df = load_shadow_log(log_path)
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nSample:\n{df.head()}")
    else:
        print(f"Shadow log not yet created: {log_path}")
        print("(Will be created when API receives requests)")
