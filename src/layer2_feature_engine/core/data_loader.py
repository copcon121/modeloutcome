"""
Data Loader - Load raw JSONL data from NinjaTrader exports
Converts JSONL format to RawBar objects
"""

import json
from pathlib import Path
from typing import Iterator, List
from datetime import datetime
import logging

from .schema import RawBar

logger = logging.getLogger(__name__)


def iter_raw_bars(jsonl_path: str) -> Iterator[RawBar]:
    """
    Load raw bars from JSONL file exported by NinjaTrader

    Args:
        jsonl_path: Path to .jsonl file

    Yields:
        RawBar objects parsed from JSON

    Example JSON format:
        {
            "symbol": "GC 02-26",
            "timeframe": "M1",
            "timestamp": "2025-11-16T23:02:00.0000000",
            "bar_index": 1,
            "bar": {
                "o": 4119.4,
                "h": 4121.2,
                "l": 4118.0,
                "c": 4118.3,
                "volume": 93,
                "delta": 0,
                "buy_volume": 46.5,
                "sell_volume": 46.5,
                "best_bid": 4118.3,
                "best_ask": 4118.3
            },
            "tick_features": {
                "tick_speed": 174,
                "aggr_buy_speed": 46.5,
                "aggr_sell_speed": 46.5,
                "price_speed": 3.2
            }
        }
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    logger.info(f"Loading bars from {jsonl_path}")

    with open(path, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Parse timestamp (handle .NET DateTime format)
                timestamp_str = data['timestamp']
                # Remove .NET DateTime trailing zeros and 'Z'
                timestamp_str = timestamp_str.split('.')[0]
                ts = datetime.fromisoformat(timestamp_str)

                # Extract bar data
                bar_data = data['bar']
                tick_data = data.get('tick_features', {})

                raw_bar = RawBar(
                    ts=ts,
                    open=float(bar_data['o']),
                    high=float(bar_data['h']),
                    low=float(bar_data['l']),
                    close=float(bar_data['c']),
                    volume=float(bar_data['volume']),
                    delta=float(bar_data.get('delta', 0.0)),
                    buy_volume=float(bar_data.get('buy_volume', 0.0)),
                    sell_volume=float(bar_data.get('sell_volume', 0.0)),
                    tick_speed=float(tick_data.get('tick_speed', 0.0)),
                    aggr_buy_speed=float(tick_data.get('aggr_buy_speed', 0.0)),
                    aggr_sell_speed=float(tick_data.get('aggr_sell_speed', 0.0)),
                    price_speed=float(tick_data.get('price_speed', 0.0))
                )

                yield raw_bar

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Error parsing line {line_num}: {e}")
                logger.debug(f"Line content: {line}")
                continue


def load_raw_bars(jsonl_path: str) -> List[RawBar]:
    """
    Load all bars from JSONL file into a list

    Args:
        jsonl_path: Path to .jsonl file

    Returns:
        List of RawBar objects, sorted by timestamp
    """
    bars = list(iter_raw_bars(jsonl_path))

    # Sort by timestamp (should already be sorted, but ensure it)
    bars.sort(key=lambda b: b.ts)

    logger.info(f"Loaded {len(bars)} bars")
    if bars:
        logger.info(f"First bar: {bars[0].ts}")
        logger.info(f"Last bar:  {bars[-1].ts}")

    return bars


def load_raw_bars_window(
    jsonl_path: str,
    start_idx: int = 0,
    end_idx: int = None
) -> List[RawBar]:
    """
    Load a window of bars from JSONL file
    Useful for processing large files in chunks

    Args:
        jsonl_path: Path to .jsonl file
        start_idx: Starting bar index (0-based)
        end_idx: Ending bar index (exclusive). If None, load to end

    Returns:
        List of RawBar objects in the specified range
    """
    bars = []

    for idx, bar in enumerate(iter_raw_bars(jsonl_path)):
        if idx < start_idx:
            continue
        if end_idx is not None and idx >= end_idx:
            break
        bars.append(bar)

    logger.info(f"Loaded {len(bars)} bars (indices {start_idx} to {start_idx + len(bars) - 1})")
    return bars


def get_bar_count(jsonl_path: str) -> int:
    """
    Count total number of bars in JSONL file

    Args:
        jsonl_path: Path to .jsonl file

    Returns:
        Number of bars in file
    """
    count = sum(1 for _ in iter_raw_bars(jsonl_path))
    return count


def print_sample_bars(jsonl_path: str, n: int = 5) -> None:
    """
    Print first n bars from JSONL file for inspection

    Args:
        jsonl_path: Path to .jsonl file
        n: Number of bars to print
    """
    print(f"\n{'='*80}")
    print(f"Sample bars from: {jsonl_path}")
    print(f"{'='*80}\n")

    for i, bar in enumerate(iter_raw_bars(jsonl_path)):
        if i >= n:
            break

        print(f"Bar {i+1}:")
        print(f"  Timestamp:   {bar.ts}")
        print(f"  OHLC:        O={bar.open:.2f}, H={bar.high:.2f}, L={bar.low:.2f}, C={bar.close:.2f}")
        print(f"  Volume:      {bar.volume:.0f}")
        print(f"  Delta:       {bar.delta:.0f} (Buy: {bar.buy_volume:.1f}, Sell: {bar.sell_volume:.1f})")
        print(f"  Tick Speed:  {bar.tick_speed:.0f} ticks")
        print(f"  Aggr Buy:    {bar.aggr_buy_speed:.1f}")
        print(f"  Aggr Sell:   {bar.aggr_sell_speed:.1f}")
        print(f"  Price Speed: {bar.price_speed:.2f}")
        print(f"  Range:       {bar.range_size:.2f}")
        print(f"  Body:        {bar.body_size:.2f} ({'Bullish' if bar.is_bullish else 'Bearish'})")
        print()

    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Test the loader
    import sys

    if len(sys.argv) > 1:
        jsonl_path = sys.argv[1]
    else:
        jsonl_path = "/home/user/modeloutcome/data/raw/smc_export_gc_m1_v3.jsonl"

    print(f"Testing data loader with: {jsonl_path}\n")

    # Count bars
    print(f"Counting bars...")
    total = get_bar_count(jsonl_path)
    print(f"Total bars: {total}\n")

    # Print sample
    print_sample_bars(jsonl_path, n=5)

    # Load all bars
    print(f"Loading all bars...")
    bars = load_raw_bars(jsonl_path)
    print(f"Loaded {len(bars)} bars successfully!")
