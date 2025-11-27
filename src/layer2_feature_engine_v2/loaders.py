"""
Phase 2 Feature Engine V2 - JSONL Loader
Parse JSONL files from NinjaTrader exporter to RawBar objects
"""

import json
import logging
from datetime import datetime
from typing import Iterator, Optional
from pathlib import Path

from .schema import RawBar


logger = logging.getLogger(__name__)


def parse_timestamp(ts_str: str) -> datetime:
    """
    Parse timestamp from exporter format
    Format: "2025-11-17T20:01:00.0000000"
    """
    try:
        # Handle microseconds (7 digits after decimal)
        if '.' in ts_str:
            base, frac = ts_str.split('.')
            # Take only first 6 digits for microseconds
            frac = frac[:6].ljust(6, '0')
            ts_str = f"{base}.{frac}"
        
        return datetime.fromisoformat(ts_str)
    except Exception as e:
        logger.warning(f"Failed to parse timestamp '{ts_str}': {e}")
        # Fallback: try without microseconds
        try:
            return datetime.fromisoformat(ts_str.split('.')[0])
        except:
            raise ValueError(f"Cannot parse timestamp: {ts_str}")


def parse_raw_bar(json_obj: dict) -> Optional[RawBar]:
    """
    Parse JSON object to RawBar
    
    Expected format:
    {
      "symbol": "GC 02-26",
      "timeframe": "M1",
      "timestamp": "2025-11-17T20:01:00.0000000",
      "bar_index": 1260,
      "bar": {
        "o": 4047.8,
        "h": 4049.1,
        "l": 4043.2,
        "c": 4048.9,
        "volume": 850,
        "delta": -77,
        "buy_volume": 386.5,
        "sell_volume": 463.5,
        "best_bid": 4048.9,
        "best_ask": 4048.9
      },
      "tick_features": {
        "tick_speed": 1404,
        "aggr_buy_speed": 386.5,
        "aggr_sell_speed": 463.5,
        "price_speed": 5.9
      }
    }
    """
    try:
        # Extract metadata
        symbol = json_obj.get('symbol', 'UNKNOWN')
        timeframe = json_obj.get('timeframe', 'M1')
        timestamp = parse_timestamp(json_obj['timestamp'])
        bar_index = int(json_obj.get('bar_index', 0))
        
        # Extract bar data (shorthand: o/h/l/c)
        bar_data = json_obj['bar']
        o = float(bar_data['o'])
        h = float(bar_data['h'])
        l = float(bar_data['l'])
        c = float(bar_data['c'])
        volume = float(bar_data['volume'])
        delta = float(bar_data.get('delta', 0.0))
        buy_volume = float(bar_data.get('buy_volume', 0.0))
        sell_volume = float(bar_data.get('sell_volume', 0.0))
        best_bid = float(bar_data.get('best_bid', c))
        best_ask = float(bar_data.get('best_ask', c))
        
        # Extract tick features
        tick_features = json_obj['tick_features']
        tick_speed = float(tick_features.get('tick_speed', 0.0))
        aggr_buy_speed = float(tick_features.get('aggr_buy_speed', 0.0))
        aggr_sell_speed = float(tick_features.get('aggr_sell_speed', 0.0))
        price_speed = float(tick_features.get('price_speed', 0.0))
        
        # Create RawBar
        return RawBar(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            bar_index=bar_index,
            o=o,
            h=h,
            l=l,
            c=c,
            volume=volume,
            delta=delta,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            best_bid=best_bid,
            best_ask=best_ask,
            tick_speed=tick_speed,
            aggr_buy_speed=aggr_buy_speed,
            aggr_sell_speed=aggr_sell_speed,
            price_speed=price_speed
        )
        
    except KeyError as e:
        logger.warning(f"Missing required field in JSON: {e}")
        return None
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid data type in JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        return None


def iter_raw_bars(path: str, skip_errors: bool = True) -> Iterator[RawBar]:
    """
    Iterator over RawBar objects from JSONL file
    
    Args:
        path: Path to JSONL file
        skip_errors: If True, skip malformed lines; if False, raise on error
        
    Yields:
        RawBar objects
        
    Example:
        >>> for bar in iter_raw_bars("gc_export.jsonl"):
        ...     print(f"{bar.timestamp} {bar.c}")
    """
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    logger.info(f"Loading bars from {path}")
    
    total_lines = 0
    parsed_bars = 0
    skipped_lines = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip comments (lines starting with #)
            if line.startswith('#'):
                continue
            
            try:
                # Parse JSON
                json_obj = json.loads(line)
                
                # Parse to RawBar
                bar = parse_raw_bar(json_obj)
                
                if bar is not None:
                    parsed_bars += 1
                    yield bar
                else:
                    skipped_lines += 1
                    if not skip_errors:
                        raise ValueError(f"Failed to parse line {line_num}")
                        
            except json.JSONDecodeError as e:
                skipped_lines += 1
                logger.warning(f"Invalid JSON at line {line_num}: {e}")
                if not skip_errors:
                    raise
            except Exception as e:
                skipped_lines += 1
                logger.warning(f"Error at line {line_num}: {e}")
                if not skip_errors:
                    raise
    
    logger.info(f"Loaded {parsed_bars} bars from {total_lines} lines (skipped {skipped_lines})")


def load_raw_bars_list(path: str, max_bars: Optional[int] = None) -> list:
    """
    Load all bars into a list (convenience function)
    
    Args:
        path: Path to JSONL file
        max_bars: Maximum number of bars to load (None = all)
        
    Returns:
        List of RawBar objects
    """
    bars = []
    for bar in iter_raw_bars(path):
        bars.append(bar)
        if max_bars and len(bars) >= max_bars:
            break
    return bars


def validate_jsonl_file(path: str) -> dict:
    """
    Validate JSONL file and return statistics
    
    Returns:
        dict with:
        - total_lines: int
        - valid_bars: int
        - invalid_lines: int
        - first_timestamp: datetime
        - last_timestamp: datetime
        - symbols: set
        - timeframes: set
    """
    bars = list(iter_raw_bars(path))
    
    if not bars:
        return {
            'total_lines': 0,
            'valid_bars': 0,
            'invalid_lines': 0,
            'first_timestamp': None,
            'last_timestamp': None,
            'symbols': set(),
            'timeframes': set()
        }
    
    symbols = set(bar.symbol for bar in bars)
    timeframes = set(bar.timeframe for bar in bars)
    
    return {
        'total_lines': len(bars),
        'valid_bars': len(bars),
        'invalid_lines': 0,  # Tracked in iter_raw_bars logger
        'first_timestamp': bars[0].timestamp,
        'last_timestamp': bars[-1].timestamp,
        'symbols': symbols,
        'timeframes': timeframes,
        'bar_count': len(bars)
    }


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python loaders.py <jsonl_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Validate file
    print("Validating JSONL file...")
    stats = validate_jsonl_file(file_path)
    print(f"\nFile Statistics:")
    print(f"  Total bars: {stats['bar_count']}")
    print(f"  Symbols: {stats['symbols']}")
    print(f"  Timeframes: {stats['timeframes']}")
    print(f"  Time range: {stats['first_timestamp']} to {stats['last_timestamp']}")
    
    # Show first 5 bars
    print("\nFirst 5 bars:")
    for i, bar in enumerate(iter_raw_bars(file_path)):
        if i >= 5:
            break
        print(f"  {bar.timestamp} | O:{bar.o} H:{bar.h} L:{bar.l} C:{bar.c} | "
              f"V:{bar.volume} D:{bar.delta}")
