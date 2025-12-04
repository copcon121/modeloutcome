"""
Dataset builder for STATE-ENC v1

Builds sequence samples from raw bar-level JSONL files.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import numpy as np

from .config import StateEncDatasetConfig
from .features_spec import (
    FEATURE_SPEC,
    get_feature_names,
    get_feature_defaults,
    validate_bar_features
)
from .normalization import FeatureNormalizer, compute_session_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO8601 timestamp string"""
    # Handle various formats
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts}")


def determine_session(hour: int, config: StateEncDatasetConfig) -> str:
    """Determine session based on hour"""
    sessions = config.sessions
    
    # ASIA: typically 18:00 - 02:00 (crosses midnight)
    asia = sessions.get("ASIA", {"start_hour": 18, "end_hour": 2})
    if asia["start_hour"] > asia["end_hour"]:
        # Crosses midnight
        if hour >= asia["start_hour"] or hour < asia["end_hour"]:
            return "ASIA"
    else:
        if asia["start_hour"] <= hour < asia["end_hour"]:
            return "ASIA"
    
    # LDN: typically 02:00 - 08:00
    ldn = sessions.get("LDN", {"start_hour": 2, "end_hour": 8})
    if ldn["start_hour"] <= hour < ldn["end_hour"]:
        return "LDN"
    
    # NY: typically 08:00 - 17:00
    ny = sessions.get("NY", {"start_hour": 8, "end_hour": 17})
    if ny["start_hour"] <= hour < ny["end_hour"]:
        return "NY"
    
    return "UNKNOWN"


def get_session_id(session: str) -> int:
    """Convert session string to ID"""
    mapping = {"ASIA": 0, "LDN": 1, "NY": 2, "UNKNOWN": -1}
    return mapping.get(session, -1)


def load_raw_bars(path: str) -> List[Dict[str, Any]]:
    """Load raw bars from JSONL file"""
    bars = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                bar = json.loads(line)
                bars.append(bar)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
    
    logger.info(f"Loaded {len(bars)} bars from {path}")
    return bars


def enrich_bar_features(bar: Dict[str, Any], 
                        session_stats: Dict[str, float],
                        config: StateEncDatasetConfig) -> Dict[str, Any]:
    """
    Enrich bar with computed features if missing.
    
    Computes derived features like:
    - hl_range, body, wicks
    - volume_vs_session_avg
    - session position features
    """
    enriched = bar.copy()
    defaults = get_feature_defaults()
    
    # Core OHLCV
    o = enriched.get("o", enriched.get("open", 0))
    h = enriched.get("h", enriched.get("high", 0))
    l = enriched.get("l", enriched.get("low", 0))
    c = enriched.get("c", enriched.get("close", 0))
    
    enriched["o"] = o
    enriched["h"] = h
    enriched["l"] = l
    enriched["c"] = c
    
    # Derived OHLCV features
    if "hl_range" not in enriched:
        enriched["hl_range"] = h - l
    
    if "body" not in enriched:
        enriched["body"] = c - o
    
    if "upper_wick" not in enriched:
        enriched["upper_wick"] = h - max(o, c)
    
    if "lower_wick" not in enriched:
        enriched["lower_wick"] = min(o, c) - l
    
    if "bar_type" not in enriched:
        body = enriched["body"]
        hl = enriched["hl_range"]
        if hl > 0 and abs(body) / hl < 0.1:
            enriched["bar_type"] = 0  # doji
        elif body > 0:
            enriched["bar_type"] = 1  # bull
        else:
            enriched["bar_type"] = 2  # bear
    
    # Volume features
    volume = enriched.get("volume", 0)
    enriched["volume"] = volume
    
    session_avg_vol = session_stats.get("session_avg_volume", 1.0)
    if "volume_vs_session_avg" not in enriched:
        enriched["volume_vs_session_avg"] = volume / max(session_avg_vol, 1)
    
    # Range vs session
    session_avg_range = session_stats.get("session_avg_range", 1.0)
    if "range_vs_session_avg" not in enriched:
        enriched["range_vs_session_avg"] = enriched["hl_range"] / max(session_avg_range, 0.01)
    
    # True range (simplified if prev bar not available)
    if "true_range" not in enriched:
        enriched["true_range"] = enriched["hl_range"]
    
    # Delta features
    delta = enriched.get("delta", 0)
    enriched["delta"] = delta
    
    if "delta_abs" not in enriched:
        enriched["delta_abs"] = abs(delta)
    
    if "delta_sign" not in enriched:
        if delta > 0:
            enriched["delta_sign"] = 1
        elif delta < 0:
            enriched["delta_sign"] = -1
        else:
            enriched["delta_sign"] = 0
    
    if "delta_vs_volume" not in enriched:
        enriched["delta_vs_volume"] = delta / max(volume, 1)
    
    # Buy/sell volume
    buy_vol = enriched.get("buy_volume", 0)
    sell_vol = enriched.get("sell_volume", 0)
    enriched["buy_volume"] = buy_vol
    enriched["sell_volume"] = sell_vol
    
    if "buy_ratio" not in enriched:
        total = buy_vol + sell_vol
        enriched["buy_ratio"] = buy_vol / max(total, 1)
    
    if "sell_ratio" not in enriched:
        total = buy_vol + sell_vol
        enriched["sell_ratio"] = sell_vol / max(total, 1)
    
    if "imbalance_buy_sell" not in enriched:
        enriched["imbalance_buy_sell"] = (buy_vol - sell_vol) / max(volume, 1)
    
    # Session position
    session_high = session_stats.get("session_high", h)
    session_low = session_stats.get("session_low", l)
    enriched["session_high"] = session_high
    enriched["session_low"] = session_low
    
    if "pos_in_session_range" not in enriched:
        range_val = session_high - session_low
        if range_val > 0:
            enriched["pos_in_session_range"] = (c - session_low) / range_val
        else:
            enriched["pos_in_session_range"] = 0.5
    
    # Fill missing features with defaults
    for fname in get_feature_names():
        if fname not in enriched:
            enriched[fname] = defaults.get(fname, 0.0)
    
    return enriched


def compute_future_labels(bars: List[Dict], 
                          idx: int, 
                          config: StateEncDatasetConfig) -> Optional[Dict[str, Any]]:
    """
    Compute future labels for bar at index idx.
    
    Returns None if not enough future bars.
    """
    future_bars = config.future_bars
    
    if idx + future_bars >= len(bars):
        return None
    
    current_close = bars[idx].get("c", bars[idx].get("close", 0))
    
    # Future bars
    future_slice = bars[idx + 1: idx + 1 + future_bars]
    
    if len(future_slice) < future_bars:
        return None
    
    future_close = future_slice[-1].get("c", future_slice[-1].get("close", 0))
    future_highs = [b.get("h", b.get("high", 0)) for b in future_slice]
    future_lows = [b.get("l", b.get("low", 0)) for b in future_slice]
    
    # Future return
    if current_close > 0:
        future_return = (future_close - current_close) / current_close
    else:
        future_return = 0.0
    
    # Future range
    future_range = (max(future_highs) - min(future_lows)) / config.tick_size
    
    # Future direction
    if future_return > config.future_dir_threshold_up:
        future_dir = 1
    elif future_return < config.future_dir_threshold_down:
        future_dir = -1
    else:
        future_dir = 0
    
    return {
        "future_return_5": future_return,
        "future_range_5": future_range,
        "future_dir_5": future_dir
    }


def group_bars_by_session(bars: List[Dict], 
                          config: StateEncDatasetConfig) -> Dict[Tuple[str, str, str], List[Dict]]:
    """
    Group bars by (symbol, date, session).
    
    Returns dict mapping (symbol, date, session) -> list of bars
    """
    groups = defaultdict(list)
    
    for bar in bars:
        # Get symbol
        symbol = bar.get("symbol", config.symbol)
        
        # Get timestamp
        ts_str = bar.get("time", bar.get("timestamp", bar.get("datetime", "")))
        if not ts_str:
            logger.warning("Bar missing timestamp, skipping")
            continue
        
        try:
            ts = parse_timestamp(ts_str)
        except ValueError as e:
            logger.warning(f"Cannot parse timestamp '{ts_str}': {e}")
            continue
        
        date_str = ts.strftime("%Y-%m-%d")
        
        # Determine session
        session = bar.get("session", determine_session(ts.hour, config))
        
        # Add to group
        key = (symbol, date_str, session)
        groups[key].append(bar)
    
    # Sort each group by time
    for key in groups:
        groups[key].sort(key=lambda b: b.get("time", b.get("timestamp", "")))
    
    return groups


def build_samples_from_group(bars: List[Dict],
                             symbol: str,
                             date: str,
                             session: str,
                             config: StateEncDatasetConfig) -> List[Dict[str, Any]]:
    """
    Build sequence samples from a group of bars using sliding window.
    """
    samples = []
    N = config.sequence_length
    stride = config.stride
    future_bars = config.future_bars
    
    if len(bars) < N + future_bars:
        logger.debug(f"Group {symbol}/{date}/{session} has only {len(bars)} bars, need {N + future_bars}")
        return samples
    
    # Compute session stats for normalization
    session_stats = compute_session_stats(bars)
    
    # Enrich all bars
    enriched_bars = [enrich_bar_features(b, session_stats, config) for b in bars]
    
    # Sliding window
    for i in range(0, len(enriched_bars) - N - future_bars + 1, stride):
        seq_bars = enriched_bars[i: i + N]
        last_bar_idx = i + N - 1
        
        # Compute future labels
        aux = compute_future_labels(enriched_bars, last_bar_idx, config)
        if aux is None:
            continue
        
        # Get timestamps
        start_time = seq_bars[0].get("time", seq_bars[0].get("timestamp", ""))
        end_time = seq_bars[-1].get("time", seq_bars[-1].get("timestamp", ""))
        
        # Build sample
        sample = {
            "symbol": symbol,
            "tf": config.timeframe,
            "date": date,
            "session": session,
            "start_time": start_time,
            "end_time": end_time,
            "seq": seq_bars,
            "aux": aux
        }
        
        samples.append(sample)
    
    return samples


def build_state_enc_dataset(raw_bars_path: str,
                            output_path: str,
                            config: StateEncDatasetConfig) -> Dict[str, Any]:
    """
    Main function to build encoder dataset.
    
    Args:
        raw_bars_path: Path to input JSONL file with bar data
        output_path: Path to output JSONL file for samples
        config: Dataset configuration
        
    Returns:
        Summary statistics dict
    """
    logger.info(f"Building STATE-ENC dataset from {raw_bars_path}")
    logger.info(f"Config: N={config.sequence_length}, stride={config.stride}, future={config.future_bars}")
    
    # Load raw bars
    bars = load_raw_bars(raw_bars_path)
    
    if not bars:
        raise ValueError(f"No bars loaded from {raw_bars_path}")
    
    # Check for missing features (sample first bar)
    is_valid, missing = validate_bar_features(bars[0])
    if missing:
        logger.warning(f"Sample bar missing {len(missing)} features: {missing[:10]}...")
    
    # Group by session
    groups = group_bars_by_session(bars, config)
    logger.info(f"Found {len(groups)} session groups")
    
    # Build samples
    all_samples = []
    session_counts = defaultdict(int)
    dir_counts = {-1: 0, 0: 0, 1: 0}
    
    for (symbol, date, session), group_bars in groups.items():
        samples = build_samples_from_group(group_bars, symbol, date, session, config)
        all_samples.extend(samples)
        session_counts[session] += len(samples)
        
        for s in samples:
            dir_counts[s["aux"]["future_dir_5"]] += 1
    
    logger.info(f"Built {len(all_samples)} samples total")
    
    # Fit normalizer on all bars
    logger.info("Fitting normalizer...")
    all_bars_flat = []
    for sample in all_samples:
        all_bars_flat.extend(sample["seq"])
    
    normalizer = FeatureNormalizer()
    normalizer.fit(all_bars_flat)
    
    # Save normalizer config
    feature_config = {
        "feature_names": get_feature_names(),
        "feature_dim": len(get_feature_names()),
        "normalization": {
            "config": {
                "method": normalizer.config.method,
                "clip_zscore": normalizer.config.clip_zscore,
                "eps": normalizer.config.eps
            },
            "stats": {
                name: {
                    "mean": s.mean,
                    "std": s.std,
                    "min_val": s.min_val,
                    "max_val": s.max_val
                }
                for name, s in normalizer.stats.items()
            },
            "is_fitted": True
        }
    }
    
    # Save feature config
    Path(config.feature_config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config.feature_config_path, "w") as f:
        json.dump(feature_config, f, indent=2)
    logger.info(f"Saved feature config to {config.feature_config_path}")
    
    # Write samples to output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    
    logger.info(f"Saved {len(all_samples)} samples to {output_path}")
    
    # Summary
    summary = {
        "total_samples": len(all_samples),
        "total_bars": len(bars),
        "num_groups": len(groups),
        "samples_by_session": dict(session_counts),
        "future_dir_distribution": dir_counts,
        "sequence_length": config.sequence_length,
        "stride": config.stride,
        "feature_dim": len(get_feature_names())
    }
    
    logger.info("=" * 50)
    logger.info("Dataset Summary:")
    logger.info(f"  Total samples: {summary['total_samples']}")
    logger.info(f"  By session: {dict(session_counts)}")
    logger.info(f"  Future dir distribution: {dir_counts}")
    logger.info("=" * 50)
    
    return summary


if __name__ == "__main__":
    # Test with sample config
    config = StateEncDatasetConfig(
        raw_bars_path="data/bars_enhanced.jsonl",
        output_path="state_enc_v1/artifacts/encoder_dataset.jsonl",
        feature_config_path="state_enc_v1/artifacts/feature_config.json"
    )
    
    summary = build_state_enc_dataset(
        config.raw_bars_path,
        config.output_path,
        config
    )
    print(json.dumps(summary, indent=2))
