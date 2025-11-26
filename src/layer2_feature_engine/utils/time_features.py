"""
Time-based Feature Extraction
Session detection and cyclical time encoding
"""

from datetime import datetime
from typing import Dict
import math


def extract_time_features(ts: datetime) -> Dict[str, float]:
    """
    Extract time-based features from timestamp

    Args:
        ts: Datetime timestamp

    Returns:
        Dict of time features
    """
    # Convert to EST/EDT (US market time)
    # For simplicity, we'll use UTC hour as-is
    # TODO: Add proper timezone conversion

    hour = ts.hour
    minute = ts.minute
    day_of_week = ts.weekday()  # 0=Monday, 6=Sunday

    features = {}

    # Session detection (based on UTC approximation)
    # Asia: 22:00-08:00 UTC (approx)
    # Europe: 08:00-16:00 UTC
    # US: 14:00-22:00 UTC (9:30-17:00 EST)

    features['session_asia'] = 1.0 if (hour >= 22 or hour < 8) else 0.0
    features['session_europe'] = 1.0 if (8 <= hour < 16) else 0.0
    features['session_us'] = 1.0 if (14 <= hour < 22) else 0.0

    # Cyclical encoding of time (sin/cos for periodicity)
    minutes_of_day = hour * 60 + minute
    max_minutes = 24 * 60

    features['time_sin'] = math.sin(2 * math.pi * minutes_of_day / max_minutes)
    features['time_cos'] = math.cos(2 * math.pi * minutes_of_day / max_minutes)

    # Day of week (one-hot style, but normalized)
    features['day_of_week'] = day_of_week / 6.0  # Normalize to [0, 1]

    # Is weekend (Saturday or Sunday)
    features['is_weekend'] = 1.0 if day_of_week >= 5 else 0.0

    # Time of day categories
    features['is_morning'] = 1.0 if 6 <= hour < 12 else 0.0
    features['is_afternoon'] = 1.0 if 12 <= hour < 18 else 0.0
    features['is_evening'] = 1.0 if 18 <= hour < 24 else 0.0
    features['is_night'] = 1.0 if 0 <= hour < 6 else 0.0

    return features


def is_market_hours(ts: datetime, market: str = 'US') -> bool:
    """
    Check if timestamp is during market hours

    Args:
        ts: Datetime timestamp
        market: 'US', 'Europe', or 'Asia'

    Returns:
        True if during market hours
    """
    hour = ts.hour
    day_of_week = ts.weekday()

    # Skip weekends
    if day_of_week >= 5:
        return False

    if market == 'US':
        # US market: 9:30-16:00 EST (approx 14:30-21:00 UTC)
        return 14 <= hour < 21
    elif market == 'Europe':
        # Europe market: 8:00-16:30 GMT (approx)
        return 8 <= hour < 17
    elif market == 'Asia':
        # Asia market: varies, approximate Tokyo hours
        return 0 <= hour < 6 or 23 <= hour < 24
    else:
        return True
