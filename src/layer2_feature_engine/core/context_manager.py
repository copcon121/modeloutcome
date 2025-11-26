"""
Context Manager - Orchestrates feature building across all modules
Maintains sliding window of bars and coordinates feature extraction
"""

from collections import deque
from typing import List, Dict, Optional
import logging
import numpy as np

from .schema import RawBar, FeatureBar, Record, SMCStructure, VolumeProfileState, L2DepthState
from .normalizer import Normalizer

# Import feature extraction modules
from ..smc import swing, structure, zones
from ..volume_profile import vp_builder
from ..orderflow_l2 import l2_features
from ..utils import time_features


logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages a sliding window of bars and builds feature vectors
    Orchestrates all feature extraction modules
    """

    def __init__(
        self,
        context_len: int = 60,
        max_history: int = 100,
        normalizer: Optional[Normalizer] = None
    ):
        """
        Args:
            context_len: Number of bars to use for ML context
            max_history: Maximum bars to keep in memory
            normalizer: Optional normalizer (if None, features are unnormalized)
        """
        self.context_len = context_len
        self.max_history = max_history
        self.normalizer = normalizer

        # Sliding window of raw bars
        self.bars: deque[RawBar] = deque(maxlen=max_history)

        # Cached states (updated incrementally)
        self.smc_structure: Optional[SMCStructure] = None
        self.vp_state: Optional[VolumeProfileState] = None
        self.l2_state: Optional[L2DepthState] = None

    def add_bar(self, bar: RawBar) -> None:
        """
        Add a new bar to the context window

        Args:
            bar: RawBar to add
        """
        self.bars.append(bar)
        logger.debug(f"Added bar at {bar.ts}, total bars: {len(self.bars)}")

    def add_bars_batch(self, bars: List[RawBar]) -> None:
        """
        Add multiple bars at once (useful for initialization)

        Args:
            bars: List of RawBar objects
        """
        for bar in bars:
            self.bars.append(bar)
        logger.info(f"Added {len(bars)} bars, total bars: {len(self.bars)}")

    def build_features(self) -> List[FeatureBar]:
        """
        Build feature vectors for all bars in the current window
        Orchestrates all feature extraction modules

        Returns:
            List of FeatureBar objects (one per bar in window)
        """
        if len(self.bars) < self.context_len:
            logger.warning(f"Not enough bars: {len(self.bars)} < {self.context_len}")
            return []

        # Convert deque to list for easier indexing
        bars_list = list(self.bars)

        # Step 1: Detect SMC structure (swings, BOS, CHoCH, zones)
        self.smc_structure = self._build_smc_structure(bars_list)

        # Step 2: Build Volume Profile
        self.vp_state = self._build_volume_profile(bars_list)

        # Step 3: Aggregate L2 depth (if available)
        self.l2_state = self._build_l2_state(bars_list)

        # Step 4: Extract features for each bar
        feature_bars = []
        for i in range(len(bars_list)):
            features = self._extract_bar_features(i, bars_list)

            # Apply normalization if available
            if self.normalizer:
                features = self.normalizer.transform(features)

            feature_bar = FeatureBar(
                ts=bars_list[i].ts,
                features=features
            )
            feature_bars.append(feature_bar)

        logger.info(f"Built features for {len(feature_bars)} bars")
        return feature_bars

    def get_latest_context(self) -> List[FeatureBar]:
        """
        Get the latest context_len bars as features

        Returns:
            List of FeatureBar (length = context_len)
        """
        all_features = self.build_features()
        if len(all_features) < self.context_len:
            return all_features
        return all_features[-self.context_len:]

    def create_record(self, label: Optional[int] = None) -> Record:
        """
        Create a Record object with current context

        Args:
            label: Optional label (for training data)

        Returns:
            Record object
        """
        context = self.get_latest_context()

        return Record(
            context=context,
            label=label,
            entry_price=self.bars[-1].close if self.bars else None,
            atr=self._compute_atr(list(self.bars)) if self.bars else None
        )

    # ========== Private Helper Methods ==========

    def _build_smc_structure(self, bars_list: List[RawBar]) -> SMCStructure:
        """Build SMC structure (swings, BOS, CHoCH, zones)"""
        # Detect swing points
        swing_highs, swing_lows = swing.detect_swings(bars_list, lookback=2)

        # Detect structure breaks
        struct_flags = structure.compute_structure_flags(bars_list, swing_highs, swing_lows)

        # Detect zones (OB, FVG)
        obs_up, obs_down = zones.detect_order_blocks(bars_list)
        fvgs_up, fvgs_down = zones.detect_fair_value_gaps(bars_list)

        return SMCStructure(
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            bos_up_indices=struct_flags.get('bos_up', []),
            bos_down_indices=struct_flags.get('bos_down', []),
            choch_up_indices=struct_flags.get('choch_up', []),
            choch_down_indices=struct_flags.get('choch_down', []),
            active_obs_up=obs_up,
            active_obs_down=obs_down,
            active_fvgs_up=fvgs_up,
            active_fvgs_down=fvgs_down
        )

    def _build_volume_profile(self, bars_list: List[RawBar]) -> VolumeProfileState:
        """Build Volume Profile for the current window"""
        return vp_builder.build_volume_profile(bars_list, price_bins=50)

    def _build_l2_state(self, bars_list: List[RawBar]) -> L2DepthState:
        """Aggregate Level 2 depth data"""
        return l2_features.compute_l2_state(bars_list)

    def _extract_bar_features(self, i: int, bars_list: List[RawBar]) -> Dict[str, float]:
        """
        Extract all features for bar at index i
        Combines OHLCV, SMC, VP, L2, and time features
        """
        bar = bars_list[i]
        features = {}

        # 1. Basic OHLCV features (normalized)
        features.update(self._extract_ohlcv_features(i, bars_list))

        # 2. SMC features
        features.update(structure.extract_bar_structure_features(i, self.smc_structure))
        features.update(zones.extract_zone_features(i, bars_list, self.smc_structure))

        # 3. Volume Profile features
        features.update(vp_builder.extract_bar_vp_features(i, self.vp_state, bar))

        # 4. Level 2 features
        features.update(l2_features.extract_bar_l2_features(i, self.l2_state, bar))

        # 5. Time features
        features.update(time_features.extract_time_features(bar.ts))

        return features

    def _extract_ohlcv_features(self, i: int, bars_list: List[RawBar]) -> Dict[str, float]:
        """Extract basic OHLCV and derived features"""
        bar = bars_list[i]

        # Compute ATR for normalization (use last 14 bars)
        atr = self._compute_atr(bars_list[max(0, i-13):i+1])
        if atr < 1e-8:
            atr = 1.0  # Avoid division by zero

        features = {
            # Raw OHLCV (normalized by close)
            'open_norm': bar.open / bar.close if bar.close != 0 else 1.0,
            'high_norm': bar.high / bar.close if bar.close != 0 else 1.0,
            'low_norm': bar.low / bar.close if bar.close != 0 else 1.0,
            'close_norm': 1.0,  # Reference

            # Volume (log-scaled)
            'volume_log': np.log1p(bar.volume),

            # Bar characteristics
            'range_atr': bar.range_size / atr,
            'body_atr': bar.body_size / atr,
            'wick_upper_atr': (bar.high - max(bar.open, bar.close)) / atr,
            'wick_lower_atr': (min(bar.open, bar.close) - bar.low) / atr,
            'is_bullish': 1.0 if bar.is_bullish else 0.0,

            # Orderflow (if available)
            'delta': bar.delta if bar.delta is not None else 0.0,
            'buy_volume_pct': bar.buy_volume / bar.volume if bar.volume > 0 else 0.5,
            'sell_volume_pct': bar.sell_volume / bar.volume if bar.volume > 0 else 0.5,
        }

        return features

    def _compute_atr(self, bars: List[RawBar], period: int = 14) -> float:
        """Compute Average True Range"""
        if len(bars) < 2:
            return 0.0

        true_ranges = []
        for i in range(1, len(bars)):
            high_low = bars[i].high - bars[i].low
            high_close = abs(bars[i].high - bars[i-1].close)
            low_close = abs(bars[i].low - bars[i-1].close)
            tr = max(high_low, high_close, low_close)
            true_ranges.append(tr)

        # Take average of last 'period' true ranges
        atr = sum(true_ranges[-period:]) / min(len(true_ranges), period)
        return atr
