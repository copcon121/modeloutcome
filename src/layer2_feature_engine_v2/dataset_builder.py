"""
Phase 2 Feature Engine V2 - Dataset Builder
Main entry point for generating ML-ready datasets
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from .schema import RawBar, FeatureBar
from .config import SMCConfig
from .context_manager import SMCContextManager

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Dataset Builder
    
    Main entry point for building ML datasets from JSONL exports.
    
    Workflow:
    1. Load raw bars from JSONL
    2. Process each bar through Context Manager
    3. Build feature sequences (sliding window)
    4. Export to CSV/NPY format
    
    Output formats:
    - CSV: Human-readable, for validation
    - NPY: Efficient numpy arrays for ML training
    """
    
    def __init__(self, config: SMCConfig, tick_size: float):
        """
        Initialize dataset builder
        
        Args:
            config: SMC configuration
            tick_size: Instrument tick size
        """
        self.config = config
        self.tick_size = tick_size
        self.context_manager = SMCContextManager(config, tick_size)
        
        logger.info("DatasetBuilder initialized")
    
    def load_jsonl(self, file_path: str, max_bars: Optional[int] = None) -> List[RawBar]:
        """
        Load raw bars from JSONL file
        
        Args:
            file_path: Path to JSONL file
            max_bars: Optional limit on number of bars
            
        Returns:
            List of RawBar objects
        """
        raw_bars = []
        
        logger.info(f"Loading bars from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_bars and i >= max_bars:
                    break
                
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    timestamp_str = data.get('timestamp', '')
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
                    raw_bar = RawBar(
                        symbol=data.get('symbol', 'GC'),
                        timeframe=data.get('timeframe', 'M1'),
                        timestamp=timestamp,
                        bar_index=data.get('bar_index', i),
                        o=data.get('open', 0.0),
                        h=data.get('high', 0.0),
                        l=data.get('low', 0.0),
                        c=data.get('close', 0.0),
                        volume=data.get('volume', 0.0),
                        delta=data.get('delta', 0.0),
                        buy_volume=data.get('buy_volume', 0.0),
                        sell_volume=data.get('sell_volume', 0.0),
                        best_bid=data.get('best_bid', data.get('close', 0.0)),
                        best_ask=data.get('best_ask', data.get('close', 0.0)),
                        tick_speed=data.get('tick_speed', 0.0),
                        aggr_buy_speed=data.get('aggr_buy_speed', 0.0),
                        aggr_sell_speed=data.get('aggr_sell_speed', 0.0),
                        price_speed=data.get('price_speed', 0.0)
                    )
                    
                    raw_bars.append(raw_bar)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse bar {i}: {e}")
                    continue
        
        logger.info(f"Loaded {len(raw_bars)} bars")
        return raw_bars
    
    def build_features(self, raw_bars: List[RawBar]) -> List[FeatureBar]:
        """
        Process raw bars through Context Manager to build features
        
        Args:
            raw_bars: List of raw bars
            
        Returns:
            List of feature bars
        """
        feature_bars = []
        
        logger.info(f"Processing {len(raw_bars)} bars through Context Manager")
        
        for i, raw_bar in enumerate(raw_bars):
            feature_bar = self.context_manager.update(raw_bar)
            feature_bars.append(feature_bar)
            
            if (i + 1) % 100 == 0:
                logger.debug(f"Processed {i + 1}/{len(raw_bars)} bars")
        
        logger.info(f"Built {len(feature_bars)} feature bars")
        return feature_bars
    
    def build_sequences(
        self,
        feature_bars: List[FeatureBar],
        window_size: int = 60,
        stride: int = 1
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Build sliding window sequences for ML
        
        Args:
            feature_bars: List of feature bars
            window_size: Sequence length (default 60 bars = 1 hour on M1)
            stride: Step size for sliding window
            
        Returns:
            sequences: Array of shape [N, window_size, num_features]
            indices: List of ending bar indices for each sequence
        """
        logger.info(f"Building sequences: window={window_size}, stride={stride}")
        
        # Convert feature bars to arrays
        feature_arrays = []
        for fb in feature_bars:
            feature_arrays.append(fb.to_array())
        
        feature_arrays = np.array(feature_arrays)  # [num_bars, num_features]
        num_bars, num_features = feature_arrays.shape
        
        logger.info(f"Feature array shape: {feature_arrays.shape}")
        
        # Build sequences using sliding window
        sequences = []
        indices = []
        
        for i in range(window_size - 1, num_bars, stride):
            sequence = feature_arrays[i - window_size + 1:i + 1]  # [window_size, num_features]
            sequences.append(sequence)
            indices.append(i)
        
        sequences = np.array(sequences)  # [N, window_size, num_features]
        
        logger.info(f"Built {len(sequences)} sequences of shape {sequences.shape}")
        return sequences, indices
    
    def export_csv(
        self,
        feature_bars: List[FeatureBar],
        output_path: str
    ):
        """
        Export feature bars to CSV (for validation/inspection)
        
        Args:
            feature_bars: List of feature bars
            output_path: Path to output CSV file
        """
        logger.info(f"Exporting {len(feature_bars)} feature bars to CSV: {output_path}")
        
        # Convert to list of dicts
        records = [fb.to_dict() for fb in feature_bars]
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Export
        df.to_csv(output_path, index=False)
        logger.info(f"Exported to {output_path}")
    
    def export_npy(
        self,
        sequences: np.ndarray,
        indices: List[int],
        output_dir: str,
        prefix: str = "dataset"
    ):
        """
        Export sequences to NPY format (for ML training)
        
        Args:
            sequences: Array of shape [N, window_size, num_features]
            indices: List of ending bar indices
            output_dir: Directory to save files
            prefix: Prefix for output files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sequences_path = output_dir / f"{prefix}_sequences.npy"
        indices_path = output_dir / f"{prefix}_indices.npy"
        
        logger.info(f"Exporting sequences to: {sequences_path}")
        np.save(sequences_path, sequences)
        np.save(indices_path, np.array(indices))
        
        logger.info(f"Exported {len(sequences)} sequences")
        logger.info(f"  Shape: {sequences.shape}")
        logger.info(f"  Files: {sequences_path}, {indices_path}")
    
    def build_dataset(
        self,
        jsonl_path: str,
        output_dir: str,
        window_size: int = 60,
        stride: int = 1,
        max_bars: Optional[int] = None,
        export_csv_flag: bool = True,
        export_npy_flag: bool = True
    ) -> Tuple[np.ndarray, List[int], List[FeatureBar]]:
        """
        Main entry point: Build complete dataset from JSONL
        
        Args:
            jsonl_path: Path to input JSONL file
            output_dir: Directory for outputs
            window_size: Sequence window size
            stride: Sliding window stride
            max_bars: Optional limit on bars to process
            export_csv_flag: Export CSV for inspection
            export_npy_flag: Export NPY for ML
            
        Returns:
            sequences: [N, window_size, num_features]
            indices: Ending bar indices for each sequence
            feature_bars: All feature bars (for debugging)
        """
        logger.info("="*80)
        logger.info("DATASET BUILDER - Starting")
        logger.info("="*80)
        
        # 1. Load raw bars
        raw_bars = self.load_jsonl(jsonl_path, max_bars)
        
        # 2. Build features
        feature_bars = self.build_features(raw_bars)
        
        # 3. Build sequences
        sequences, indices = self.build_sequences(feature_bars, window_size, stride)
        
        # 4. Export
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if export_csv_flag:
            csv_path = output_dir / "features.csv"
            self.export_csv(feature_bars, str(csv_path))
        
        if export_npy_flag:
            self.export_npy(sequences, indices, str(output_dir))
        
        logger.info("="*80)
        logger.info("DATASET BUILDER - Complete")
        logger.info(f"  Input bars: {len(raw_bars)}")
        logger.info(f"  Feature bars: {len(feature_bars)}")
        logger.info(f"  Sequences: {len(sequences)}")
        logger.info(f"  Output: {output_dir}")
        logger.info("="*80)
        
        return sequences, indices, feature_bars


def build_context_dataset(
    jsonl_path: str,
    output_dir: str,
    config: SMCConfig,
    tick_size: float,
    window_size: int = 60,
    stride: int = 1,
    max_bars: Optional[int] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Convenience function: Build dataset with default settings
    
    Args:
        jsonl_path: Path to JSONL file
        output_dir: Output directory
        config: SMC configuration
        tick_size: Instrument tick size
        window_size: Sequence length (default 60)
        stride: Window stride (default 1)
        max_bars: Optional bar limit
        
    Returns:
        sequences: [N, window_size, num_features]
        indices: Ending bar indices
    """
    builder = DatasetBuilder(config, tick_size)
    sequences, indices, _ = builder.build_dataset(
        jsonl_path, output_dir, window_size, stride, max_bars
    )
    return sequences, indices
