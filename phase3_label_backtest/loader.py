"""
Data Loader for Phase 3
Loads P2 sequences, full bar stream, and manages data alignment
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase3_label_backtest import config

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


@dataclass
class P2EventData:
    """Data for a single P2 event"""
    event_id: int
    global_bar_index: int  # Index in full stream
    timestamp: str
    sequence: np.ndarray  # [60, 62] context
    features: Dict[str, float]  # Current bar features


class DataLoader:
    """
    Phase 3 Data Loader
    
    Loads and manages:
    - Full bar stream (67k bars)
    - P2 sequences (20k events)
    - Mapping between them
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        
        # Paths
        self.features_all_path = self.project_root / config.FEATURES_ALL_PATH
        self.features_ohlcv_path = self.project_root / config.FEATURES_WITH_OHLCV_PATH
        self.sequences_path = self.project_root / config.SEQUENCES_P2_PATH
        self.indices_path = self.project_root / config.SEQUENCES_INDICES_PATH
        
        # Data containers
        self.df_all: Optional[pd.DataFrame] = None
        self.df_ohlcv: Optional[pd.DataFrame] = None
        self.sequences: Optional[np.ndarray] = None
        self.p2_indices: Optional[np.ndarray] = None
        
    def load_all(self):
        """Load all required data"""
        logger.info("Loading Phase 3 data...")
        
        # Load full stream with OHLCV
        logger.info(f"Loading full stream: {self.features_ohlcv_path}")
        self.df_ohlcv = pd.read_csv(self.features_ohlcv_path)
        
        # Add timestamp column if not present (features_all doesn't have it)
        if 'timestamp' not in self.df_ohlcv.columns:
            logger.warning("Timestamp column not found in CSV - will use P2 filtered CSV for timestamps")
            # Create dummy timestamps for now
            self.df_ohlcv['timestamp'] = pd.date_range(start='2025-09-01', periods=len(self.df_ohlcv), freq='1min')
        else:
            self.df_ohlcv['timestamp'] = pd.to_datetime(self.df_ohlcv['timestamp'])
        
        logger.info(f"  Loaded {len(self.df_ohlcv):,} bars")
        
        # Load P2 sequences
        logger.info(f"Loading P2 sequences: {self.sequences_path}")
        self.sequences = np.load(self.sequences_path)
        logger.info(f"  Loaded sequences: {self.sequences.shape}")
        
        # Load P2 indices (if exists)
        if self.indices_path.exists():
            logger.info(f"Loading P2 indices: {self.indices_path}")
            self.p2_indices = np.load(self.indices_path)
            logger.info(f"  Loaded {len(self.p2_indices):,} indices")
        else:
            logger.warning(f"P2 indices file not found: {self.indices_path}")
            logger.info("  Assuming sequential P2 events from start")
            # Create dummy indices (will need to be corrected)
            self.p2_indices = np.arange(len(self.sequences))
        
        self._verify_data()
        
    def _verify_data(self):
        """Verify data integrity"""
        logger.info("\nVerifying data integrity...")
        
        # Check shapes
        n_events, seq_len, n_features = self.sequences.shape
        logger.info(f"  P2 events: {n_events:,}")
        logger.info(f"  Sequence length: {seq_len}")
        logger.info(f"  Features per bar: {n_features}")
        
        # Check alignment
        if self.p2_indices is not None:
            if len(self.p2_indices) != n_events:
                logger.error(f"  Mismatch: {len(self.p2_indices)} indices vs {n_events} sequences")
            else:
                logger.info(f"  ✓ Indices align with sequences")
            
            # Check indices are valid
            max_idx = self.p2_indices.max()
            if max_idx >= len(self.df_ohlcv):
                logger.error(f"  Max P2 index {max_idx} exceeds bar count {len(self.df_ohlcv)}")
            else:
                logger.info(f"  ✓ All indices within bounds (max: {max_idx})")
        
        # Check feature columns
        expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'ext_trend_dir', 'in_bull_fvg', 'in_bear_fvg', 
                        'in_bull_ob', 'in_bear_ob']
        missing = [c for c in expected_cols if c not in self.df_ohlcv.columns]
        if missing:
            logger.error(f"  Missing columns: {missing}")
        else:
            logger.info(f"  ✓ All required columns present")
        
        logger.info("Data verification complete!\n")
    
    def get_p2_event(self, event_id: int) -> P2EventData:
        """Get data for a specific P2 event"""
        if event_id < 0 or event_id >= len(self.sequences):
            raise IndexError(f"Event ID {event_id} out of range [0, {len(self.sequences)})")
        
        # Get sequence
        sequence = self.sequences[event_id]  # [60, 62]
        
        # Get global bar index
        global_bar_index = int(self.p2_indices[event_id])
        
        # Get bar data
        bar_row = self.df_ohlcv.iloc[global_bar_index]
        
        # Extract features as dict
        features = bar_row.to_dict()
        
        return P2EventData(
            event_id=event_id,
            global_bar_index=global_bar_index,
            timestamp=str(bar_row['timestamp']),
            sequence=sequence,
            features=features
        )
    
    def get_future_bars(self, global_bar_index: int, n_bars: int = 100) -> pd.DataFrame:
        """Get future bars after a given index"""
        start_idx = global_bar_index + 1
        end_idx = min(start_idx + n_bars, len(self.df_ohlcv))
        
        return self.df_ohlcv.iloc[start_idx:end_idx]
    
    def get_all_p2_events(self) -> List[P2EventData]:
        """Get all P2 events"""
        logger.info(f"Loading all {len(self.sequences)} P2 events...")
        events = [self.get_p2_event(i) for i in range(len(self.sequences))]
        logger.info(f"Loaded {len(events):,} events")
        return events


def main():
    """Test data loader"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and verify Phase 3 data')
    parser.add_argument('--verify', action='store_true', help='Verify data only')
    parser.add_argument('--sample', type=int, default=5, help='Show N sample events')
    args = parser.parse_args()
    
    print("="*80)
    print("PHASE 3 DATA LOADER")
    print("="*80)
    
    loader = DataLoader()
    loader.load_all()
    
    if args.verify:
        print("\n✓ Data verification complete!")
        return
    
    # Show sample events
    print(f"\nShowing {args.sample} sample events:\n")
    for i in range(min(args.sample, len(loader.sequences))):
        event = loader.get_p2_event(i)
        print(f"Event {event.event_id}:")
        print(f"  Timestamp: {event.timestamp}")
        print(f"  Global bar index: {event.global_bar_index}")
        print(f"  Sequence shape: {event.sequence.shape}")
        print(f"  ext_trend_dir: {event.features.get('ext_trend_dir', 'N/A')}")
        print(f"  in_bull_fvg: {event.features.get('in_bull_fvg', 'N/A')}")
        print(f"  in_bear_fvg: {event.features.get('in_bear_fvg', 'N/A')}")
        print(f"  close: {event.features.get('close', 'N/A')}")
        print()


if __name__ == '__main__':
    main()
