"""
Phase 2 Feature Engine V2
Clean rewrite matching NinjaTrader SMC indicator logic
"""

__version__ = "2.0.0"

from .config import SMCConfig, GC_M1_SMC_CONFIG, NQ_M1_SMC_CONFIG, DEFAULT_SMC_CONFIG
from .schema import (
    RawBar,
    InternalSwingState,
    ExternalSwingState,
    SMCState,
    FVGZone,
    OBZone,
    ZonesState,
    VolumeProfileState,
    FeatureBar
)
from .loaders import iter_raw_bars, load_raw_bars_list, validate_jsonl_file

__all__ = [
    # Config
    'SMCConfig',
    'GC_M1_SMC_CONFIG',
    'NQ_M1_SMC_CONFIG',
    'DEFAULT_SMC_CONFIG',
    
    # Schema
    'RawBar',
    'InternalSwingState',
    'ExternalSwingState',
    'SMCState',
    'FVGZone',
    'OBZone',
    'ZonesState',
    'VolumeProfileState',
    'FeatureBar',
    
    # Loaders
    'iter_raw_bars',
    'load_raw_bars_list',
    'validate_jsonl_file',
]
