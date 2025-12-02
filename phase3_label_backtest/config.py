"""
Phase 3 Configuration
Centralized settings for SMC rule-based labeling and backtesting
"""

# Instrument Settings
SYMBOL = "GC"
TICK_SIZE = 0.1  # GC futures tick size

# Signal Generation Rules
BUFFER_TICKS = 2  # Buffer for SL/TP beyond zone boundaries
TARGET_RR = 3.0   # Risk-reward ratio for take profit
MAX_HOLD_BARS = 100  # Maximum bars to hold position before force close

# Zone Detection
NEAR_ZONE_TICKS = 5  # Distance to consider "near" a zone (in ticks)

# Outcome Computation
MIN_FUTURE_BARS = 10  # Minimum future bars required to compute outcome

# Dataset Paths (relative to project root) with env overrides for custom runs
import os
FEATURES_ALL_PATH = os.getenv("PHASE2_FEATURES_ALL", "output/production_10weeks_v2/features_all_10weeks.csv")
# Full stream with timestamp/OHLCV + key SMC flags aligned 1:1 with bars
FEATURES_WITH_OHLCV_PATH = os.getenv("PHASE2_FEATURES_WITH_OHLCV", "output/production_10weeks_v2/features_all_10weeks.csv")
SEQUENCES_P2_PATH = os.getenv("PHASE2_SEQUENCES", "output/production_10weeks_v2/sequences_p2_10weeks_sequences.npy")
SEQUENCES_INDICES_PATH = os.getenv("PHASE2_INDICES", "output/production_10weeks_v2/sequences_p2_10weeks_indices.npy")

# Output Paths
OUTPUT_DIR = os.getenv("PHASE3_OUTPUT_DIR", "output/phase3_labeled")
EVENTS_LABELED_PATH = f"{OUTPUT_DIR}/events_p2_labeled_10weeks.jsonl"
DATASET_PT_PATH = f"{OUTPUT_DIR}/dataset_p2_labeled.pt"
BACKTEST_REPORT_PATH = f"{OUTPUT_DIR}/backtest_report.txt"
EQUITY_CURVE_PATH = f"{OUTPUT_DIR}/equity_curve.csv"

# Session Times (for backtest grouping)
SESSIONS = {
    'Asia': (0, 8),    # 00:00 - 08:00 UTC
    'Europe': (8, 14),  # 08:00 - 14:00 UTC
    'NY': (14, 22),     # 14:00 - 22:00 UTC
}

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "logs/phase3_labeling.log"
