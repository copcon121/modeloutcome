"""
Build Labeled Dataset
Processes all P2 events with SMC labeler + outcome calculator
Exports labeled events for ML training and backtesting
"""

import sys
from pathlib import Path
import json
import torch
import logging
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase3_label_backtest import config
from phase3_label_backtest.loader import DataLoader
from phase3_label_backtest.labeler import SMCLabeler
from phase3_label_backtest.outcome import OutcomeCalculator

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


def main():
    print("="*80)
    print("PHASE 3: BUILD LABELED DATASET")
    print("="*80)
    
    # Load data
    print("\n[1/4] Loading data...")
    loader = DataLoader()
    loader.load_all()
    
    # Initialize labeler and outcome calculator
    print("\n[2/4] Initializing SMC labeler and outcome calculator...")
    labeler = SMCLabeler()
    outcome_calc = OutcomeCalculator()
    
    # Process all P2 events
    print(f"\n[3/4] Processing {len(loader.sequences)} P2 events...")
    
    labeled_events = []
    sequences_list = []
    labels_list = []
    meta_list = []
    
    for event_id in tqdm(range(len(loader.sequences)), desc="Labeling events"):
        try:
            # Get event data
            event = loader.get_p2_event(event_id)
            
            # Determine signal
            signal = labeler.determine_signal(event.features)
            
            # Get future bars for outcome
            future_bars = loader.get_future_bars(event.global_bar_index, config.MAX_HOLD_BARS)
            
            # Calculate outcome (returns None for skip signals)
            outcome = outcome_calc.calculate_trade(signal, event.features, future_bars)
            
           # Prepare event record
            event_record = {
                'event_id': event_id,
                'timestamp': event.timestamp,
                'global_bar_index': event.global_bar_index,
                'signal_side': signal.side,
                'signal_reason': signal.reason,
            }
            
            if outcome:
                event_record.update({
                    'entry_price': outcome.entry_price,
                    'sl_price': outcome.sl_price,
                    'tp_price': outcome.tp_price,
                    'risk_ticks': outcome.risk_ticks,
                    'rr_ratio': outcome.rr_ratio,
                    'hit': outcome.hit,
                    'outcome_rr': outcome.outcome_rr,
                    'hold_bars': outcome.hold_bars,
                    'exit_price': outcome.exit_price,
                    'max_runup_rr': outcome.max_runup_rr,
                    'max_drawdown_rr': outcome.max_drawdown_rr,
                })
            else:
                event_record.update({
                    'entry_price': None,
                    'sl_price': None,
                    'tp_price': None,
                    'risk_ticks': None,
                    'rr_ratio': None,
                    'hit': 'skip',
                    'outcome_rr': 0.0,
                    'hold_bars': 0,
                    'exit_price': None,
                    'max_runup_rr': 0.0,
                    'max_drawdown_rr': 0.0,
                })
            
            labeled_events.append(event_record)
            
            # For ML dataset
            sequences_list.append(event.sequence)
            
            # Map label to int: 0=long, 1=short, 2=skip
            label_map = {'long': 0, 'short': 1, 'skip': 2}
            labels_list.append(label_map[signal.side])
            
            meta_list.append({
                'timestamp': event.timestamp,
                'signal_side': signal.side,
                'outcome_rr': event_record['outcome_rr'],
                'hit': event_record['hit']
            })
            
        except Exception as e:
            logger.error(f"Error processing event {event_id}: {e}")
            continue
    
    # Export labeled events (JSONL)
    print(f"\n[4/4] Exporting labeled dataset...")
    
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export JSONL
    jsonl_path = Path(config.EVENTS_LABELED_PATH)
    with open(jsonl_path, 'w') as f:
        for event in labeled_events:
            f.write(json.dumps(event) + '\n')
    
    print(f"\n[OK] Exported {len(labeled_events):,} events to {jsonl_path}")
    
    # Export PyTorch dataset
    dataset = {
        'sequences': torch.tensor(sequences_list, dtype=torch.float32),
        'labels': torch.tensor(labels_list, dtype=torch.long),
        'meta': meta_list
    }
    
    torch.save(dataset, config.DATASET_PT_PATH)
    print(f"[OK] Exported ML dataset to {config.DATASET_PT_PATH}")
    print(f"     Sequences shape: {dataset['sequences'].shape}")
    print(f"     Labels shape: {dataset['labels'].shape}")
    
    # Print statistics
    labeler.print_stats()
    
    print("\n" + "="*80)
    print("DATASET BUILD COMPLETE!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - {jsonl_path}")
    print(f"  - {config.DATASET_PT_PATH}")
    print(f"\nReady for backtest and ML training!")


if __name__ == '__main__':
    main()
