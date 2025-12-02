import pandas as pd
import numpy as np
import glob
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("label_outcome.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def compute_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr.fillna(method='bfill')

def compute_outcomes(df, config):
    """
    Compute R-based outcomes for every bar.
    Label: 1 (Long), -1 (Short), 0 (Skip)
    """
    future_window = config.get('future_window', 60)
    target_R = config.get('target_R', 3.0)
    stop_R = config.get('stop_R', 1.0)
    
    # Calculate ATR
    df['atr'] = compute_atr(df)
    
    labels = np.zeros(len(df), dtype=int)
    max_up_Rs = np.zeros(len(df))
    max_down_Rs = np.zeros(len(df))
    
    # Iterate (vectorization is hard for path-dependent target/stop logic)
    # We'll use a loop for clarity and correctness
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atrs = df['atr'].values
    
    for i in range(len(df) - future_window):
        if i % 5000 == 0:
            logger.info(f"Processing bar {i}/{len(df)}")
            
        entry_price = closes[i]
        atr = atrs[i]
        
        if atr < 1e-8: continue
        
        # Future window
        window_highs = highs[i+1 : i+1+future_window]
        window_lows = lows[i+1 : i+1+future_window]
        
        # Check for Target/Stop hit
        # Long: Target = Entry + 3R, Stop = Entry - 1R
        long_target = entry_price + (target_R * atr)
        long_stop = entry_price - (stop_R * atr)
        
        # Short: Target = Entry - 3R, Stop = Entry + 1R
        short_target = entry_price - (target_R * atr)
        short_stop = entry_price + (stop_R * atr)
        
        # Find first hit index
        # This is slow in Python loop. 
        # Optimization: We can just check if High >= Target before Low <= Stop
        
        # Long Outcome
        long_hit_idx = np.argmax(window_highs >= long_target)
        long_stop_idx = np.argmax(window_lows <= long_stop)
        
        # argmax returns 0 if not found, so we check values
        long_hit = window_highs[long_hit_idx] >= long_target
        long_stopped = window_lows[long_stop_idx] <= long_stop
        
        label = 0
        
        if long_hit and (not long_stopped or long_hit_idx < long_stop_idx):
            label = 1 # Long Win
        else:
            # Check Short
            short_hit_idx = np.argmax(window_lows <= short_target)
            short_stop_idx = np.argmax(window_highs >= short_stop)
            
            short_hit = window_lows[short_hit_idx] <= short_target
            short_stopped = window_highs[short_stop_idx] >= short_stop
            
            if short_hit and (not short_stopped or short_hit_idx < short_stop_idx):
                label = -1 # Short Win
        
        labels[i] = label
        
        # Max Excursion (Optional metadata)
        max_up = np.max(window_highs) - entry_price
        max_down = entry_price - np.min(window_lows)
        max_up_Rs[i] = max_up / atr
        max_down_Rs[i] = max_down / atr
        
    df['label'] = labels
    df['max_up_R'] = max_up_Rs
    df['max_down_R'] = max_down_Rs
    
    return df

def process_files():
    input_dir = r"c:\Users\Administrator\Desktop\modeloutcome\data\processed_v2"
    output_dir = r"c:\Users\Administrator\Desktop\modeloutcome\data\labeled_outcome"
    
    os.makedirs(output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(input_dir, "*_features.csv"))
    logger.info(f"Found {len(files)} feature files.")
    
    config = {
        'future_window': 60, # 1 hour
        'target_R': 2.0,     # OPTIMIZATION: Lower target to increase winrate
        'stop_R': 1.0
    }
    
    total_labeled = 0
    
    for f in files:
        try:
            output_path = os.path.join(output_dir, os.path.basename(f).replace("_features.csv", "_labeled.csv"))
            
            if os.path.exists(output_path):
                logger.info(f"Skipping {f} (already labeled)")
                continue
                
            df = pd.read_csv(f)
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Compute Outcomes
            df_labeled = compute_outcomes(df, config)
            
            # output_path defined above
            df_labeled.to_csv(output_path, index=False)
            
            n_labeled = len(df_labeled[df_labeled['label'] != 0])
            total_labeled += n_labeled
            logger.info(f"Processed {os.path.basename(f)}: {n_labeled} outcomes found")
            
        except Exception as e:
            logger.error(f"Error processing {f}: {e}")
            
    logger.info(f"Labeling complete. Total labeled samples: {total_labeled}")

if __name__ == "__main__":
    process_files()
