import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_dir):
    logger.info("Loading data...")
    files = sorted(glob.glob(os.path.join(data_dir, "*_labeled.csv")))
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        dfs.append(df)
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Feature Cols
    exclude = ['timestamp', 'bar_index', 'label', 'max_up_R', 'max_down_R', 'signal_type', 'open', 'high', 'low', 'close'] 
    sample_df = dfs[0]
    feature_cols = [c for c in sample_df.columns if c not in exclude and sample_df[c].dtype in [np.float64, np.float32, np.int64]]
    
    logger.info(f"Features: {len(feature_cols)}")
    
    # Split (Train Only for Analysis)
    split_idx = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split_idx]
    
    X_train = train_df[feature_cols]
    y_train = train_df['label'] + 1 # -1,0,1 -> 0,1,2
    
    return X_train, y_train, feature_cols

def analyze():
    DATA_DIR = r"c:\Users\Administrator\Desktop\modeloutcome\data\labeled_outcome"
    
    X_train, y_train, feature_cols = load_data(DATA_DIR)
    
    logger.info("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=3,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Feature Importance
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importance})
    feat_imp = feat_imp.sort_values('importance', ascending=False).reset_index(drop=True)
    
    print("\n--- Top 20 Features ---")
    print(feat_imp.head(20))
    
    print("\n--- Bottom 20 Features ---")
    print(feat_imp.tail(20))
    
    # Save to CSV
    feat_imp.to_csv("feature_importance.csv", index=False)
    logger.info("Feature importance saved to feature_importance.csv")
    
    # Check for "Context" features
    context_keywords = ['trend', 'bias', 'daily', 'vwap', 'ext_']
    print("\n--- Context Feature Ranks ---")
    for idx, row in feat_imp.iterrows():
        if any(k in row['feature'] for k in context_keywords):
            print(f"{idx+1}. {row['feature']} ({row['importance']:.4f})")

if __name__ == "__main__":
    analyze()
