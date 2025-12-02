import matplotlib.pyplot as plt
import re
import pandas as pd
import os

def parse_log(log_path):
    epochs = []
    train_losses = []
    train_accs = []
    val_accs = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match: Epoch 1: Train Loss=1.0967, Train Acc=0.3658, Val Acc=0.3395
            match = re.search(r"Epoch (\d+): Train Loss=([\d.]+), Train Acc=([\d.]+), Val Acc=([\d.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                train_accs.append(float(match.group(3)))
                val_accs.append(float(match.group(4)))
                
    return pd.DataFrame({
        'Epoch': epochs,
        'Train Loss': train_losses,
        'Train Acc': train_accs,
        'Val Acc': val_accs
    })

def plot_metrics(df, output_path='training_metrics.png'):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(df['Epoch'], df['Train Loss'], color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(df['Epoch'], df['Train Acc'], color=color, linestyle='--', label='Train Acc')
    ax2.plot(df['Epoch'], df['Val Acc'], color='tab:green', linestyle='-', label='Val Acc')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Training Metrics (GRU v3)')
    fig.tight_layout()
    plt.grid(True)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    log_file = r"c:\Users\Administrator\Desktop\modeloutcome\training_csv_v2.log"
    if os.path.exists(log_file):
        df = parse_log(log_file)
        if not df.empty:
            print(df.tail())
            plot_metrics(df)
        else:
            print("No metrics found in log.")
    else:
        print(f"Log file not found: {log_file}")
