import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set premium styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def format_length(val):
    if val >= 1000:
        return f"{val/1000:.0f}K" if val % 1000 == 0 else f"{val/1000:.1f}K"
    return str(val)

def plot_vram():
    print("Plotting VRAM comparison...")
    # Load CSV files
    try:
        df_trans = pd.read_csv("results/vram_transformer.csv")
        df_hybrid = pd.read_csv("results/vram_hybrid.csv")
    except FileNotFoundError as e:
        print(f"Skipping VRAM plot: file not found ({e}). Run benchmarks first.")
        return

    df = pd.merge(df_trans, df_hybrid, on="sequence_length", how="outer").sort_values("sequence_length")
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Extract x and y values
    x = df["sequence_length"].values
    x_labels = [format_length(v) for v in x]
    
    y_trans = df["transformer_vram_mb"].values
    y_hybrid = df["hybrid_gdn2_vram_mb"].values
    
    # Process Transformer VRAM (handle OOM)
    y_trans_numeric = []
    oom_idx = []
    for idx, val in enumerate(y_trans):
        if str(val).strip().upper() == "OOM":
            y_trans_numeric.append(np.nan)
            oom_idx.append(idx)
        else:
            y_trans_numeric.append(float(val))
            
    # Process Hybrid VRAM
    y_hybrid_numeric = [float(val) if str(val).strip().upper() != "OOM" else np.nan for val in y_hybrid]
    
    # Plot Hybrid GDN-2 (Model B) - sleek teal/blue line
    ax.plot(x, y_hybrid_numeric, marker='o', linewidth=3, color='#009688', label='Model B: Hybrid Gated DeltaNet-2', markersize=8)
    
    # Plot Transformer (Model A) - reddish coral line
    ax.plot(x, y_trans_numeric, marker='s', linewidth=3, color='#E63946', label='Model A: Pure Transformer (Baseline)', markersize=8)
    
    # Add OOM markers if any
    for idx in oom_idx:
        x_val = x[idx]
        ax.axvline(x=x_val, color='#E63946', linestyle='--', alpha=0.7)
        # Position label at 85% of max y or a reasonable default height
        max_y = np.nanmax(y_trans_numeric + y_hybrid_numeric) if not np.isnan(np.nanmax(y_trans_numeric)) else 16000.0
        ax.text(x_val, max_y * 0.7, '  OUT OF MEMORY (OOM) 💥', color='#E63946', weight='bold', verticalalignment='center')
        ax.plot(x_val, max_y * 0.95, marker='x', color='red', markersize=12, markeredgewidth=3)
        
    ax.set_title("VRAM Scaling Comparison (Batch Size = 2)", pad=15)
    ax.set_xlabel("Sequence Length (Context Length)")
    ax.set_ylabel("Peak VRAM Usage (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ddd")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_path = "results/vram_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved VRAM plot to {plot_path}")

def plot_speed():
    print("Plotting throughput comparison...")
    try:
        df_trans = pd.read_csv("results/speed_transformer.csv")
        df_hybrid = pd.read_csv("results/speed_hybrid.csv")
    except FileNotFoundError as e:
        print(f"Skipping speed plot: file not found ({e}). Run benchmarks first.")
        return

    df = pd.merge(df_trans, df_hybrid, on="sequence_length", how="outer").sort_values("sequence_length")
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    x = df["sequence_length"].values
    x_labels = [format_length(v) for v in x]
    
    # Process numeric values
    y_trans = [float(v) if str(v).strip().upper() != "OOM" else 0.0 for v in df["transformer_speed_tok_per_sec"]]
    y_hybrid = [float(v) if str(v).strip().upper() != "OOM" else 0.0 for v in df["hybrid_speed_tok_per_sec"]]
    
    # Plot lines
    ax.plot(x, y_hybrid, marker='o', linewidth=3, color='#009688', label='Model B: Hybrid Gated DeltaNet-2', markersize=8)
    
    # Draw transformer up to successful runs
    y_trans_filtered = [v if v > 0 else np.nan for v in y_trans]
    ax.plot(x, y_trans_filtered, marker='s', linewidth=3, color='#E63946', label='Model A: Pure Transformer (Baseline)', markersize=8)
    
    # Mark OOM in speed as 0 / disconnected
    for idx, v in enumerate(y_trans):
        if v == 0.0:
            ax.text(x[idx], 100, '  OOM 💥', color='#E63946', weight='bold', horizontalalignment='center')
            
    ax.set_title("Training Throughput vs Context Length", pad=15)
    ax.set_xlabel("Sequence Length (Context Length)")
    ax.set_ylabel("Throughput (Tokens per Second)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#ddd")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_path = "results/speed_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved speed plot to {plot_path}")

def plot_loss():
    print("Plotting training loss comparison...")
    try:
        df_trans = pd.read_csv("results/loss_transformer.csv")
        df_hybrid = pd.read_csv("results/loss_hybrid.csv")
    except FileNotFoundError as e:
        print(f"Skipping loss plot: file not found ({e}). Run training runs first.")
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Loss curves
    ax.plot(df_trans["step"], df_trans["loss"], linewidth=2, color='#E63946', alpha=0.8, label='Model A: Pure Transformer')
    ax.plot(df_hybrid["step"], df_hybrid["loss"], linewidth=2, color='#009688', alpha=0.8, label='Model B: Hybrid Gated DeltaNet-2')
    
    # Smooth curves using running average for cleaner visualization
    if len(df_trans) > 10:
        smooth_trans = df_trans["loss"].rolling(window=10, min_periods=1).mean()
        smooth_hybrid = df_hybrid["loss"].rolling(window=10, min_periods=1).mean()
        ax.plot(df_trans["step"], smooth_trans, linewidth=3, color='#B22222', label='Model A: Transformer (Smoothed)')
        ax.plot(df_hybrid["step"], smooth_hybrid, linewidth=3, color='#004D40', label='Model B: Hybrid GDN-2 (Smoothed)')
        
    ax.set_title("Training Loss Convergence Curve (Causal LM Pretraining)", pad=15)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cross Entropy Loss")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#ddd")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_path = "results/loss_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss plot to {plot_path}")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    plot_vram()
    plot_speed()
    plot_loss()
