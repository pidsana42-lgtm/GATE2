# GATE2: Pure Transformer vs Hybrid Gated DeltaNet-2 (GDN-2) Benchmark

A project to benchmark and compare a **Pure Transformer (Model A)** and a **Hybrid Gated DeltaNet-2 (Model B)** under a "Fair Play" parameter-matched setup (~100M parameters) using context length scaling (2K to 8K) and next-token prediction pretraining on Thai old books dataset.

## Project Structure
```
GATE/
├── models/
│   ├── transformer.py          # Pure Transformer (Baseline)
│   ├── gated_deltanet2.py      # Hybrid Gated DeltaNet-2
│   └── model_utils.py          # Parameter matching and helper utilities
├── utils/
│   ├── trainer.py              # Next-token prediction pretraining loop
│   └── benchmark.py            # VRAM & Speed throughput metrics logger
├── train.py                    # Pretraining script (Thai old books dataset)
├── download_dataset.py         # Dynamic dataset & tokenizer caching script
├── benchmark_vram.py           # VRAM scaling benchmark script
├── benchmark_speed.py          # Speed (tokens/sec) scaling benchmark script
├── plot_results.py             # Generates comparison charts
├── requirements.txt            # Package dependencies
└── experiment.ipynb            # Kaggle/Colab pipeline notebook
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Tokenizer and Dataset
The pretraining script uses the **Typhoon-7b Tokenizer** and the **pythainlp/thai-tnhc2-books** dataset. You can fetch and cache them locally using:
```bash
python download_dataset.py
```

### 3. Run VRAM & Speed Benchmarks
To run memory and speed benchmarks across different context lengths:
```bash
# Benchmark VRAM
python benchmark_vram.py --model transformer --device cuda:0
python benchmark_vram.py --model hybrid --device cuda:0

# Benchmark Speed
python benchmark_speed.py --model transformer --device cuda:0
python benchmark_speed.py --model hybrid --device cuda:0
```

*Note: On dual-GPU environments (like Kaggle T4 x2), you can run them concurrently by specifying different devices (`cuda:0` and `cuda:1`) in parallel.*

### 4. Plot Comparison Results
```bash
python plot_results.py
```
This saves comparison charts in the `results/` folder.
