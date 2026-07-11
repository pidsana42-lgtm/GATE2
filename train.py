import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from models.transformer import PureTransformerDecoder
from models.gated_deltanet2 import HybridGatedDeltaNet2Decoder
from models.model_utils import match_parameters
from utils.trainer import train_one_epoch

class DummyTokenizedDataset(Dataset):
    """A synthetic dataset yielding random tokens for fallback/offline testing."""
    def __init__(self, num_samples=1000, seq_len=2048, vocab_size=35219):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": input_ids}

def get_dataloader(batch_size, seq_len, vocab_size):
    """Attempts to load local dataset from ./dataset or download pythainlp/thai-tnhc2-books, otherwise dummy."""
    try:
        from datasets import load_from_disk, load_dataset
        dataset_path = "./dataset"
        
        if os.path.exists(dataset_path) and os.listdir(dataset_path):
            print(f"Loading dataset from local disk: {dataset_path}...")
            dataset = load_from_disk(dataset_path)
        else:
            print("Local dataset not found. Downloading pythainlp/thai-tnhc2-books from Hugging Face...")
            dataset = load_dataset("pythainlp/thai-tnhc2-books")
        
        # Extract split if it's a DatasetDict
        if isinstance(dataset, dict) or hasattr(dataset, "keys"):
            if "train" in dataset:
                dataset = dataset["train"]
            else:
                first_key = list(dataset.keys())[0]
                dataset = dataset[first_key]
                
        # Load tokenizer from local Tokeniz or download Typhoon
        tokenizer_path = "./Tokeniz" if os.path.exists("./Tokeniz") else "typhoon-ai/typhoon-7b"
        print(f"Loading tokenizer from: {tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("Tokenizing dataset...")
        # Get list of columns to remove
        cols_to_remove = [c for c in dataset.column_names if c != "input_ids"]
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=seq_len, padding="max_length")
            
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=cols_to_remove,
            desc="Running tokenizer on dataset"
        )
        
        tokenized_dataset.set_format("torch")
        
        def collate_fn(batch):
            input_ids = torch.stack([x["input_ids"][:seq_len] for x in batch])
            return {"input_ids": input_ids}
            
        dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        print(f"Successfully loaded dataset dataloader. Total samples: {len(tokenized_dataset)}")
        return dataloader
    except Exception as e:
        print(f"Could not load dataset ({e}). Falling back to dummy synthetic dataset...")
        dataset = DummyTokenizedDataset(num_samples=100, seq_len=seq_len, vocab_size=vocab_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

def main():
    parser = argparse.ArgumentParser(description="Train Transformer or Hybrid Gated DeltaNet-2")
    parser.add_argument("--model", type=str, required=True, choices=["transformer", "hybrid"],
                        help="Which model to train: 'transformer' or 'hybrid'")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to train on (e.g. 'cuda:0', 'cuda:1', 'cpu')")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Maximum training steps per epoch")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Sequence length for training context")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--vocab_size", type=int, default=35219,
                        help="Default vocabulary size (matches Typhoon-7b)")
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=12,
                        help="Number of heads")
    parser.add_argument("--use_fla", action="store_true",
                        help="Use Flash Linear Attention Triton kernels for hybrid model")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable logging to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="transformer-vs-gated-deltanet2",
                        help="W&B project name")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training {args.model.upper()} on device: {device}")
    
    # Dynamically determine vocab size from local tokenizer if possible
    vocab_size = args.vocab_size
    tokenizer_path = "./Tokeniz" if os.path.exists("./Tokeniz") else "typhoon-ai/typhoon-7b"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        vocab_size = len(tokenizer)
        print(f"Detected tokenizer vocabulary size: {vocab_size}")
    except Exception as e:
        print(f"Could not load tokenizer for parameter counting ({e}). Using default: {vocab_size}")

    # 1. Match parameter counts
    print("Matching FFN dimension for target parameters...")
    matched_ffn_size, a_params, b_params = match_parameters(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        model_a_ffn_size=2048,
        verbose=True
    )
    
    # 2. Instantiate target model
    if args.model == "transformer":
        model = PureTransformerDecoder(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            intermediate_size=2048,
            num_layers=args.num_layers
        )
        model_name = "Transformer_Baseline"
    else:
        model = HybridGatedDeltaNet2Decoder(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            intermediate_size=matched_ffn_size,
            num_layers=args.num_layers
        )
        model_name = "Hybrid_Gated_DeltaNet2"
        
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # 3. Handle W&B initialization
    wandb_active = False
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=f"{model_name}_seq{args.seq_len}",
                config={
                    "model": args.model,
                    "vocab_size": vocab_size,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "batch_size": args.batch_size,
                    "seq_len": args.seq_len,
                    "lr": args.lr,
                    "use_fla": args.use_fla,
                    "parameters": count_parameters(model)
                }
            )
            wandb_active = True
            print("W&B logger initialized successfully.")
        except ImportError:
            print("W&B package is not installed. Continuing training without W&B.")
            
    # 4. Load dataset
    dataloader = get_dataloader(args.batch_size, args.seq_len, vocab_size)
    
    # 5. Train
    print(f"Starting training run ({args.max_steps} steps max per epoch)...")
    history_all = []
    
    for epoch in range(args.epochs):
        epoch_history = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_steps=args.max_steps,
            log_interval=10,
            use_fla=args.use_fla,
            wandb_active=wandb_active
        )
        history_all.extend(epoch_history)
        
    # Save training logs locally
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(history_all)
    csv_path = f"results/loss_{args.model}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Training complete. Saved loss logs to {csv_path}")
    
    if wandb_active:
        wandb.finish()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    main()
