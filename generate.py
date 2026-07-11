import os
import argparse
import torch
from transformers import AutoTokenizer
from models.transformer import PureTransformerDecoder
from models.gated_deltanet2 import HybridGatedDeltaNet2Decoder
from models.model_utils import match_parameters

def main():
    parser = argparse.ArgumentParser(description="Autoregressive text generation for pre-trained models")
    parser.add_argument("--model", type=str, required=True, choices=["transformer", "hybrid"],
                        help="Which model to use: 'transformer' or 'hybrid'")
    parser.add_argument("--prompt", type=str, default="กาลครั้งหนึ่งนานมาแล้ว",
                        help="Thai text prompt to start generation")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run inference on")
    parser.add_argument("--use_fla", action="store_true",
                        help="Enable Flash Linear Attention Triton kernels for Hybrid model")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load tokenizer
    tokenizer_path = "./Tokeniz" if os.path.exists("./Tokeniz") else "typhoon-ai/typhoon-7b"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        vocab_size = len(tokenizer)
        print(f"Loaded tokenizer from {tokenizer_path}. Vocab size: {vocab_size}")
    except Exception as e:
        print(f"Failed to load tokenizer ({e}). Cannot proceed.")
        return

    # 2. Get matched model size and instantiate
    matched_ffn_size, _, _ = match_parameters(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        model_a_ffn_size=2048,
        verbose=False
    )

    if args.model == "transformer":
        model = PureTransformerDecoder(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            intermediate_size=2048,
            num_layers=args.num_layers
        )
    else:
        model = HybridGatedDeltaNet2Decoder(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            intermediate_size=matched_ffn_size,
            num_layers=args.num_layers
        )

    # 3. Load model weights checkpoint
    checkpoint_path = f"results/model_{args.model}.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found. Please train the model first.")
        return

    print(f"Loading weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # 4. Run autoregressive generation loop
    print(f"\n--- Generating text from prompt: '{args.prompt}' ---")
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    
    generated_tokens = []
    
    with torch.no_grad():
        for step in range(args.max_tokens):
            # Forward pass
            if args.model == "hybrid":
                logits = model(input_ids, use_fla=args.use_fla)
            else:
                logits = model(input_ids)
            
            # Extract last token logits
            next_token_logits = logits[:, -1, :]
            
            # Simple greedy search (argmax)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append next token to input history
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            token_id = next_token.item()
            generated_tokens.append(token_id)
            
            # Stop if End-of-Text token is generated
            if token_id == tokenizer.eos_token_id:
                break
                
    # 5. Decode generated text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Generated output:\n{generated_text}")
    print("-------------------------------------------------")

if __name__ == "__main__":
    main()
