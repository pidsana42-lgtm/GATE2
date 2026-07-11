import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from models.transformer import PureTransformerDecoder
from models.gated_deltanet2 import HybridGatedDeltaNet2Decoder
from models.model_utils import match_parameters

def sample_next_token(logits, temperature=1.0, top_p=1.0):
    """Apply temperature scaling + top-p (nucleus) sampling."""
    # Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    if top_p < 1.0:
        # Sort logits descending
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative prob above threshold (nucleus)
        sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[sorted_indices_to_remove] = float('-inf')

        # Scatter back to original ordering
        logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

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
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (>1 = more random, <1 = more deterministic)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling: keep top tokens whose cumulative prob >= top_p")
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

    model.to(device)

    # Pre-instantiate FLA layers if using FLA, so they exist in model.state_dict()
    if args.model == "hybrid" and args.use_fla:
        print("Pre-instantiating FLA layers for state_dict matching...")
        dummy_x = torch.zeros(1, 10, dtype=torch.long, device=device)
        with torch.no_grad():
            model(dummy_x, use_fla=True)

    print(f"Loading weights from {checkpoint_path}...")
    # Use strict=False to gracefully load model parameters regardless of dynamic submodules
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
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
            
            # Temperature + Top-p sampling (greedy when temperature=1.0, top_p=1.0)
            next_token = sample_next_token(
                next_token_logits[0],
                temperature=args.temperature,
                top_p=args.top_p
            ).unsqueeze(0)

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
