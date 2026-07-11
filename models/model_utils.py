import torch
import torch.nn as nn
from models.transformer import PureTransformerDecoder
from models.gated_deltanet2 import HybridGatedDeltaNet2Decoder

def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model: nn.Module, name: str):
    """Print the model summary including parameter count and layers."""
    params = count_parameters(model)
    print(f"==================================================")
    print(f"Model Summary: {name}")
    print(f"==================================================")
    print(f"Total Parameters: {params:,} ({params/1e6:.2f}M)")
    
    # Print architecture details
    if hasattr(model, 'layers'):
        print(f"Number of layers: {len(model.layers)}")
        if name.lower() == "hybrid_gated_deltanet2":
            layer_types = [layer.layer_type for layer in model.layers]
            print(f"Layer distribution: {layer_types}")
    print(f"==================================================\n")

def match_parameters(
    vocab_size=50257, # Default GPT-2 vocab size
    hidden_size=768,
    num_heads=12,
    num_layers=12,
    target_params=None,
    model_a_ffn_size=2048,
    verbose=True
):
    """
    Finds the correct FFN intermediate size for the Hybrid Model to match the
    parameter count of the Baseline Transformer model.
    """
    # 1. Initialize Baseline Transformer
    model_a = PureTransformerDecoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=model_a_ffn_size,
        num_layers=num_layers
    )
    
    a_params = count_parameters(model_a)
    if verbose:
        print(f"Baseline Transformer Parameter Count: {a_params:,} ({a_params/1e6:.2f}M)")
        
    # If target_params is specified, use that, otherwise use Baseline Transformer count
    target = target_params if target_params is not None else a_params
    
    # 2. Binary search for Model B's intermediate size (FFN size)
    low = 128
    high = 8192
    best_ffn_size = None
    best_diff = float('inf')
    best_b_params = None
    
    while low <= high:
        mid = (low + high) // 2
        
        # Test model B with intermediate_size = mid
        model_b = HybridGatedDeltaNet2Decoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=mid,
            num_layers=num_layers
        )
        b_params = count_parameters(model_b)
        diff = b_params - target
        
        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best_ffn_size = mid
            best_b_params = b_params
            
        if diff > 0:
            high = mid - 1
        else:
            low = mid + 1
            
    if verbose:
        print(f"Matched Hybrid Model Gated DeltaNet-2:")
        print(f" - Optimal FFN size: {best_ffn_size}")
        print(f" - Hybrid Model Parameter Count: {best_b_params:,} ({best_b_params/1e6:.2f}M)")
        diff_pct = (best_b_params - a_params) / a_params * 100
        print(f" - Parameter Difference: {best_b_params - a_params:,} ({diff_pct:.4f}%)")
        
    return best_ffn_size, a_params, best_b_params
