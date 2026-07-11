import time
import gc
import torch
import torch.nn as nn
from contextlib import nullcontext

def clean_gpu_memory():
    """Clear memory garbage and PyTorch CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def benchmark_memory(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    use_fla: bool = False,
    inference: bool = False
):
    """
    Measures the peak VRAM memory usage (in MB) for a forward pass (and backward pass if not inference).
    Returns "OOM" if the run runs out of memory.
    """
    model = model.to(device)
    if inference:
        model.eval()
        grad_ctx = torch.no_grad()
    else:
        model.train()
        grad_ctx = nullcontext()
        
    clean_gpu_memory()
    
    # Check if GPU is available to measure VRAM
    if device.type != "cuda":
        return 0.0
        
    try:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Reset peak memory stats before forward pass
        torch.cuda.reset_peak_memory_stats()
        
        # Forward pass (Mixed Precision)
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        
        with grad_ctx:
            with autocast_ctx:
                if 'gated' in model.__class__.__name__.lower():
                    logits = model(input_ids, use_fla=use_fla)
                else:
                    logits = model(input_ids)
                
                vocab_dim = logits.shape[-1]
                loss = nn.functional.cross_entropy(logits.view(-1, vocab_dim), targets.view(-1))
                
            # Backward pass (only during training mode)
            if not inference:
                loss.backward()
        
        # Query peak memory
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_mb = peak_bytes / (1024 * 1024)
        
        return peak_mb
        
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        err_msg = str(e)
        if "out of memory" in err_msg.lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            return "OOM"
        else:
            raise e
            
    finally:
        if 'logits' in locals():
            del logits
        if 'loss' in locals():
            del loss
        if 'input_ids' in locals():
            del input_ids
        if 'targets' in locals():
            del targets
        clean_gpu_memory()

def benchmark_throughput(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    num_warmup: int = 3,
    num_steps: int = 10,
    use_fla: bool = False,
    inference: bool = False
):
    """
    Measures throughput in tokens/second over a number of steps.
    Supports both training (with backward) and inference (no grad, forward only).
    """
    model = model.to(device)
    if inference:
        model.eval()
        grad_ctx = torch.no_grad()
    else:
        model.train()
        grad_ctx = nullcontext()
        
    clean_gpu_memory()
    
    try:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Warmup steps
        autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.float16) if device.type == "cuda" else nullcontext()
        
        for _ in range(num_warmup):
            with grad_ctx:
                with autocast_ctx:
                    if 'gated' in model.__class__.__name__.lower():
                        logits = model(input_ids, use_fla=use_fla)
                    else:
                        logits = model(input_ids)
                    vocab_dim = logits.shape[-1]
                    loss = nn.functional.cross_entropy(logits.view(-1, vocab_dim), targets.view(-1))
                if not inference:
                    loss.backward()
                    model.zero_grad(set_to_none=True)
            
        # Synchronization if using CUDA
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        # Timing steps
        start_time = time.time()
        
        for _ in range(num_steps):
            with grad_ctx:
                with autocast_ctx:
                    if 'gated' in model.__class__.__name__.lower():
                        logits = model(input_ids, use_fla=use_fla)
                    else:
                        logits = model(input_ids)
                    vocab_dim = logits.shape[-1]
                    loss = nn.functional.cross_entropy(logits.view(-1, vocab_dim), targets.view(-1))
                if not inference:
                    loss.backward()
                    model.zero_grad(set_to_none=True)
            
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        total_time = time.time() - start_time
        total_tokens = batch_size * seq_len * num_steps
        tokens_per_sec = total_tokens / total_time
        
        return tokens_per_sec
        
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        err_msg = str(e)
        if "out of memory" in err_msg.lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            return "OOM"
        else:
            raise e
            
    finally:
        if 'logits' in locals():
            del logits
        if 'loss' in locals():
            del loss
        if 'input_ids' in locals():
            del input_ids
        if 'targets' in locals():
            del targets
        clean_gpu_memory()
