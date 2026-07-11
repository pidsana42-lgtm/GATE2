import time
import json
import torch
import torch.nn as nn
from contextlib import nullcontext

def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_steps: int = None,
    log_interval: int = 10,
    use_fla: bool = False,
    wandb_active: bool = False
):
    """
    Trains the model for one epoch on causal language modeling (next-token prediction).
    Logs metrics locally and to W&B if active.
    """
    model.train()
    total_loss = 0
    start_time = time.time()
    step = 0
    history = []
    
    # Choose autocast based on device
    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.float16) if device.type == "cuda" else nullcontext()
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    
    for batch_idx, batch in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break
            
        # Move to device
        # dataloader is expected to yield dict with "input_ids"
        input_ids = batch["input_ids"].to(device)
        
        # Next-token prediction: input is token [0...N-1], target is token [1...N]
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast_ctx:
            # Gated DeltaNet-2 uses use_fla flag
            if hasattr(model, 'lm_head') and 'gated' in model.__class__.__name__.lower():
                logits = model(inputs, use_fla=use_fla)
            else:
                logits = model(inputs)
                
            # Logits shape: [B, L, Vocab]
            # Targets shape: [B, L]
            loss = F_cross_entropy(logits, targets)
            
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        step += 1
        total_loss += loss.item()
        
        # Calculate speed metrics
        tokens_processed = inputs.numel()
        elapsed = time.time() - start_time
        tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0.0
        
        # Reset timer
        start_time = time.time()
        
        if step % log_interval == 0:
            current_loss = loss.item()
            lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch} | Step {step} | Loss: {current_loss:.4f} | Speed: {tokens_per_sec:.2f} tok/s | LR: {lr:.6f}")
            
            # Record history
            history_item = {
                "epoch": epoch,
                "step": step,
                "loss": current_loss,
                "tokens_per_sec": tokens_per_sec,
                "lr": lr
            }
            history.append(history_item)
            
            # W&B Logging
            if wandb_active:
                import wandb
                wandb.log({
                    "train/loss": current_loss,
                    "train/tokens_per_second": tokens_per_sec,
                    "train/lr": lr,
                    "train/step": step + (epoch * len(dataloader))
                })
                
    return history

def F_cross_entropy(logits, targets):
    """Reshape logits and targets and apply cross entropy loss."""
    vocab_size = logits.shape[-1]
    return nn.functional.cross_entropy(
        logits.view(-1, vocab_size), 
        targets.reshape(-1)
    )
