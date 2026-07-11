import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import RMSNorm, CausalSelfAttention, SwiGLUMLP

class CausalConv1d(nn.Module):
    """A standard 1D Causal Convolution layer."""
    def __init__(self, channels, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels, 
            channels, 
            kernel_size=kernel_size, 
            padding=kernel_size - 1, 
            groups=channels, 
            bias=True
        )

    def forward(self, x):
        # x: [B, L, D]
        # Transpose to [B, D, L] for PyTorch Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Slice to keep it causal (remove the future padding)
        x = x[..., :-(self.kernel_size - 1)]
        return x.transpose(1, 2)

def gated_deltanet2_pytorch(q, k, v, b, w, alpha, chunk_size=64):
    r"""
    Decoupled Gated DeltaNet-2 recurrent state update:
    S_t = (I - k_t (b_t \odot k_t)^T) D_t S_{t-1} + k_t (w_t \odot v_t)^T
    y_t = q_t S_t
    
    Uses chunked gradient checkpointing to keep VRAM usage low during training.
    """
    B, H, L, d_k = q.shape
    d_v = v.shape[-1]
    
    # Recurrent state S of shape [B, H, d_k, d_v]
    S = torch.zeros(B, H, d_k, d_v, device=q.device, dtype=q.dtype)
    outputs = []
    
    def chunk_forward(S_prev, q_c, k_c, v_c, b_c, w_c, alpha_c):
        S_curr = S_prev
        y_c = []
        for t in range(q_c.shape[2]):
            q_t = q_c[:, :, t]
            k_t = k_c[:, :, t]
            v_t = v_c[:, :, t]
            b_t = b_c[:, :, t]
            w_t = w_c[:, :, t]
            alpha_t = alpha_c[:, :, t]
            
            S_decayed = alpha_t.unsqueeze(-1) * S_curr
            u_t = b_t * k_t
            p_t = torch.einsum('bhkd,bhk->bhd', S_decayed, u_t)
            
            erase_term = k_t.unsqueeze(-1) * p_t.unsqueeze(-2)
            write_term = k_t.unsqueeze(-1) * (w_t * v_t).unsqueeze(-2)
            
            S_curr = S_decayed - erase_term + write_term
            y_t = torch.einsum('bhk,bhkd->bhd', q_t, S_curr)
            y_c.append(y_t)
            
        y_c = torch.stack(y_c, dim=2)
        return S_curr, y_c

    # Process in chunks using PyTorch Gradient Checkpointing
    from torch.utils.checkpoint import checkpoint
    
    for i in range(0, L, chunk_size):
        q_chunk = q[:, :, i:i+chunk_size]
        k_chunk = k[:, :, i:i+chunk_size]
        v_chunk = v[:, :, i:i+chunk_size]
        b_chunk = b[:, :, i:i+chunk_size]
        w_chunk = w[:, :, i:i+chunk_size]
        alpha_chunk = alpha[:, :, i:i+chunk_size]
        
        if S.requires_grad:
            # use_reentrant=False is the modern and recommended checkpointing mode
            S, y_chunk = checkpoint(
                chunk_forward,
                S, q_chunk, k_chunk, v_chunk, b_chunk, w_chunk, alpha_chunk,
                use_reentrant=False
            )
        else:
            S, y_chunk = chunk_forward(S, q_chunk, k_chunk, v_chunk, b_chunk, w_chunk, alpha_chunk)
            
        outputs.append(y_chunk)
        
    outputs = torch.cat(outputs, dim=2) # [B, H, L, d_v]
    return outputs

class GatedDeltaNet2Attention(nn.Module):
    """
    Gated DeltaNet-2 Attention Layer (Decoupled Erase & Write).
    Falls back to FLA's Triton kernels if installed and use_fla=True.
    """
    def __init__(self, hidden_size, num_heads, head_dim=64, head_dim_v=64, use_short_conv=True, conv_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        
        # Decoupled Gates
        self.b_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)     # erase gate
        self.w_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)   # write gate
        self.alpha_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False) # decay gate
        
        # Output projections
        self.o_proj = nn.Linear(num_heads * head_dim_v, hidden_size, bias=False)
        
        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.q_conv = CausalConv1d(hidden_size, conv_size)
            self.k_conv = CausalConv1d(hidden_size, conv_size)
            self.v_conv = CausalConv1d(hidden_size, conv_size)
            
        self.gate_norm = nn.LayerNorm(num_heads * head_dim_v)

    def forward(self, x, attention_mask=None, use_fla=False):
        # x: [B, L, D]
        B, L, D = x.shape
        
        # If FLA is requested and available, we attempt using flash-linear-attention
        if use_fla:
            try:
                # Note: FLA library uses fused Triton kernels for training efficiency.
                # Here we import and route to GatedDeltaNet layer if available.
                # GatedDeltaNet-2 has dynamic cuda compilation.
                from fla.layers import GatedDeltaNet
                # We dynamically construct the FLA layer on first forward pass or cache it
                if not hasattr(self, 'fla_layer'):
                    self.fla_layer = GatedDeltaNet(
                        hidden_size=self.hidden_size, 
                        num_heads=self.num_heads,
                        mode='chunk'
                    ).to(x.device).to(x.dtype)
                return self.fla_layer(x)
            except Exception as e:
                # If FLA fails or is not installed, fallback gracefully to PyTorch
                pass
        
        # Causal convolutions
        if self.use_short_conv:
            q_in = self.q_conv(x)
            k_in = self.k_conv(x)
            v_in = self.v_conv(x)
        else:
            q_in, k_in, v_in = x, x, x
            
        # Get Q, K, V projections
        q = self.q_proj(q_in).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_in).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_in).view(B, L, self.num_heads, self.head_dim_v).transpose(1, 2)
        
        # Compute gates
        b = torch.sigmoid(self.b_proj(x)).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        w = torch.sigmoid(self.w_proj(x)).view(B, L, self.num_heads, self.head_dim_v).transpose(1, 2)
        alpha = torch.sigmoid(self.alpha_proj(x)).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Run state update recurrence
        out = gated_deltanet2_pytorch(q, k, v, b, w, alpha) # [B, H, L, d_v]
        
        # Reshape and norm
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.gate_norm(out)
        
        # Project output
        return self.o_proj(out)

class HybridDecoderLayer(nn.Module):
    """
    A single block in the Hybrid Model.
    Can be configured as a Gated DeltaNet-2 layer OR a Standard Self-Attention layer.
    """
    def __init__(self, hidden_size, num_heads, intermediate_size, layer_type="gdn"):
        super().__init__()
        self.layer_type = layer_type
        self.norm1 = RMSNorm(hidden_size)
        
        if layer_type == "gdn":
            self.attn = GatedDeltaNet2Attention(hidden_size, num_heads)
        elif layer_type == "attn":
            self.attn = CausalSelfAttention(hidden_size, num_heads)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
            
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x, attention_mask=None, use_fla=False):
        if self.layer_type == "gdn":
            # Gated DeltaNet-2 layer
            x = x + self.attn(self.norm1(x), use_fla=use_fla)
        else:
            # Standard Self-Attention layer
            x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
            
        x = x + self.mlp(self.norm2(x))
        return x

class HybridGatedDeltaNet2Decoder(nn.Module):
    """
    Hybrid Gated DeltaNet-2 Decoder Model.
    Mixes standard Self-Attention layers and Gated DeltaNet-2 layers.
    """
    def __init__(self, vocab_size, hidden_size, num_heads, intermediate_size, num_layers, layer_types=None):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Default layer interleaving: 9 GDN layers and 3 Attention layers (attn at indices 2, 5, 8)
        if layer_types is None:
            layer_types = []
            for i in range(num_layers):
                if i in [2, 5, 8]:
                    layer_types.append("attn")
                else:
                    layer_types.append("gdn")
                    
        self.layers = nn.ModuleList([
            HybridDecoderLayer(hidden_size, num_heads, intermediate_size, layer_type=layer_types[i])
            for i in range(num_layers)
        ])
        
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, use_fla=False):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, use_fla=use_fla)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
