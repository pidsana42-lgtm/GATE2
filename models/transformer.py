import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (standard in Llama and Mistral)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for causal self-attention."""
    def __init__(self, dim, max_seq_len=8192, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len):
        return self.cos_cached[:seq_len].to(x.device), self.sin_cached[:seq_len].to(x.device)

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(1) # [1, 1, L, d]
    sin = sin.unsqueeze(0).unsqueeze(1) # [1, 1, L, d]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class CausalSelfAttention(nn.Module):
    """Standard causal self-attention using PyTorch scaled_dot_product_attention (supporting FlashAttention)."""
    def __init__(self, hidden_size, num_heads, head_dim=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x, attention_mask=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, L, d]
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, L, d]
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, L, d]
        
        cos, sin = self.rotary_emb(q, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Next-token causal self-attention
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attention_mask, 
            is_causal=True if attention_mask is None else False
        )
        
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)

class SwiGLUMLP(nn.Module):
    """SwiGLU MLP block as used in Llama-3 and Mistral."""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerDecoderLayer(nn.Module):
    """Single layer of causal Transformer decoder."""
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads)
        
        self.mlp_norm = RMSNorm(hidden_size)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class PureTransformerDecoder(nn.Module):
    """Decoder-only Transformer Model."""
    def __init__(self, vocab_size, hidden_size, num_heads, intermediate_size, num_layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
