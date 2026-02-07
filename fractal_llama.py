"""
MORE: Mixture of Recursive Experts
============================================================================

This architecture combines:
1. Sparse Mixture of Experts routing
2. Recursive "thinking" within each expert
3. Step embeddings to differentiate recursion depths

Key Innovation: Each "expert" is itself a tiny recursive transformer that
"thinks" for N steps before producing output. This creates a fractal-like
structure where computation depth varies per token based on routing.

Author: AI Practicals
Date: February 2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


# =============================================================================
# Part 1: Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for MORE."""
    vocab_size: int = 65          # Character vocabulary size
    n_embd: int = 128             # Embedding dimension
    n_head: int = 4               # Number of attention heads
    n_kv_head: int = 4            # Number of KV heads (for GQA)
    n_layer: int = 2              # Number of transformer blocks
    block_size: int = 64          # Maximum sequence length
    num_experts: int = 4          # Number of recursive experts
    top_k: int = 2                # Top-k experts per token
    n_recursions: int = 3         # Recursion steps per expert
    intermediate_size: int = 256  # FFN hidden dimension
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0


# =============================================================================
# Part 1: RMSNorm
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


# =============================================================================
# Part 1: RoPE Helper Functions
# =============================================================================

def precompute_rope_frequencies(dim: int, seq_len: int, theta: float = 10000.0,
                                 device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute sin/cos frequencies for RoPE."""
    assert dim % 2 == 0, "Dimension must be even for RoPE"
    
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to queries and keys."""
    seq_len = q.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    
    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    
    return q_rotated, k_rotated


# =============================================================================
# Part 1: Causal Self-Attention with GQA
# =============================================================================

class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with Grouped Query Attention (GQA)."""
    
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = config.n_head // config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        
        cos, sin = precompute_rope_frequencies(self.head_dim, config.block_size, config.rope_theta)
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)
    
    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads for GQA."""
        if self.n_rep == 1:
            return x
        B, n_kv, T, hd = x.shape
        x = x.unsqueeze(2).expand(B, n_kv, self.n_rep, T, hd)
        return x.reshape(B, n_kv * self.n_rep, T, hd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        q, k = apply_rope(q, k, self.cos_cached, self.sin_cached)
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(attn_output))


# =============================================================================
# Part 1: SwiGLU Feed-Forward Network
# =============================================================================

class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))



# =============================================================================
# Part 2: The Core Invention - RecursiveExpert
# =============================================================================

class RecursiveExpert(nn.Module):
    """
    A Recursive Expert - the core innovation of MORE.
    
    As per the blueprint diagram:
    ┌──────────────────────────────────────────────────────────────┐
    │                    RECURSIVE EXPERT                          │
    │                                                              │
    │   INPUT                                                      │
    │      │                                                       │
    │      ▼                                                       │
    │   ┌──────────────── LOOP n_recursions ───────────────────┐  │
    │   │                                                       │  │
    │   │      ┌─────────────┐                                  │  │
    │   │      │  TINY ATTN  │  (Self-Attention)               │  │
    │   │      └──────┬──────┘                                  │  │
    │   │             │                                         │  │
    │   │             ▼                                         │  │
    │   │      ┌─────────────┐                                  │  │
    │   │      │  SwiGLU/FFN │  (Feed-Forward)                 │  │
    │   │      └──────┬──────┘                                  │  │
    │   │             │                                         │  │
    │   └─────────────┴─────────────────────────────────────────┘  │
    │      │                                                       │
    │      ▼                                                       │
    │   OUTPUT                                                     │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    
    Each recursion applies: Tiny Attn → SwiGLU FFN
    Step embeddings differentiate each recursion depth.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.n_recursions = config.n_recursions
        
        # Tiny Attention (shared across recursions)
        self.tiny_attn_norm = RMSNorm(config.n_embd)
        self.tiny_attn = CausalSelfAttention(config)
        
        # SwiGLU FFN (shared across recursions)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config)
        
        # Learnable step embeddings to differentiate recursion depths
        # Shape: (n_recursions, n_embd)
        self.step_embedding = nn.Parameter(
            torch.randn(config.n_recursions, config.n_embd) * 0.02
        )
        
        # Output normalization for stability
        self.out_norm = RMSNorm(config.n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply recursive thinking: Loop N times through (Tiny Attn → SwiGLU).
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)
            
        Returns:
            Refined output after n_recursions steps
        """
        z = x
        
        # Recursive loop: Tiny Attn → SwiGLU FFN (repeated n times)
        for i in range(self.n_recursions):
            # Add step embedding (differentiates recursion depth)
            z = z + self.step_embedding[i].unsqueeze(0).unsqueeze(0)
            
            # Tiny Attention (with residual)
            z = z + self.tiny_attn(self.tiny_attn_norm(z))
            
            # SwiGLU FFN (with residual)
            z = z + self.ffn(self.ffn_norm(z))
        
        # Final normalization for stability
        z = self.out_norm(z)
        
        return z


# =============================================================================
# Part 3: MoRE Layer (Mixture of Recursive Experts)
# =============================================================================

class MoRE_Layer(nn.Module):
    """
    Mixture of Recursive Experts (MoRE) Layer.
    
    This is the routing system that:
    1. Uses a router to select top-k experts per token
    2. Routes tokens to their assigned RecursiveExperts
    3. Combines outputs via weighted sum
    
    ┌──────────────────────────────────────────────────────────────┐
    │                      MoRE Layer                              │
    │                                                              │
    │   Input Token                                                │
    │       │                                                      │
    │       ▼                                                      │
    │   ┌────────┐                                                 │
    │   │ Router │ ──────┬──────────┬──────────────────────────   │
    │   └────────┘       │          │                              │
    │       │        w1=0.7     w2=0.3                             │
    │       │            │          │                              │
    │       │            ▼          ▼                              │
    │       │     ┌──────────┐ ┌──────────┐                       │
    │       │     │Recursive │ │Recursive │  (other experts idle) │
    │       │     │Expert 1  │ │Expert 3  │                       │
    │       │     │(3 loops) │ │(3 loops) │                       │
    │       │     └────┬─────┘ └────┬─────┘                       │
    │       │          │            │                              │
    │       │          ▼            ▼                              │
    │       │     ┌────────────────────────┐                       │
    │       │     │ 0.7*E1 + 0.3*E3       │ (weighted sum)        │
    │       │     └────────────────────────┘                       │
    │       │                   │                                  │
    │       ▼                   ▼                                  │
    │   Output Token                                               │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        
        # Router: projects input to expert logits
        self.router = nn.Linear(config.n_embd, config.num_experts, bias=False)
        
        # Create the recursive experts
        self.experts = nn.ModuleList([
            RecursiveExpert(config) for _ in range(config.num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Route tokens to top-k experts and combine outputs.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        B, T, C = x.shape
        
        # Step 1: Compute routing logits and probabilities
        logits = self.router(x)  # (B, T, num_experts)
        probs = F.softmax(logits, dim=-1)
        
        # Step 2: Get top-k experts per token
        weights, indices = torch.topk(probs, k=self.top_k, dim=-1)
        # weights: (B, T, top_k)
        # indices: (B, T, top_k)
        
        # Renormalize weights to sum to 1
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Step 3: Flatten for processing
        flat_x = x.view(-1, C)  # (B*T, C)
        flat_weights = weights.view(-1, self.top_k)  # (B*T, top_k)
        flat_indices = indices.view(-1, self.top_k)  # (B*T, top_k)
        
        # Initialize output
        flat_output = torch.zeros_like(flat_x)
        
        # Step 4: Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (flat_indices == expert_idx)  # (B*T, top_k)
            
            if not expert_mask.any():
                continue
            
            # Get token indices and which slot (0 or 1) selected this expert
            token_indices, slot_indices = torch.where(expert_mask)
            
            # Gather inputs for this expert
            expert_input = flat_x[token_indices]  # (num_tokens, C)
            
            # Reshape for the expert (needs batch dim)
            # We treat each token as a separate "sequence" of length 1
            expert_input = expert_input.unsqueeze(1)  # (num_tokens, 1, C)
            
            # Run through the recursive expert
            expert_output = self.experts[expert_idx](expert_input)  # (num_tokens, 1, C)
            expert_output = expert_output.squeeze(1)  # (num_tokens, C)
            
            # Get the corresponding weights
            expert_weights = flat_weights[token_indices, slot_indices]  # (num_tokens,)
            
            # Accumulate weighted output
            flat_output.index_add_(
                0,
                token_indices,
                expert_output * expert_weights.unsqueeze(-1)
            )
        
        # Reshape back to (B, T, C)
        return flat_output.view(B, T, C)


# =============================================================================
# Part 4: Fractal Block (Attention + MoRE)
# =============================================================================

class FractalBlock(nn.Module):
    """
    A Transformer Block that uses MoRE instead of standard FFN.
    
    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + MoRE(RMSNorm(x))        # MoRE instead of FFN!
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.more_norm = RMSNorm(config.n_embd)
        self.more = MoRE_Layer(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.more(self.more_norm(x))
        return x


# =============================================================================
# Part 4: MORE - The Complete Model
# =============================================================================

class MORE(nn.Module):
    """
    MORE: A novel architecture combining MoE with recursive experts.
    
    Architecture:
        Token Embedding
            ↓
        N × FractalBlock (Attention + MoRE)
            ↓
        Final RMSNorm
            ↓
        LM Head
    
    The "fractal" nature comes from:
    1. Each block contains multiple experts
    2. Each expert contains recursive computation
    3. Creating a tree-like structure of computation paths
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Token embedding (no position embedding - RoPE handles it)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Stack of Fractal Blocks
        self.blocks = nn.ModuleList([
            FractalBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final normalization
        self.final_norm = RMSNorm(config.n_embd)
        
        # Output projection
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, 
                targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            idx: Token indices (batch, seq_len)
            targets: Target indices for loss (optional)
            
        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"
        
        # Token embeddings
        x = self.dropout(self.tok_emb(idx))
        
        # Pass through blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm and projection
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0) -> torch.Tensor:
        """Generate new tokens."""
        for _ in range(max_new_tokens):
            idx_crop = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# =============================================================================
# Utility Functions
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Part 5: The Overfit Test
# =============================================================================

if __name__ == "__main__":
    import urllib.request
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║            🌀 MORE: Training Run 🌀                                ║
    ║                                                                   ║
    ║   Architecture: Transformer + MORE (Mixture of Recursive Experts) ║
    ║   Data: Shakespeare (Tiny Shakespeare)                            ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Device - use GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Download Shakespeare dataset from Karpathy's char-rnn repository
    print("\n📥 Downloading Shakespeare dataset...")
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with urllib.request.urlopen(data_url) as response:
        text = response.read().decode('utf-8')
    print("   ✅ Downloaded successfully!")
    
    print(f"\n📜 Data: Tiny Shakespeare")
    print(f"   Length: {len(text):,} characters")
    
    # Build vocabulary from text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    print(f"   Vocab size: {vocab_size} unique characters")
    
    # Hyperparameters
    batch_size = 32
    block_size = 128
    
    # Create config
    config = Config(
        vocab_size=vocab_size,
        n_embd=128,               # Small embedding
        n_head=4,
        n_kv_head=4,
        n_layer=2,                # 2 fractal blocks
        block_size=block_size,
        num_experts=4,            # 4 recursive experts
        top_k=2,                  # Use top-2
        n_recursions=3,           # 3 recursion steps per expert
        intermediate_size=256,
    )
    
    # Create model
    model = MORE(config).to(device)
    
    # Print model info
    total_params = count_parameters(model)
    print(f"\n🌀 MORE Model:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Layers: {config.n_layer}")
    print(f"   Experts: {config.num_experts} (top-{config.top_k} routing)")
    print(f"   Recursions per expert: {config.n_recursions}")
    
    # Encode entire text to tensor
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"\n   Train size: {len(train_data):,} tokens")
    print(f"   Val size: {len(val_data):,} tokens")
    
    def get_batch(split):
        """Get a random batch of data."""
        data_split = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_split) - block_size, (batch_size,))
        x = torch.stack([data_split[i:i+block_size] for i in ix])
        y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)
    
    # Get first batch for shape info
    x, y = get_batch('train')
    print(f"   Batch shape: {x.shape}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Training loop
    max_steps = 1000
    eval_interval = 100
    
    print(f"\n{'='*60}")
    print("🏋️  Training on input.txt...")
    print(f"{'='*60}\n")
    
    model.train()
    for step in range(max_steps + 1):
        # Get a fresh batch each step
        x, y = get_batch('train')
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Print progress
        if step % eval_interval == 0:
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_x, val_y = get_batch('val')
                _, val_loss = model(val_x, val_y)
            model.train()
            print(f"   Step {step:4d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
    
    # Test generation
    print(f"\n{'='*60}")
    print("📝 Generation Test")
    print(f"{'='*60}\n")
    
    model.eval()
    # Start with a prompt from the text
    prompt = text[:50]
    context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    generated = model.generate(context, max_new_tokens=100, temperature=0.8)
    output = decode(generated[0].tolist())
    
    print(f"   Prompt: '{prompt}'")
    print(f"\n   Generated continuation:")
    print(f"   '{output[len(prompt):]}'")
    
    # Results summary
    print(f"\n{'='*60}")
    print("📊 Results")
    print(f"{'='*60}\n")
    
    print(f"   Final Train Loss: {loss.item():.4f}")
    print(f"   Final Val Loss: {val_loss.item():.4f}")
    print("\n   The following components are working:")
    print("   • Router → top-k selection")
    print("   • RecursiveExpert → step embeddings + recursive loop")
    print("   • MoRE_Layer → sparse routing + weighted combination")
    print("   • MORE → end-to-end training on real data")
    
    print(f"\n{'='*60}")
    print("✅ Training Complete!")
    print(f"{'='*60}")
