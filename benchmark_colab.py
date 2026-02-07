"""
MORE: Benchmark & Visualization Script for Google Colab
========================================================

This script:
1. Trains the MORE model on Tiny Shakespeare
2. Trains a baseline transformer for comparison
3. Benchmarks training speed, inference speed, and memory usage
4. Generates learning curves and comparison charts

Run this in Google Colab with GPU for best results.
"""

# =============================================================================
# Install dependencies (uncomment if needed in Colab)
# =============================================================================
# !pip install torch matplotlib

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
import urllib.request
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for MORE."""
    vocab_size: int = 65
    n_embd: int = 128
    n_head: int = 4
    n_kv_head: int = 4
    n_layer: int = 2
    block_size: int = 128
    num_experts: int = 4
    top_k: int = 2
    n_recursions: int = 3
    intermediate_size: int = 256
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0


# =============================================================================
# Building Blocks
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


def precompute_rope_frequencies(dim: int, seq_len: int, theta: float = 10000.0,
                                 device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = q.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
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
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0, is_causal=True
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(attn_output))


class SwiGLU(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# =============================================================================
# RecursiveExpert - The Core Innovation
# =============================================================================

class RecursiveExpert(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_recursions = config.n_recursions
        self.tiny_attn_norm = RMSNorm(config.n_embd)
        self.tiny_attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config)
        self.step_embedding = nn.Parameter(torch.randn(config.n_recursions, config.n_embd) * 0.02)
        self.out_norm = RMSNorm(config.n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for i in range(self.n_recursions):
            z = z + self.step_embedding[i].unsqueeze(0).unsqueeze(0)
            z = z + self.tiny_attn(self.tiny_attn_norm(z))
            z = z + self.ffn(self.ffn_norm(z))
        return self.out_norm(z)


# =============================================================================
# MoRE Layer (Mixture of Recursive Experts)
# =============================================================================

class MoRE_Layer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.router = nn.Linear(config.n_embd, config.num_experts, bias=False)
        self.experts = nn.ModuleList([RecursiveExpert(config) for _ in range(config.num_experts)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, k=self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        flat_x = x.view(-1, C)
        flat_weights = weights.view(-1, self.top_k)
        flat_indices = indices.view(-1, self.top_k)
        flat_output = torch.zeros_like(flat_x)
        
        for expert_idx in range(self.num_experts):
            expert_mask = (flat_indices == expert_idx)
            if not expert_mask.any():
                continue
            token_indices, slot_indices = torch.where(expert_mask)
            expert_input = flat_x[token_indices].unsqueeze(1)
            expert_output = self.experts[expert_idx](expert_input).squeeze(1)
            expert_weights = flat_weights[token_indices, slot_indices]
            flat_output.index_add_(0, token_indices, expert_output * expert_weights.unsqueeze(-1))
        
        return flat_output.view(B, T, C)


# =============================================================================
# Fractal Block
# =============================================================================

class FractalBlock(nn.Module):
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
# MORE - The Complete Model
# =============================================================================

class MORE(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([FractalBlock(config) for _ in range(config.n_layer)])
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.dropout(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_crop = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# =============================================================================
# Baseline Transformer (for comparison)
# =============================================================================

class BaselineBlock(nn.Module):
    """Standard Transformer Block with FFN instead of MoRE."""
    def __init__(self, config: Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class BaselineTransformer(nn.Module):
    """Standard Transformer for baseline comparison."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([BaselineBlock(config) for _ in range(config.n_layer)])
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.dropout(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


# =============================================================================
# Main Benchmark Script
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🌀 MORE: Mixture of Recursive Experts - Benchmark Suite 🌀")
    print("=" * 70)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Download data
    print("\n📥 Downloading Shakespeare dataset...")
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with urllib.request.urlopen(data_url) as response:
        text = response.read().decode('utf-8')
    print(f"   ✅ Downloaded {len(text):,} characters")
    
    # Build vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Prepare data
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Hyperparameters
    batch_size = 32
    block_size = 128
    max_steps = 1000
    eval_interval = 50
    
    # Create config
    config = Config(vocab_size=vocab_size, block_size=block_size)
    
    # Create models
    print("\n🔧 Creating models...")
    more_model = MORE(config).to(device)
    baseline_model = BaselineTransformer(config).to(device)
    
    more_params = count_parameters(more_model)
    baseline_params = count_parameters(baseline_model)
    
    print(f"   MORE Model: {more_params:,} parameters")
    print(f"   Baseline Model: {baseline_params:,} parameters")
    
    def get_batch(split):
        data_split = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_split) - block_size, (batch_size,))
        x = torch.stack([data_split[i:i+block_size] for i in ix])
        y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)
    
    # ==========================================================================
    # Training Function
    # ==========================================================================
    
    def train_model(model, name, max_steps, eval_interval):
        """Train a model and collect metrics."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        
        train_losses = []
        val_losses = []
        steps_list = []
        
        print(f"\n{'='*60}")
        print(f"🏋️  Training {name}...")
        print(f"{'='*60}")
        
        model.train()
        start_time = time.time()
        
        for step in range(max_steps + 1):
            x, y = get_batch('train')
            logits, loss = model(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_x, val_y = get_batch('val')
                    _, val_loss = model(val_x, val_y)
                model.train()
                
                train_losses.append(loss.item())
                val_losses.append(val_loss.item())
                steps_list.append(step)
                
                elapsed = time.time() - start_time
                print(f"   Step {step:4d} | Train: {loss.item():.4f} | Val: {val_loss.item():.4f} | Time: {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        tokens_processed = max_steps * batch_size * block_size
        training_speed = tokens_processed / total_time
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'steps': steps_list,
            'total_time': total_time,
            'training_speed': training_speed,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }
    
    # ==========================================================================
    # Inference Benchmark
    # ==========================================================================
    
    def benchmark_inference(model, name, num_tokens=100, num_runs=10):
        """Benchmark inference speed."""
        model.eval()
        prompt = torch.tensor(encode("ROMEO: "), dtype=torch.long).unsqueeze(0).to(device)
        
        # Warmup
        with torch.no_grad():
            _ = model.generate(prompt.clone(), max_new_tokens=10)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times = []
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = model.generate(prompt.clone(), max_new_tokens=num_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        tokens_per_sec = num_tokens / avg_time
        
        return {'avg_time': avg_time, 'tokens_per_sec': tokens_per_sec}
    
    # ==========================================================================
    # Memory Benchmark
    # ==========================================================================
    
    def benchmark_memory(model, name):
        """Benchmark memory usage."""
        if not torch.cuda.is_available():
            return {'peak_memory_mb': 0}
        
        torch.cuda.reset_peak_memory_stats()
        model.train()
        x, y = get_batch('train')
        logits, loss = model(x, y)
        loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        return {'peak_memory_mb': peak_memory}
    
    # ==========================================================================
    # Run Benchmarks
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("📊 BENCHMARK RESULTS")
    print("=" * 70)
    
    # Train both models
    more_metrics = train_model(more_model, "MORE", max_steps, eval_interval)
    baseline_metrics = train_model(baseline_model, "Baseline Transformer", max_steps, eval_interval)
    
    # Inference benchmarks
    print("\n⚡ Inference Speed Benchmark...")
    more_inference = benchmark_inference(more_model, "MORE")
    baseline_inference = benchmark_inference(baseline_model, "Baseline")
    
    # Memory benchmarks
    print("\n💾 Memory Usage Benchmark...")
    more_memory = benchmark_memory(more_model, "MORE")
    baseline_memory = benchmark_memory(baseline_model, "Baseline")
    
    # ==========================================================================
    # Print Summary
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("📈 SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Metric':<30} {'MORE':<20} {'Baseline':<20}")
    print("-" * 70)
    print(f"{'Parameters':<30} {more_params:,<20} {baseline_params:,<20}")
    print(f"{'Final Train Loss':<30} {more_metrics['final_train_loss']:<20.4f} {baseline_metrics['final_train_loss']:<20.4f}")
    print(f"{'Final Val Loss':<30} {more_metrics['final_val_loss']:<20.4f} {baseline_metrics['final_val_loss']:<20.4f}")
    print(f"{'Training Time (s)':<30} {more_metrics['total_time']:<20.1f} {baseline_metrics['total_time']:<20.1f}")
    print(f"{'Training Speed (tok/s)':<30} {more_metrics['training_speed']:<20.0f} {baseline_metrics['training_speed']:<20.0f}")
    print(f"{'Inference Speed (tok/s)':<30} {more_inference['tokens_per_sec']:<20.1f} {baseline_inference['tokens_per_sec']:<20.1f}")
    if torch.cuda.is_available():
        print(f"{'Peak Memory (MB)':<30} {more_memory['peak_memory_mb']:<20.1f} {baseline_memory['peak_memory_mb']:<20.1f}")
    
    # ==========================================================================
    # Generate Plots
    # ==========================================================================
    
    print("\n📊 Generating plots...")
    
    # Set style
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 12
    
    # 1. Learning Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training Loss
    axes[0].plot(more_metrics['steps'], more_metrics['train_losses'], 'b-', linewidth=2, label='MORE', marker='o', markersize=4)
    axes[0].plot(baseline_metrics['steps'], baseline_metrics['train_losses'], 'r--', linewidth=2, label='Baseline', marker='s', markersize=4)
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Validation Loss
    axes[1].plot(more_metrics['steps'], more_metrics['val_losses'], 'b-', linewidth=2, label='MORE', marker='o', markersize=4)
    axes[1].plot(baseline_metrics['steps'], baseline_metrics['val_losses'], 'r--', linewidth=2, label='Baseline', marker='s', markersize=4)
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: learning_curves.png")
    plt.show()
    
    # 2. Speed Comparison Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training Speed
    models = ['MORE', 'Baseline']
    train_speeds = [more_metrics['training_speed'], baseline_metrics['training_speed']]
    colors = ['#3498db', '#e74c3c']
    
    bars1 = axes[0].bar(models, train_speeds, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Tokens/Second')
    axes[0].set_title('Training Speed Comparison')
    for bar, speed in zip(bars1, train_speeds):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f'{speed:.0f}', 
                     ha='center', va='bottom', fontweight='bold')
    
    # Inference Speed
    infer_speeds = [more_inference['tokens_per_sec'], baseline_inference['tokens_per_sec']]
    
    bars2 = axes[1].bar(models, infer_speeds, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Tokens/Second')
    axes[1].set_title('Inference Speed Comparison')
    for bar, speed in zip(bars2, infer_speeds):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{speed:.1f}', 
                     ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('speed_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: speed_comparison.png")
    plt.show()
    
    # 3. Memory Comparison (if GPU available)
    if torch.cuda.is_available():
        fig, ax = plt.subplots(figsize=(8, 5))
        
        memories = [more_memory['peak_memory_mb'], baseline_memory['peak_memory_mb']]
        bars = ax.bar(models, memories, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Peak Memory (MB)')
        ax.set_title('GPU Memory Usage Comparison')
        for bar, mem in zip(bars, memories):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{mem:.1f} MB', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('memory_comparison.png', dpi=150, bbox_inches='tight')
        print("   ✅ Saved: memory_comparison.png")
        plt.show()
    
    # ==========================================================================
    # Sample Generation
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("📝 SAMPLE GENERATION")
    print("=" * 70)
    
    prompt_text = "ROMEO: "
    prompt = torch.tensor(encode(prompt_text), dtype=torch.long).unsqueeze(0).to(device)
    
    more_model.eval()
    baseline_model.eval()
    
    with torch.no_grad():
        more_output = more_model.generate(prompt.clone(), max_new_tokens=200, temperature=0.8)
        baseline_output = baseline_model.generate(prompt.clone(), max_new_tokens=200, temperature=0.8)
    
    print(f"\n🌀 MORE Output:")
    print("-" * 40)
    print(decode(more_output[0].tolist()))
    
    print(f"\n📦 Baseline Output:")
    print("-" * 40)
    print(decode(baseline_output[0].tolist()))
    
    print("\n" + "=" * 70)
    print("✅ BENCHMARK COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("   • learning_curves.png")
    print("   • speed_comparison.png")
    if torch.cuda.is_available():
        print("   • memory_comparison.png")
    print("\nPlease take screenshots of the output and graphs for the README!")
