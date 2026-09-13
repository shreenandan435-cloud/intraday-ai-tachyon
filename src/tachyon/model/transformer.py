"""Causal Transformer backbone — the Transformer-PPO policy trunk.

Architecture
------------
Pre-LayerNorm Transformer blocks stacked into a causal autoregressive trunk:

    x → LN → CausalSelfAttention → Dropout → +x → LN → GEGLU → Dropout → +x

Key design decisions
--------------------
* **Pre-LN**: Stabilises training at depth; the residual stream sees unit-scale inputs.
* **Causal SDPA**: ``F.scaled_dot_product_attention(..., is_causal=True)`` — no explicit
  ``T×T`` mask tensor allocated. FlashAttention kernel receives the causal flag directly.
* **RoPE on Q/K only**: The ``RotaryPositionalEmbedding`` from :mod:`embedding` is applied
  to queries and keys; values pass through unrotated (content, not position).
* **GEGLU MLP**: Single fused matmul ``d_model → 4*d_model*2`` → split → ``GELU(gate)*up``
  → ``Linear(4*d_model, d_model)``. Export-friendly; avoids ``SiLU`` + separate matmuls.
* **KV Cache**: Optional ``(k_cache, v_cache)`` tuple for ``seq_len=1`` autoregressive
  inference. Handled via a separate ``step`` method to keep ``forward`` export-clean.
* **FP16 Safe**: LayerNorm computed in FP32 (functional call with upcast weights).
  RoPE tables already FP32 via :class:`embedding.RotaryPositionalEmbedding._apply`.

Export discipline
-----------------
All modules survive ``torch.export`` → ONNX dynamo → TensorRT:
* No tensor-dependent control flow (no ``if x > 0:`` on tensors)
* No data-dependent shapes (KV cache uses ``torch.cat`` on symbolic dimension in ``step``)
* No Python-side state during forward
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812 - conventional PyTorch alias

from tachyon.model.embedding import RotaryPositionalEmbedding

# ─── constants ────────────────────────────────────────────────────────────────

DEFAULT_D_MODEL: Final[int] = 256
DEFAULT_N_HEADS: Final[int] = 8
DEFAULT_MLP_RATIO: Final[int] = 4
DEFAULT_DROPOUT: Final[float] = 0.1
DEFAULT_MAX_SEQ_LEN: Final[int] = 1024
DEFAULT_ROPE_THETA: Final[float] = 10_000.0

# ─── KV Cache ─────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class _KVCache:
    """Key-Value cache for autoregressive decoding.

    Shapes: (B, n_heads, T_past, head_dim)
    """
    k: Tensor
    v: Tensor


# ─── GEGLU MLP ────────────────────────────────────────────────────────────────


class GEGLU(nn.Module):
    """Gated Linear Unit with GELU activation.

    ``x → Linear(d, 4d*2) → [gate, up] → GELU(gate) * up → Linear(4d, d)``

    The first linear projects to *twice* the hidden width so we can split into
    gate and up projections in one matmul. This is the standard GEGLU formulation
    used in PaLM, LLaMA, etc., and compiles cleanly to ONNX/TensorRT.
    """

    def __init__(
        self,
        *,
        d_model: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        self.d_model = d_model
        self.mlp_ratio = mlp_ratio
        hidden = d_model * mlp_ratio

        # Single fused matmul for gate + up
        self.gate_up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Small init for gate/up (like LLaMA); down gets normal init
        nn.init.normal_(self.gate_up.weight, std=0.02)
        nn.init.normal_(self.down.weight, std=0.02 / math.sqrt(2 * self.mlp_ratio))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, d_model)
        gate_up = self.gate_up(x)  # (B, T, 2*hidden)
        gate, up = gate_up.chunk(2, dim=-1)  # each (B, T, hidden)

        # GEGLU: GELU(gate) * up
        hidden = F.gelu(gate, approximate="tanh") * up
        hidden = self.down(hidden)
        return self.dropout(hidden)  # type: ignore[no-any-return]


# ─── Causal Self-Attention ───────────────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE on Q/K.

    Uses PyTorch's ``scaled_dot_product_attention`` with ``is_causal=True`` for
    FlashAttention dispatch. No explicit causal mask tensor is constructed.

    RoPE is applied via :class:`RotaryPositionalEmbedding` to Q and K only.
    V passes through unrotated.

    KV Cache:
        Handled via the ``step`` method for autoregressive decoding (seq_len=1).
        The main ``forward`` is used for training and export (full window, no cache).
    """

    def __init__(
        self,
        *,
        d_model: int = DEFAULT_D_MODEL,
        n_heads: int = DEFAULT_N_HEADS,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        rope_theta: float = DEFAULT_ROPE_THETA,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Q, K, V projections (no bias for export friendliness)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

        # RoPE applied to Q and K (shared across heads)
        self.rope = RotaryPositionalEmbedding(
            self.head_dim, max_seq_len=max_seq_len, theta=rope_theta
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Causal self-attention forward pass (training/export path).

        Args:
            x: ``(B, T, d_model)`` input tensor.

        Returns:
            Output tensor ``(B, T, d_model)``.
        """
        b, t, _ = x.shape  # noqa: N806 - standard attention notation

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (broadcasts over head dim)
        q = self.rope(q)
        k = self.rope(k)
        # V is NOT rotated

        # Scaled dot-product attention with causal masking
        # is_causal=True triggers FlashAttention kernel when available
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Reshape back: (B, n_heads, T, head_dim) -> (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, self.d_model)

        # Output projection
        out = self.out_proj(attn_out)
        return self.dropout_layer(out)  # type: ignore[no-any-return]

    def step(
        self, x: Tensor, kv_cache: _KVCache | None = None
    ) -> tuple[Tensor, _KVCache]:
        """Single-step autoregressive forward with KV cache.

        Args:
            x: ``(B, 1, d_model)`` single token input.
            kv_cache: Optional ``_KVCache(k, v)`` from previous steps.

        Returns:
            Tuple of ``(output, new_kv_cache)`` where:
            - output: ``(B, 1, d_model)``
            - new_kv_cache: Updated ``_KVCache`` with shapes
              ``(B, n_heads, T_past+1, head_dim)``.
        """
        b, t, _ = x.shape  # noqa: N806 - standard attention notation
        if t != 1:
            raise ValueError(f"step() expects T=1, got T={t}")

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, 1, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, n_heads, 1, head_dim)
        q = q.view(b, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, 1, self.n_heads, self.head_dim).transpose(1, 2)

        # Absolute position of this token = depth of the existing cache, clamped to the
        # RoPE table. Once the cache saturates at ``max_seq_len`` the unclamped offset
        # would slice past the table (``cos_table[max:max+1]`` is empty -> broadcast
        # crash at step max_seq_len+1). While in range the clamp is an identity.
        max_cache = self.rope.max_seq_len
        offset = kv_cache.k.size(2) if kv_cache is not None else 0
        offset = min(offset, max_cache - 1)

        # RoPE at the true absolute position, applied BEFORE the new K/V enter the cache.
        # Cached keys were already rotated when they were appended; rotating them again here
        # would compound the rotation (R_j from step j applied (t-j) times -> R_{(t-j)*j}).
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        # V is NOT rotated

        # Append the already-rotated key to the cache, capped at the RoPE table length.
        # The slice is an identity while T <= max_cache; beyond it the oldest keys/values
        # are evicted (sliding window). This strictly bounds VRAM, keeps the per-step
        # ``cat`` copy cost bounded, and keeps the TensorRT KV profile (T_past <= 1024)
        # satisfiable when the present caches are fed back. Unconditional slice rather than
        # a branch: no tensor-value control flow, so the export graph stays single-path.
        if kv_cache is not None:
            k = torch.cat([kv_cache.k, k], dim=2)[:, :, -max_cache:, :]
            v = torch.cat([kv_cache.v, v], dim=2)[:, :, -max_cache:, :]

        # Attention (no causal mask needed for single step with cache)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # New cache
        new_kv_cache = _KVCache(k=k, v=v)

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, 1, self.d_model)

        # Output projection
        return self.dropout_layer(self.out_proj(attn_out)), new_kv_cache


# ─── Transformer Block (Pre-LN) ──────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: Attn + MLP with residuals.

    Structure:
        x → LN1 → CausalSelfAttention → Dropout → +x
          → LN2 → GEGLU → Dropout → +x

    Both LayerNorms are computed in FP32 for numerical stability under .half().
    """

    def __init__(
        self,
        *,
        d_model: int = DEFAULT_D_MODEL,
        n_heads: int = DEFAULT_N_HEADS,
        mlp_ratio: int = DEFAULT_MLP_RATIO,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        rope_theta: float = DEFAULT_ROPE_THETA,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Pre-attention LayerNorm
        self.norm1 = nn.LayerNorm(d_model)

        # Causal self-attention
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            dropout=dropout,
        )

        # Pre-MLP LayerNorm
        self.norm2 = nn.LayerNorm(d_model)

        # GEGLU MLP
        self.mlp = GEGLU(d_model=d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass (training/export path)."""
        # Pre-LN attention block
        normed = self._layer_norm_fp32(x, self.norm1)
        attn_out = self.attn(normed)
        x = x + attn_out

        # Pre-LN MLP block
        normed = self._layer_norm_fp32(x, self.norm2)
        mlp_out = self.mlp(normed)
        x = x + mlp_out

        return x  # noqa: RET504 - intentional in-place accumulation pattern

    def step(
        self, x: Tensor, kv_cache: _KVCache | None = None
    ) -> tuple[Tensor, _KVCache]:
        """Single-step autoregressive forward with KV cache."""
        # Pre-LN attention block
        normed = self._layer_norm_fp32(x, self.norm1)
        attn_out, new_kv_cache = self.attn.step(normed, kv_cache)
        x = x + attn_out

        # Pre-LN MLP block
        normed = self._layer_norm_fp32(x, self.norm2)
        mlp_out = self.mlp(normed)
        x = x + mlp_out

        return x, new_kv_cache

    def _layer_norm_fp32(self, x: Tensor, ln: nn.LayerNorm) -> Tensor:
        """LayerNorm computed in FP32 for numerical stability.

        TorchScript can resolve ln.normalized_shape as a constant attribute
        since ln is a known submodule.
        """
        return F.layer_norm(
            x.float(),
            ln.normalized_shape,
            ln.weight.float(),
            ln.bias.float(),
            ln.eps,
        ).to(x.dtype)


# ─── Causal Transformer Stack ────────────────────────────────────────────────


class CausalTransformer(nn.Module):
    """Stack of Pre-LN Transformer blocks with shared RoPE.

    The full trunk for the PPO actor-critic. Takes embedded LOB windows
    ``(B, T, d_model)`` and returns the same shape. The policy/value heads
    (in :mod:`ppo`) take the last timestep ``[:, -1]``.

    KV Cache:
        Handled via the ``step`` method for autoregressive decoding.
    """

    def __init__(
        self,
        *,
        d_model: int = DEFAULT_D_MODEL,
        n_heads: int = DEFAULT_N_HEADS,
        n_layers: int = 6,
        mlp_ratio: int = DEFAULT_MLP_RATIO,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        rope_theta: float = DEFAULT_ROPE_THETA,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")

        self.d_model = d_model
        self.n_layers = n_layers

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    max_seq_len=max_seq_len,
                    rope_theta=rope_theta,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Final LayerNorm (Pre-LN convention)
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the full transformer stack (training/export)."""
        for layer in self.layers:
            x = layer(x)

        # Final LayerNorm in FP32
        x = F.layer_norm(
            x.float(),
            self.norm_f.normalized_shape,
            self.norm_f.weight.float(),
            self.norm_f.bias.float(),
            self.norm_f.eps,
        ).to(x.dtype)

        return x  # noqa: RET504 - intentional in-place accumulation pattern

    def step(
        self, x: Tensor, kv_caches: list[_KVCache] | None = None
    ) -> tuple[Tensor, list[_KVCache]]:
        """Single-step autoregressive forward with KV cache.

        Args:
            x: ``(B, 1, d_model)`` single token input.
            kv_caches: Optional list of ``_KVCache`` (length = n_layers).

        Returns:
            Tuple of ``(output, new_kv_caches)``.
        """
        if kv_caches is not None and len(kv_caches) != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} KV caches, got {len(kv_caches)}"
            )

        new_kv_caches: list[_KVCache] = []

        hidden: Tensor = x
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            # mypy doesn't infer ModuleList element type correctly; cast explicitly
            layer_typed = cast(TransformerBlock, layer)
            next_hidden, new_cache = layer_typed.step(hidden, layer_cache)
            hidden = next_hidden
            new_kv_caches.append(new_cache)

        # Final LayerNorm in FP32 — applied to the stack OUTPUT (`hidden`), not the input
        # `x`. Normalising `x` here would discard the entire transformer stack and return
        # LayerNorm(embedding) to the PPO heads: the KV caches would update but the policy
        # would be running on the raw input at every live step.
        hidden = F.layer_norm(
            hidden.float(),
            self.norm_f.normalized_shape,
            self.norm_f.weight.float(),
            self.norm_f.bias.float(),
            self.norm_f.eps,
        ).to(hidden.dtype)

        return hidden, new_kv_caches
