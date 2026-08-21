"""LOB state embedding with rotary position encoding — the Transformer's input stage.

Projects the ``(B, T, 25)`` normalised state (22 order book features + 3 portfolio state)
that :mod:`tachyon.model.dataset` yields into the ``(B, T, d_model)`` residual stream
the policy trunk consumes.

Export discipline
-----------------
Everything here is written to survive ``torch.export`` → ONNX → TensorRT unchanged, because the
alternative is discovering at conversion time that the trained graph cannot be compiled. Three
rules, enforced by :class:`tests.unit.test_model.TestExportSafety`:

* **No tensor-dependent control flow.** Not one ``if`` reads a tensor *value*. Every branch that
  exists is resolved in ``__init__`` against Python ints, so the traced graph is the only graph.
* **No data-dependent shapes.** The rotary tables are buffers sized at construction and sliced by
  ``x.size(-2)``. A slice by a symbolic dimension is a single ``Slice`` node; a table *built* per
  call would bake ``arange``/``outer`` into the graph and defeat constant folding.
* **No Python-side state.** Nothing is cached on ``self`` during ``forward``, so the first call
  and the ten-thousandth produce identical graphs.

Half precision
--------------
The whole module is safe under ``torch.autocast`` and under an explicit ``.half()``, which are
not the same thing — under ``.half()`` the *parameters* are fp16 too, so relying on autocast's
fp32 op list is not enough. Two places are pinned to fp32 regardless of input dtype:

* **LayerNorm**, computed through :func:`torch.nn.functional.layer_norm` on up-cast inputs *and*
  up-cast affine parameters. Its reduction accumulates ``d_model`` squared values; in fp16 the
  variance of a wide layer is exactly the kind of sum that saturates at 65 504.
* **The rotary rotation**, in :class:`RotaryPositionalEmbedding`. The tables are held in fp32
  because ``cos``/``sin`` at large positions need mantissa the fp16 format does not have — fp16's
  spacing near 1.0 is ~5e-4, which at ``seq_len=128`` quantises adjacent positions into each
  other and silently destroys the position signal rather than raising anything.

Both cast back to the input dtype on the way out, so the module is dtype-transparent: the matmul
that dominates the cost still runs in fp16 and still hits the tensor cores.
"""

# NOTE: no ``from __future__ import annotations`` here — the one file in the package without it,
# and deliberately so.
#
# PEP 563 (that import) turns every annotation into a *string*, and TorchScript's annotation
# resolver cannot look those strings back up: a class-body ``cos_table: Tensor`` dies with
# ``ValueError: Unknown type annotation: 'Tensor'`` at ``torch.jit.script`` time. Those buffer
# declarations are not optional either — without them mypy sees the ``Tensor | Module`` union
# that ``nn.Module.__getattr__`` is typed to return, and rejects both the slice in ``forward``
# and the ``.float()`` in ``_apply``.
#
# Python 3.14 resolves the conflict for free. PEP 649 defers annotation evaluation natively, so
# omitting the import costs nothing at import time and still yields real type objects when
# TorchScript asks for them. Forward references need no quoting; ruff's UP037 enforces that.
from collections.abc import Callable
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812 - the conventional torch alias

from tachyon.model.dataset import LOB_FEATURES

DEFAULT_D_MODEL: Final[int] = 128

#: Embedding input width: 22 order book features (from dataset: 20 base + spread_tick + obi_l1)
#: + 3 portfolio state features (position, entry_price_bps, holding_bars) injected by RL env.
EMBEDDING_IN_FEATURES: Final[int] = LOB_FEATURES + 3

#: Rotary base. 10 000 is the value RoFormer introduced and every later model inherited; it sets
#: how fast the wavelength spectrum decays across the head dimension.
DEFAULT_ROPE_THETA: Final[float] = 10_000.0

#: Longest window the rotary tables are precomputed for. Sized generously against
#: :data:`~tachyon.model.dataset.DEFAULT_SEQ_LEN`; the cost is
#: ``max_seq_len * d_model * 2 * 4`` bytes, which at the defaults is 1 MB.
DEFAULT_MAX_SEQ_LEN: Final[int] = 1024


def _rotate_half(x: Tensor) -> Tensor:
    """``[a, b] -> [-b, a]`` over the two halves of the last dimension.

    ``narrow`` rather than slicing or ``chunk``: it is unambiguously scriptable, and it lowers to
    exactly one ``Slice`` node per half in ONNX. ``chunk`` lowers to ``Split`` with an output
    count that some TensorRT parser versions treat as dynamic.
    """
    half = x.size(-1) // 2
    x1 = torch.narrow(x, -1, 0, half)
    x2 = torch.narrow(x, -1, half, half)
    return torch.cat((-x2, x1), dim=-1)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary position embedding (RoPE), applied over the last two dimensions.

    Absolute sinusoidal encodings are *added* to the content vector, which entangles position with
    value and leaves the model to disentangle them. RoPE *rotates* instead, and the rotation has
    the property that ``<R_m q, R_n k>`` depends only on ``m - n``. Attention therefore sees
    relative offsets directly, which is what a book wants: what matters is that a sweep happened
    four updates ago, not that it happened at index 71 of the window.

    Shape-agnostic by construction. The tables broadcast as ``(T, dim)``, so this works unchanged
    on ``(B, T, dim)`` here and on ``(B, heads, T, head_dim)`` inside attention later — which is
    where RoPE properly belongs, applied to queries and keys only.

    Args:
        dim: rotated width. Must be even — the rotation pairs dimension ``i`` with ``i + dim/2``.
        max_seq_len: table length. A longer input raises on the broadcast rather than silently
            truncating, which is the failure mode worth having.
        theta: rotary base.

    Raises:
        ValueError: ``dim`` is odd or non-positive, or ``max_seq_len`` is non-positive.
    """

    # Declared so both mypy and TorchScript see a Tensor rather than the ``Tensor | Module``
    # union that ``nn.Module.__getattr__`` returns for a registered buffer.
    cos_table: Tensor
    sin_table: Tensor

    def __init__(
        self,
        dim: int,
        *,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        theta: float = DEFAULT_ROPE_THETA,
    ) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"RoPE dim must be positive and even, got {dim}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / float(dim)))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        # Duplicated rather than interleaved, which is what pairs dimension i with i + dim/2 and
        # makes _rotate_half the correct partner. The interleaved formulation is equivalent up to
        # a permutation of the head dimension, but mixing the two silently trains a model whose
        # positions are scrambled — so the pairing convention is fixed here and nowhere else.
        table = torch.cat((freqs, freqs), dim=-1)

        # persistent=False: these are a pure function of (dim, max_seq_len, theta), so writing
        # them into every checkpoint would add megabytes of derivable data and, worse, let a
        # stale checkpoint override a corrected table on load.
        self.register_buffer("cos_table", table.cos(), persistent=False)
        self.register_buffer("sin_table", table.sin(), persistent=False)

    def _apply(
        self, fn: Callable[[Tensor], Tensor], recurse: bool = True
    ) -> RotaryPositionalEmbedding:
        """Follow device moves, but refuse dtype demotion of the rotary tables.

        ``Module.half()`` casts every floating-point *buffer* as well as every parameter, so
        without this the tables silently become fp16 the moment the model is halved for TensorRT —
        precisely the deployment path this module exists to serve, and precisely the case the
        fp32 promise in the class docstring is about. The damage is invisible: no error, no NaN,
        just neighbouring positions rounding onto each other and the relative-position signal
        quietly degrading.

        ``fn`` is still applied first so ``.cuda()``, ``.to(device)`` and pinning all work
        normally; only the dtype is put back.
        """
        super()._apply(fn, recurse)  # type: ignore[no-untyped-call]
        self.cos_table = self.cos_table.float()
        self.sin_table = self.sin_table.float()
        return self

    def forward(self, x: Tensor) -> Tensor:
        """Rotate ``x``. Last dimension must be ``dim``; second-to-last is the time axis."""
        seq_len = x.size(-2)
        cos = self.cos_table[:seq_len]
        sin = self.sin_table[:seq_len]
        # fp32 throughout the rotation, then back — see the module docstring on why fp16 cos/sin
        # quantises adjacent positions together.
        rotated = x.float() * cos + _rotate_half(x.float()) * sin
        return rotated.to(x.dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, theta={self.theta}"


class LOBEmbedding(nn.Module):
    """``(B, T, 25)`` normalised state (22 order book + 3 portfolio) to ``(B, T, d_model)``.

    Pipeline, and the ordering is load-bearing::

        Linear -> GELU(tanh) -> LayerNorm(fp32) -> Dropout -> RoPE

    **RoPE is strictly last.** Its whole guarantee is that the inner product of two rotated
    vectors depends only on their separation; any elementwise nonlinearity applied *after* the
    rotation breaks that identity, and it breaks it quietly — the model still trains, just without
    a usable relative-position signal.

    **LayerNorm sits after the activation**, not before it. GELU's output is not zero-mean, so
    normalising first leaves the trunk's first pre-norm block receiving a shifted distribution;
    normalising last hands it unit scale whatever the projection learned.

    Args:
        n_features: input width. Defaults to :data:`EMBEDDING_IN_FEATURES` (25 = 22 order book
            features from dataset + 3 portfolio state features injected by RL environment).
        d_model: residual stream width. Must be even (RoPE pairs halves).
        max_seq_len: longest window the rotary tables cover.
        dropout: applied before the rotation. Identity in ``eval()``, so exported graphs are
            unaffected — but export from ``eval()`` regardless.
        rope_theta: rotary base.
        apply_rope: ``False`` returns the projection without rotation, for a trunk that applies
            RoPE inside attention to queries and keys — the more standard placement. The
            :class:`RotaryPositionalEmbedding` module is exported for exactly that use.

    Raises:
        ValueError: ``d_model`` is odd or non-positive, or ``n_features`` is non-positive.
    """

    def __init__(
        self,
        *,
        n_features: int = EMBEDDING_IN_FEATURES,
        d_model: int = DEFAULT_D_MODEL,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        dropout: float = 0.0,
        rope_theta: float = DEFAULT_ROPE_THETA,
        apply_rope: bool = True,
    ) -> None:
        super().__init__()
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        if d_model <= 0 or d_model % 2 != 0:
            raise ValueError(f"d_model must be positive and even, got {d_model}")

        self.n_features = n_features
        self.d_model = d_model
        # Read in forward as a plain bool. Resolved here, against a Python value, so the exported
        # graph contains one branch or the other and never a conditional.
        self.apply_rope = apply_rope

        self.projection = nn.Linear(n_features, d_model)
        self.activation = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(d_model, max_seq_len=max_seq_len, theta=rope_theta)

        # LayerNorm's affine parameters live on a real submodule so checkpoints carry the
        # conventional `norm.weight` / `norm.bias` names, but the op is invoked functionally in
        # forward to force fp32 accumulation. Calling self.norm(x) would run in whatever dtype
        # the module was cast to.
        self.norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier-uniform on the projection.

        The default ``nn.Linear`` init scales by ``1/sqrt(fan_in)``, which is tuned for the deep
        stacks it sits inside. This is a single widening projection from 25 to ``d_model``, and
        Xavier keeps the output variance at the input's — which the feature design already fixed
        near unit scale by construction (bps, log-quantities, spread_tick, obi_l1, portfolio).
        """
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Embed a batch of windows.

        Args:
            x: ``(B, T, n_features)``, float32 or float16.

        Returns:
            ``(B, T, d_model)``, in ``x``'s dtype.
        """
        # Annotated because nn.Module.__call__ is typed as returning Any; without these the
        # module's declared -> Tensor return degrades to Any and stops being checked at all.
        hidden: Tensor = self.activation(self.projection(x))

        # fp32 LayerNorm, parameters included, so this holds under .half() as well as autocast.
        normed = F.layer_norm(
            hidden.float(),
            self.norm.normalized_shape,
            self.norm.weight.float(),
            self.norm.bias.float(),
            self.norm.eps,
        ).to(hidden.dtype)

        embedded: Tensor = self.dropout(normed)
        if self.apply_rope:
            embedded = self.rope(embedded)
        return embedded

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}, d_model={self.d_model}, apply_rope={self.apply_rope}"
