"""
Crossword-adapted Tiny Recursive Reasoning Model.

Extends the base TRM with clue embedding injection:
- Pre-computed sentence-transformer embeddings for each cell's across/down clues
  are projected to hidden_size and added to the input token embeddings.
- The rest of the recursive reasoning loop is identical to the base TRM.
"""
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm, SwiGLU, Attention, RotaryEmbedding,
    CosSin, CastedEmbedding, CastedLinear,
)
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


# ---------------------------------------------------------------------------
# Carry dataclasses (same as base TRM)
# ---------------------------------------------------------------------------

@dataclass
class CrosswordTRM_InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class CrosswordTRM_Carry:
    inner_carry: CrosswordTRM_InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class CrosswordTRM_Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int = 1
    vocab_size: int = 29  # PAD + BLACK + BLANK + A-Z

    H_cycles: int
    L_cycles: int

    H_layers: int  # unused, kept for interface compat
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    mlp_t: bool = False
    puzzle_emb_len: int = 0
    no_ACT_continue: bool = True

    # ---- Crossword-specific ----
    clue_emb_dim: int = 384  # dimension of pre-computed clue embeddings


# ---------------------------------------------------------------------------
# Blocks (reused from base TRM logic)
# ---------------------------------------------------------------------------

class CrosswordTRM_Block(nn.Module):
    def __init__(self, config: CrosswordTRM_Config) -> None:
        super().__init__()
        self.config = config

        if self.config.mlp_t:
            self.puzzle_emb_len = (
                -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
                if self.config.puzzle_emb_len == 0
                else self.config.puzzle_emb_len
            )
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )

        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class CrosswordTRM_ReasoningModule(nn.Module):
    def __init__(self, layers: List[CrosswordTRM_Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


# ---------------------------------------------------------------------------
# Inner model (with clue injection)
# ---------------------------------------------------------------------------

class CrosswordTRM_Inner(nn.Module):
    def __init__(self, config: CrosswordTRM_Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # ---- I/O layers ----
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, self.config.hidden_size,
            init_std=embed_init_std, cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # ---- Clue embedding projection ----
        self.clue_proj = CastedLinear(self.config.clue_emb_dim, self.config.hidden_size, bias=False)

        # ---- Puzzle embedding (optional, usually disabled for crosswords) ----
        self.puzzle_emb_len = (
            -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            if self.config.puzzle_emb_len == 0
            else self.config.puzzle_emb_len
        )
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # ---- Positional encodings ----
        total_seq = self.config.seq_len + self.puzzle_emb_len
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=total_seq,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                total_seq, self.config.hidden_size,
                init_std=embed_init_std, cast_to=self.forward_dtype,
            )

        # ---- Reasoning layers ----
        self.L_level = CrosswordTRM_ReasoningModule(
            layers=[CrosswordTRM_Block(self.config) for _ in range(self.config.L_layers)]
        )

        # ---- Initial latent states ----
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # ---- Q head init (biased toward not halting early) ----
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    # ---- Input embedding (with clue injection) ----

    def _input_embeddings(
        self,
        input_ids: torch.Tensor,
        puzzle_identifiers: torch.Tensor,
        clue_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Token embedding
        embedding = self.embed_tokens(input_ids.to(torch.int32))

        # Inject clue embeddings
        if clue_embeddings is not None:
            clue_proj = self.clue_proj(clue_embeddings.to(self.forward_dtype))
            embedding = embedding + clue_proj

        # Puzzle embeddings (optional)
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = (
                self.puzzle_emb_len * self.config.hidden_size
                - puzzle_embedding.shape[-1]
            )
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (
                    puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size),
                    embedding,
                ),
                dim=-2,
            )

        # Learned positional embeddings
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
            )

        return self.embed_scale * embedding

    # ---- Carry management ----

    def empty_carry(self, batch_size: int):
        total_seq = self.config.seq_len + self.puzzle_emb_len
        return CrosswordTRM_InnerCarry(
            z_H=torch.empty(batch_size, total_seq, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, total_seq, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: CrosswordTRM_InnerCarry):
        return CrosswordTRM_InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    # ---- Forward ----

    def forward(
        self,
        carry: CrosswordTRM_InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[CrosswordTRM_InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding (with clue injection)
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch["puzzle_identifiers"],
            clue_embeddings=batch.get("clue_embeddings"),
        )

        # Recursive reasoning
        z_H, z_L = carry.z_H, carry.z_L

        # H_cycles - 1 without grad
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)

        # Last H cycle with grad
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # Output heads
        new_carry = CrosswordTRM_InnerCarry(
            z_H=z_H.detach(), z_L=z_L.detach()
        )
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


# ---------------------------------------------------------------------------
# ACT wrapper
# ---------------------------------------------------------------------------

class CrosswordTRM(nn.Module):
    """Adaptive Computation Time wrapper for the crossword TRM."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = CrosswordTRM_Config(**config_dict)
        self.inner = CrosswordTRM_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return CrosswordTRM_Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: CrosswordTRM_Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[CrosswordTRM_Carry, Dict[str, torch.Tensor]]:

        # Reset carry for halted sequences (start fresh with new puzzle data)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(
                        new_inner_carry, new_current_data
                    )
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        return (
            CrosswordTRM_Carry(new_inner_carry, new_steps, halted, new_current_data),
            outputs,
        )
