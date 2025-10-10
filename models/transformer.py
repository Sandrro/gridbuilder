"""Autoregressive Transformer model for grid cell synthesis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from data.dataset import EDGE_TOKENS


@dataclass
class ModelConfig:
    """Configuration container for the Transformer."""

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    dropout: float = 0.1
    max_positions: int = 4096


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[start_pos : start_pos + seq_len]


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention with optional additive bias."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim * n_heads != d_model:
            raise ValueError("d_model must be divisible by n_heads")
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            scores += attn_mask.unsqueeze(0)
        if attn_bias is not None:
            scores += attn_bias.unsqueeze(1)
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Single decoder block with residual connections."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        attn_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_out = self.attn(
            self.norm1(x),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            attn_bias=attn_bias,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class PromptEncoder(nn.Module):
    """Encodes zone-level conditioning information into prompt tokens."""

    def __init__(self, zone_vocab: int, service_vocab: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.zone_embedding = nn.Embedding(zone_vocab, d_model)
        self.living_proj = nn.Linear(1, d_model)
        self.living_norm = nn.LayerNorm(1)
        if service_vocab > 0:
            self.service_proj = nn.Linear(service_vocab, d_model)
            self.service_norm = nn.LayerNorm(service_vocab)
        else:
            self.service_proj = None
            self.service_norm = None
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        zone_type_ids: torch.Tensor,
        living_prompt: torch.Tensor,
        living_mask: torch.Tensor,
        service_prompt: torch.Tensor,
        service_mask: torch.Tensor,
    ) -> torch.Tensor:
        zone_tok = self.zone_embedding(zone_type_ids).unsqueeze(1)
        living_normed = self.living_norm(living_prompt)
        living_tok = self.living_proj(living_normed) * living_mask.unsqueeze(-1)
        if self.service_proj is not None and service_prompt.size(1) > 0:
            service_normed = self.service_norm(service_prompt)
            has_budget = (service_mask.sum(dim=1, keepdim=True) > 0).float()
            service_tok = self.service_proj(service_normed) * has_budget
        else:
            service_tok = torch.zeros_like(living_tok)
        tokens = torch.cat([zone_tok, living_tok, service_tok], dim=1)
        return self.dropout(tokens)


class AutoregressiveTransformer(nn.Module):
    """Decoder-only Transformer model operating on grid cell sequences."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        service_vocab: int,
        zone_vocab: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.prompt_encoder = PromptEncoder(zone_vocab, service_vocab, config.d_model, config.dropout)
        self.edge_embeddings = nn.Parameter(torch.randn(len(EDGE_TOKENS), config.d_model))
        self.cell_embedding = nn.Embedding(5, config.d_model)  # 4 classes + start token
        self.pos_encoding = PositionalEncoding(config.d_model, max_len=config.max_positions + 32)

        self.layers = nn.ModuleList(
            [TransformerBlock(config.d_model, config.n_heads, config.dropout) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

        self.cell_class_head = nn.Linear(config.d_model, 4)
        self.is_living_head = nn.Linear(config.d_model, 1)
        self.storeys_head = nn.Linear(config.d_model, 1)
        self.living_area_head = nn.Linear(config.d_model, 1)
        self.service_type_head = nn.Linear(config.d_model, service_vocab)
        self.service_capacity_head = nn.Linear(config.d_model, 1)
        self.edge_bias_scale = nn.Parameter(torch.tensor(1.0))

    @property
    def num_edge_tokens(self) -> int:
        return len(EDGE_TOKENS)

    def build_attention_bias(
        self,
        edge_distances: torch.Tensor,
        seq_mask: torch.Tensor,
        prompt_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch, seq_len, _ = edge_distances.size()
        total_len = prompt_tokens + self.num_edge_tokens + seq_len
        bias = torch.zeros(batch, total_len, total_len, device=device)
        prefix = prompt_tokens
        for b in range(batch):
            length = int(seq_mask[b].sum().item())
            for i in range(length):
                dist = edge_distances[b, i]
                target = prefix
                bias[b, prefix + self.num_edge_tokens + i, target : target + self.num_edge_tokens] = (
                    -dist * self.edge_bias_scale
                )
        return bias

    def build_attention_mask(
        self,
        seq_len: int,
        prompt_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_len = prompt_tokens + self.num_edge_tokens + seq_len
        mask = torch.zeros(total_len, total_len, device=device)
        prefix = prompt_tokens + self.num_edge_tokens
        for i in range(prefix, total_len):
            mask[i, i + 1 :] = float("-inf")
        return mask

    def forward(
        self,
        cell_classes: torch.Tensor,
        sequence_mask: torch.Tensor,
        *,
        zone_type_ids: torch.Tensor,
        living_prompt: torch.Tensor,
        living_prompt_mask: torch.Tensor,
        service_prompt: torch.Tensor,
        service_prompt_mask: torch.Tensor,
        edge_distances: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = cell_classes.device
        batch, seq_len = cell_classes.shape

        prompts = self.prompt_encoder(
            zone_type_ids=zone_type_ids,
            living_prompt=living_prompt,
            living_mask=living_prompt_mask,
            service_prompt=service_prompt,
            service_mask=service_prompt_mask,
        )
        edge_tokens = self.edge_embeddings.unsqueeze(0).expand(batch, -1, -1)

        start_ids = torch.full((batch, 1), 4, dtype=torch.long, device=device)
        prev_tokens = torch.cat([start_ids, cell_classes[:, :-1].clamp(min=0)], dim=1)
        cell_emb = self.cell_embedding(prev_tokens)
        cell_emb = self.pos_encoding(cell_emb)

        x = torch.cat([prompts, edge_tokens, cell_emb], dim=1)

        prompt_tokens = prompts.size(1)
        attn_mask = self.build_attention_mask(seq_len, prompt_tokens, device)
        bias = self.build_attention_bias(edge_distances, sequence_mask, prompt_tokens, device)
        key_padding_mask = torch.zeros(batch, x.size(1), dtype=torch.bool, device=device)
        prefix = prompt_tokens + self.num_edge_tokens
        pad = ~(sequence_mask.bool())
        key_padding_mask[:, prefix : prefix + seq_len] = pad

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, attn_bias=bias)
        x = self.final_norm(x)

        cells = x[:, prefix:]
        outputs = {
            "cell_class_logits": self.cell_class_head(cells),
            "is_living_logits": self.is_living_head(cells),
            "storeys": self.storeys_head(cells),
            "living_area": self.living_area_head(cells),
            "service_type_logits": self.service_type_head(cells),
            "service_capacity": self.service_capacity_head(cells),
        }
        return outputs

    def decode_step(
        self,
        generated: torch.Tensor,
        *,
        zone_type_ids: torch.Tensor,
        living_prompt: torch.Tensor,
        living_prompt_mask: torch.Tensor,
        service_prompt: torch.Tensor,
        service_prompt_mask: torch.Tensor,
        edge_distances: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Convenience wrapper used at inference time."""
        return self.forward(
            generated,
            sequence_mask,
            zone_type_ids=zone_type_ids,
            living_prompt=living_prompt,
            living_prompt_mask=living_prompt_mask,
            service_prompt=service_prompt,
            service_prompt_mask=service_prompt_mask,
            edge_distances=edge_distances,
        )
