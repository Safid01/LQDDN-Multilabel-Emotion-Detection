from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from lqddn.hf_utils import disable_hf_safetensors_auto_conversion


class FeedForwardBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states + self.dropout(self.net(hidden_states)))


class LabelQueryDecoder(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForwardBlock(hidden_size, dropout)

    def forward(
        self,
        queries: torch.Tensor,
        token_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        attended, weights = self.cross_attention(
            queries,
            token_states,
            token_states,
            key_padding_mask=key_padding_mask,
            need_weights=return_weights,
            average_attn_weights=True,
        )
        hidden_states = self.norm(queries + self.dropout(attended))
        return self.ffn(hidden_states), weights


class StaticGraphRefiner(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        propagated = torch.einsum("lk,bkh->blh", adjacency, hidden_states)
        updated = F.gelu(self.linear(propagated))
        return self.norm(hidden_states + self.dropout(updated))


class DynamicDependencyRefiner(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForwardBlock(hidden_size, dropout)

    def forward(self, hidden_states: torch.Tensor, return_weights: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, weights = self.self_attention(
            hidden_states,
            hidden_states,
            hidden_states,
            need_weights=return_weights,
            average_attn_weights=True,
        )
        hidden_states = self.norm(hidden_states + self.dropout(attended))
        return self.ffn(hidden_states), weights


@dataclass
class LQDDNOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    label_states: torch.Tensor
    cross_attention_weights: torch.Tensor | None = None
    dynamic_attention_weights: torch.Tensor | None = None
    fusion_gates: torch.Tensor | None = None
    base_states: torch.Tensor | None = None
    static_states: torch.Tensor | None = None
    dynamic_states: torch.Tensor | None = None


class LQDDNModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_labels: int,
        num_attention_heads: int,
        dropout: float,
        use_label_queries: bool,
        use_static_graph: bool,
        use_dynamic_dependency: bool,
    ) -> None:
        super().__init__()
        disable_hf_safetensors_auto_conversion()
        self.encoder = AutoModel.from_pretrained(backbone_name, use_safetensors=False)
        hidden_size = self.encoder.config.hidden_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.use_label_queries = use_label_queries
        self.use_static_graph = use_static_graph
        self.use_dynamic_dependency = use_dynamic_dependency

        self.label_queries = nn.Parameter(torch.randn(num_labels, hidden_size) * 0.02)
        self.decoder = LabelQueryDecoder(hidden_size, num_attention_heads, dropout)
        self.static_refiner = StaticGraphRefiner(hidden_size, dropout)
        self.dynamic_refiner = DynamicDependencyRefiner(hidden_size, num_attention_heads, dropout)
        self.fusion_gate = nn.Linear(hidden_size * 3, 2)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier_weight = nn.Parameter(torch.randn(num_labels, hidden_size) * 0.02)
        self.classifier_bias = nn.Parameter(torch.zeros(num_labels))

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids"),
        )
        token_states = outputs.last_hidden_state
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = token_states[:, 0]
        return token_states, pooled

    def forward(self, batch: dict[str, torch.Tensor], adjacency: torch.Tensor, return_analysis: bool = False) -> LQDDNOutput:
        token_states, pooled = self.encode(batch)
        batch_size = pooled.size(0)
        label_queries = self.label_queries.unsqueeze(0).expand(batch_size, -1, -1)
        cross_attention_weights = None
        dynamic_attention_weights = None
        gates = None

        if self.use_label_queries:
            base_states, cross_attention_weights = self.decoder(
                label_queries,
                token_states,
                batch.get("attention_mask"),
                return_weights=return_analysis,
            )
        else:
            base_states = label_queries + pooled.unsqueeze(1)

        static_states = base_states
        dynamic_states = base_states
        if self.use_static_graph:
            static_states = self.static_refiner(base_states, adjacency)
        if self.use_dynamic_dependency:
            dynamic_states, dynamic_attention_weights = self.dynamic_refiner(
                base_states,
                return_weights=return_analysis,
            )

        if self.use_static_graph and self.use_dynamic_dependency:
            gate_logits = self.fusion_gate(torch.cat([base_states, static_states, dynamic_states], dim=-1))
            gates = torch.softmax(gate_logits, dim=-1)
            fused_states = base_states + gates[..., :1] * static_states + gates[..., 1:] * dynamic_states
        elif self.use_static_graph:
            fused_states = base_states + static_states
        elif self.use_dynamic_dependency:
            fused_states = base_states + dynamic_states
        else:
            fused_states = base_states

        fused_states = self.final_norm(self.dropout(fused_states))
        logits = (fused_states * self.classifier_weight.unsqueeze(0)).sum(dim=-1) + self.classifier_bias.unsqueeze(0)
        probabilities = torch.sigmoid(logits)
        return LQDDNOutput(
            logits=logits,
            probabilities=probabilities,
            label_states=fused_states,
            cross_attention_weights=cross_attention_weights if return_analysis else None,
            dynamic_attention_weights=dynamic_attention_weights if return_analysis else None,
            fusion_gates=gates if return_analysis else None,
            base_states=base_states if return_analysis else None,
            static_states=static_states if return_analysis else None,
            dynamic_states=dynamic_states if return_analysis else None,
        )
