import torch
import torch.nn as nn


class ReferTransformer(nn.Module):
    def __init__(
        self,
        d_model=128,
        nhead=4,
        dim_feedforward=256,
        dropout=0.1,
        num_queries=16,
    ):
        super().__init__()
        self.query_text_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.query_text_norm = nn.LayerNorm(d_model)
        self.query_gauss_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.query_gauss_norm = nn.LayerNorm(d_model)
        self.query_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, d_model),
        )
        self.query_ffn_norm = nn.LayerNorm(d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.query_norm = nn.LayerNorm(d_model)

    def forward(self, gauss_tokens, text_tokens):
        if text_tokens.dim() == 2:
            text_tokens = text_tokens.unsqueeze(0)

        gauss_tokens = gauss_tokens.unsqueeze(0)

        queries = self.query_embed.weight.unsqueeze(0)
        q_text, _ = self.query_text_attn(
            query=queries, key=text_tokens, value=text_tokens
        )
        queries = self.query_text_norm(queries + q_text)

        q_gauss, attn_weights = self.query_gauss_attn(
            query=queries, key=gauss_tokens, value=gauss_tokens, need_weights=True
        )
        queries = self.query_gauss_norm(queries + q_gauss)
        queries = self.query_ffn_norm(queries + self.query_ffn(queries))
        queries = self.query_norm(queries)

        relation_logits = torch.matmul(queries, gauss_tokens.transpose(1, 2)).squeeze(0)
        # gauss_context = torch.bmm(attn_weights.transpose(1, 2), queries).squeeze(0)
        return queries.squeeze(0), relation_logits
