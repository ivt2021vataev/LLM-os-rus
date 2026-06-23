import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_out = cfg["emb_dim"]
        self.num_heads = cfg["n_heads"]
        self.head_dim = self.d_out // self.num_heads

        self.W_query = nn.Linear(cfg["emb_dim"], self.d_out, bias=cfg["qkv_bias"])
        self.W_key = nn.Linear(cfg["emb_dim"], self.d_out, bias=cfg["qkv_bias"])
        self.W_value = nn.Linear(cfg["emb_dim"], self.d_out, bias=cfg["qkv_bias"])

        self.out_proj = nn.Linear(self.d_out, self.d_out)
        self.dropout = nn.Dropout(cfg["drop_rate"])

        self.register_buffer("mask",
                             torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1).bool())

    def forward(self, x):
        batch_size, num_tokens, emb_dim = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3) / (self.head_dim ** 0.5)
        mask_bool = self.mask[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(batch_size, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x): 
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

    def forward(self, x):
        x = x + self.att(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        batch, seq = x.shape
        x = self.tok_emb(x) + self.pos_emb(torch.arange(seq, device=x.device))
        x = self.trf_blocks(x)
        return self.out_head(self.final_norm(x))

    def generate(self, idx, max_new_tokens, context_size):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]

            with torch.no_grad():
                logits = self(idx_cond)

            logits = logits[:, -1, :]

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx
