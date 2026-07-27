import copy
import math

import torch
import torch.nn as nn


def _sincos_from_values(values, embed_dim):
    values = values.reshape(-1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, device=values.device, dtype=values.dtype)
        * -(math.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(values.numel(), embed_dim, device=values.device, dtype=values.dtype)
    pe[:, 0::2] = torch.sin(values.unsqueeze(1) * div_term)
    pe[:, 1::2] = torch.cos(values.unsqueeze(1) * div_term[:pe[:, 1::2].shape[1]])
    return pe


def _compatible_num_heads(embed_dim, requested_heads):
    requested_heads = int(requested_heads)
    if embed_dim % requested_heads == 0:
        return requested_heads
    for heads in range(min(requested_heads, embed_dim), 0, -1):
        if embed_dim % heads == 0:
            return heads
    return 1


def _make_transformer_encoder(embed_dim, heads, depth, dropout):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=_compatible_num_heads(embed_dim, heads),
        dim_feedforward=4 * embed_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(
        encoder_layer,
        num_layers=max(1, int(depth)),
        norm=nn.LayerNorm(embed_dim),
    )


def _parse_harmonic_ratios(value):
    ratios = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        ratio = float(item)
        if ratio <= 1.0:
            raise ValueError("freq_pope_harmonics must contain ratios > 1")
        ratios.append(ratio)
    if not ratios:
        raise ValueError("freq_pope_harmonics must not be empty")
    return ratios


def _apply_rope(x, positions, rope_base):
    if x.size(-1) % 2 != 0:
        raise ValueError("RoPE expects an even per-head dimension")
    theta = torch.exp(
        torch.arange(
            0,
            x.size(-1),
            2,
            device=x.device,
            dtype=x.dtype,
        )
        * -(math.log(float(rope_base)) / x.size(-1))
    )
    phase = positions.to(device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
    phase = phase * theta.view(1, 1, 1, -1)
    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = x_even * cos_phase - x_odd * sin_phase
    rotated[..., 1::2] = x_even * sin_phase + x_odd * cos_phase
    return rotated


class HarmonicPoPEEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        heads,
        dropout,
        window_seconds,
        harmonic_ratios,
        harmonic_sigma,
        rope_base,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = _compatible_num_heads(self.embed_dim, int(heads))
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by heads")
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                "Per-head dimension (%d) must be even for frequency RoPE"
                % self.head_dim
            )
        self.window_seconds = float(window_seconds)
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be > 0 for harmonic PoPE")
        self.harmonic_sigma = float(harmonic_sigma)
        if self.harmonic_sigma <= 0:
            raise ValueError("freq_pope_sigma must be > 0")
        self.rope_base = float(rope_base)
        if self.rope_base <= 1:
            raise ValueError("freq_rope_base must be > 1")

        ratios = torch.tensor(harmonic_ratios, dtype=torch.float32)
        self.register_buffer("harmonic_log_ratios", torch.log(ratios), persistent=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.harmonic_weight = nn.Parameter(
            torch.zeros(self.num_heads, len(harmonic_ratios))
        )
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(4 * self.embed_dim, self.embed_dim),
        )
        self.dropout = nn.Dropout(p=float(dropout))

    def _frequency_coordinates(self, seq_len, device, dtype):
        freqs = torch.arange(1, seq_len + 1, device=device, dtype=dtype)
        freqs = freqs / self.window_seconds
        log_freqs = torch.log(freqs.clamp_min(1e-12))
        denom = (log_freqs[-1] - log_freqs[0]).clamp_min(1e-6)
        norm_log_freqs = (log_freqs - log_freqs[0]) / denom
        rope_positions = norm_log_freqs * max(1, seq_len - 1)
        return log_freqs, norm_log_freqs, rope_positions

    def _pope_bias(self, seq_len, device, dtype):
        log_freqs, _, _ = self._frequency_coordinates(
            seq_len,
            device,
            dtype,
        )
        log_ratio = (log_freqs[:, None] - log_freqs[None, :]).abs()
        harmonic_delta = log_ratio.unsqueeze(-1) - self.harmonic_log_ratios.to(
            device=device,
            dtype=dtype,
        ).view(1, 1, -1)
        harmonic_features = torch.exp(
            -0.5 * (harmonic_delta / self.harmonic_sigma) ** 2
        )
        harmonic_bias = torch.einsum(
            "ijr,hr->hij",
            harmonic_features,
            self.harmonic_weight.to(dtype=dtype),
        ).unsqueeze(0)
        return harmonic_bias

    def forward(self, x, token_mask=None, return_attention=False):
        batch_size, seq_len, _ = x.shape
        residual = x
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        _, _, rope_positions = self._frequency_coordinates(seq_len, x.device, q.dtype)
        q = _apply_rope(q, rope_positions, self.rope_base)
        k = _apply_rope(k, rope_positions, self.rope_base)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        scores = scores + self._pope_bias(seq_len, x.device, scores.dtype)

        query_mask = None
        if token_mask is not None:
            token_mask = token_mask.to(device=x.device, dtype=torch.bool)
            query_mask = token_mask.unsqueeze(1).unsqueeze(-1)
            key_mask = token_mask.unsqueeze(1).unsqueeze(2)
            empty_rows = token_mask.sum(dim=1) == 0
            if empty_rows.any():
                key_mask = key_mask.clone()
                key_mask[empty_rows, :, :, 0] = True
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        if query_mask is not None:
            attn = attn * query_mask.to(dtype=attn.dtype)
        attended = torch.matmul(attn, v)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.embed_dim,
        )
        x = residual + self.dropout(self.out_proj(attended))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        if token_mask is not None:
            x = x.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        if return_attention:
            return x, attn
        return x


class HarmonicPoPEEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        heads,
        depth,
        dropout,
        window_seconds,
        harmonic_ratios,
        harmonic_sigma,
        rope_base,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                HarmonicPoPEEncoderLayer(
                    embed_dim,
                    heads,
                    dropout,
                    window_seconds,
                    harmonic_ratios,
                    harmonic_sigma,
                    rope_base,
                )
                for _ in range(max(1, int(depth)))
            ]
        )
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, src, src_key_padding_mask=None, return_attentions=False):
        token_mask = None
        if src_key_padding_mask is not None:
            token_mask = ~src_key_padding_mask.to(device=src.device, dtype=torch.bool)
        x = src
        attentions = []
        for layer in self.layers:
            if return_attentions:
                x, attention = layer(
                    x,
                    token_mask=token_mask,
                    return_attention=True,
                )
                attentions.append(attention)
            else:
                x = layer(x, token_mask=token_mask)
        x = self.norm(x)
        if token_mask is not None:
            x = x.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        if return_attentions:
            return x, attentions
        return x


def _make_frequency_encoder(args, embed_dim, heads, depth, dropout, use_global_fft):
    if (
        use_global_fft
        and getattr(args, "freq_position_encoding", "additive_log_sincos")
        == "harmonic_pope"
    ):
        return HarmonicPoPEEncoder(
            embed_dim,
            heads,
            depth,
            dropout,
            float(args.window_seconds),
            _parse_harmonic_ratios(getattr(args, "freq_pope_harmonics", "2,3,4,5")),
            float(getattr(args, "freq_pope_sigma", 0.08)),
            float(getattr(args, "freq_rope_base", 10000.0)),
        )
    return _make_transformer_encoder(embed_dim, heads, depth, dropout)


def _masked_mean(tokens, token_mask):
    if token_mask is None:
        return tokens.mean(dim=1)
    weights = token_mask.to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _single_channel_args(args):
    branch_args = copy.copy(args)
    branch_args.num_channels = 1
    return branch_args


def _flatten_channel_samples(x, mask=None):
    if x.ndim != 3:
        raise ValueError("Expected input shape [B, C, L], got %s" % (tuple(x.shape),))
    batch_size, num_channels, seq_len = x.shape
    x = x.reshape(batch_size * num_channels, 1, seq_len)
    if mask is not None:
        mask = mask.to(device=x.device, dtype=torch.bool)
        mask = mask.repeat_interleave(num_channels, dim=0)
    return x, mask, batch_size, num_channels


def _mean_channel_features(features, batch_size, num_channels):
    features = features.view(batch_size, num_channels, -1)
    return features.mean(dim=1)


def _make_single_classifier(embed_dim, num_classes, dropout):
    if num_classes <= 0:
        return None
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, embed_dim),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(embed_dim, num_classes),
    )


def _make_dual_classifier(embed_dim, num_classes, dropout):
    if num_classes <= 0:
        return None
    return nn.Sequential(
        nn.LayerNorm(2 * embed_dim),
        nn.Linear(2 * embed_dim, embed_dim),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(embed_dim, num_classes),
    )


class LearnedGatedFusion(nn.Module):
    """Sample-wise learned fusion for time and frequency branch features."""

    def __init__(self, embed_dim, dropout):
        super().__init__()
        embed_dim = int(embed_dim)
        self.time_adapter = nn.Linear(embed_dim, embed_dim)
        self.freq_adapter = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(embed_dim, 2),
        )
        nn.init.eye_(self.time_adapter.weight)
        nn.init.zeros_(self.time_adapter.bias)
        nn.init.eye_(self.freq_adapter.weight)
        nn.init.zeros_(self.freq_adapter.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, time_feature, freq_feature, return_weights=False):
        if time_feature.shape != freq_feature.shape:
            raise ValueError(
                "Learned fusion expects matching branch features, got %s and %s"
                % (tuple(time_feature.shape), tuple(freq_feature.shape))
            )
        gate_input = torch.cat([time_feature, freq_feature], dim=1)
        weights = torch.softmax(self.gate(gate_input), dim=1)
        time_feature = self.time_adapter(time_feature)
        freq_feature = self.freq_adapter(freq_feature)
        fused = (
            weights[:, 0:1] * time_feature
            + weights[:, 1:2] * freq_feature
        )
        if return_weights:
            return fused, weights
        return fused


class TimeFrequencyFeatureMixin:
    def _encode_transformer(self, encoder, tokens, token_mask):
        if token_mask is None:
            return encoder(tokens)
        key_padding_mask = ~token_mask
        empty_rows = token_mask.sum(dim=1) == 0
        if empty_rows.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[empty_rows, 0] = False

        encoded = encoder(tokens, src_key_padding_mask=key_padding_mask)
        encoded = encoded.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        return encoded

    def encode_time_feature(self, time_x, time_mask=None):
        tokens, token_mask = self.time_tokenizer(time_x, time_mask=time_mask)
        encoded = self._encode_transformer(self.time_encoder, tokens, token_mask)
        return _masked_mean(encoded, token_mask)

    def encode_frequency_feature(self, freq_x, freq_mask=None):
        tokens, token_mask = self.freq_tokenizer(freq_x, freq_mask=freq_mask)
        encoded = self._encode_transformer(self.freq_encoder, tokens, token_mask)
        return _masked_mean(encoded, token_mask)

    def encode_patch_frequency_feature(self, time_x, time_mask=None):
        tokens, token_mask = self.freq_tokenizer(time_x, time_mask=time_mask)
        encoded = self._encode_transformer(self.freq_encoder, tokens, token_mask)
        return _masked_mean(encoded, token_mask)
