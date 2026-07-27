import math

import torch
import torch.nn as nn

from model.time_frequency_common import _compatible_num_heads
from model.time_frequency_common import _sincos_from_values


class ContinuousTimeTokenizer(nn.Module):
    """Patch-by-physical-time tokenizer with selectable intra-patch attention."""

    def __init__(self, args):
        super().__init__()
        self.num_channels = int(args.num_channels)
        self.embed_dim = int(args.embed_dim)
        self.num_heads = int(args.heads)
        self.window_seconds = float(args.window_seconds)
        self.time_patch_seconds = float(args.time_patch_seconds)
        self.time_patch_attention = getattr(args, "time_patch_attention", "lipm_mhsa")

        if self.time_patch_attention not in ["lipm_mhsa", "mha"]:
            raise ValueError(
                "time_patch_attention must be lipm_mhsa or mha, got %s"
                % self.time_patch_attention
            )
        if self.num_heads <= 0:
            raise ValueError("heads must be > 0")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                "embed_dim (%d) must be divisible by heads (%d) for "
                "intra-patch attention"
                % (self.embed_dim, self.num_heads)
            )
        self.head_dim = self.embed_dim // self.num_heads
        if self.time_patch_attention == "lipm_mhsa" and self.head_dim % 2 != 0:
            raise ValueError(
                "Per-head q/k dimension (%d) must be even for IrregularRoPE"
                % self.head_dim
            )
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if self.time_patch_seconds <= 0:
            raise ValueError("time_patch_seconds must be > 0")

        self.num_time_tokens = max(
            1,
            int(math.ceil(self.window_seconds / self.time_patch_seconds - 1e-9)),
        )
        if self.time_patch_attention == "lipm_mhsa":
            self.emb_token = nn.Parameter(torch.zeros(1, 1, self.num_channels))
            self.q_proj = nn.Linear(self.num_channels, self.embed_dim, bias=False)
            self.k_proj = nn.Linear(self.num_channels, self.embed_dim, bias=False)
            self.v_proj = nn.Linear(self.num_channels, self.embed_dim, bias=False)
        else:
            self.point_proj = nn.Linear(self.num_channels, self.embed_dim, bias=False)
            self.attn_query = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.mha_q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            self.mha_k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            self.mha_v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            self.mha_out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            self.patch_attention_norm = nn.LayerNorm(self.embed_dim)
        self.attn_drop = nn.Dropout(p=float(args.dropout))
        self.pos_drop = nn.Dropout(p=float(args.dropout))

    def _apply_irregular_rope(self, x, time_values):
        if x.size(-1) % 2 != 0:
            raise ValueError("IrregularRoPE expects an even last dimension")
        theta = torch.exp(
            torch.arange(
                0,
                x.size(-1),
                2,
                device=x.device,
                dtype=x.dtype,
            )
            * -(math.log(1000.0) / x.size(-1))
        )
        phase = time_values.to(device=x.device, dtype=x.dtype)
        phase = phase.unsqueeze(1).unsqueeze(-1) * theta.view(1, 1, 1, -1)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.empty_like(x)
        rotated[..., 0::2] = x_even * cos_phase - x_odd * sin_phase
        rotated[..., 1::2] = x_even * sin_phase + x_odd * cos_phase
        return rotated

    def _lipm_mhsa_patch_tokens(self, segment_tensor, segment_valid, time_values):
        num_segments = segment_tensor.size(0)
        emb = self.emb_token.expand(num_segments, 1, self.num_channels)
        x_all = torch.cat([emb.to(dtype=segment_tensor.dtype), segment_tensor], dim=1)
        t_all = torch.cat(
            [
                time_values.new_zeros(num_segments, 1),
                time_values,
            ],
            dim=1,
        )
        valid_all = torch.cat(
            [
                torch.ones(
                    num_segments,
                    1,
                    device=segment_tensor.device,
                    dtype=torch.bool,
                ),
                segment_valid,
            ],
            dim=1,
        )
        seq_len = x_all.size(1)

        q = self.q_proj(x_all)
        k = self.k_proj(x_all)
        v = self.v_proj(x_all)
        q = q.view(num_segments, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(num_segments, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(num_segments, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self._apply_irregular_rope(q, t_all)
        k = self._apply_irregular_rope(k, t_all)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        key_mask = valid_all.unsqueeze(1).unsqueeze(2)
        query_mask = valid_all.unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        attn = attn * query_mask.to(dtype=attn.dtype)
        attended = torch.matmul(attn, v)
        attended = attended.transpose(1, 2).contiguous().view(
            num_segments,
            seq_len,
            self.embed_dim,
        )
        return attended[:, 0, :]

    def _mha_patch_tokens(self, segment_tensor, segment_valid, time_values):
        point_features = self.point_proj(segment_tensor)
        time_encoding = _sincos_from_values(
            time_values,
            self.embed_dim,
        ).view(segment_tensor.size(0), segment_tensor.size(1), self.embed_dim)
        point_features = point_features + time_encoding
        num_segments, seq_len, _ = point_features.shape
        query = self.attn_query.expand(num_segments, -1, -1)

        q = self.mha_q_proj(query)
        k = self.mha_k_proj(point_features)
        v = self.mha_v_proj(point_features)
        q = q.view(num_segments, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(num_segments, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(num_segments, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        key_mask = segment_valid.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        token_values = torch.matmul(attn, v)
        token_values = token_values.transpose(1, 2).contiguous().view(
            num_segments,
            1,
            self.embed_dim,
        )
        token_values = self.mha_out_proj(token_values)
        return self.patch_attention_norm(token_values.squeeze(1))

    def _window_plan(self, valid_lengths):
        plans = []
        for valid_len in valid_lengths:
            valid_len = max(1, int(valid_len))
            if valid_len < self.num_time_tokens:
                raise ValueError(
                    "valid time length (%d) is shorter than num_time_tokens (%d)"
                    % (valid_len, self.num_time_tokens)
                )
            windows = []
            dt = self.window_seconds / float(valid_len)
            for window_id in range(self.num_time_tokens):
                start = (window_id * valid_len) // self.num_time_tokens
                end = ((window_id + 1) * valid_len) // self.num_time_tokens
                windows.append((start, end, dt))
            plans.append(windows)
        return plans

    def forward(self, x, time_mask=None):
        if x.ndim != 3:
            raise ValueError("Expected input shape [B, C, L], got %s" % (tuple(x.shape),))
        batch_size, num_channels, seq_len = x.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Input channels is %d, but tokenizer was built for %d"
                % (num_channels, self.num_channels)
            )

        if time_mask is None:
            time_mask = torch.ones(
                batch_size,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            )
        else:
            time_mask = time_mask.to(device=x.device, dtype=torch.bool)

        valid_lengths = time_mask.sum(dim=1).detach().cpu().tolist()
        plans = self._window_plan(valid_lengths)

        num_segments = batch_size * self.num_time_tokens
        max_segment_len = 1
        for batch_id, windows in enumerate(plans):
            for start, end, _ in windows:
                length = int(end - start)
                max_segment_len = max(max_segment_len, length)

        segment_tensor = x.new_zeros(num_segments, max_segment_len, self.num_channels)
        segment_valid = torch.zeros(
            num_segments,
            max_segment_len,
            device=x.device,
            dtype=torch.bool,
        )
        time_values = x.new_zeros(num_segments, max_segment_len)
        for batch_id, windows in enumerate(plans):
            for window_id, (start, end, dt) in enumerate(windows):
                segment_id = batch_id * self.num_time_tokens + window_id
                length = int(end - start)
                segment_tensor[segment_id, :length, :] = x[
                    batch_id,
                    :,
                    start:end,
                ].transpose(0, 1)
                segment_valid[segment_id, :length] = time_mask[batch_id, start:end]
                local_time = (
                    torch.arange(length, device=x.device, dtype=x.dtype) * float(dt)
                )
                time_scale = max(float(dt) * float(max(length - 1, 1)), 1e-6)
                time_values[segment_id, :length] = local_time / time_scale

        if self.time_patch_attention == "lipm_mhsa":
            patch_token = self._lipm_mhsa_patch_tokens(
                segment_tensor,
                segment_valid,
                time_values,
            )
        else:
            patch_token = self._mha_patch_tokens(
                segment_tensor,
                segment_valid,
                time_values,
            )
        tokens = patch_token.view(batch_size, self.num_time_tokens, self.embed_dim)

        positions = torch.arange(
            self.num_time_tokens,
            device=x.device,
            dtype=tokens.dtype,
        )
        tokens = tokens + _sincos_from_values(positions, self.embed_dim).unsqueeze(0)
        tokens = self.pos_drop(tokens)
        return tokens, None


class FrequencyBinTokenizer(nn.Module):
    """One FFT bin per token for the global FFT frequency branch."""

    def __init__(self, args):
        super().__init__()
        self.num_channels = int(args.num_channels)
        self.embed_dim = int(args.embed_dim)
        self.window_seconds = float(args.window_seconds)
        self.position_encoding = getattr(
            args,
            "freq_position_encoding",
            "additive_log_sincos",
        )
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if self.position_encoding not in [
            "additive_log_sincos",
            "sequence_sincos",
            "none",
            "harmonic_pope",
        ]:
            raise ValueError(
                "freq_position_encoding must be additive_log_sincos, "
                "sequence_sincos, none, or harmonic_pope"
            )
        self.bin_proj = nn.Linear(self.num_channels, self.embed_dim, bias=False)
        self.pos_drop = nn.Dropout(p=float(args.dropout))

    def forward(self, x, freq_mask=None):
        if x.ndim != 3:
            raise ValueError("Expected input shape [B, C, F], got %s" % (tuple(x.shape),))
        batch_size, num_channels, freq_len = x.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Input channels is %d, but tokenizer was built for %d"
                % (num_channels, self.num_channels)
            )

        if freq_mask is None:
            freq_mask = torch.ones(
                batch_size,
                freq_len,
                device=x.device,
                dtype=torch.bool,
            )
        else:
            freq_mask = freq_mask.to(device=x.device, dtype=torch.bool)

        tokens = self.bin_proj(x.transpose(1, 2))
        if self.position_encoding == "additive_log_sincos":
            freqs = torch.arange(1, freq_len + 1, device=x.device, dtype=tokens.dtype)
            freqs = freqs / self.window_seconds
            log_freqs = torch.log(freqs.clamp_min(1e-12))
            denom = (log_freqs[-1] - log_freqs[0]).clamp_min(1e-6)
            values = (log_freqs - log_freqs[0]) / denom
            values = values * max(1, freq_len - 1)
            tokens = tokens + _sincos_from_values(values, self.embed_dim).unsqueeze(0)
        elif self.position_encoding == "sequence_sincos":
            positions = torch.arange(
                1,
                freq_len + 1,
                device=x.device,
                dtype=tokens.dtype,
            )
            tokens = tokens + _sincos_from_values(
                positions,
                self.embed_dim,
            ).unsqueeze(0)
        tokens = self.pos_drop(tokens)
        tokens = tokens.masked_fill(~freq_mask.unsqueeze(-1), 0.0)
        return tokens, freq_mask


class PatchFFTTokenizer(nn.Module):
    """Physical-time local FFT tokenizer for the experimental frequency branch."""

    def __init__(self, args):
        super().__init__()
        self.num_channels = int(args.num_channels)
        self.embed_dim = int(args.embed_dim)
        self.window_seconds = float(args.window_seconds)
        self.time_patch_seconds = float(args.time_patch_seconds)
        self.patch_fft_window = getattr(args, "patch_fft_window", "rect")
        self.num_queries = int(getattr(args, "patch_fft_num_queries", 1))

        if self.num_channels != 1:
            raise ValueError("PatchFFTTokenizer expects single-channel inputs")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if self.time_patch_seconds <= 0:
            raise ValueError("time_patch_seconds must be > 0")
        if self.patch_fft_window not in ["rect", "hann"]:
            raise ValueError("patch_fft_window must be 'rect' or 'hann'")
        if self.num_queries <= 0:
            raise ValueError("patch_fft_num_queries must be > 0")

        self.num_time_tokens = max(
            1,
            int(math.ceil(self.window_seconds / self.time_patch_seconds - 1e-9)),
        )
        self.bin_proj = nn.Linear(1, self.embed_dim)
        self.freq_queries = nn.Parameter(torch.zeros(1, self.num_queries, self.embed_dim))
        self.patch_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=_compatible_num_heads(self.embed_dim, int(args.heads)),
            dropout=float(args.dropout),
            batch_first=True,
        )
        self.patch_attention_norm = nn.LayerNorm(self.embed_dim)
        self.pos_drop = nn.Dropout(p=float(args.dropout))

    def _window_plan(self, valid_lengths):
        plans = []
        for valid_len in valid_lengths:
            valid_len = max(0, int(valid_len))
            windows = []
            for window_id in range(self.num_time_tokens):
                start = (window_id * valid_len) // self.num_time_tokens
                end = ((window_id + 1) * valid_len) // self.num_time_tokens
                windows.append((start, end))
            plans.append(windows)
        return plans

    def forward(self, time_x, time_mask=None):
        if time_x.ndim != 3:
            raise ValueError(
                "Expected time_x shape [B, C, L], got %s" % (tuple(time_x.shape),)
            )
        batch_size, num_channels, seq_len = time_x.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Input channels is %d, but PatchFFTTokenizer was built for %d"
                % (num_channels, self.num_channels)
            )

        if time_mask is None:
            time_mask = torch.ones(
                batch_size,
                seq_len,
                device=time_x.device,
                dtype=torch.bool,
            )
        else:
            time_mask = time_mask.to(device=time_x.device, dtype=torch.bool)

        valid_lengths = time_mask.sum(dim=1).detach().cpu().tolist()
        plans = self._window_plan(valid_lengths)
        num_segments = batch_size * self.num_time_tokens

        specs = []
        freqs_hz = []
        segment_ids = []
        patch_mask = torch.zeros(
            num_segments,
            device=time_x.device,
            dtype=torch.bool,
        )
        max_bins = 0
        for batch_id, windows in enumerate(plans):
            valid_len = max(0, int(valid_lengths[batch_id]))
            fs = float(valid_len) / self.window_seconds if valid_len > 0 else 0.0
            for window_id, (start, end) in enumerate(windows):
                segment_id = batch_id * self.num_time_tokens + window_id
                segment_len = max(0, int(end - start))
                if segment_len < 2 or fs <= 0:
                    continue

                segment = time_x[batch_id, 0, start:end].float()
                if self.patch_fft_window == "hann":
                    window = torch.hann_window(
                        segment_len,
                        periodic=True,
                        device=time_x.device,
                        dtype=segment.dtype,
                    )
                    segment = segment * window

                spec = torch.fft.rfft(segment, dim=-1).abs() / float(segment_len)
                spec = spec[1:]
                if spec.numel() == 0:
                    continue

                bin_indices = torch.arange(
                    1,
                    spec.numel() + 1,
                    device=time_x.device,
                    dtype=spec.dtype,
                )
                freq_hz = bin_indices * (fs / float(segment_len))

                specs.append(spec)
                freqs_hz.append(freq_hz)
                segment_ids.append(segment_id)
                patch_mask[segment_id] = True
                max_bins = max(max_bins, int(spec.numel()))

        tokens = time_x.new_zeros(batch_size, self.num_time_tokens, self.embed_dim)
        token_mask = patch_mask.view(batch_size, self.num_time_tokens)
        if max_bins <= 0:
            return tokens, token_mask

        local_mag = time_x.new_zeros(num_segments, max_bins)
        local_freq_hz = time_x.new_zeros(num_segments, max_bins)
        local_mask = torch.zeros(
            num_segments,
            max_bins,
            device=time_x.device,
            dtype=torch.bool,
        )
        for spec, freq_hz, segment_id in zip(specs, freqs_hz, segment_ids):
            num_bins = int(spec.numel())
            local_mag[segment_id, :num_bins] = spec.to(dtype=local_mag.dtype)
            local_freq_hz[segment_id, :num_bins] = freq_hz.to(dtype=local_freq_hz.dtype)
            local_mask[segment_id, :num_bins] = True

        local_tokens = self.bin_proj(local_mag.unsqueeze(-1))
        valid_rows = local_mask.sum(dim=1) > 0
        if valid_rows.any():
            nyquist = time_x.new_zeros(num_segments)
            valid_lengths_tensor = time_mask.sum(dim=1).to(dtype=time_x.dtype)
            sample_nyquist = valid_lengths_tensor / self.window_seconds * 0.5
            nyquist = sample_nyquist.repeat_interleave(self.num_time_tokens)
            denom = torch.log1p(nyquist).clamp_min(1e-6).unsqueeze(1)
            values = torch.log1p(local_freq_hz) / denom
            pos_embed = _sincos_from_values(
                values,
                self.embed_dim,
            ).view(num_segments, max_bins, self.embed_dim)
            local_tokens = local_tokens + pos_embed

        local_tokens = self.pos_drop(local_tokens)
        local_tokens = local_tokens.masked_fill(~local_mask.unsqueeze(-1), 0.0)

        valid_indices = torch.nonzero(valid_rows, as_tuple=False).flatten()
        query = self.freq_queries.expand(valid_indices.numel(), -1, -1)
        query_out, _ = self.patch_attention(
            query,
            local_tokens.index_select(0, valid_indices),
            local_tokens.index_select(0, valid_indices),
            key_padding_mask=~local_mask.index_select(0, valid_indices),
            need_weights=False,
        )
        query_out = self.patch_attention_norm(query_out)
        if self.num_queries == 1:
            patch_values = query_out.squeeze(1)
        else:
            patch_values = query_out.mean(dim=1)

        flat_tokens = time_x.new_zeros(num_segments, self.embed_dim)
        flat_tokens[valid_indices] = patch_values.to(dtype=flat_tokens.dtype)
        tokens = flat_tokens.view(batch_size, self.num_time_tokens, self.embed_dim)
        tokens = tokens.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        return tokens, token_mask


class FFTPatchConvTokenizer(nn.Module):
    """Global FFT tokenizer that encodes fixed-bin frequency patches with Conv1d."""

    def __init__(self, args):
        super().__init__()
        self.num_channels = int(args.num_channels)
        self.embed_dim = int(args.embed_dim)
        self.window_seconds = float(args.window_seconds)
        self.freq_max_bins = int(getattr(args, "freq_max_bins", 0))
        self.patch_size = int(getattr(args, "freq_patch_size", 16))
        self.position_encoding = str(
            getattr(args, "freq_position_encoding", "none")
        )
        self.include_log_position = bool(
            getattr(args, "freq_patch_include_log_position", True)
        )
        self.patch_projection = str(
            getattr(args, "freq_patch_projection", "conv")
        )

        if self.num_channels != 1:
            raise ValueError("FFTPatchConvTokenizer expects single-channel inputs")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if self.freq_max_bins < 0:
            raise ValueError("freq_max_bins must be >= 0")
        if self.patch_size <= 0:
            raise ValueError("freq_patch_size must be > 0")
        if self.position_encoding not in [
            "additive_log_sincos",
            "sequence_sincos",
            "none",
            "harmonic_pope",
        ]:
            raise ValueError("Unsupported freq_position_encoding")
        if self.patch_projection not in ["conv", "linear"]:
            raise ValueError("freq_patch_projection must be conv or linear")

        self.patch_input_channels = 2 if self.include_log_position else 1
        if self.patch_projection == "conv":
            self.patch_proj = nn.Conv1d(
                in_channels=self.patch_input_channels,
                out_channels=self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
        else:
            self.patch_proj = nn.Linear(
                self.patch_input_channels * self.patch_size,
                self.embed_dim,
            )
        self.pos_drop = nn.Dropout(p=float(args.dropout))

    def _prepare_time_mask(self, time_x, time_mask):
        batch_size, _, seq_len = time_x.shape
        if time_mask is None:
            return torch.ones(
                batch_size,
                seq_len,
                device=time_x.device,
                dtype=torch.bool,
            )
        return time_mask.to(device=time_x.device, dtype=torch.bool)

    def _frequency_grid(self, time_x, time_mask=None):
        if time_x.ndim != 3:
            raise ValueError(
                "Expected time_x shape [B, C, L], got %s" % (tuple(time_x.shape),)
            )
        batch_size, num_channels, _ = time_x.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Input channels is %d, but FFTPatchConvTokenizer was built for %d"
                % (num_channels, self.num_channels)
            )

        time_mask = self._prepare_time_mask(time_x, time_mask)
        valid_lengths = time_mask.sum(dim=1).to(dtype=torch.long)
        if self.freq_max_bins > 0:
            freq_len = self.freq_max_bins
        else:
            non_dc_bins = torch.div(valid_lengths, 2, rounding_mode="floor")
            freq_len = int(non_dc_bins.max().item()) if batch_size > 0 else 0

        padded_len = self.patch_size
        if freq_len > 0:
            padded_len = int(math.ceil(freq_len / self.patch_size)) * self.patch_size

        magnitudes = time_x.new_zeros(batch_size, padded_len)
        log_freq_pos = time_x.new_zeros(batch_size, padded_len)
        bin_mask = torch.zeros(
            batch_size,
            padded_len,
            device=time_x.device,
            dtype=torch.bool,
        )
        if freq_len <= 0:
            return magnitudes, log_freq_pos, bin_mask

        for batch_id, valid_len_tensor in enumerate(valid_lengths.detach().cpu().tolist()):
            valid_len = int(valid_len_tensor)
            if valid_len < 2:
                continue

            spec = (
                torch.fft.rfft(time_x[batch_id, 0, :valid_len].float(), dim=-1).abs()
                / float(valid_len)
            )
            spec = spec[1:]
            copy_len = min(int(spec.numel()), freq_len)
            if copy_len <= 0:
                continue

            magnitudes[batch_id, :copy_len] = spec[:copy_len].to(dtype=magnitudes.dtype)
            freq_hz = (
                torch.arange(
                    1,
                    copy_len + 1,
                    device=time_x.device,
                    dtype=log_freq_pos.dtype,
                )
                / self.window_seconds
            )
            nyquist_hz = max(float(valid_len) / self.window_seconds * 0.5, 1.0)
            denom = math.log1p(nyquist_hz)
            log_freq_pos[batch_id, :copy_len] = torch.log1p(freq_hz) / denom
            bin_mask[batch_id, :copy_len] = True

        return magnitudes, log_freq_pos, bin_mask

    def _patchify_grid(self, magnitudes, log_freq_pos, bin_mask):
        mag_patches = magnitudes.unfold(1, self.patch_size, self.patch_size).contiguous()
        pos_patches = log_freq_pos.unfold(1, self.patch_size, self.patch_size).contiguous()
        bin_patch_mask = bin_mask.unfold(1, self.patch_size, self.patch_size).contiguous()
        patch_mask = bin_patch_mask.any(dim=-1)
        return mag_patches, pos_patches, bin_patch_mask, patch_mask

    def spectrum_patches(self, time_x, time_mask=None):
        magnitudes, log_freq_pos, bin_mask = self._frequency_grid(time_x, time_mask)
        mag_patches, _, bin_patch_mask, patch_mask = self._patchify_grid(
            magnitudes,
            log_freq_pos,
            bin_mask,
        )
        return mag_patches, patch_mask, bin_patch_mask

    def forward(self, time_x, time_mask=None):
        magnitudes, log_freq_pos, bin_mask = self._frequency_grid(time_x, time_mask)
        _, _, _, patch_mask = self._patchify_grid(
            magnitudes,
            log_freq_pos,
            bin_mask,
        )
        if self.include_log_position:
            patch_input = torch.stack([magnitudes, log_freq_pos], dim=1)
        else:
            patch_input = magnitudes.unsqueeze(1)
        if self.patch_projection == "conv":
            tokens = self.patch_proj(patch_input).transpose(1, 2).contiguous()
        else:
            patches = patch_input.unfold(
                -1,
                self.patch_size,
                self.patch_size,
            )
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(
                patches.shape[0],
                patches.shape[1],
                self.patch_input_channels * self.patch_size,
            )
            tokens = self.patch_proj(patches)
        if self.position_encoding == "sequence_sincos":
            positions = torch.arange(
                1,
                tokens.shape[1] + 1,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            tokens = tokens + _sincos_from_values(
                positions,
                self.embed_dim,
            ).unsqueeze(0)
        tokens = self.pos_drop(tokens)
        tokens = tokens.masked_fill(~patch_mask.unsqueeze(-1), 0.0)
        return tokens, patch_mask
