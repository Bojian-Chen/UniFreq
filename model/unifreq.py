import torch
import torch.nn as nn

from model.time_frequency_common import HarmonicPoPEEncoder
from model.time_frequency_common import LearnedGatedFusion
from model.time_frequency_common import TimeFrequencyFeatureMixin
from model.time_frequency_common import _flatten_channel_samples
from model.time_frequency_common import _make_frequency_encoder
from model.time_frequency_common import _make_dual_classifier
from model.time_frequency_common import _make_single_classifier
from model.time_frequency_common import _make_transformer_encoder
from model.time_frequency_common import _mean_channel_features
from model.time_frequency_common import _single_channel_args
from model.time_frequency_tokenizers import ContinuousTimeTokenizer
from model.time_frequency_tokenizers import FFTPatchConvTokenizer
from model.time_frequency_tokenizers import FrequencyBinTokenizer
from model.time_frequency_tokenizers import PatchFFTTokenizer


class UniFreqModel(nn.Module, TimeFrequencyFeatureMixin):
    def __init__(self, args):
        super().__init__()
        self._finetuning_enabled = False
        self.args = args
        self.embed_dim = int(args.embed_dim)
        self.num_classes = int(args.num_classes)
        self.fusion_mode = str(
            getattr(args, "timefreq_fusion", "concat")
        )
        self.freq_tokenizer_name = getattr(args, "freq_tokenizer", "global_fft")
        self.use_patch_fft = self.freq_tokenizer_name == "patch_fft"
        self.use_fft_patch_conv = self.freq_tokenizer_name == "fft_patch_conv"
        self.use_model_fft = self.use_patch_fft or self.use_fft_patch_conv
        self.use_freq_tail_stats = (
            (not self.use_model_fft)
            and bool(getattr(args, "freq_tail_stats", True))
        )

        branch_args = _single_channel_args(args)
        self.time_tokenizer = ContinuousTimeTokenizer(branch_args)
        if self.use_patch_fft:
            self.freq_tokenizer = PatchFFTTokenizer(branch_args)
        elif self.use_fft_patch_conv:
            self.freq_tokenizer = FFTPatchConvTokenizer(branch_args)
        else:
            self.freq_tokenizer = FrequencyBinTokenizer(branch_args)
        time_layers = int(getattr(args, "time_transformer_layers", args.depth))
        freq_layers = int(getattr(args, "freq_transformer_layers", args.depth))
        self.time_encoder = _make_transformer_encoder(
            self.embed_dim,
            int(args.heads),
            time_layers,
            float(args.dropout),
        )
        self.freq_encoder = _make_frequency_encoder(
            args,
            self.embed_dim,
            int(args.heads),
            freq_layers,
            float(args.dropout),
            not self.use_model_fft,
        )
        if self.use_freq_tail_stats:
            self.freq_tail_encoder = nn.Sequential(
                nn.LayerNorm(5),
                nn.Linear(5, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
            )
            self.freq_tail_scale = nn.Parameter(torch.zeros(1))
        else:
            self.freq_tail_encoder = None
            self.register_parameter("freq_tail_scale", None)
        if self.fusion_mode == "concat":
            self.fusion = None
            self.head = _make_dual_classifier(
                self.embed_dim,
                self.num_classes,
                float(args.dropout),
            )
        elif self.fusion_mode == "learned_gate":
            self.fusion = (
                LearnedGatedFusion(self.embed_dim, float(args.dropout))
                if self.num_classes > 0
                else None
            )
            self.head = _make_single_classifier(
                self.embed_dim,
                self.num_classes,
                float(args.dropout),
            )
        else:
            raise ValueError(
                "Unsupported timefreq_fusion: %s" % self.fusion_mode
            )
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode=True):
        """Remain in evaluation mode unless downstream fine-tuning is enabled."""
        return super().train(bool(mode) if self._finetuning_enabled else False)

    def enable_finetuning(self):
        """Enable gradients and training mode for downstream supervised tuning."""
        self._finetuning_enabled = True
        self.requires_grad_(True)
        return super().train(True)

    def disable_finetuning(self):
        """Restore the frozen inference behavior used by feature extraction."""
        self._finetuning_enabled = False
        self.requires_grad_(False)
        return super().train(False)

    def _unpack(self, batch):
        if not isinstance(batch, dict):
            raise TypeError("timefreq expects a dict batch from pad_timefreq_collate")
        if self.use_model_fft:
            if "time_x" not in batch:
                raise KeyError(
                    "timefreq with model-side FFT tokenizer expects a batch "
                    "containing time_x/time_mask"
                )
            return (
                batch["time_x"],
                batch.get("time_mask"),
                None,
                None,
                None,
            )
        return (
            batch["time_x"],
            batch.get("time_mask"),
            batch["freq_x"],
            batch.get("freq_mask"),
            batch.get("freq_tail_stats"),
        )

    def _flatten_freq_tail_stats(self, freq_tail_stats, batch_size, num_channels,
                                 device, dtype):
        num_flat = int(batch_size) * int(num_channels)
        if freq_tail_stats is None:
            return torch.zeros(num_flat, 5, device=device, dtype=dtype)
        freq_tail_stats = freq_tail_stats.to(device=device, dtype=dtype)
        if freq_tail_stats.ndim == 3:
            return freq_tail_stats.reshape(-1, freq_tail_stats.size(-1))
        if freq_tail_stats.ndim == 2:
            if int(freq_tail_stats.size(0)) == num_flat:
                return freq_tail_stats
            if int(freq_tail_stats.size(0)) == int(batch_size) and int(num_channels) == 1:
                return freq_tail_stats
        raise ValueError(
            "freq_tail_stats shape %s is incompatible with batch=%d channels=%d"
            % (tuple(freq_tail_stats.shape), int(batch_size), int(num_channels))
        )

    def _apply_freq_tail_feature(self, freq_feature, freq_tail_stats):
        if not self.use_freq_tail_stats:
            return freq_feature
        if freq_tail_stats is None:
            freq_tail_stats = freq_feature.new_zeros(freq_feature.size(0), 5)
        else:
            freq_tail_stats = freq_tail_stats.to(
                device=freq_feature.device,
                dtype=freq_feature.dtype,
            )
        tail_feature = self.freq_tail_encoder(freq_tail_stats)
        tail_mask = (freq_tail_stats[:, 0:1] > 0).to(dtype=tail_feature.dtype)
        tail_gate = torch.tanh(self.freq_tail_scale).to(dtype=tail_feature.dtype)
        return freq_feature + tail_gate * tail_mask * tail_feature

    def branch_features(self, batch):
        time_x, time_mask, freq_x, freq_mask, freq_tail_stats = self._unpack(batch)
        time_x, time_mask, batch_size, num_channels = _flatten_channel_samples(
            time_x,
            time_mask,
        )
        time_feature = self.encode_time_feature(time_x, time_mask=time_mask)
        if self.use_model_fft:
            freq_feature = self.encode_patch_frequency_feature(time_x, time_mask=time_mask)
            freq_batch_size = batch_size
            freq_num_channels = num_channels
        else:
            freq_x, freq_mask, freq_batch_size, freq_num_channels = _flatten_channel_samples(
                freq_x,
                freq_mask,
            )
            freq_feature = self.encode_frequency_feature(freq_x, freq_mask=freq_mask)
            freq_tail_stats = self._flatten_freq_tail_stats(
                freq_tail_stats,
                freq_batch_size,
                freq_num_channels,
                freq_feature.device,
                freq_feature.dtype,
            )
            freq_feature = self._apply_freq_tail_feature(
                freq_feature,
                freq_tail_stats,
            )
        time_feature = _mean_channel_features(time_feature, batch_size, num_channels)
        freq_feature = _mean_channel_features(
            freq_feature,
            freq_batch_size,
            freq_num_channels,
        )
        return time_feature, freq_feature

    def frequency_features(self, batch):
        """Encode only the frequency branch and average original channels."""
        time_x, time_mask, freq_x, freq_mask, freq_tail_stats = self._unpack(batch)
        if self.use_model_fft:
            time_x, time_mask, batch_size, num_channels = _flatten_channel_samples(
                time_x,
                time_mask,
            )
            freq_feature = self.encode_patch_frequency_feature(
                time_x,
                time_mask=time_mask,
            )
        else:
            freq_x, freq_mask, batch_size, num_channels = _flatten_channel_samples(
                freq_x,
                freq_mask,
            )
            freq_feature = self.encode_frequency_feature(
                freq_x,
                freq_mask=freq_mask,
            )
            freq_tail_stats = self._flatten_freq_tail_stats(
                freq_tail_stats,
                batch_size,
                num_channels,
                freq_feature.device,
                freq_feature.dtype,
            )
            freq_feature = self._apply_freq_tail_feature(
                freq_feature,
                freq_tail_stats,
            )
        return _mean_channel_features(freq_feature, batch_size, num_channels)

    def frequency_features_with_attentions(self, batch):
        """Encode global-FFT features and return per-layer PoPE attentions."""
        if self.use_model_fft:
            raise ValueError(
                "Frequency attention export requires freq_tokenizer=global_fft"
            )
        if not isinstance(self.freq_encoder, HarmonicPoPEEncoder):
            raise ValueError(
                "Frequency attention export requires "
                "freq_position_encoding=harmonic_pope"
            )

        _, _, freq_x, freq_mask, freq_tail_stats = self._unpack(batch)
        freq_x, freq_mask, batch_size, num_channels = _flatten_channel_samples(
            freq_x,
            freq_mask,
        )
        tokens, token_mask = self.freq_tokenizer(freq_x, freq_mask=freq_mask)
        key_padding_mask = None if token_mask is None else ~token_mask
        encoded, attentions = self.freq_encoder(
            tokens,
            src_key_padding_mask=key_padding_mask,
            return_attentions=True,
        )
        if token_mask is None:
            freq_feature = encoded.mean(dim=1)
        else:
            weights = token_mask.to(dtype=encoded.dtype).unsqueeze(-1)
            freq_feature = (
                (encoded * weights).sum(dim=1)
                / weights.sum(dim=1).clamp_min(1.0)
            )
        freq_tail_stats = self._flatten_freq_tail_stats(
            freq_tail_stats,
            batch_size,
            num_channels,
            freq_feature.device,
            freq_feature.dtype,
        )
        freq_feature = self._apply_freq_tail_feature(
            freq_feature,
            freq_tail_stats,
        )
        freq_feature = _mean_channel_features(
            freq_feature,
            batch_size,
            num_channels,
        )
        return freq_feature, attentions, token_mask, batch_size, num_channels

    def forward(self, batch):
        time_feature, freq_feature = self.branch_features(batch)
        if self.fusion_mode == "concat":
            return torch.cat([time_feature, freq_feature], dim=1)
        if self.fusion is None:
            raise RuntimeError(
                "learned_gate fusion requires a supervised model with "
                "num_classes > 0"
            )
        return self.fusion(time_feature, freq_feature)

    def fusion_weights(self, batch):
        if self.fusion_mode != "learned_gate" or self.fusion is None:
            raise ValueError(
                "fusion_weights requires timefreq_fusion=learned_gate "
                "and num_classes > 0"
            )
        time_feature, freq_feature = self.branch_features(batch)
        _, weights = self.fusion(
            time_feature,
            freq_feature,
            return_weights=True,
        )
        return weights

    def predict(self, features):
        if self.head is None:
            return features
        return self.head(features)

    def classify(self, batch):
        return self.predict(self.forward(batch))
