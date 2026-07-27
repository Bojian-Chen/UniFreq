import json
import os
from types import SimpleNamespace


DEFAULT_CONFIG = {
    "model_type": "phmfd-timefreq",
    "architectures": ["TimeFrequencyBackbone"],
    "model_arch": "timefreq",
    "num_classes": 0,
    "num_channels": 1,
    "embed_dim": 128,
    "heads": 4,
    "depth": 4,
    "dropout": 0.1,
    "window_seconds": 0.1,
    "time_patch_seconds": 0.00625,
    "time_patch_attention": "mha",
    "time_transformer_layers": 4,
    "freq_tokenizer": "global_fft",
    "freq_max_bins": 800,
    "freq_tail_stats": True,
    "freq_transformer_layers": 4,
    "freq_position_encoding": "harmonic_pope",
    "freq_pope_harmonics": "2,3,4,5",
    "freq_pope_sigma": 0.08,
    "freq_rope_base": 1000.0,
    "freq_patch_size": 16,
    "freq_patch_include_log_position": True,
    "freq_patch_projection": "conv",
    "patch_fft_window": "rect",
    "patch_fft_num_queries": 1,
    "timefreq_fusion": "concat",
    "input_length_mode": "preserve",
    "input_target_len": 0,
    "time_data_normalize": "none",
    "freq_fft_source_normalize": "none",
    "freq_data_normalize": "sample_minmax",
}


class PHMFDConfig(SimpleNamespace):
    config_name = "config.json"

    def __init__(self, **values):
        merged = dict(DEFAULT_CONFIG)
        merged.update(values)
        super().__init__(**merged)
        self._validate()

    def _validate(self):
        if self.model_arch != "timefreq":
            raise ValueError("This branch only supports model_arch=timefreq")
        if int(self.num_classes) < 0:
            raise ValueError("num_classes must be >= 0")
        if int(self.embed_dim) <= 0:
            raise ValueError("embed_dim must be > 0")
        if int(self.heads) <= 0 or int(self.embed_dim) % int(self.heads) != 0:
            raise ValueError("embed_dim must be divisible by heads")
        if float(self.window_seconds) <= 0:
            raise ValueError("window_seconds must be > 0")
        if float(self.time_patch_seconds) <= 0:
            raise ValueError("time_patch_seconds must be > 0")
        if self.time_patch_attention not in ["mha", "lipm_mhsa"]:
            raise ValueError("Unsupported time_patch_attention")
        if self.freq_tokenizer not in [
            "global_fft",
            "patch_fft",
            "fft_patch_conv",
        ]:
            raise ValueError("Unsupported freq_tokenizer")
        if int(self.freq_max_bins) < 0:
            raise ValueError("freq_max_bins must be >= 0")
        if self.timefreq_fusion not in ["concat", "learned_gate"]:
            raise ValueError("Unsupported timefreq_fusion")
        if self.input_length_mode not in [
            "preserve",
            "resample",
            "truncate",
            "resample_short_truncate_long",
        ]:
            raise ValueError("Unsupported input_length_mode")
        if (
            self.input_length_mode != "preserve"
            and int(self.input_target_len) <= 0
        ):
            raise ValueError(
                "input_target_len must be > 0 when input length is transformed"
            )
        for name in [
            "time_data_normalize",
            "freq_fft_source_normalize",
            "freq_data_normalize",
        ]:
            if getattr(self, name) not in [
                "none",
                "sample_zscore",
                "sample_minmax",
            ]:
                raise ValueError(
                    "%s must be none, sample_zscore, or sample_minmax" % name
                )

    @classmethod
    def from_dict(cls, values):
        return cls(**dict(values))

    @classmethod
    def from_json_file(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("config.json must contain a JSON object")
        return cls.from_dict(values)

    def to_dict(self):
        return dict(vars(self))

    def save_pretrained(self, directory):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, self.config_name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                self.to_dict(),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        return path
