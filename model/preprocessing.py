import torch
import torch.nn.functional as F


def _as_channel_first(sample):
    sample = torch.as_tensor(sample, dtype=torch.float32)
    if sample.ndim == 1:
        return sample.unsqueeze(0)
    if sample.ndim != 2:
        raise ValueError(
            "Each sample must have shape [L], [C, L], or [L, C], got %s"
            % (tuple(sample.shape),)
        )
    if sample.shape[0] >= sample.shape[1]:
        return sample.transpose(0, 1).contiguous()
    return sample.contiguous()


def _normalize(sample, mode, eps=1e-8):
    if mode == "none":
        return sample
    if mode == "sample_zscore":
        mean = sample.mean(dim=-1, keepdim=True)
        std = sample.std(dim=-1, keepdim=True, unbiased=False)
        return (sample - mean) / std.clamp_min(eps)
    if mode == "sample_minmax":
        minimum = sample.amin(dim=-1, keepdim=True)
        maximum = sample.amax(dim=-1, keepdim=True)
        return (sample - minimum) / (maximum - minimum).clamp_min(eps)
    raise ValueError("Unsupported normalization mode: %s" % mode)


def _resize(sample, mode, target_len):
    length = int(sample.shape[-1])
    target_len = int(target_len)
    if mode == "preserve" or length == target_len:
        return sample
    if mode == "truncate":
        if length < target_len:
            raise ValueError(
                "Cannot truncate length %d to longer length %d"
                % (length, target_len)
            )
        return sample[..., :target_len].contiguous()
    if mode == "resample_short_truncate_long" and length > target_len:
        return sample[..., :target_len].contiguous()
    return F.interpolate(
        sample.unsqueeze(0),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).squeeze(0).contiguous()


def _tail_statistics(spectrum, kept_len):
    total_len = int(spectrum.shape[-1])
    tail = spectrum[..., int(kept_len):]
    result = spectrum.new_zeros(spectrum.shape[:-1] + (5,))
    if total_len <= 0 or tail.shape[-1] <= 0:
        return result
    tail = tail.float()
    result[..., 0] = float(tail.shape[-1]) / float(total_len)
    result[..., 1] = tail.mean(dim=-1)
    result[..., 2] = tail.std(dim=-1, unbiased=False)
    result[..., 3] = tail.amax(dim=-1)
    result[..., 4] = torch.sqrt(tail.square().mean(dim=-1).clamp_min(0.0))
    return result


class UniFreqProcessor:
    def __init__(self, config):
        self.config = config

    def _prepare_sample(self, sample):
        raw = _as_channel_first(sample)
        if not torch.isfinite(raw).all():
            raise ValueError("Input contains NaN or infinity")
        raw = _resize(
            raw,
            self.config.input_length_mode,
            self.config.input_target_len,
        )
        if raw.shape[-1] < 2:
            raise ValueError("Each sample must contain at least two time points")

        if self.config.model_arch != "unifreq":
            expected = (self.config.seq_len if self.config.model_arch == "unifault"
                        else self.config.tf_profm_input_len)
            if raw.shape[-1] != int(expected):
                raise ValueError("Input length must match model input length %d" % expected)
            if (self.config.model_arch == "unifault"
                    and raw.shape[0] != int(self.config.num_channels)):
                raise ValueError("Input channels must match num_channels")
        time_x = _normalize(raw, self.config.time_data_normalize)
        result = {"time_x": time_x.contiguous()}
        if (self.config.model_arch != "unifreq"
                or self.config.freq_tokenizer != "global_fft"):
            return result

        freq_source = _normalize(
            raw,
            self.config.freq_fft_source_normalize,
        )
        freq_x = (
            torch.fft.rfft(freq_source.float(), dim=-1).abs()
            / float(freq_source.shape[-1])
        )
        freq_x = _normalize(freq_x, self.config.freq_data_normalize)
        result["freq_x"] = freq_x[..., 1:].contiguous()
        return result

    def __call__(self, samples):
        if torch.is_tensor(samples) and samples.ndim <= 2:
            samples = [samples]
        elif not isinstance(samples, (list, tuple)):
            samples = list(samples)
        if not samples:
            raise ValueError("samples must not be empty")

        prepared = [self._prepare_sample(sample) for sample in samples]
        channel_counts = {int(item["time_x"].shape[0]) for item in prepared}
        if len(channel_counts) != 1:
            raise ValueError("All samples in one batch must have equal channel counts")

        max_time_len = max(int(item["time_x"].shape[-1]) for item in prepared)
        time_values = []
        time_masks = []
        for item in prepared:
            value = item["time_x"]
            valid_len = int(value.shape[-1])
            time_values.append(F.pad(value, (0, max_time_len - valid_len)))
            mask = torch.zeros(max_time_len, dtype=torch.bool)
            mask[:valid_len] = True
            time_masks.append(mask)
        batch = {
            "time_x": torch.stack(time_values),
            "time_mask": torch.stack(time_masks),
        }

        if (self.config.model_arch != "unifreq"
                or self.config.freq_tokenizer != "global_fft"):
            return batch

        max_bins = int(self.config.freq_max_bins)
        if max_bins <= 0:
            max_bins = max(int(item["freq_x"].shape[-1]) for item in prepared)
        freq_values = []
        freq_masks = []
        tail_values = []
        for item in prepared:
            full_freq = item["freq_x"]
            kept_len = min(int(full_freq.shape[-1]), max_bins)
            tail_values.append(_tail_statistics(full_freq, kept_len))
            value = full_freq[..., :kept_len]
            freq_values.append(F.pad(value, (0, max_bins - kept_len)))
            mask = torch.zeros(max_bins, dtype=torch.bool)
            mask[:kept_len] = True
            freq_masks.append(mask)
        batch.update(
            {
                "freq_x": torch.stack(freq_values),
                "freq_mask": torch.stack(freq_masks),
                "freq_tail_stats": torch.stack(tail_values),
            }
        )
        return batch
