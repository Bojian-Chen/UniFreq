# UniFreq

[中文说明](README_zh.md)

UniFreq is an inference-only time-frequency encoder for time-series fault
diagnosis. It converts variable-length, multi-channel signals into time-domain,
frequency-domain, and fused embeddings. Pretrained checkpoints are hosted on
Hugging Face and loaded directly with `safetensors`.

This repository contains no training loop, optimizer, loss function, or
fine-tuning pipeline. Model parameters are frozen and the model always remains
in evaluation mode.

## Model variants

| Model | Time dim | Frequency dim | Fused dim | Heads | Layers per branch | Parameters | Weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [UniFreq-tiny](https://huggingface.co/bojian1/UniFreq-tiny) | 64 | 64 | 128 | 2 | 4 | 420K | 1.69 MB |
| [UniFreq-small](https://huggingface.co/bojian1/UniFreq-small) | 96 | 96 | 192 | 3 | 4 | 941K | 3.78 MB |
| [UniFreq-base](https://huggingface.co/bojian1/UniFreq-base) | 128 | 128 | 256 | 4 | 4 | 1.67M | 6.69 MB |

All variants use a 0.1-second input window, 800 non-DC frequency bins, four
time-branch Transformer layers, four frequency-branch Transformer layers, and
harmonic-aware frequency positional encoding.

## Installation

Python 3.9 or newer is required.

```bash
git clone https://github.com/Bojian-Chen/UniFreq.git
cd UniFreq

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA inference, install the PyTorch build matching your CUDA version before
installing the remaining requirements.

## Python API

```python
import numpy as np
import torch

from model import UniFreqProcessor, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, config = load_model("bojian1/UniFreq-base")
model = model.to(device)

signals = [
    np.random.randn(2048).astype("float32"),
    np.random.randn(1600).astype("float32"),
]
batch = UniFreqProcessor(config)(signals)
batch = {name: value.to(device) for name, value in batch.items()}

with torch.inference_mode():
    features = model(batch)

print(features.shape)  # [2, 256] for UniFreq-base
```

Samples in one batch may have different lengths, but they must have the same
number of channels. Preprocessing follows each checkpoint's `config.json`,
including normalization, FFT scaling, DC removal, frequency truncation, tail
statistics, masks, and padding.

## End-to-end inference with PHMFD data

The processed datasets are documented in
[PHMFD-data-preprocessing](https://github.com/Bojian-Chen/PHMFD-data-preprocessing)
and hosted in the
[PHMFD-data](https://huggingface.co/datasets/bojian1/PHMFD-data) dataset
repository. Download one processed Parquet split:

```bash
hf download bojian1/PHMFD-data \
  --repo-type dataset \
  --include "Process_data/PU/train_1shot.parquet" \
  --local-dir data
```

Run UniFreq-base directly on the downloaded signals. The command downloads the
model checkpoint from Hugging Face, streams the `samples` column from Parquet,
and writes the extracted features:

```bash
python inference.py \
  --model-id bojian1/UniFreq-base \
  --input data/Process_data/PU/train_1shot.parquet \
  --input-key samples \
  --batch-size 8 \
  --limit 8 \
  --device auto \
  --output pu_unifreq_base_features.npz
```

For this example, `features`, `time_features`, and `frequency_features` have
shapes `[8, 256]`, `[8, 128]`, and `[8, 128]`, respectively. Remove
`--limit 8` to process the entire split.

## Command-line inference

Inputs may be `.parquet`, `.npy`, or `.npz` files.

```bash
python inference.py \
  --model-id bojian1/UniFreq-base \
  --input samples.npy \
  --input-layout NL \
  --batch-size 32 \
  --device auto \
  --output features.npz
```

For Parquet files, `--input-key` selects the signal column and defaults to
`samples`; records are streamed in batches. For `.npz` archives, it selects the
stored array. `--model-id` may also point to a local directory containing
`config.json` and `model.safetensors`.

### Input layouts

| Layout | Meaning | Shape |
| --- | --- | --- |
| `L` | One single-channel sample | `[L]` |
| `NL` | Batch of single-channel samples | `[N, L]` |
| `CL` | One channel-first sample | `[C, L]` |
| `LC` | One channel-last sample | `[L, C]` |
| `NCL` | Channel-first batch | `[N, C, L]` |
| `NLC` | Channel-last batch | `[N, L, C]` |

### Output arrays

The generated `.npz` archive contains:

| Array | Description |
| --- | --- |
| `features` | Concatenated time and frequency embeddings |
| `time_features` | Time-branch embeddings |
| `frequency_features` | Frequency-branch embeddings |
| `logits` | Optional classifier output when a compatible head is present |

## Repository layout

```text
UniFreq/
├── inference.py
├── config.example.json
├── model/
│   ├── config.py
│   ├── hub.py
│   ├── preprocessing.py
│   ├── unifreq.py
│   ├── time_frequency_common.py
│   └── time_frequency_tokenizers.py
└── requirements.txt
```

## Checkpoint format

Each Hugging Face model repository contains:

```text
config.json
model.safetensors
```

Only `safetensors` weights are accepted. Pickled checkpoints are intentionally
unsupported.

## License

UniFreq is released under the [MIT License](LICENSE).
