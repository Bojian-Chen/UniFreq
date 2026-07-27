# UniFreq

[English](README.md)

UniFreq 是一个面向时序故障诊断的纯推理时频编码器，可以将变长、多通道时序
转换为时域、频域和融合特征。已发布的权重托管在 Hugging Face，并通过
`safetensors` 直接加载。

本仓库不包含训练循环、优化器、损失函数或微调流程。模型参数已冻结，并始终
保持在评估模式。

## 模型版本

| 模型 | 时域维度 | 频域维度 | 融合维度 | 注意力头 | 每分支层数 | 参数量 | 权重大小 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [UniFreq-tiny](https://huggingface.co/bojian1/UniFreq-tiny) | 64 | 64 | 128 | 2 | 4 | 42 万 | 1.69 MB |
| [UniFreq-small](https://huggingface.co/bojian1/UniFreq-small) | 96 | 96 | 192 | 3 | 4 | 94 万 | 3.78 MB |
| [UniFreq-base](https://huggingface.co/bojian1/UniFreq-base) | 128 | 128 | 256 | 4 | 4 | 167 万 | 6.69 MB |

三个版本均使用 0.1 秒输入窗口、800 个非直流频点、4 层时域 Transformer、
4 层频域 Transformer，以及谐波感知的频率位置编码。

## 安装

需要 Python 3.9 或更高版本。

```bash
git clone https://github.com/Bojian-Chen/UniFreq.git
cd UniFreq

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

使用 CUDA 推理时，建议先安装与本机 CUDA 版本匹配的 PyTorch，再安装其余依赖。

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

print(features.shape)  # UniFreq-base 输出 [2, 256]
```

同一个 batch 内的样本可以具有不同长度，但通道数必须一致。处理器会按照每个
checkpoint 的 `config.json` 完成归一化、FFT 缩放、直流分量移除、频点截取、
尾部统计、mask 和 padding。

## 使用 PHMFD 数据进行端到端推理

处理后数据的格式与目录结构见
[PHMFD-data-preprocessing](https://github.com/Bojian-Chen/PHMFD-data-preprocessing)，
数据文件托管在
[PHMFD-data](https://huggingface.co/datasets/bojian1/PHMFD-data)。
先下载一个处理后的 Parquet split：

```bash
hf download bojian1/PHMFD-data \
  --repo-type dataset \
  --include "Process_data/PU/train_1shot.parquet" \
  --local-dir data
```

直接对下载的数据运行 UniFreq-base。以下命令会从 Hugging Face 下载模型权重，
流式读取 Parquet 的 `samples` 列并保存特征：

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

在该示例中，`features`、`time_features` 和 `frequency_features` 的形状分别为
`[8, 256]`、`[8, 128]` 和 `[8, 128]`。移除 `--limit 8` 即可处理完整 split。

## 命令行推理

输入支持 `.parquet`、`.npy` 和 `.npz`：

```bash
python inference.py \
  --model-id bojian1/UniFreq-base \
  --input samples.npy \
  --input-layout NL \
  --batch-size 32 \
  --device auto \
  --output features.npz
```

对于 Parquet 文件，`--input-key` 用于选择信号列，默认值为 `samples`，数据会
按 batch 流式读取；对于 `.npz`，该参数用于选择其中保存的数组。`--model-id`
也可以指向包含 `config.json` 和 `model.safetensors` 的本地目录。

### 输入布局

| 布局 | 含义 | 形状 |
| --- | --- | --- |
| `L` | 一个单通道样本 | `[L]` |
| `NL` | 一批单通道样本 | `[N, L]` |
| `CL` | 一个通道优先样本 | `[C, L]` |
| `LC` | 一个通道置后样本 | `[L, C]` |
| `NCL` | 一批通道优先样本 | `[N, C, L]` |
| `NLC` | 一批通道置后样本 | `[N, L, C]` |

### 输出数组

生成的 `.npz` 包含：

| 数组 | 说明 |
| --- | --- |
| `features` | 时域与频域特征的拼接结果 |
| `time_features` | 时域分支特征 |
| `frequency_features` | 频域分支特征 |
| `logits` | checkpoint 包含兼容分类头时的可选分类结果 |

## 仓库结构

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

## Checkpoint 格式

每个 Hugging Face 模型仓库包含：

```text
config.json
model.safetensors
```

加载器只接受 `safetensors` 权重，有意不支持 pickle checkpoint。

## 许可证

UniFreq 使用 [MIT License](LICENSE)。
