# UniFreq

[English](README.md)

UniFreq 是一个面向时序故障诊断的时频编码器，可以将变长、多通道时序转换为
时域、频域和融合特征。已发布的权重托管在 Hugging Face，并通过
`safetensors` 直接加载。

本仓库提供推理以及两种下游协议：监督分类器微调和无参数 KNN，但不包含任何
预训练代码。KNN 始终冻结已发布的 UniFreq 编码器；监督微调会增加分类器，
并默认同时更新编码器和分类器。

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

## 下游评估

处理后的 PHMFD Parquet 文件包含 `samples` 和 `labels` 列。UniFreq 支持以下
两种下游协议：

| 协议 | 训练参数 | 使用的数据划分 | 输出 |
| --- | --- | --- | --- |
| 分类器微调 | UniFreq 编码器与新增的 MLP 分类器 | train / validation / test | 最佳模型、指标、测试集预测 |
| KNN | 无 | reference / test | 指标、测试集预测 |

两种协议都会下载所选 UniFreq checkpoint、流式读取 Parquet 并使用融合特征。
标签可以是不连续整数；程序会在内部映射，并在输出时恢复原始标签。

### 分类器微调

分类器沿用 TPFFT 的下游设计：
`LayerNorm → Linear → GELU → Dropout → Linear`。程序按验证集 macro-F1 选择
最佳 epoch，以安全张量格式保存编码器和分类器，并在最终测试前重新加载最佳
权重。与 TPFFT 一样，编码器使用独立的学习率倍率：

```bash
python downstream.py finetune \
  --model-id bojian1/UniFreq-base \
  --train-file data/Process_data/IMS_FD/train.parquet \
  --val-file data/Process_data/IMS_FD/val.parquet \
  --test-file data/Process_data/IMS_FD/test.parquet \
  --epochs 100 \
  --batch-size 64 \
  --backbone-lr-scale 1.0 \
  --device auto \
  --output-dir outputs/ims_fd_finetune
```

设置 `--backbone-lr-scale 0` 可以冻结 UniFreq，仅训练新增分类器。

输出目录包含：

| 文件 | 说明 |
| --- | --- |
| `config.json` | UniFreq 编码器配置 |
| `model.safetensors` | 验证集表现最佳的微调后编码器权重 |
| `classifier.safetensors` | 验证集表现最佳的分类器权重 |
| `classifier_config.json` | 特征维度、原始类别标签和分类器结构 |
| `metrics.json` | 验证与测试集的准确率、macro-F1 和混淆矩阵 |
| `predictions.csv` | 每个测试样本的真实标签与预测标签 |

### KNN

KNN 会对 reference 和 query 特征进行 L2 归一化，计算余弦相似度，再执行无权重
多数投票。如果指定的 `k` 大于 reference 样本数，程序会自动缩小 `k`。

```bash
python downstream.py knn \
  --model-id bojian1/UniFreq-base \
  --reference-file data/Process_data/IMS_FD/train.parquet \
  --test-file data/Process_data/IMS_FD/test.parquet \
  --neighbors 5 \
  --device auto \
  --output-dir outputs/ims_fd_knn
```

该命令生成 `metrics.json` 和 `predictions.csv`。进行 K-shot 评估时，将对应的
处理后划分（例如 `train_5shot.parquet`）传给 `--reference-file` 或
`--train-file`。

### 端到端验证结果

两条路径均已在完整 IMS_FD 划分上使用 `UniFreq-tiny`、seed 42 和 CPU
完成集成验证。下表用于验证流程正确性，并非基准结果：

| 协议 | 训练 / reference | 验证 | 测试 | 测试准确率 | 测试 macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 分类器微调（`backbone-lr-scale=1`，最多 10 epochs） | 144 | 48 | 48 | 95.83% | 95.80% |
| KNN（`k=5`） | 144 | — | 48 | 100.00% | 100.00% |

验证还重新加载了保存的微调编码器和分类器，并成功得到形状为 `[1, 4]` 的有限
logits。

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
├── downstream.py
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
