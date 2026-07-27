# PHMFD Inference

这是 PHMFD 的纯推理分支，用于从 Hugging Face Hub 加载已发布的 TimeFreq
权重并提取时域、频域及融合特征。

本分支不包含预训练、微调、优化器、损失函数、训练数据管线或实验 sweep
代码，也不提供任何参数更新入口。权重请在私有训练分支完成导出后上传到
Hugging Face。

## 安装

需要 Python 3.9 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Hugging Face 仓库格式

模型仓库至少需要两个文件：

```text
your-name/your-model
├── config.json
└── model.safetensors
```

`config.json` 描述推理结构和输入预处理。仓库中的
[`config.example.json`](config.example.json) 是当前默认 TimeFreq 模型的模板；
上传前请确认参数与导出权重完全一致。

`model.safetensors` 应使用模型的原始 state-dict 键名。加载器会兼容 DDP
导出的 `module.` 前缀，并忽略旧 checkpoint 中仅供训练使用的两个 projector；
其余键仍会严格校验缺失、额外项和张量形状。
这里有意只接受 `safetensors`，避免对 Hub 上的 pickle checkpoint 执行反序列化。

私有 Hugging Face 仓库可先设置：

```bash
export HF_TOKEN=hf_xxx
```

## 命令行推理

输入支持 `.npy` 和 `.npz`。下面的例子将 `[N, L]` 单通道时序转换为特征：

```bash
python inference.py \
  --model-id your-name/your-model \
  --input samples.npy \
  --input-layout NL \
  --output features.npz
```

如需固定 Hub revision：

```bash
python inference.py \
  --model-id your-name/your-model \
  --revision COMMIT_SHA \
  --input samples.npy \
  --input-layout NCL \
  --batch-size 32 \
  --device cuda \
  --output features.npz
```

`--model-id` 也可以是本地模型目录，便于上传前离线验证。

输入布局：

| 参数 | 输入形状 |
| --- | --- |
| `L` | 单个单通道样本 `[L]` |
| `NL` | 一批单通道样本 `[N, L]` |
| `CL` | 单个多通道样本 `[C, L]` |
| `LC` | 单个多通道样本 `[L, C]` |
| `NCL` | 一批多通道样本 `[N, C, L]` |
| `NLC` | 一批多通道样本 `[N, L, C]` |

输出 `.npz` 包含：

- `features`：时域与频域融合后的 backbone 特征；
- `time_features`：时域分支特征；
- `frequency_features`：频域分支特征；
- `logits`：仅当 Hub 权重包含分类头时写出。

## Python API

```python
import numpy as np
import torch

from model import PHMFDProcessor, load_model

model, config = load_model("your-name/your-model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

signals = [np.random.randn(2048), np.random.randn(1600)]
batch = PHMFDProcessor(config)(signals)
batch = {name: value.to(device) for name, value in batch.items()}

with torch.inference_mode():
    features = model(batch)
```

同一 batch 内样本可以变长，但通道数必须一致。处理器会按 `config.json`
完成归一化、FFT、非 DC 频点截取、tail statistics 和 padding，确保预处理与
发布权重一致。

## 本地验证

```bash
python -m py_compile inference.py model/*.py tests/*.py
python -m pytest -q
```

纯前向 smoke test 不需要下载数据集或访问 Hugging Face。
