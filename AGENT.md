# AGENT.md

This branch is inference-only.

## Scope

- `inference.py`: command-line feature extraction.
- `model/config.py`: portable Hub configuration.
- `model/preprocessing.py`: deterministic inference preprocessing.
- `model/hub.py`: local directory and Hugging Face Hub loading.
- `model/timefreq.py`: forward-only TimeFreq model.
- `model/time_frequency_common.py` and `model/time_frequency_tokenizers.py`:
  inference components required by published checkpoints.

Do not add optimization, parameter updates, losses, data augmentation, dataset
splits, experiment sweeps, or training entry points to this branch.

Hub artifacts must use `config.json` plus `model.safetensors`. Do not add
pickle checkpoint loading.

## Validation

```bash
python -m py_compile inference.py model/*.py tests/*.py
python -m flake8 inference.py model tests --select=E9,F
python -m pytest -q
```
