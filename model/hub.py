import os

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from model.config import UniFreqConfig
from model.unifreq import UniFreqModel


def _resolve_file(
    model_id,
    filename,
    revision=None,
    token=None,
    cache_dir=None,
    local_files_only=False,
):
    if os.path.isdir(model_id):
        path = os.path.join(model_id, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError("Missing model file: %s" % path)
        return path
    return hf_hub_download(
        repo_id=model_id,
        filename=filename,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def _without_ddp_prefix(state_dict):
    if state_dict and all(name.startswith("module.") for name in state_dict):
        return {name[len("module."):]: value for name, value in state_dict.items()}
    return state_dict


def _inference_state_dict(state_dict):
    ignored_prefixes = ("time_projector.", "freq_projector.")
    return {
        name: value
        for name, value in state_dict.items()
        if not name.startswith(ignored_prefixes)
    }


def load_model(
    model_id,
    revision=None,
    token=None,
    cache_dir=None,
    local_files_only=False,
    config_filename="config.json",
    weights_filename="model.safetensors",
    strict=True,
):
    """Load a frozen UniFreq model from a local or Hugging Face directory."""
    config_path = _resolve_file(
        model_id,
        config_filename,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    weights_path = _resolve_file(
        model_id,
        weights_filename,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    config = UniFreqConfig.from_json_file(config_path)
    model = UniFreqModel(config)
    state_dict = _without_ddp_prefix(load_file(weights_path, device="cpu"))
    state_dict = _inference_state_dict(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=bool(strict))
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(
            "Loaded with missing keys %s and unexpected keys %s"
            % (incompatible.missing_keys, incompatible.unexpected_keys)
        )
    model.eval()
    return model, config
