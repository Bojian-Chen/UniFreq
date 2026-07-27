import argparse
import os

import numpy as np
import torch

from model import PHMFDProcessor
from model import load_model


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract PHMFD features from a Hugging Face checkpoint."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--config-filename", default="config.json")
    parser.add_argument("--weights-filename", default="model.safetensors")
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-key", default="x")
    parser.add_argument(
        "--input-layout",
        choices=["L", "NL", "CL", "LC", "NCL", "NLC"],
        default="NL",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="features.npz")
    return parser


def _load_array(path, key):
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                raise KeyError("%s has no array named %s" % (path, key))
            return archive[key]
    raise ValueError("Input must be a .npy or .npz file")


def _samples_from_layout(array, layout):
    array = np.asarray(array)
    expected_dims = {
        "L": 1,
        "NL": 2,
        "CL": 2,
        "LC": 2,
        "NCL": 3,
        "NLC": 3,
    }
    if array.ndim != expected_dims[layout]:
        raise ValueError(
            "Layout %s expects %d dimensions, got shape %s"
            % (layout, expected_dims[layout], tuple(array.shape))
        )
    if layout == "L":
        return [array]
    if layout == "NL":
        return [array[index] for index in range(array.shape[0])]
    if layout == "CL":
        return [array]
    if layout == "LC":
        return [array.T]
    if layout == "NCL":
        return [array[index] for index in range(array.shape[0])]
    return [array[index].T for index in range(array.shape[0])]


def _select_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _to_device(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def run(args):
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    array = _load_array(args.input, args.input_key)
    samples = _samples_from_layout(array, args.input_layout)
    model, config = load_model(
        args.model_id,
        revision=args.revision,
        token=args.token,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        config_filename=args.config_filename,
        weights_filename=args.weights_filename,
    )
    device = _select_device(args.device)
    model = model.to(device).eval()
    processor = PHMFDProcessor(config)

    outputs = {
        "features": [],
        "time_features": [],
        "frequency_features": [],
    }
    if model.head is not None:
        outputs["logits"] = []

    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            batch = processor(samples[start:start + args.batch_size])
            batch = _to_device(batch, device)
            time_features, frequency_features = model.branch_features(batch)
            if model.fusion_mode == "concat":
                features = torch.cat(
                    [time_features, frequency_features],
                    dim=1,
                )
            else:
                features = model.fusion(time_features, frequency_features)
            outputs["features"].append(features.cpu())
            outputs["time_features"].append(time_features.cpu())
            outputs["frequency_features"].append(frequency_features.cpu())
            if model.head is not None:
                outputs["logits"].append(model.predict(features).cpu())

    arrays = {
        name: torch.cat(values, dim=0).numpy()
        for name, values in outputs.items()
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(
        "Saved %d samples to %s on %s"
        % (len(samples), args.output, device)
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
