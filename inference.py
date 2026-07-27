import argparse
import os

import numpy as np
import torch

from model import UniFreqProcessor
from model import load_model


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract UniFreq features from a Hugging Face checkpoint."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--config-filename", default="config.json")
    parser.add_argument("--weights-filename", default="model.safetensors")
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--input-key",
        default="samples",
        help="Array or Parquet column containing signals.",
    )
    parser.add_argument(
        "--input-layout",
        choices=["L", "NL", "CL", "LC", "NCL", "NLC"],
        default="NL",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum samples to process. Use 0 for all samples.",
    )
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
    raise ValueError("Array input must be a .npy or .npz file")


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


def _array_sample_batches(args):
    array = _load_array(args.input, args.input_key)
    samples = _samples_from_layout(array, args.input_layout)
    if args.limit > 0:
        samples = samples[:args.limit]
    for start in range(0, len(samples), args.batch_size):
        yield samples[start:start + args.batch_size]


def _parquet_sample_batches(args):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for Parquet input. Install requirements.txt."
        ) from exc

    parquet = pq.ParquetFile(args.input)
    if args.input_key not in parquet.schema_arrow.names:
        raise KeyError(
            "%s has no Parquet column named %s"
            % (args.input, args.input_key)
        )

    processed = 0
    for record_batch in parquet.iter_batches(
        batch_size=args.batch_size,
        columns=[args.input_key],
    ):
        samples = record_batch.column(0).to_pylist()
        if args.limit > 0:
            remaining = args.limit - processed
            if remaining <= 0:
                break
            samples = samples[:remaining]
        if samples:
            yield samples
            processed += len(samples)


def _sample_batches(args):
    suffix = os.path.splitext(args.input)[1].lower()
    if suffix == ".parquet":
        yield from _parquet_sample_batches(args)
        return
    yield from _array_sample_batches(args)


def _select_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _to_device(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def run(args):
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if args.limit < 0:
        raise ValueError("limit must be >= 0")
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
    processor = UniFreqProcessor(config)

    outputs = {
        "features": [],
        "time_features": [],
        "frequency_features": [],
    }
    if model.head is not None:
        outputs["logits"] = []

    sample_count = 0
    with torch.inference_mode():
        for samples in _sample_batches(args):
            batch = processor(samples)
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
            sample_count += len(samples)

    if sample_count == 0:
        raise ValueError("Input contains no samples")
    arrays = {
        name: torch.cat(values, dim=0).numpy()
        for name, values in outputs.items()
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(
        "Saved %d samples to %s on %s"
        % (sample_count, args.output, device)
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
