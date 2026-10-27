import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from safetensors.torch import save_file

from model import UniFreqProcessor
from model import load_model


class UniFreqClassifier(nn.Module):
    """MLP classifier used by the supervised downstream protocol."""

    def __init__(self, feature_dim, num_classes, dropout=0.1):
        super().__init__()
        hidden_dim = max(1, int(feature_dim) // 2)
        self.network = nn.Sequential(
            nn.LayerNorm(int(feature_dim)),
            nn.Linear(int(feature_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, int(num_classes)),
        )

    def forward(self, features):
        return self.network(features)


def _add_model_arguments(parser):
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--config-filename", default="config.json")
    parser.add_argument("--weights-filename", default="model.safetensors")
    parser.add_argument("--input-key", default="samples")
    parser.add_argument("--label-key", default="labels")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run supervised classifier fine-tuning or KNN with UniFreq."
    )
    subparsers = parser.add_subparsers(dest="method", required=True)

    finetune = subparsers.add_parser(
        "finetune",
        help="Add a classifier and fine-tune it with UniFreq.",
    )
    _add_model_arguments(finetune)
    finetune.add_argument("--train-file", required=True)
    finetune.add_argument("--val-file", required=True)
    finetune.add_argument("--test-file", required=True)
    finetune.add_argument("--epochs", type=int, default=100)
    finetune.add_argument("--batch-size", type=int, default=64)
    finetune.add_argument("--learning-rate", type=float, default=1e-3)
    finetune.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=1.0,
        help=(
            "Encoder learning-rate multiplier. Use 0 to train only the "
            "classifier."
        ),
    )
    finetune.add_argument("--weight-decay", type=float, default=1e-4)
    finetune.add_argument("--dropout", type=float, default=0.1)
    finetune.add_argument("--patience", type=int, default=20)
    finetune.add_argument("--limit-train", type=int, default=0)
    finetune.add_argument("--limit-val", type=int, default=0)
    finetune.add_argument("--limit-test", type=int, default=0)

    knn = subparsers.add_parser(
        "knn",
        help="Classify frozen UniFreq features with cosine KNN.",
    )
    _add_model_arguments(knn)
    knn.add_argument("--reference-file", required=True)
    knn.add_argument("--test-file", required=True)
    knn.add_argument("--neighbors", type=int, default=5)
    knn.add_argument("--limit-reference", type=int, default=0)
    knn.add_argument("--limit-test", type=int, default=0)
    return parser


def _select_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _to_device(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parquet_batches(
    path,
    input_key,
    label_key,
    batch_size,
    limit,
    shuffle=False,
    seed=0,
):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for downstream Parquet data."
        ) from exc

    parquet = pq.ParquetFile(path)
    missing = {
        input_key,
        label_key,
    }.difference(parquet.schema_arrow.names)
    if missing:
        raise KeyError(
            "%s is missing Parquet columns: %s"
            % (path, ", ".join(sorted(missing)))
        )

    processed = 0
    for record_batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=[input_key, label_key],
    ):
        samples = record_batch.column(0).to_pylist()
        labels = record_batch.column(1).to_pylist()
        if limit > 0:
            remaining = limit - processed
            if remaining <= 0:
                break
            samples = samples[:remaining]
            labels = labels[:remaining]
        if shuffle and len(samples) > 1:
            order = np.random.default_rng(seed + processed).permutation(
                len(samples)
            )
            samples = [samples[index] for index in order]
            labels = [labels[index] for index in order]
        if samples:
            yield samples, labels
            processed += len(samples)


def _as_integer_labels(labels, path):
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("%s labels must be scalar integers" % path)
    if not np.all(np.isfinite(values)):
        raise ValueError("%s labels contain NaN or infinity" % path)
    integer_values = values.astype(np.int64)
    if not np.array_equal(values, integer_values):
        raise ValueError("%s labels must be integers" % path)
    return torch.from_numpy(integer_values)


def _extract_features(
    model,
    processor,
    path,
    input_key,
    label_key,
    batch_size,
    limit,
    device,
):
    feature_batches = []
    label_batches = []
    with torch.inference_mode():
        for samples, labels in _parquet_batches(
            path,
            input_key,
            label_key,
            batch_size,
            limit,
        ):
            batch = _to_device(processor(samples), device)
            features = model(batch)
            if features.ndim != 2 or not torch.isfinite(features).all():
                raise ValueError("UniFreq produced invalid features for %s" % path)
            feature_batches.append(features.cpu())
            label_batches.append(_as_integer_labels(labels, path))
    if not feature_batches:
        raise ValueError("%s contains no samples" % path)
    features = torch.cat(feature_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    print(
        "Extracted %d x %d features from %s"
        % (features.shape[0], features.shape[1], path)
    )
    return features, labels


def _load_labels(path, label_key, batch_size, limit):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for downstream Parquet data."
        ) from exc

    parquet = pq.ParquetFile(path)
    if label_key not in parquet.schema_arrow.names:
        raise KeyError("%s has no Parquet column named %s" % (path, label_key))
    batches = []
    processed = 0
    for record_batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=[label_key],
    ):
        labels = record_batch.column(0).to_pylist()
        if limit > 0:
            remaining = limit - processed
            if remaining <= 0:
                break
            labels = labels[:remaining]
        if labels:
            batches.append(_as_integer_labels(labels, path))
            processed += len(labels)
    if not batches:
        raise ValueError("%s contains no labels" % path)
    return torch.cat(batches)


def _class_labels(reference_labels, *query_labels):
    classes = torch.unique(reference_labels, sorted=True)
    if classes.numel() < 2:
        raise ValueError("The reference/training split must contain at least 2 classes")
    class_set = set(classes.tolist())
    for labels in query_labels:
        unseen = sorted(set(labels.tolist()).difference(class_set))
        if unseen:
            raise ValueError(
                "Query split contains labels absent from training/reference: %s"
                % unseen
            )
    return classes


def _encode_labels(labels, classes):
    indices = torch.searchsorted(classes, labels)
    if not torch.equal(classes[indices], labels):
        raise ValueError("Could not map labels to contiguous class indices")
    return indices


def _metrics(targets, predictions, num_classes):
    targets = targets.to(dtype=torch.long, device="cpu")
    predictions = predictions.to(dtype=torch.long, device="cpu")
    confusion = torch.zeros(
        int(num_classes),
        int(num_classes),
        dtype=torch.long,
    )
    flat = targets * int(num_classes) + predictions
    confusion += torch.bincount(
        flat,
        minlength=int(num_classes) ** 2,
    ).view(int(num_classes), int(num_classes))
    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(dim=0).clamp_min(1).float()
    recall = true_positive / confusion.sum(dim=1).clamp_min(1).float()
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "accuracy": float((targets == predictions).float().mean().item()),
        "macro_f1": float(f1.mean().item()),
        "confusion_matrix": confusion.tolist(),
    }


def _run_supervised_epoch(
    model,
    classifier,
    processor,
    path,
    input_key,
    label_key,
    classes,
    batch_size,
    limit,
    device,
    optimizer=None,
    seed=0,
):
    is_train = optimizer is not None
    model.train(is_train)
    classifier.train(is_train)
    loss_sum = 0.0
    sample_count = 0
    predictions = []
    targets = []
    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for samples, labels in _parquet_batches(
            path,
            input_key,
            label_key,
            batch_size,
            limit,
            shuffle=is_train,
            seed=seed,
        ):
            batch = _to_device(processor(samples), device)
            batch_targets = _encode_labels(
                _as_integer_labels(labels, path),
                classes,
            ).to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = classifier(model(batch))
            loss = F.cross_entropy(logits, batch_targets)
            if is_train:
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.item()) * batch_targets.shape[0]
            sample_count += batch_targets.shape[0]
            predictions.append(logits.argmax(dim=1).detach().cpu())
            targets.append(batch_targets.cpu())
    if sample_count == 0:
        raise ValueError("%s contains no samples" % path)
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    epoch_metrics = _metrics(targets, predictions, classes.numel())
    epoch_metrics["loss"] = loss_sum / sample_count
    return epoch_metrics, predictions


def _write_predictions(path, encoded_targets, encoded_predictions, classes):
    targets = classes[encoded_targets].tolist()
    predictions = classes[encoded_predictions].tolist()
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "target", "prediction", "correct"])
        for index, (target, prediction) in enumerate(zip(targets, predictions)):
            writer.writerow([index, target, prediction, int(target == prediction)])


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_encoder(args, device):
    model, config = load_model(
        args.model_id,
        revision=args.revision,
        token=args.token,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        config_filename=args.config_filename,
        weights_filename=args.weights_filename,
    )
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, UniFreqProcessor(config), config


def _validate_common_args(args):
    if args.feature_batch_size <= 0:
        raise ValueError("feature-batch-size must be > 0")
    for name, value in vars(args).items():
        if name.startswith("limit_") and value < 0:
            raise ValueError("%s must be >= 0" % name.replace("_", "-"))


def run_finetune(args):
    _validate_common_args(args)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be > 0")
    if (
        args.learning_rate <= 0
        or args.weight_decay < 0
        or args.backbone_lr_scale < 0
    ):
        raise ValueError(
            "learning-rate must be > 0; weight-decay and "
            "backbone-lr-scale must be >= 0"
        )
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if args.patience < 0:
        raise ValueError("patience must be >= 0")

    _set_seed(args.seed)
    device = _select_device(args.device)
    model, processor, config = _load_encoder(args, device)
    train_labels = _load_labels(
        args.train_file,
        args.label_key,
        args.feature_batch_size,
        args.limit_train,
    )
    val_labels = _load_labels(
        args.val_file,
        args.label_key,
        args.feature_batch_size,
        args.limit_val,
    )
    test_labels = _load_labels(
        args.test_file,
        args.label_key,
        args.feature_batch_size,
        args.limit_test,
    )
    classes = _class_labels(train_labels, val_labels, test_labels)
    feature_dim = int(config.embed_dim)
    if config.model_arch == "TF-ProFM":
        feature_dim = int(config.tf_profm_feature_dim)
    elif config.model_arch == "unifreq" and config.timefreq_fusion == "concat":
        feature_dim *= 2
    classifier = UniFreqClassifier(
        feature_dim,
        classes.numel(),
        args.dropout,
    ).to(device)
    parameter_groups = [
        {
            "params": classifier.parameters(),
            "lr": args.learning_rate,
        }
    ]
    if args.backbone_lr_scale > 0:
        model.enable_finetuning()
        parameter_groups.insert(
            0,
            {
                "params": model.parameters(),
                "lr": args.learning_rate * args.backbone_lr_scale,
            },
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=args.weight_decay,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    classifier_path = os.path.join(args.output_dir, "classifier.safetensors")
    model_path = os.path.join(args.output_dir, "model.safetensors")
    best_epoch = 0
    best_macro_f1 = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = _run_supervised_epoch(
            model,
            classifier,
            processor,
            args.train_file,
            args.input_key,
            args.label_key,
            classes,
            args.batch_size,
            args.limit_train,
            device,
            optimizer=optimizer,
            seed=args.seed + epoch,
        )
        val_metrics, _ = _run_supervised_epoch(
            model,
            classifier,
            processor,
            args.val_file,
            args.input_key,
            args.label_key,
            classes,
            args.feature_batch_size,
            args.limit_val,
            device,
        )
        print(
            "Epoch %d: train_loss=%.6f val_accuracy=%.4f val_macro_f1=%.4f"
            % (
                epoch,
                train_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
            )
        )
        if val_metrics["macro_f1"] > best_macro_f1:
            best_epoch = epoch
            best_macro_f1 = val_metrics["macro_f1"]
            stale_epochs = 0
            state = {
                name: value.detach().cpu().contiguous()
                for name, value in classifier.state_dict().items()
            }
            save_file(state, classifier_path)
            model_state = {
                name: value.detach().cpu().contiguous()
                for name, value in model.state_dict().items()
            }
            save_file(model_state, model_path)
        else:
            stale_epochs += 1
            if args.patience > 0 and stale_epochs >= args.patience:
                print("Early stopping at epoch %d" % epoch)
                break

    model.load_state_dict(load_file(model_path, device="cpu"))
    classifier.load_state_dict(load_file(classifier_path, device="cpu"))
    model = model.to(device)
    classifier = classifier.to(device)
    model.disable_finetuning()
    val_metrics, _ = _run_supervised_epoch(
        model,
        classifier,
        processor,
        args.val_file,
        args.input_key,
        args.label_key,
        classes,
        args.feature_batch_size,
        args.limit_val,
        device,
    )
    test_metrics, test_predictions = _run_supervised_epoch(
        model,
        classifier,
        processor,
        args.test_file,
        args.input_key,
        args.label_key,
        classes,
        args.feature_batch_size,
        args.limit_test,
        device,
    )
    config.save_pretrained(args.output_dir)
    classifier_config = {
        "model_id": args.model_id,
        "feature_dim": feature_dim,
        "num_classes": int(classes.numel()),
        "class_labels": classes.tolist(),
        "dropout": float(args.dropout),
        "backbone_lr_scale": float(args.backbone_lr_scale),
        "architecture": "LayerNorm-Linear-GELU-Dropout-Linear",
    }
    results = {
        "method": "classifier_finetune",
        "model_id": args.model_id,
        "encoder_frozen": args.backbone_lr_scale == 0,
        "backbone_lr_scale": float(args.backbone_lr_scale),
        "best_epoch": best_epoch,
        "samples": {
            "train": int(train_labels.shape[0]),
            "validation": int(val_labels.shape[0]),
            "test": int(test_labels.shape[0]),
        },
        "validation": val_metrics,
        "test": test_metrics,
    }
    _write_json(
        os.path.join(args.output_dir, "classifier_config.json"),
        classifier_config,
    )
    _write_json(os.path.join(args.output_dir, "metrics.json"), results)
    _write_predictions(
        os.path.join(args.output_dir, "predictions.csv"),
        _encode_labels(test_labels, classes),
        test_predictions,
        classes,
    )
    print(
        "Best epoch %d; test_accuracy=%.4f test_macro_f1=%.4f"
        % (best_epoch, test_metrics["accuracy"], test_metrics["macro_f1"])
    )
    return results


def run_knn(args):
    _validate_common_args(args)
    if args.neighbors <= 0:
        raise ValueError("neighbors must be > 0")

    _set_seed(args.seed)
    device = _select_device(args.device)
    model, processor, _ = _load_encoder(args, device)
    reference_features, reference_labels = _extract_features(
        model,
        processor,
        args.reference_file,
        args.input_key,
        args.label_key,
        args.feature_batch_size,
        args.limit_reference,
        device,
    )
    test_features, test_labels = _extract_features(
        model,
        processor,
        args.test_file,
        args.input_key,
        args.label_key,
        args.feature_batch_size,
        args.limit_test,
        device,
    )
    classes = _class_labels(reference_labels, test_labels)
    reference_targets = _encode_labels(reference_labels, classes).to(device)
    test_targets = _encode_labels(test_labels, classes)
    effective_neighbors = min(args.neighbors, reference_features.shape[0])
    reference_features = F.normalize(reference_features.to(device), dim=1)
    test_features = F.normalize(test_features.to(device), dim=1)
    predictions = []
    with torch.inference_mode():
        for start in range(0, test_features.shape[0], args.feature_batch_size):
            query = test_features[start:start + args.feature_batch_size]
            similarity = query @ reference_features.T
            indices = similarity.topk(
                effective_neighbors,
                dim=1,
                largest=True,
                sorted=True,
            ).indices
            neighbor_targets = reference_targets[indices]
            votes = torch.zeros(
                query.shape[0],
                classes.numel(),
                device=device,
                dtype=torch.long,
            )
            votes.scatter_add_(
                1,
                neighbor_targets,
                torch.ones_like(neighbor_targets),
            )
            predictions.append(votes.argmax(dim=1).cpu())
    predictions = torch.cat(predictions)
    test_metrics = _metrics(test_targets, predictions, classes.numel())
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "method": "knn",
        "model_id": args.model_id,
        "distance": "cosine",
        "neighbors": int(effective_neighbors),
        "requested_neighbors": int(args.neighbors),
        "class_labels": classes.tolist(),
        "samples": {
            "reference": int(reference_features.shape[0]),
            "test": int(test_features.shape[0]),
        },
        "test": test_metrics,
    }
    _write_json(os.path.join(args.output_dir, "metrics.json"), results)
    _write_predictions(
        os.path.join(args.output_dir, "predictions.csv"),
        test_targets,
        predictions,
        classes,
    )
    print(
        "KNN test_accuracy=%.4f test_macro_f1=%.4f (k=%d)"
        % (
            test_metrics["accuracy"],
            test_metrics["macro_f1"],
            effective_neighbors,
        )
    )
    return results


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.method == "finetune":
        run_finetune(args)
    else:
        run_knn(args)


if __name__ == "__main__":
    main()
