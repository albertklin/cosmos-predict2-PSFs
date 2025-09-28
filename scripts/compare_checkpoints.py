"""Utility for comparing the tensor contents of two PyTorch checkpoints.

This script is designed for inspecting video2world LoRA checkpoints, but it
works with any PyTorch checkpoint that stores a state dict (either directly or
nested under common keys such as ``state_dict`` or ``model``).

Example usage::

    python scripts/compare_checkpoints.py /path/to/base.pt /path/to/lora.pt

The script prints a high level summary of the tensors contained in each
checkpoint along with any key, shape, or dtype mismatches between them. Use the
``--prefix`` flag to restrict the comparison to keys that start with a given
prefix (e.g. ``net.`` or ``net_ema.``).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import Dict, Iterable, Tuple

import torch

TensorInfo = Tuple[Tuple[int, ...], torch.dtype]


def _extract_state_dict(obj) -> Dict[str, torch.Tensor]:
    """Best-effort extraction of a tensor-only state dict from ``obj``.

    The function handles common checkpoint layouts produced by PyTorch's
    ``torch.save``. It first checks if ``obj`` itself looks like a state dict and
    otherwise searches a few well-known keys (``state_dict``, ``model``,
    ``module``, ``ema``).
    """

    if isinstance(obj, dict):
        tensor_items = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
        if tensor_items and len(tensor_items) == len(obj):
            return obj  # Already a pure state dict.

        for key in ("state_dict", "model", "module", "ema", "net"):
            if key in obj and isinstance(obj[key], dict):
                maybe_state = _extract_state_dict(obj[key])
                if maybe_state:
                    return maybe_state

    raise ValueError(
        "Could not locate a tensor state dict in the checkpoint. "
        "Inspect the file manually and update this script if necessary."
    )


def _load_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - informational message only.
        raise RuntimeError(f"Failed to load checkpoint '{path}': {exc}") from exc

    return _extract_state_dict(obj)


def _collect_info(
    state_dict: Dict[str, torch.Tensor],
    *,
    prefix: str | None,
) -> Dict[str, TensorInfo]:
    info: Dict[str, TensorInfo] = {}
    for key, tensor in state_dict.items():
        if prefix and not key.startswith(prefix):
            continue
        info[key] = (tuple(tensor.shape), tensor.dtype)
    return info


def _format_shape(shape: Iterable[int]) -> str:
    return "(" + ", ".join(str(dim) for dim in shape) + ")"


def summarize(info: Dict[str, TensorInfo], label: str) -> None:
    print(f"=== {label} ===")
    print(f"Total tensors: {len(info)}")
    dtype_counts = Counter(dtype for _, dtype in info.values())
    if dtype_counts:
        print("Dtype counts:")
        for dtype, count in sorted(dtype_counts.items(), key=lambda item: str(item[0])):
            print(f"  {dtype}: {count}")
    else:
        print("(No tensors matched the provided filters)")
    print()


def compare(
    info_a: Dict[str, TensorInfo],
    info_b: Dict[str, TensorInfo],
    label_a: str,
    label_b: str,
) -> None:
    keys_a = set(info_a)
    keys_b = set(info_b)

    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    shared = keys_a & keys_b

    print("=== Key differences ===")
    print(f"Only in {label_a}: {len(only_a)}")
    if only_a:
        for name in only_a:
            shape, dtype = info_a[name]
            print(f"  {name} :: shape={_format_shape(shape)}, dtype={dtype}")
    print()

    print(f"Only in {label_b}: {len(only_b)}")
    if only_b:
        for name in only_b:
            shape, dtype = info_b[name]
            print(f"  {name} :: shape={_format_shape(shape)}, dtype={dtype}")
    print()

    dtype_mismatch = []
    shape_mismatch = []
    for name in sorted(shared):
        shape_a, dtype_a = info_a[name]
        shape_b, dtype_b = info_b[name]
        if shape_a != shape_b:
            shape_mismatch.append((name, shape_a, shape_b))
        if dtype_a != dtype_b:
            dtype_mismatch.append((name, dtype_a, dtype_b))

    print("=== Shared key mismatches ===")
    print(f"Shape mismatches: {len(shape_mismatch)}")
    for name, shape_a, shape_b in shape_mismatch:
        print(
            f"  {name}\n"
            f"    {label_a}: {_format_shape(shape_a)}\n"
            f"    {label_b}: {_format_shape(shape_b)}"
        )

    print()
    print(f"Dtype mismatches: {len(dtype_mismatch)}")
    for name, dtype_a, dtype_b in dtype_mismatch:
        print(
            f"  {name}\n"
            f"    {label_a}: {dtype_a}\n"
            f"    {label_b}: {dtype_b}"
        )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_a", help="Path to the first checkpoint")
    parser.add_argument("checkpoint_b", help="Path to the second checkpoint")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional key prefix to filter tensors before comparison (e.g. 'net.')",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the detailed lists of differing keys; only print counts.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    state_a = _load_checkpoint(args.checkpoint_a)
    state_b = _load_checkpoint(args.checkpoint_b)

    info_a = _collect_info(state_a, prefix=args.prefix)
    info_b = _collect_info(state_b, prefix=args.prefix)

    summarize(info_a, label="Checkpoint A")
    summarize(info_b, label="Checkpoint B")

    if args.quiet:
        print("(quiet mode: skipping detailed diff)")
        print(f"Keys only in A: {len(set(info_a) - set(info_b))}")
        print(f"Keys only in B: {len(set(info_b) - set(info_a))}")
        shared = set(info_a) & set(info_b)
        shape_mismatches = sum(
            info_a[name][0] != info_b[name][0] for name in shared
        )
        dtype_mismatches = sum(
            info_a[name][1] != info_b[name][1] for name in shared
        )
        print(f"Shared keys with shape mismatches: {shape_mismatches}")
        print(f"Shared keys with dtype mismatches: {dtype_mismatches}")
        return 0

    compare(info_a, info_b, label_a="Checkpoint A", label_b="Checkpoint B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
