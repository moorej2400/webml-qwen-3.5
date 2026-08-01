"""Regenerate the independent PyTorch values in the vision CPU-oracle tests.

Run with Python and PyTorch 2.8.0 on CPU. The TypeScript oracle is deliberately
not imported here, so shared implementation mistakes cannot regenerate their
own expected values.
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as functional


PYTORCH_VERSION = "2.8.0"
TRANSFORMERS_REVISION = "3717b9cda226af7377da1944f395af0eac6e2b51"


def require_reference_runtime() -> None:
    if torch.__version__.split("+")[0] != PYTORCH_VERSION:
        raise RuntimeError(
            f"fixtures require torch {PYTORCH_VERSION}, found {torch.__version__}"
        )


def production_linear() -> list[float]:
    width = 1_024
    values = torch.tensor(
        [math.sin(index * 0.13) * 3 for index in range(width)],
        dtype=torch.float32,
    )
    words = torch.tensor(
        [0x3C00 + ((index * 37) % 0x600) for index in range(width * 8)],
        dtype=torch.int32,
    )
    weights = (words << 16).view(torch.float32).reshape(8, width)
    bias = torch.tensor([index * 0.1 for index in range(8)], dtype=torch.float32)
    return functional.linear(values, weights, bias).tolist()


def production_layer_norm() -> list[float]:
    width = 1_024
    values = torch.tensor(
        [1_000 + math.sin(index * 0.17) * 1.5 for index in range(width)],
        dtype=torch.float32,
    )
    normalized = functional.layer_norm(values, (width,), eps=1e-6)
    return normalized[[0, 1, 2, 127, 511, 512, 1_022, 1_023]].tolist()


def long_position_rope() -> dict[str, list[float]]:
    dimension = 64
    half_dimension = dimension // 2
    frequencies_per_axis = half_dimension // 2
    query = torch.arange(1, dimension + 1, dtype=torch.float32)
    key = torch.arange(dimension, 0, -1, dtype=torch.float32)
    frequency = torch.arange(frequencies_per_axis, dtype=torch.float32)
    inverse_frequency = 1.0 / (10_000 ** (2 * frequency / half_dimension))
    angles = torch.cat((1_023 * inverse_frequency, 2_047 * inverse_frequency))
    cosine = angles.cos()
    sine = angles.sin()

    def rotate(values: torch.Tensor) -> torch.Tensor:
        left, right = values[:half_dimension], values[half_dimension:]
        return torch.cat((left * cosine - right * sine, right * cosine + left * sine))

    indexes = [0, 1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63]
    return {
        "query": rotate(query)[indexes].tolist(),
        "key": rotate(key)[indexes].tolist(),
    }


def segmented_attention() -> list[float]:
    query = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=torch.float32)
    key = torch.tensor([[1, 0], [0, 1], [100, 0], [0, 100]], dtype=torch.float32)
    value = torch.tensor([[1, 2], [3, 4], [100, 200], [300, 400]], dtype=torch.float32)
    outputs = []
    for start in (0, 2):
        scores = query[start : start + 2] @ key[start : start + 2].T / math.sqrt(2)
        outputs.append(scores.softmax(dim=-1) @ value[start : start + 2])
    return torch.cat(outputs).reshape(-1).tolist()


def merger_orientation() -> list[float]:
    patches = torch.tensor(
        [[0, 1, 4], [2, 7, 3], [9, 1, 5], [4, 6, 12]],
        dtype=torch.float32,
    )
    shuffled = functional.layer_norm(patches, (3,), eps=1e-6).reshape(-1)
    return functional.gelu(shuffled, approximate="none").tolist()


def main() -> None:
    require_reference_runtime()
    print(json.dumps({
        "reference": {
            "transformersRevision": TRANSFORMERS_REVISION,
            "pytorchVersion": PYTORCH_VERSION,
            "device": "cpu",
            "dtype": "float32",
        },
        "productionLinear": production_linear(),
        "productionLayerNorm": production_layer_norm(),
        "longPositionRope": long_position_rope(),
        "segmentedAttention": segmented_attention(),
        "mergerOrientation": merger_orientation(),
    }, indent=2))


if __name__ == "__main__":
    main()
