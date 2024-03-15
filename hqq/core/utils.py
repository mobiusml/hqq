# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
import torch
import gc
import numpy as np


def cleanup() -> None:
    torch.cuda.empty_cache()
    gc.collect()


def is_divisible(val1, val2) -> bool:
    return int(val2 * np.ceil(val1 / val2)) == val1


def make_multiple(val, multiple) -> int:
    return int(multiple * np.ceil(val / float(multiple)))


def zero_pad_row(
    tensor: torch.Tensor, num_rows: int, dtype: torch.dtype | None = None
) -> Tensor:
    out = torch.zeros(
        [num_rows, tensor.shape[1]],
        device=tensor.device,
        dtype=tensor.dtype if (dtype is None) else dtype,
    )
    out[: len(tensor)] = tensor

    return out
