# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
import torch
import gc
import numpy as np


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def is_divisible(val1, val2):
    return int(val2 * np.ceil(val1 / val2)) == val1


def make_multiple(val, multiple):
    return int(multiple * np.ceil(val / float(multiple)))


def zero_pad_row(tensor, num_rows, dtype=None):
    out = torch.zeros(
        [num_rows, tensor.shape[1]],
        device=tensor.device,
        dtype=tensor.dtype if (dtype is None) else dtype,
    )
    out[: len(tensor)] = tensor
    return out
