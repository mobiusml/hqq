# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
import torch
import gc
import math
from typing import Union


def cleanup() -> None:
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def is_divisible(val1: int, val2: int) -> bool:
    return int(val2 * math.ceil(val1 / val2)) == val1


def zero_pad_row(
    tensor: torch.Tensor, num_rows: int, dtype: Union[torch.dtype, None] = None
) -> torch.Tensor:
    out = torch.zeros(
        [num_rows, tensor.shape[1]],
        device=tensor.device,
        dtype=tensor.dtype if (dtype is None) else dtype,
    )
    out[: len(tensor)] = tensor

    return out


# Map a Pytorch dtype into a safetensor dtype
def encode_safetensor_type(data):
    if isinstance(data, (torch.Tensor, torch.nn.Parameter)):
        return data
    if isinstance(data, torch.Size):
        return torch.tensor(data)
    if isinstance(data, torch.dtype):
        data = str(data)
    if isinstance(data, (bool, int)):
        return torch.tensor(int(data), dtype=torch.uint8)
    if isinstance(data, float):
        return torch.tensor(float(data), dtype=torch.float32)
    if isinstance(data, str):
        return torch.tensor([ord(i) for i in data], dtype=torch.uint8)


# Decode a safetensor dtype into a Pytorch dtype
def decode_safetensor_type(data, data_type):
    if data_type in [torch.Tensor, torch.nn.Parameter]:
        return data
    if data_type is torch.Size:
        return torch.Size(data)
    if data_type is bool:
        return bool(data.item())
    if data_type is int:
        return int(data.item())
    if data_type is float:
        return float(data.item())
    if data_type is str:
        return "".join([chr(i) for i in data])
    if data_type is torch.dtype:
        return eval("".join([chr(i) for i in data]))
