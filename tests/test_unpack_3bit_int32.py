# about hqq/hqq/core/bitpack.py

from time import time
import torch
from torch import Tensor, LongTensor


def unpack_3bit_int32(
        input_tensor: LongTensor, is_test: bool = False
) -> Tensor:

    def drag_binary_3bits_int(n: int) -> int:
        i = 3 * (n - 1)
        prefix = i * '0'
        postfix = (27 - i) * '0'
        b_str = f'0b00{prefix}111{postfix}'

        return int(b_str, 2)

    def make_bitwise_step(x: LongTensor, n: int):
        return (x & drag_binary_3bits_int(n)) >> 3 * (10 - n)

    _step = input_tensor.shape[0]
    out = torch.empty([10 * _step, input_tensor.shape[1]], dtype=torch.uint8)

    if is_test:
        for k in range(1, 10):
            out[(k - 1) * _step : k * _step] = make_bitwise_step(input_tensor, k)

    else:
        out[:_step] = (input_tensor & 0b00111000000000000000000000000000) >> 27
        out[1 * _step : 2 * _step] = (input_tensor & 0b00000111000000000000000000000000) >> 24
        out[2 * _step : 3 * _step] = (input_tensor & 0b00000000111000000000000000000000) >> 21
        out[3 * _step : 4 * _step] = (input_tensor & 0b00000000000111000000000000000000) >> 18
        out[4 * _step : 5 * _step] = (input_tensor & 0b00000000000000111000000000000000) >> 15
        out[5 * _step : 6 * _step] = (input_tensor & 0b00000000000000000111000000000000) >> 12
        out[6 * _step : 7 * _step] = (input_tensor & 0b00000000000000000000111000000000) >> 9
        out[7 * _step : 8 * _step] = (input_tensor & 0b00000000000000000000000111000000) >> 6
        out[8 * _step : 9 * _step] = (input_tensor & 0b00000000000000000000000000111000) >> 3

    out[9 * _step :] = input_tensor & 0b00_000_000_000_000_000_000_000_000_000_111

    return out


if __name__ == '__main__':
    test_tensor = LongTensor(10, 21)

    start = time()
    normal_ = unpack_3bit_int32(test_tensor, True)
    end = time()
    print(end - start)
    print(normal_)
