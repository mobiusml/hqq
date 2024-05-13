# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
import torch
import numpy as np
from torch import float32, float16, Tensor
from functools import partial
from typing import Union


# re-estimate the scale based on the inverse median: Only tested with axis==0
def update_scale_inverse_median(
    W_f: Tensor, scale: Tensor, zero: Tensor, axis: int, min_max: list
) -> tuple:
    scale_rng = 2e4
    z_val = 1e-4
    delta = 1e-2

    W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])

    # Correct zero to avoid W_q==zero
    zero_c = zero.clone()
    zero_c_indx = torch.sum(1.0 * ((W_q - zero) == 0), axis=axis, keepdim=True) > 0
    zero_c[zero_c_indx] = zero_c[zero_c_indx] + delta

    # Build scale tensor
    W_f_c = W_f.clone()
    W_f_c_mask = torch.abs(W_f_c) < z_val
    W_f_c[W_f_c_mask] = z_val

    scale_tensor = (W_q - zero_c).float() / W_f_c.float()
    # W_r               = (W_q - zero_c)/scale_tensor

    # Normalize scale_tensor
    scale_b = torch.median(scale_tensor, axis=axis, keepdim=True)[0]
    scale_b = scale_b.clamp(min=-scale_rng, max=scale_rng).half()

    # Mix with older scale
    W_r = (W_q - zero_c) / scale_b
    err_b = torch.abs(W_f - W_r).mean(axis=axis, keepdim=True)

    W_r = (W_q - zero_c) / scale
    err_a = torch.abs(W_f - W_r).mean(axis=axis, keepdim=True)

    mask = (err_b < err_a).half()
    scale_b = mask * scale_b + (1 - mask) * scale

    # W_r   = (W_q - zero_c)/scale_b
    return scale_b, zero_c


# Greedy local search: Only tested with axis==0
def update_scale_grid_search(
    W_f: Tensor, scale: Tensor, zero: Tensor, axis: int, min_max: list, N: int = 128 + 1
) -> Tensor:
    # Make sure it's an odd number so that the original scale is included
    assert N % 2 == 1, "Please check whether N: odd number"
    rng_dump = 0.05  # 0.05 / 1.
    z_val = 2e-4

    device = scale.device
    dtype = scale.dtype
    ###############################
    W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])
    n_clusters = max(W_q.shape[0], W_q.shape[1])
    rng = torch.abs(scale).mean() * rng_dump if (rng_dump < 1.0) else rng_dump

    scale_shifted = (
        torch.linspace(-rng, rng, N)[:, None]
        .to(dtype=dtype, device=device)
        .repeat(1, n_clusters)
        + scale
    )

    # Safe inverse
    scale_shifted[
        torch.logical_and(scale_shifted >= 0, torch.abs(scale_shifted) <= z_val)
    ] = z_val
    scale_shifted[
        torch.logical_and(scale_shifted < 0, torch.abs(scale_shifted) <= z_val)
    ] = -z_val

    err = torch.empty([N, n_clusters], dtype=dtype, device=device)
    for i in range(N):
        W_r = (W_q - zero) / scale_shifted[i][None, :]
        err[i] = torch.abs(W_f - W_r).mean(axis=axis, keepdim=True)

    ind_r = torch.argmin(err, axis=axis).to(torch.int32)
    ind_c = torch.arange(len(ind_r), dtype=torch.int32, device=device)
    scale_b = scale_shifted[ind_r, ind_c]

    return scale_b


# Shrinking operator
def shrink_lp_op(x: Tensor, beta: float, lp_norm: float) -> Tensor:
    if lp_norm == 1:
        return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
    else:
        return torch.sign(x) * torch.nn.functional.relu(
            torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), lp_norm - 1)
        )


# Proximal solver || W - dequantize(quantize(W))||_p^p - Experimental
@torch.inference_mode()
def optimize_weights_proximal_v2(
    tensor: Tensor,
    scale: Tensor,
    zero: Tensor,
    min_max: list,
    axis: int = 0,
    device: Union[str, None] = None,
    dtype: Union[torch.dtype, None] = None,
    opt_params: dict = {
        "lp_norm": 0.7,
        "beta": 1e1,
        "kappa": 1.01,
        "iters": 20,
        "tol": 0.0,
        "early_stop": True,
        "scale_gridsearch": False,
    },
    verbose: bool = False,
) -> tuple:
    # Params
    lp_norm = max(opt_params["lp_norm"], 0.1)
    beta = opt_params["beta"]
    kappa = opt_params["kappa"]
    iters = opt_params["iters"]
    early_stop = opt_params["early_stop"]
    tol = opt_params["tol"]

    # Check
    assert lp_norm <= 1.0, "lp_norm should be <=1"
    assert beta > 0.0, "beta should be > 0"
    assert kappa > 1.0, "kappa should be > 1"
    assert iters > 1, "iters should be > 1"

    # Cast/device
    if device is None:
        device = tensor.device
    else:
        device = torch.device(device)

    if dtype is None:
        dtype = float16 if (device.type == "cuda") else float32

    W_f = tensor.to(device=device, dtype=dtype)
    scale = scale.to(device=device, dtype=dtype)
    zero = zero.to(device=device, dtype=dtype)

    # Update scale: works slightly better. Tested on Llama2 only
    if opt_params["scale_gridsearch"]:
        scale = update_scale_grid_search(W_f, scale, zero, axis, min_max)

    # Optimize for zero-point
    best_error = 1e4
    scale_prev, zero_prev = scale.clone(), zero.clone()
    for i in range(iters):
        W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])
        W_r = (W_q - zero) / scale

        # current_error = float(torch.pow(torch.abs(W_f - W_r), max(0.80, lp_norm)).mean())
        current_error = float(torch.abs(W_f - W_r).mean())

        if verbose:
            print(i, np.round(current_error, 6))

        if early_stop:
            if best_error - current_error > tol:
                best_error = current_error
                scale_prev, zero_prev = scale.clone(), zero.clone()
            else:
                scale, zero = scale_prev.clone(), zero_prev.clone()
                break

        W_e = shrink_lp_op(W_f - W_r, beta, lp_norm)
        zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
        beta *= kappa

    # Clean-up
    scale = scale.to(tensor.device)
    zero = zero.to(tensor.device)
    del W_f, W_q, W_r, W_e, scale_prev, zero_prev
    torch.cuda.empty_cache()

    W_q = torch.round(tensor * scale + zero).clamp(min_max[0], min_max[1])

    return W_q, scale, zero


# Proximal solver || W - dequantize(quantize(W))||_p^p
@torch.inference_mode()
def optimize_weights_proximal_legacy(
    tensor: Tensor,
    scale: Tensor,
    zero: Tensor,
    min_max: list,
    axis: int = 0,
    device: Union[str, None] = None,
    opt_params: dict = {"lp_norm": 0.7, "beta": 1e1, "kappa": 1.01, "iters": 20},
    verbose: bool = False,
) -> tuple:
    lp_norm, beta, kappa, iters = (
        opt_params["lp_norm"],
        opt_params["beta"],
        opt_params["kappa"],
        opt_params["iters"],
    )

    if device is None:
        device = tensor.device
    else:
        device = torch.device(device)

    dtype = float16 if (device.type == "cuda") else float32
    W_f = tensor.to(dtype=dtype, device=device)
    scale = scale.to(dtype=dtype, device=device)
    zero = zero.to(dtype=dtype, device=device)

    best_error = 1e4
    for i in range(iters):
        W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])
        W_r = (W_q - zero) / scale
        W_e = shrink_lp_op(W_f - W_r, beta, lp_norm)
        zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
        beta *= kappa

        current_error = float(torch.abs(W_f - W_r).mean())
        if verbose:
            print(i, np.round(current_error, 6))
        if current_error < best_error:
            best_error = current_error
        else:
            break

    scale = scale.to(tensor.device)
    zero = zero.to(tensor.device)
    del W_f, W_q, W_r, W_e
    torch.cuda.empty_cache()

    W_q = torch.round(tensor * scale + zero).clamp(min_max[0], min_max[1])
    return W_q, scale, zero


# Default: fast with early stopping
optimize_weights_proximal = optimize_weights_proximal_legacy

# Slower, better quality: no early stoppping, more iterations
optimize_weights_proximal_slow = partial(
    optimize_weights_proximal_v2,
    dtype=torch.float32,
    opt_params={
        "lp_norm": 0.7,
        "beta": 1e1,
        "kappa": 1.01,
        "iters": 100,
        "tol": 0.0,
        "early_stop": False,
        "scale_gridsearch": False,
    },
)

##############################################################################################################
# L1 with SGD optimizer: supports scale and W_q updates. L{p<1} fails with SGD


class LinearSchedulerWithWarmStart(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_start, lr_end, iters, warm_start=0, last_epoch=-1):
        iters_wrm = max(0, int(iters * warm_start))
        self.lr_wrm = (
            np.linspace(lr_end, lr_start, iters_wrm)
            if (iters_wrm > 0)
            else np.array([])
        )
        self.lr_mid = np.linspace(lr_start, lr_end, iters - iters_wrm)
        self.lr_sch = np.concatenate([self.lr_wrm, self.lr_mid])
        self.idx = 0
        super(LinearSchedulerWithWarmStart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.idx = min(self.idx, len(self.lr_sch) - 1)
        out = [self.lr_sch[self.idx] for base_lr in self.base_lrs]
        self.idx += 1
        return out


# SGD solver  || W - dequantize(quantize(W))||_1 (p=1 only, with additional fake inputs x)
def optimize_weights_autograd(
    tensor: Tensor,
    scale: Tensor,
    zero: Tensor,
    min_max: list,
    axis: int = 0,
    device: Union[str, None] = None,
    dtype: Union[torch.dtype, None] = float32,
    opt_params: dict = {
        "lr": 2e-3,
        "iters": 2500,
        "lr_schedule": False,
        "update_Wq": False,
        "use_fake_data": False,
    },
    data_params: dict = {"normalize": False, "data_rng": 10.0, "data_ctx": 32},
    compile: bool = True,
    verbose: bool = False,
) -> tuple:
    ref_device = scale.device
    ref_dtype = scale.dtype

    if device is None:
        device = tensor.device
    else:
        device = torch.device(device)

    W_f = tensor.to(dtype=dtype, device=device)

    params = {}
    params["scale"] = torch.nn.Parameter(
        scale.to(dtype=dtype, device=device), requires_grad=True
    )
    params["zero"] = torch.nn.Parameter(
        zero.to(dtype=dtype, device=device), requires_grad=True
    )

    if opt_params["update_Wq"]:
        with torch.no_grad():
            params["W_q"] = torch.round(W_f * params["scale"] + params["zero"]).clamp(
                min_max[0], min_max[1]
            )
        params["W_q"] = torch.nn.Parameter(params["W_q"], requires_grad=True)

    optimizer = torch.optim.AdamW(
        [params[k] for k in params],
        lr=opt_params["lr"],
        betas=(0.9, 0.99),
        eps=1e-06,
        weight_decay=0.0,
    )

    if opt_params["lr_schedule"]:
        scheduler = LinearSchedulerWithWarmStart(
            optimizer,
            lr_start=opt_params["lr"],
            lr_end=1e-6,
            iters=opt_params["iters"],
            warm_start=0,
        )
    else:
        scheduler = None

    with torch.no_grad():
        if data_params["normalize"]:
            scale_loss = 1.0 / (tensor.abs().mean() + 1e-4)
        else:
            scale_loss = 1.0

    def _loss_fct(output, target):
        return torch.mean(torch.abs(scale_loss * (target - output)))  # L1

    def _fake_quant_fixed_Wq(W_f):
        # Quantize
        W_q = torch.round(W_f * params["scale"] + params["zero"]).clamp(
            min_max[0], min_max[1]
        )
        # Dequantize
        W_r = (W_q - params["zero"]) / params["scale"]
        return W_r

    def _fake_quant_update_Wq(W_f):
        # Quantize
        W_q = torch.round(params["W_q"]).clamp(min_max[0], min_max[1])
        # Dequantize
        W_r = (W_q - params["zero"]) / params["scale"]
        return W_r

    if opt_params["update_Wq"]:
        _fake_quant = _fake_quant_update_Wq
    else:
        _fake_quant = _fake_quant_fixed_Wq

    def _fake_quant_loss(W_f):
        return _loss_fct(_fake_quant(W_f), W_f)

    def _fake_quant_loss_with_fake_data(W_f):
        x = (
            torch.rand(
                [data_params["data_ctx"], W_f.shape[1]], device=device, dtype=dtype
            )
            - 0.5
        ) * 2 ** data_params["data_rng"]
        y_ref = torch.matmul(x, W_f.T)
        y_pred = torch.matmul(x, _fake_quant(W_f).T)
        return _loss_fct(y_pred, y_ref)

    if opt_params["use_fake_data"]:
        _fake_quant_loss = _fake_quant_loss_with_fake_data
    else:
        _fake_quant_loss = _fake_quant_loss

    if compile:
        _fake_quant_loss = torch.compile(_fake_quant_loss)

    def _step(W_f):
        optimizer.zero_grad()
        loss = _fake_quant_loss(W_f)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return np.round(loss.item(), 10)

    with torch.no_grad():
        _init_loss = _fake_quant_loss(W_f).item()

    for i in range(opt_params["iters"]):
        loss_out = _step(W_f)
        if verbose and (i % 100) == 0:
            print(i, loss_out)

    with torch.no_grad():
        _final_loss = _fake_quant_loss(W_f).item()

    if _final_loss < _init_loss:
        for k in params:
            params[k] = params[k].data.detach()
    else:
        if verbose:
            print("optimization failed...")
        params = {"scale": scale, "zero": zero}

    scale = params["scale"].to(device=ref_device, dtype=ref_dtype)
    zero = params["zero"].to(device=ref_device, dtype=ref_dtype)

    if "W_q" in params:
        W_q = params["W_q"].to(device=ref_device, dtype=ref_dtype)
    else:
        W_q = (
            torch.round(tensor * scale + zero)
            .clamp(min_max[0], min_max[1])
            .to(device=ref_device, dtype=ref_dtype)
        )

    del W_f
    torch.cuda.empty_cache()
    return W_q, scale, zero


optimize_weights_autograd_main = partial(
    optimize_weights_autograd,
    dtype=torch.float32,
    opt_params={
        "lr": 2e-3,
        "iters": 1000,
        "lr_schedule": True,
        "update_Wq": True,
        "use_fake_data": False,
    },
    verbose=False,
    compile=True,
)

optimize_weights_autograd_fakedata = partial(
    optimize_weights_autograd,
    dtype=torch.float32,
    opt_params={
        "lr": 2e-3,
        "iters": 1000,
        "lr_schedule": True,
        "update_Wq": True,
        "use_fake_data": True,
    },
    data_params={"normalize": False, "data_rng": 10.0, "data_ctx": 32},
    verbose=False,
    compile=True,
)
