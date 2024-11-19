# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
######################################################

import torch
from torch import Tensor
from tqdm import tqdm
import requests
import gc
from PIL import Image

from torch.nn.attention import sdpa_kernel, SDPBackend

from ..core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig
from ..models.hf.base import AutoHQQHFModel
from .patching import patch_hqq_to_aoint4, prepare_for_inference
from .generation_hf import patch_accelerate_device_hook


class HQQGroupedGemm(torch.nn.Module):
    def __init__(
        self,
        grouped_gemm_layer,
        num_active_experts,
        quant_config,
        compute_dtype=torch.bfloat16,
        device="cuda:0",
    ):
        super().__init__()
        self.quant_config = quant_config
        self.num_active_experts = num_active_experts
        self.compute_dtype = compute_dtype
        self.device = device
        self.quant_expert(grouped_gemm_layer)

    def quant_expert(self, mlp_layer, backend="torchao_int4"):
        weight = mlp_layer.weight
        num_experts, in_features, out_features = weight.shape

        weight_int4pack, scales_and_zeros = [], []
        for j in range(num_experts):
            hqq_layer = patch_hqq_to_aoint4(
                HQQLinear.from_weights(
                    weight=weight[j].T,
                    bias=None,
                    quant_config=self.quant_config,
                    compute_dtype=self.compute_dtype,
                    device=self.device,
                    del_orig=True,
                ),
                None,
            )
            weight_int4pack.append(hqq_layer.weight_int4pack[None, :])
            scales_and_zeros.append(hqq_layer.scales_and_zeros[None, :])

        self.weights = torch.cat(weight_int4pack)
        self.scales_and_zeros = torch.cat(scales_and_zeros)
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.group_size = self.quant_config["weight_quant_params"]["group_size"]

        del hqq_layer
        torch.cuda.empty_cache()

    def forward(self, input, tokens_per_expert):
        num_tokens = input.shape[0]
        forward_fct = (
            self.forward_seq_gemm_prefill
            if (num_tokens > self.num_active_experts)
            else self.forward_seq_gemm_decode
        )
        return forward_fct(
            input,
            tokens_per_expert,
            self.num_experts,
            self.out_features,
            self.weights,
            self.scales_and_zeros,
            self.group_size,
        )

    def forward_seq_gemm_prefill(
        self,
        input: Tensor,
        tokens_per_expert: Tensor,
        num_experts: int,
        out_features: int,
        weights: Tensor,
        meta_data: Tensor,
        group_size: int,
    ) -> Tensor:
        num_tokens = input.shape[0]
        output = torch.empty(
            num_tokens, out_features, dtype=input.dtype, device=input.device
        )

        cumsum_num_tokens = torch.zeros(
            len(tokens_per_expert) + 1, dtype=torch.long, device=input.device
        )
        cumsum_num_tokens[1:] = torch.cumsum(tokens_per_expert, dim=0)

        num_active_experts = (
            (tokens_per_expert > 0).sum().int()
        )  # this should be static for torch.compile
        selected_experts = torch.topk(
            torch.diff(cumsum_num_tokens), k=num_active_experts, dim=0, largest=True
        )[1].sort()[0]

        _weights = weights[selected_experts]
        _scales = meta_data[selected_experts]
        _start = cumsum_num_tokens[selected_experts]
        _end = cumsum_num_tokens[selected_experts + 1]

        for i in range(num_active_experts):
            _s, _e = _start[i], _end[i]
            output[_s:_e] = self.matmul(
                input[_s:_e], _weights[i], _scales[i], group_size, out_features
            )

        return output

    def forward_seq_gemm_decode(
        self,
        input: Tensor,
        tokens_per_expert: Tensor,
        num_experts: int,
        out_features: int,
        weights: Tensor,
        meta_data: Tensor,
        group_size: int,
    ) -> Tensor:
        num_tokens = input.shape[0]
        output = torch.empty(
            num_tokens, out_features, dtype=input.dtype, device=input.device
        )

        cumsum_num_tokens = torch.zeros(
            len(tokens_per_expert) + 1, dtype=torch.long, device=input.device
        )
        cumsum_num_tokens[1:] = torch.cumsum(tokens_per_expert, dim=0)

        num_active_experts = self.num_active_experts
        selected_experts = torch.topk(
            torch.diff(cumsum_num_tokens), k=num_active_experts, dim=0, largest=True
        )[1].sort()[0]

        _weights = weights[selected_experts]
        _scales = meta_data[selected_experts]
        _start = cumsum_num_tokens[selected_experts]
        _end = cumsum_num_tokens[selected_experts + 1]

        for i in range(num_active_experts):
            _s, _e = i, i + 1
            output[_s:_e] = self.matmul(
                input[_s:_e], _weights[i], _scales[i], group_size, out_features
            )

        return output

    def matmul(
        self,
        x: Tensor,
        weight_int4pack: Tensor,
        scales_and_zeros: Tensor,
        groupsize: int,
        out_features: int,
    ) -> Tensor:
        origin_x_size = x.size()
        x = x.reshape(-1, origin_x_size[-1])
        c = torch.ops.aten._weight_int4pack_mm(
            x, weight_int4pack, groupsize, scales_and_zeros
        )
        new_shape = origin_x_size[:-1] + (out_features,)
        c = c.reshape(new_shape)
        return c


def quantize_model(
    model,
    num_active_experts=6,
    attn_quant_config=BaseQuantizeConfig(nbits=4, group_size=64, axis=1),
    experts_quant_config=BaseQuantizeConfig(nbits=4, group_size=64, axis=1),
    compute_dtype=torch.bfloat16,
    device="cuda:0",
):
    for i in tqdm(range(len(model.language_model.model.layers))):
        model.language_model.model.layers[i].mlp.experts.fc1 = HQQGroupedGemm(
            model.language_model.model.layers[i].mlp.experts.fc1,
            num_active_experts=num_active_experts,
            quant_config=experts_quant_config,
            compute_dtype=compute_dtype,
            device=device,
        )
        model.language_model.model.layers[i].mlp.experts.fc2 = HQQGroupedGemm(
            model.language_model.model.layers[i].mlp.experts.fc2,
            num_active_experts=num_active_experts,
            quant_config=experts_quant_config,
            compute_dtype=compute_dtype,
            device=device,
        )
    gc.collect()

    # Quantize the rest
    AutoHQQHFModel.quantize_model(
        model.language_model,
        quant_config=attn_quant_config,
        compute_dtype=compute_dtype,
        device=device,
    )

    # Remove losses
    # import moe_llm
    # moe_llm.apply_z_loss   = lambda logits: logits
    # moe_llm.apply_aux_loss = lambda logits, tokens_per_expert, scores: scores

    # Move the vision model to the device
    model.multi_modal_projector = model.multi_modal_projector.to(device)
    model.vision_tower = model.vision_tower.to(device)

    # Optimize
    HQQLinear.set_backend(
        HQQBackend.ATEN
        if experts_quant_config["weight_quant_params"]["axis"] == 0
        else HQQBackend.PYTORCH_COMPILE
    )
    prepare_for_inference(model.language_model, backend="torchao_int4", verbose=True)


def generate(model, processor, img_path, prompt, max_new_tokens=500, do_sample=True):
    image = Image.open(requests.get(img_path, stream=True).raw)

    messages = [
        {
            "role": "user",
            "content": [
                {"text": None, "type": "image"},
                {"text": prompt, "type": "text"},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            stop_strings=["<|im_end|>"],
            tokenizer=processor.tokenizer,
            do_sample=do_sample,
            temperature=0.9,
            cache_implementation="static",
        )
        output_ids = output[0][inputs["input_ids"].shape[1] :]
        result = processor.decode(output_ids, skip_special_tokens=True)

    return result


def patch_model_for_compiled_runtime(
    model, processor, warmup=True, patch_accelerate=True
):
    if patch_accelerate:
        patch_accelerate_device_hook()

    torch._dynamo.config.inline_inbuilt_nn_modules = False  # torch 2.5.0 fix

    model.language_model.config.use_cache = True
    model.language_model.generation_config.cache_implementation = "static"
    model.eval()

    forward_compiled = torch.compile(
        model.language_model.forward, mode="reduce-overhead", fullgraph=True
    )
    forward_simple = model.language_model.forward

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens({"pad_token": "<<[PAD]>>"})

    def custom_forward(*args, **kwargs):
        # Prefill phase
        out_fct = forward_simple

        # Decoding pahse
        if (
            (len(args) > 0 and args[0].shape[-1] == 1)
            or ("input_ids" in kwargs and kwargs["input_ids"].shape[-1] == 1)
            or ("inputs_embeds" in kwargs and kwargs["inputs_embeds"].shape[1] == 1)
        ):
            out_fct = forward_compiled

        with sdpa_kernel([SDPBackend.MATH]):
            out = out_fct(*args, **kwargs)

        return out

    model.language_model.forward = custom_forward

    if warmup:
        for _ in tqdm(range(5)):
            generate(
                model,
                processor,
                img_path="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
                prompt="what is the image?",
            )
