# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
from ..base import BasePatch
from .base import BaseHQQHFModel
from tqdm import tqdm
import torch


# Patch functions
class PhiPatch(BasePatch):
    # These tags are used to specify the parameters of each layer type. For example, if you want to give different quantization parameters to different layers
    @classmethod
    def get_linear_tags(cls):
        return [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.dense",
            "mlp.fc1",
            "mlp.fc2",
        ]

    @classmethod
    def patch_nonlinearlayers(cls, model, patch_fct, verbose=True):
        base_model = model.model
        model.lm_head = patch_fct(model.lm_head)
        base_model.embed_tokens = patch_fct(base_model.embed_tokens)
        base_model.final_layernorm = patch_fct(base_model.final_layernorm)

        # Remove dropout
        base_model.embed_dropout = torch.nn.Identity()

        layers = base_model.layers
        for i in tqdm(range(len(base_model.layers)), disable=not verbose):
            layers[i].self_attn.rotary_emb = patch_fct(layers[i].self_attn.rotary_emb)
            layers[i].mlp.activation_fn = patch_fct(layers[i].mlp.activation_fn)
            layers[i].input_layernorm = patch_fct(layers[i].input_layernorm)
            layers[i].resid_dropout = torch.nn.Identity()

    @classmethod
    def patch_linearlayers(cls, model, patch_fct, patch_params, verbose=True):
        base_model = model.model
        layers = base_model.layers
        for i in tqdm(range(len(layers)), disable=not verbose):
            layers[i].self_attn.q_proj = patch_fct(
                layers[i].self_attn.q_proj, patch_params["self_attn.q_proj"]
            )
            layers[i].self_attn.k_proj = patch_fct(
                layers[i].self_attn.k_proj, patch_params["self_attn.k_proj"]
            )
            layers[i].self_attn.v_proj = patch_fct(
                layers[i].self_attn.v_proj, patch_params["self_attn.v_proj"]
            )
            layers[i].self_attn.dense = patch_fct(
                layers[i].self_attn.dense, patch_params["self_attn.dense"]
            )
            layers[i].mlp.fc1 = patch_fct(layers[i].mlp.fc1, patch_params["mlp.fc1"])
            layers[i].mlp.fc2 = patch_fct(layers[i].mlp.fc2, patch_params["mlp.fc2"])


class PhiHQQ(PhiPatch, BaseHQQHFModel):
    pass
