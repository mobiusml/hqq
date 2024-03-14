# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
from ..base import BasePatch
from .base import BaseHQQHFModel
from tqdm import tqdm
import torch

# Experimental - This is the optimized version of microsoft's Phi model which contains merged QKV layers and other things.
# Unfortunately this cannot be used automatically because the base class PhiForCausalLM has been modified. you need to use it manually by calling PhiHQQ.fct(model, args)


# Patch functions
class PhiPatch(BasePatch):
    # These tags are used to specify the parameters of each layer type. For example, if you want to give different quantization parameters to different layers
    @classmethod
    def get_linear_tags(cls):
        return ["mixer.Wqkv", "mixer.out_proj", "mlp.fc1", "mlp.fc2"]

    @classmethod
    def patch_nonlinearlayers(cls, model, patch_fct, verbose=True):
        # Remove dropout
        model.transformer.embd.drop = torch.nn.Identity()

        base_model = model.transformer
        model.lm_head = patch_fct(model.lm_head)
        base_model.embd = patch_fct(base_model.embd)

        layers = base_model.h
        for i in tqdm(range(len(layers)), disable=not verbose):
            layers[i].ln = patch_fct(layers[i].ln)
            layers[i].mixer.rotary_emb = patch_fct(layers[i].mixer.rotary_emb)
            layers[i].mlp.act = patch_fct(layers[i].mlp.act)

    @classmethod
    def patch_linearlayers(cls, model, patch_fct, patch_params, verbose=True):
        base_model = model.transformer
        layers = base_model.h
        for i in tqdm(range(len(layers)), disable=not verbose):
            layers[i].mixer.Wqkv = patch_fct(
                layers[i].mixer.Wqkv, patch_params["mixer.Wqkv"]
            )
            layers[i].mixer.out_proj = patch_fct(
                layers[i].mixer.out_proj, patch_params["mixer.out_proj"]
            )
            layers[i].mlp.fc1 = patch_fct(layers[i].mlp.fc1, patch_params["mlp.fc1"])
            layers[i].mlp.fc2 = patch_fct(layers[i].mlp.fc2, patch_params["mlp.fc2"])

            # Remove dropout
            layers[i].resid_dropout = torch.nn.Identity()
            layers[i].mixer.inner_attn.drop = torch.nn.Identity()
            layers[i].mixer.inner_cross_attn.drop = torch.nn.Identity()


class PhiHQQ(PhiPatch, BaseHQQHFModel):
    pass
