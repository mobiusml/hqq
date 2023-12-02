from ..base import *
from .base  import *

#Patch LLama functions
class LLamaPatch(BasePatch):
	#These tags are used to specify the parameters of each layer type. For example, if you want to give different quantization parameters to different layers
	@classmethod
	def get_linear_tags(cls):
		return ['self_attn.q_proj',
				'self_attn.k_proj',
				'self_attn.v_proj',
				'self_attn.o_proj',
				'mlp.gate_proj'   ,
				'mlp.up_proj'     ,
				'mlp.down_proj'   ]

	@classmethod
	def patch_nonlinearlayers(cls, model, patch_fct, verbose=True):
		base_model              = model.model
		model.lm_head           = patch_fct(model.lm_head)
		base_model.embed_tokens = patch_fct(base_model.embed_tokens)
		base_model.norm         = patch_fct(base_model.norm)

		layers = base_model.layers
		for i in tqdm(range(len(base_model.layers)), disable=not verbose):
			layers[i].self_attn.rotary_emb     = patch_fct(layers[i].self_attn.rotary_emb)
			layers[i].mlp.act_fn               = patch_fct(layers[i].mlp.act_fn)
			layers[i].input_layernorm          = patch_fct(layers[i].input_layernorm)
			layers[i].post_attention_layernorm = patch_fct(layers[i].post_attention_layernorm)

	@classmethod
	def patch_linearlayers(cls, model, patch_fct, patch_params, verbose=True):
		base_model = model.model
		layers     = base_model.layers 
		for i in tqdm(range(len(layers)), disable=not verbose):
			layers[i].self_attn.q_proj = patch_fct(layers[i].self_attn.q_proj, patch_params['self_attn.q_proj'])
			layers[i].self_attn.k_proj = patch_fct(layers[i].self_attn.k_proj, patch_params['self_attn.k_proj'])
			layers[i].self_attn.v_proj = patch_fct(layers[i].self_attn.v_proj, patch_params['self_attn.v_proj'])
			layers[i].self_attn.o_proj = patch_fct(layers[i].self_attn.o_proj, patch_params['self_attn.o_proj'])
			layers[i].mlp.gate_proj    = patch_fct(layers[i].mlp.gate_proj,    patch_params['mlp.gate_proj'])
			layers[i].mlp.up_proj      = patch_fct(layers[i].mlp.up_proj,      patch_params['mlp.up_proj'])
			layers[i].mlp.down_proj    = patch_fct(layers[i].mlp.down_proj,    patch_params['mlp.down_proj'])


class LlamaHQQ(LLamaPatch, BaseHQQHFModel):
	#layers to ignore when saving the weights
	@classmethod
	def get_ignore_layers(cls, model):
		return ['', 'model', 'model.layers'] + ['model.layers.' + str(i) for i in range(len(model.model.layers))]

	#Create empty model
	@classmethod
	def create_model(cls, save_dir):
		config = transformers.AutoConfig.from_pretrained(cls.get_config_file(save_dir))
		with init_empty_weights():
			model = transformers.LlamaForCausalLM(config)
		return model
