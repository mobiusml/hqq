from .base import *

from tqdm import tqdm 
import timm, json, os

#Patch ViT functions
class VitPatch(BasePatch):
	#These tags are used to specify the parameters of each layer type. For example, if you want to give different quantization parameters to different layers
	@classmethod
	def get_linear_tags(cls):
		return ['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2']

	@classmethod
	def freeze_model(cls, model):
		for param in model.parameters():
			param.requires_grad = False

	@classmethod
	def patch_nonlinearlayers(cls, model, patch_fct, verbose=True):
		model.patch_embed.proj = patch_fct(model.patch_embed.proj)
		model.patch_embed.norm = patch_fct(model.patch_embed.norm)
		model.norm_pre         = patch_fct(model.norm_pre)
		model.norm             = patch_fct(model.norm)
		model.head             = patch_fct(model.head)
		model.cls_token.data   = patch_fct(model.cls_token.data)
		model.pos_embed.data   = patch_fct(model.pos_embed.data)

		for i in tqdm(range(len(model.blocks)), disable=not verbose):
			model.blocks[i].norm1 = patch_fct(model.blocks[i].norm1)
			model.blocks[i].norm2 = patch_fct(model.blocks[i].norm2)

	@classmethod
	def patch_linearlayers(cls, model, patch_fct, patch_params, verbose=True):
		for i in tqdm(range(len(model.blocks))):
			model.blocks[i].attn.qkv  = patch_fct(model.blocks[i].attn.qkv,  patch_params['attn.qkv']) 
			model.blocks[i].attn.proj = patch_fct(model.blocks[i].attn.proj, patch_params['attn.proj'])
			model.blocks[i].mlp.fc1   = patch_fct(model.blocks[i].mlp.fc1,   patch_params['mlp.fc1']) 
			model.blocks[i].mlp.fc2   = patch_fct(model.blocks[i].mlp.fc2,   patch_params['mlp.fc2']) 


class ViTHQQ(VitPatch, BaseHQQModel):
	#layers to ignore when saving the weights
	@classmethod
	def get_ignore_layers(cls, model):
		return ['', 'model', 'model.blocks'] + ['model.blocks.' + str(i) for i in range(len(model.blocks))]

	#Save model architecture
	@classmethod
	def cache_model(cls, model, save_dir):
		try:
			os.makedirs(save_dir, exist_ok=True)
		except Exception as error:
			print(error)

		with open(cls.get_config_file(save_dir), "w") as file:
			json.dump(model.default_cfg, file)

	#Create empty model
	@classmethod
	def create_model(cls, save_dir):
		with open(cls.get_config_file(save_dir), "r") as file:
			config = json.load(file)
		model = timm.create_model(config['architecture'] + '.' + config['tag'], pretrained=True)
		return model
